// The per-token kernels around the matmuls: the full-attention head norms and rotary embedding, the
// linear-attention causal conv, row gathers and scatters, and the vision tower's LayerNorm, GELU, rotary and
// position embedding.

#include "common.cuh"

// --- text: q/k head RMSNorm (1 + w) then partial multimodal rotary -----------------------------------------
//
// q_proj output [N, 24, 512] (query | gate per head), k_proj output [N, 4, 256]. Writes q [N, 24, 256] and k
// [N, 4, 256] (bf16), normed and rotated on the first `rot` = 64 channels. positions: [N][3] int32 (t, h, w).
// One block per token, 256 threads (one per head channel), looping over the 28 heads.
extern "C" __global__ void head_norm_rotary(const bf16* __restrict__ qg, const bf16* __restrict__ kin, const bf16* __restrict__ qw,
                                            const bf16* __restrict__ kw, const int32_t* __restrict__ positions,
                                            bf16* __restrict__ q, bf16* __restrict__ k, uint32_t heads, uint32_t kv_heads,
                                            uint32_t rot, float theta, float eps) {
    __shared__ float scratch[32];
    __shared__ float head[256];
    const size_t tok = blockIdx.x;
    const int d = threadIdx.x; // 0..255
    const int32_t pos_t = positions[tok * 3], pos_h = positions[tok * 3 + 1], pos_w = positions[tok * 3 + 2];
    // Rotary angle of channel d (d < rot): slot j = d % (rot/2); axis by j % 3 (mrope_section 11, 11, 10 interleaved).
    const int half = rot / 2;
    const int j = d % half;
    const int axis = j % 3;
    const int32_t pos = axis == 0 ? pos_t : (axis == 1 ? pos_h : pos_w);
    const float inv_freq = powf(theta, -(float)(2 * j) / (float)rot);
    const float angle = (float)pos * inv_freq;
    const float cos_b = round_bf16(cosf(angle)), sin_b = round_bf16(sinf(angle));

    for (uint32_t hd = 0; hd < heads + kv_heads; hd++) {
        const bool is_q = hd < heads;
        const bf16* src = is_q ? qg + tok * heads * 512 + hd * 512 : kin + tok * kv_heads * 256 + (hd - heads) * 256;
        const bf16* w = is_q ? qw : kw;
        const float x = bf2f(src[d]);
        const float ss = block_sum(x * x, scratch);
        const float n = round_bf16(x * rsqrtf(ss / 256.f + eps) * (1.f + bf2f(w[d])));
        head[d] = n;
        __syncthreads();
        float outv = n;
        if (d < (int)rot) {
            // rotate_half pairs channel d with d + half (negated) or d - half.
            const float partner = d < half ? -head[d + half] : head[d - half];
            outv = round_bf16(round_bf16(n * cos_b) + round_bf16(partner * sin_b));
        }
        bf16* dst = is_q ? q + tok * heads * 256 + hd * 256 : k + tok * kv_heads * 256 + (hd - heads) * 256;
        dst[d] = f2bf(outv);
        __syncthreads();
    }
}

// --- text: depthwise causal conv1d (kernel 4) + SiLU over packed sequences --------------------------------
//
// x, y: [N, C] bf16; w: [C, 4] bf16 (conv1d.weight [C, 1, 4]); seq_of: [N] int32 start token of each token's
// sequence. y[t][c] = silu(sum_j w[c][j] * x[t - 3 + j][c]) with x = 0 before the sequence start. Sums in f32.
// The same conv reading the FP8 matmul's f32 accumulator [n, c] with its row and column scales (the input is
// bf16(acc * row_scale * col_scale), as `scale_rows_cols` would have written). One thread per 8 channels of 8
// consecutive tokens, with the last 4 rows in registers, so each row is read once rather than four times.
extern "C" __global__ void conv1d_silu_scaled(const float* __restrict__ acc, const float* __restrict__ row_scale,
                                              const float* __restrict__ col_scale, const bf16* __restrict__ w,
                                              const int32_t* __restrict__ seq_of, bf16* __restrict__ y, uint32_t n, uint32_t c) {
    const uint32_t item = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t groups = (n + 7) / 8, cg = c / 8;
    if (item >= groups * cg) return;
    const uint32_t tok_begin = (item / cg) * 8, ch = (item % cg) * 8;
    float wv[4][8], cs[8];
#pragma unroll
    for (int e = 0; e < 8; e++) {
#pragma unroll
        for (int j = 0; j < 4; j++) wv[j][e] = bf2f(w[(size_t)(ch + e) * 4 + j]);
    }
    {
        const float4 c0 = *reinterpret_cast<const float4*>(col_scale + ch);
        const float4 c1 = *reinterpret_cast<const float4*>(col_scale + ch + 4);
        cs[0] = c0.x; cs[1] = c0.y; cs[2] = c0.z; cs[3] = c0.w; cs[4] = c1.x; cs[5] = c1.y; cs[6] = c1.z; cs[7] = c1.w;
    }
    // The last four rows in registers; tok_begin is a multiple of 8, so row r lives in ring[r & 3] with the
    // index known at compile time once the token loop is unrolled.
    float ring[4][8];
    auto load_row = [&](int32_t tok, float* r) {
        if (tok < 0) {
#pragma unroll
            for (int e = 0; e < 8; e++) r[e] = 0.f;
            return;
        }
        const float rs = row_scale[tok];
        const float4 a0 = *reinterpret_cast<const float4*>(acc + (size_t)tok * c + ch);
        const float4 a1 = *reinterpret_cast<const float4*>(acc + (size_t)tok * c + ch + 4);
        const float a[8] = {a0.x, a0.y, a0.z, a0.w, a1.x, a1.y, a1.z, a1.w};
#pragma unroll
        for (int e = 0; e < 8; e++) r[e] = round_bf16(a[e] * rs * cs[e]);
    };
    load_row((int32_t)tok_begin - 3, ring[1]);
    load_row((int32_t)tok_begin - 2, ring[2]);
    load_row((int32_t)tok_begin - 1, ring[3]);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t tok = tok_begin + i;
        if (tok >= n) break;
        load_row((int32_t)tok, ring[i & 3]);
        const int32_t start = seq_of[tok];
        float out[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int32_t src = (int32_t)tok - 3 + j;
            if (src < start) continue;
            const float* r = ring[(i + 1 + j) & 3]; // (tok - 3 + j) & 3 with tok_begin a multiple of 4
#pragma unroll
            for (int e = 0; e < 8; e++) out[e] += r[e] * wv[j][e];
        }
        uint4 packed;
        bf16* op = reinterpret_cast<bf16*>(&packed);
#pragma unroll
        for (int e = 0; e < 8; e++) op[e] = f2bf(silu(round_bf16(out[e])));
        *reinterpret_cast<uint4*>(y + (size_t)tok * c + ch) = packed;
    }
}

// The same conv with each token's three predecessors given by index (`sources` [n][3]: the tokens at -3, -2 and
// -1, or -1 for none). Locate's isolated questions use it: a question block's first tokens follow the context's
// last ones, not the previous block's.
extern "C" __global__ void conv1d_silu_indexed(const bf16* __restrict__ x, const bf16* __restrict__ w, const int32_t* __restrict__ sources,
                                               bf16* __restrict__ y, uint32_t n, uint32_t c) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)n * c) return;
    const uint32_t tok = i / c, ch = i % c;
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int32_t src = j == 3 ? (int32_t)tok : sources[(size_t)tok * 3 + j];
        if (src < 0) continue;
        const uint4 raw = *reinterpret_cast<const uint4*>(x + (size_t)src * c + ch);
        const bf16* xp = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += bf2f(xp[e]) * bf2f(w[(size_t)(ch + e) * 4 + j]);
    }
    uint4 out;
    bf16* op = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; e++) op[e] = f2bf(silu(round_bf16(acc[e])));
    *reinterpret_cast<uint4*>(y + i) = out;
}

extern "C" __global__ void conv1d_silu(const bf16* __restrict__ x, const bf16* __restrict__ w, const int32_t* __restrict__ seq_of,
                                       bf16* __restrict__ y, uint32_t n, uint32_t c) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)n * c) return;
    const uint32_t tok = i / c, ch = i % c;
    const int32_t start = seq_of[tok];
    float acc[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int32_t src = (int32_t)tok - 3 + j;
        if (src < start) continue;
        const uint4 raw = *reinterpret_cast<const uint4*>(x + (size_t)src * c + ch);
        const bf16* xp = reinterpret_cast<const bf16*>(&raw);
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += bf2f(xp[e]) * bf2f(w[(size_t)(ch + e) * 4 + j]);
    }
    uint4 out;
    bf16* op = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int e = 0; e < 8; e++) op[e] = f2bf(silu(round_bf16(acc[e])));
    *reinterpret_cast<uint4*>(y + i) = out;
}

// --- rows: dst[i] = src[idx[i]] and dst[idx[i]] = src[i], rows of `width` bf16 (multiple of 8) ----------------
extern "C" __global__ void gather_rows(const bf16* __restrict__ src, const int32_t* __restrict__ idx, bf16* __restrict__ dst,
                                       uint32_t rows, uint32_t width) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)rows * width) return;
    const uint32_t r = i / width, c = i % width;
    *reinterpret_cast<uint4*>(dst + i) = *reinterpret_cast<const uint4*>(src + (size_t)idx[r] * width + c);
}

extern "C" __global__ void scatter_rows(const bf16* __restrict__ src, const int32_t* __restrict__ idx, bf16* __restrict__ dst,
                                        uint32_t rows, uint32_t width) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)rows * width) return;
    const uint32_t r = i / width, c = i % width;
    *reinterpret_cast<uint4*>(dst + (size_t)idx[r] * width + c) = *reinterpret_cast<const uint4*>(src + i);
}

// --- vision: LayerNorm with bias (f32 inside, bf16 out), optionally adding a residual first -------------------
// x: [rows, n]; one block per row, 256 threads, n <= 4096.
extern "C" __global__ void layer_norm(const bf16* __restrict__ x, const bf16* __restrict__ w, const bf16* __restrict__ b,
                                      bf16* __restrict__ y, uint32_t n, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float v[16];
    int count = 0;
    float sum = 0.f;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) { v[count] = bf2f(x[row * n + i]); sum += v[count]; count++; }
    const float mean = block_sum(sum, scratch) / (float)n;
    float var = 0.f;
    for (int c = 0; c < count; c++) { const float d = v[c] - mean; var += d * d; }
    var = block_sum(var, scratch) / (float)n;
    const float inv = rsqrtf(var + eps);
    count = 0;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        y[row * n + i] = f2bf((v[count] - mean) * inv * bf2f(w[i]) + bf2f(b[i]));
        count++;
    }
}

// In-place GELU over `total` bf16 values: tanh approximation (`tanh` != 0) or the exact erf form.
extern "C" __global__ void gelu(bf16* __restrict__ x, uint32_t total, uint32_t tanh_form) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= total) return;
    uint4 raw = *reinterpret_cast<uint4*>(x + i);
    bf16* p = reinterpret_cast<bf16*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; e++) {
        const float v = bf2f(p[e]);
        p[e] = f2bf(tanh_form ? gelu_tanh(v) : 0.5f * v * (1.f + erff(v * 0.7071067811865476f)));
    }
    *reinterpret_cast<uint4*>(x + i) = raw;
}

// --- vision: axial 2-D rotary on q and k inside the qkv buffer [N, 3, heads, 72], in place ------------------
// positions: [N][2] int32 (row, col of the patch). 36 frequency slots: slot s < 18 uses the row position,
// s >= 18 the column, inv_freq[s % 18] = theta^(-2 (s % 18) / 36); channel c pairs with c + 36. f32 math, bf16 out.
extern "C" __global__ void vision_rotary(bf16* __restrict__ qkv, const int32_t* __restrict__ positions, uint32_t heads,
                                         float theta) {
    const uint32_t tok = blockIdx.x;
    const int c = threadIdx.x; // 0..35: the pair (c, c + 36)
    if (c >= 36) return;
    const int slot = c, f = slot % 18;
    const int32_t pos = slot < 18 ? positions[tok * 2] : positions[tok * 2 + 1];
    const float angle = (float)pos * powf(theta, -(float)(2 * f) / 36.f);
    const float cs = cosf(angle), sn = sinf(angle);
    for (uint32_t which = 0; which < 2; which++) { // q then k
        for (uint32_t h = 0; h < heads; h++) {
            bf16* p = qkv + (size_t)tok * 3 * heads * 72 + which * heads * 72 + h * 72;
            const float x1 = bf2f(p[c]), x2 = bf2f(p[c + 36]);
            p[c] = f2bf(x1 * cs - x2 * sn);
            p[c + 36] = f2bf(x2 * cs + x1 * sn);
        }
    }
}

// --- vision: add the bilinearly resampled learned position embedding ----------------------------------------
// h: [N, 1152] bf16 += sum of 4 taps; taps: [N][4] int32 indices into pos [2304, 1152], weights: [N][4] f32.
extern "C" __global__ void vision_pos_embed(bf16* __restrict__ h, const bf16* __restrict__ pos, const int32_t* __restrict__ taps,
                                            const float* __restrict__ weights, uint32_t rows, uint32_t n) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)rows * n) return;
    const uint32_t tok = i / n, c = i % n;
    float acc[8];
    uint4 raw = *reinterpret_cast<uint4*>(h + i);
    bf16* hp = reinterpret_cast<bf16*>(&raw);
#pragma unroll
    for (int e = 0; e < 8; e++) acc[e] = 0.f;
#pragma unroll
    for (int tap = 0; tap < 4; tap++) {
        const float wgt = weights[tok * 4 + tap];
        const uint4 praw = *reinterpret_cast<const uint4*>(pos + (size_t)taps[tok * 4 + tap] * n + c);
        const bf16* pp = reinterpret_cast<const bf16*>(&praw);
#pragma unroll
        for (int e = 0; e < 8; e++) acc[e] += wgt * bf2f(pp[e]);
    }
#pragma unroll
    for (int e = 0; e < 8; e++) hp[e] = f2bf(bf2f(hp[e]) + round_bf16(acc[e]));
    *reinterpret_cast<uint4*>(h + i) = raw;
}
