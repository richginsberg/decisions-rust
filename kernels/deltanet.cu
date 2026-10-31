// The gated delta rule of Qwen3.5's linear-attention layers, as transformers' chunked reference
// (torch_chunk_gated_delta_rule, chunk size 64, fp32) computes it, over packed sequences.
//
// Per value head h (48) and token t, with q, k from key head h / 3, all in fp32:
//   q_t = l2norm(q_t) / sqrt(128),  k_t = l2norm(k_t),  beta_t = sigmoid(b_t),
//   g_t = -exp(A_log_h) * softplus(a_t + dt_bias_h)
//   S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t k_t v_t^T,   o_t = S_t^T q_t
//
// Two kernels:
// - `delta_chunk` (one block per chunk of 64 tokens per head): the within-chunk quantities that don't depend on
//   the state: cumulative decays, the unit-lower-triangular solves giving `new_values` and `k_cumdecay`, and the
//   intra-chunk attention matrix. l2norm and the decays are recomputed from the conv output rather than stored.
// - `delta_state` (one block per sequence per head per 64-column half of the state): walks the chunks in order
//   with S (128 x 64 per block) in registers, producing the outputs.
//
// Inputs come straight from the conv output [N, 10240] bf16 (q | k | v) and the a/b projection output [N, 96]
// bf16 (a | b). Padding tokens of a chunk (beyond the sequence) act as zeros: q = k = v = 0, beta = 0, g = 0.

#include "common.cuh"

#define CHUNK 64
#define DK 128
#define DV 128
#define KV_HEADS 16
#define V_HEADS 48
#define QKV_STRIDE 10240
#define AB_STRIDE 96

struct DeltaChunk {
    uint32_t seq_start; // first token of the sequence
    uint32_t seq_len;
    uint32_t chunk_start; // first token of this chunk, relative to seq_start
    uint32_t seq_index;   // which sequence (for delta_state's chunk ranges)
};

struct DeltaSeq {
    uint32_t seq_start;
    uint32_t seq_len;
    uint32_t first_chunk; // index into the chunk table
    uint32_t chunks;
};

__device__ __forceinline__ float softplus(float v) { return v > 20.f ? v : log1pf(__expf(v)); }

// g_t for token `tok` (absolute) and head h; 0 beyond the sequence.
__device__ __forceinline__ float decay_of(const bf16* __restrict__ ab, const float* __restrict__ a_log_exp,
                                          const float* __restrict__ dt_bias, uint32_t tok, uint32_t h) {
    const float a = bf2f(ab[(size_t)tok * AB_STRIDE + h]);
    return -a_log_exp[h] * softplus(a + dt_bias[h]);
}

__device__ __forceinline__ float beta_of(const bf16* __restrict__ ab, uint32_t tok, uint32_t h) {
    return sigmoid(bf2f(ab[(size_t)tok * AB_STRIDE + V_HEADS + h]));
}

// Load a token's q (or k) vector of key head kh, l2-normalized, into `v` (128 values); the caller multiplies
// q by 1/sqrt(128). Each of 32 lanes holds 4 values: lane l holds dims 4l..4l+3.
__device__ __forceinline__ void load_l2norm_lane(const bf16* __restrict__ qkv, uint32_t tok, uint32_t offset,
                                                 uint32_t kh, float* v4) {
    const bf16* p = qkv + (size_t)tok * QKV_STRIDE + offset + kh * DK + (threadIdx.x & 31) * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(p);
    const bf16* pb = reinterpret_cast<const bf16*>(&raw);
    float ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { v4[i] = bf2f(pb[i]); ss += v4[i] * v4[i]; }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss + 1e-6f);
#pragma unroll
    for (int i = 0; i < 4; i++) v4[i] *= inv;
}

// Block: 256 threads = 8 warps. Smem: k [64][128] f32, ut [64][64] f32 (strict lower part used), attn [64][64].
#define DELTA_CHUNK_SMEM ((CHUNK * DK + 2 * CHUNK * CHUNK) * 4)

extern "C" __global__ void delta_chunk(const DeltaChunk* __restrict__ chunks, const bf16* __restrict__ qkv,
                                       const bf16* __restrict__ ab, const float* __restrict__ a_log_exp,
                                       const float* __restrict__ dt_bias, float* __restrict__ new_values,
                                       float* __restrict__ k_cumdecay, float* __restrict__ intra,
                                       float* __restrict__ cum_decay) {
    extern __shared__ float smem[];
    float* ks = smem;                       // [64][128]
    float* ut = ks + CHUNK * DK;            // [64][64]
    float* at = ut + CHUNK * CHUNK;         // [64][64]
    __shared__ float cum[CHUNK];
    __shared__ float beta[CHUNK];

    const DeltaChunk c = chunks[blockIdx.x];
    const uint32_t h = blockIdx.y, kh = h / 3;
    const uint32_t remaining = c.seq_len - c.chunk_start; // tokens of this chunk inside the sequence
    const uint32_t tok0 = c.seq_start + c.chunk_start;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    // Decays and betas, cumulative decay (a scan over 64 values by one warp).
    if (warp == 0) {
        float g0 = 0.f, g1 = 0.f, b0 = 0.f, b1 = 0.f;
        if (lane < remaining) { g0 = decay_of(ab, a_log_exp, dt_bias, tok0 + lane, h); b0 = beta_of(ab, tok0 + lane, h); }
        if (lane + 32 < remaining) { g1 = decay_of(ab, a_log_exp, dt_bias, tok0 + lane + 32, h); b1 = beta_of(ab, tok0 + lane + 32, h); }
        float s0 = g0, s1 = g1;
#pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const float t0 = __shfl_up_sync(0xffffffffu, s0, off), t1 = __shfl_up_sync(0xffffffffu, s1, off);
            if (lane >= off) { s0 += t0; s1 += t1; }
        }
        const float total0 = __shfl_sync(0xffffffffu, s0, 31);
        cum[lane] = s0;
        cum[lane + 32] = s1 + total0;
        beta[lane] = b0;
        beta[lane + 32] = b1;
    }
    // k rows (l2-normalized) into smem: warp w loads rows w, w+8, ...
    for (int row = warp; row < CHUNK; row += 8) {
        float v4[4] = {0.f, 0.f, 0.f, 0.f};
        if ((uint32_t)row < remaining) load_l2norm_lane(qkv, tok0 + row, 2048, kh, v4);
#pragma unroll
        for (int i = 0; i < 4; i++) ks[row * DK + lane * 4 + i] = v4[i];
    }
    __syncthreads();
    // cum_decay out: [N][48]
    for (int i = threadIdx.x; i < CHUNK; i += blockDim.x)
        if ((uint32_t)i < remaining) cum_decay[(size_t)(tok0 + i) * V_HEADS + h] = cum[i];

    // ut[i][j] = beta_i (k_i . k_j) exp(cum_i - cum_j) for j < i; attn[i][j] = (q_i . k_j) exp(cum_i - cum_j) for j <= i.
    // Each warp takes rows i = warp, warp + 8, ...; lanes split the columns j.
    for (int i = warp; i < CHUNK; i += 8) {
        float q4[4] = {0.f, 0.f, 0.f, 0.f};
        if ((uint32_t)i < remaining) load_l2norm_lane(qkv, tok0 + i, 0, kh, q4);
        const float qscale = rsqrtf((float)DK);
        // For each column j, every lane contributes its 4 dims of row i and row j, then the warp reduces.
        for (int j = 0; j < CHUNK; j++) {
            float kk = 0.f, qk = 0.f;
#pragma unroll
            for (int d = 0; d < 4; d++) {
                const float kj = ks[j * DK + lane * 4 + d];
                kk += ks[i * DK + lane * 4 + d] * kj;
                qk += q4[d] * kj;
            }
            kk = warp_sum(kk);
            qk = warp_sum(qk);
            if (lane == 0) {
                const float dec = (j <= i) ? __expf(cum[i] - cum[j]) : 0.f;
                ut[i * CHUNK + j] = (j < i) ? beta[i] * kk * dec : 0.f;
                at[i * CHUNK + j] = (j <= i) ? qk * qscale * dec : 0.f;
            }
        }
    }
    __syncthreads();

    // Forward substitution with the unit lower triangular (I + ut): threads 0..127 solve new_values (v beta),
    // threads 128..255 solve k_cumdecay (k beta exp(cum)); each thread owns one column.
    {
        const int col = threadIdx.x & 127;
        const bool kc = threadIdx.x >= 128;
        float x[CHUNK];
        for (int i = 0; i < CHUNK; i++) {
            float rhs = 0.f;
            if ((uint32_t)i < remaining) {
                if (kc) rhs = ks[i * DK + col] * beta[i] * __expf(cum[i]);
                else rhs = bf2f(qkv[(size_t)(tok0 + i) * QKV_STRIDE + 4096 + h * DV + col]) * beta[i];
            }
            float acc = rhs;
            for (int j = 0; j < i; j++) acc -= ut[i * CHUNK + j] * x[j];
            x[i] = acc;
        }
        float* dst = kc ? k_cumdecay : new_values;
        for (int i = 0; i < CHUNK; i++)
            if ((uint32_t)i < remaining) dst[((size_t)(tok0 + i) * V_HEADS + h) * DV + col] = x[i];
    }
    // The intra-chunk attention matrix: [chunk][48][64][64].
    for (int i = threadIdx.x; i < CHUNK * CHUNK; i += blockDim.x)
        intra[((size_t)blockIdx.x * V_HEADS + h) * CHUNK * CHUNK + i] = at[i];
}

// Block: 128 threads; column vd = half * 64 + t / 2; k-dims (t & 1) * 64 .. + 64 of S held in registers.
#define DELTA_STATE_SMEM ((CHUNK * DK + CHUNK * CHUNK + CHUNK) * 4)

extern "C" __global__ void delta_state(const DeltaSeq* __restrict__ seqs, const DeltaChunk* __restrict__ chunks,
                                       const bf16* __restrict__ qkv, const float* __restrict__ new_values,
                                       const float* __restrict__ k_cumdecay, const float* __restrict__ intra,
                                       const float* __restrict__ cum_decay, bf16* __restrict__ out) {
    extern __shared__ float smem[];
    float* buf = smem;                 // [64][128]: kc, then q, then k of the chunk
    float* at = buf + CHUNK * DK;      // [64][64]
    float* cum = at + CHUNK * CHUNK;   // [64]
    const DeltaSeq s = seqs[blockIdx.x];
    const uint32_t h = blockIdx.y, kh = h / 3, half = blockIdx.z;
    const int t = threadIdx.x, col = half * 64 + (t >> 1), k0 = (t & 1) * 64;
    float S[64];
#pragma unroll
    for (int i = 0; i < 64; i++) S[i] = 0.f;

    for (uint32_t ci = 0; ci < s.chunks; ci++) {
        const DeltaChunk c = chunks[s.first_chunk + ci];
        const uint32_t tok0 = c.seq_start + c.chunk_start, remaining = c.seq_len - c.chunk_start;
        const uint32_t len = min(remaining, (uint32_t)CHUNK);
        // cum decays and the intra matrix.
        __syncthreads();
        for (int i = t; i < CHUNK; i += 128) cum[i] = (uint32_t)i < len ? cum_decay[(size_t)(tok0 + i) * V_HEADS + h] : 0.f;
        for (int i = t; i < CHUNK * CHUNK; i += 128) at[i] = intra[((size_t)(s.first_chunk + ci) * V_HEADS + h) * CHUNK * CHUNK + i];
        // kc rows of the chunk.
        for (int i = t; i < CHUNK * DK; i += 128) {
            const int row = i / DK, d = i % DK;
            buf[i] = (uint32_t)row < len ? k_cumdecay[((size_t)(tok0 + row) * V_HEADS + h) * DV + d] : 0.f;
        }
        __syncthreads();
        // v_new[j][col] = new_values[j][col] - sum_d kc[j][d] S[d][col]
        float vnew[CHUNK];
        for (int j = 0; j < CHUNK; j++) {
            float acc = 0.f;
#pragma unroll 16
            for (int d = 0; d < 64; d++) acc += buf[j * DK + k0 + d] * S[d];
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            const float nv = (uint32_t)j < len ? new_values[((size_t)(tok0 + j) * V_HEADS + h) * DV + col] : 0.f;
            vnew[j] = nv - acc;
        }
        // q (l2-normalized, scaled, decayed by exp(cum_j)) rows into buf.
        __syncthreads();
        for (int row = t >> 5; row < CHUNK; row += 4) {
            float v4[4] = {0.f, 0.f, 0.f, 0.f};
            if ((uint32_t)row < len) load_l2norm_lane(qkv, tok0 + row, 0, kh, v4);
            const float f = rsqrtf((float)DK) * __expf(cum[row]);
            const int lane = t & 31;
#pragma unroll
            for (int i = 0; i < 4; i++) buf[row * DK + lane * 4 + i] = v4[i] * f;
        }
        __syncthreads();
        // out[j][col] = sum_d q[j][d] S[d][col] + sum_i attn[j][i] vnew[i][col]
        for (int j = 0; j < CHUNK; j++) {
            float acc = 0.f;
#pragma unroll 16
            for (int d = 0; d < 64; d++) acc += buf[j * DK + k0 + d] * S[d];
            acc += __shfl_xor_sync(0xffffffffu, acc, 1);
            float ia = 0.f;
            for (int i = 0; i <= j; i++) ia += at[j * CHUNK + i] * vnew[i];
            if ((uint32_t)j < len && (t & 1) == 0)
                out[((size_t)(tok0 + j) * V_HEADS + h) * DV + col] = f2bf(acc + ia);
        }
        // k (l2-normalized, decayed by exp(cum_last - cum_j)) rows into buf; then S = S * exp(cum_last) + k^T vnew.
        __syncthreads();
        const float cum_last = cum[len > 0 ? len - 1 : 0];
        for (int row = t >> 5; row < CHUNK; row += 4) {
            float v4[4] = {0.f, 0.f, 0.f, 0.f};
            if ((uint32_t)row < len) load_l2norm_lane(qkv, tok0 + row, 2048, kh, v4);
            const float f = __expf(cum_last - cum[row]);
            const int lane = t & 31;
#pragma unroll
            for (int i = 0; i < 4; i++) buf[row * DK + lane * 4 + i] = v4[i] * f;
        }
        __syncthreads();
        const float chunk_decay = __expf(cum_last);
#pragma unroll 8
        for (int d = 0; d < 64; d++) {
            float acc = 0.f;
            for (int j = 0; j < CHUNK; j++) acc += buf[j * DK + k0 + d] * vnew[j];
            S[d] = S[d] * chunk_decay + acc;
        }
    }
}

// The gated RMSNorm after the delta rule: per (token, head), y = bf16(w * bf16(o * rsqrt(mean(o^2) + eps))) * silu(z)
// -> bf16. o, z: [N, 48, 128] bf16 (z from the in_proj_z output [N, 6144]); y: [N, 6144] bf16. One warp per (token, head).
extern "C" __global__ void gated_norm(const bf16* __restrict__ o, const bf16* __restrict__ z, const bf16* __restrict__ w,
                                      bf16* __restrict__ y, uint32_t rows, float eps) {
    const uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (item >= rows * V_HEADS) return;
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)item * DV + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(o + base);
    const uint2 zraw = *reinterpret_cast<const uint2*>(z + base);
    const uint2 wraw = *reinterpret_cast<const uint2*>(w + lane * 4);
    const bf16* op = reinterpret_cast<const bf16*>(&raw);
    const bf16* zp = reinterpret_cast<const bf16*>(&zraw);
    const bf16* wp = reinterpret_cast<const bf16*>(&wraw);
    float v[4], ss = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) { v[i] = bf2f(op[i]); ss += v[i] * v[i]; }
    ss = warp_sum(ss);
    const float inv = rsqrtf(ss / (float)DV + eps);
    uint2 outraw;
    bf16* outp = reinterpret_cast<bf16*>(&outraw);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float n = round_bf16(v[i] * inv);
        const float g = round_bf16(bf2f(wp[i]) * n);
        outp[i] = f2bf(g * silu(bf2f(zp[i])));
    }
    *reinterpret_cast<uint2*>(y + base) = outraw;
}
