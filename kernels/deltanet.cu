// The gated delta rule of Qwen3.5's linear-attention layers over packed sequences, on the tensor cores, in the
// structure of flash-linear-attention's chunk_gated_delta_rule (chunk size 64), which the Python server runs:
//
// Per value head h (48) and token t, with q, k from key head h / 3:
//   q_t = l2norm(q_t) / sqrt(128),  k_t = l2norm(k_t),  beta_t = sigmoid(b_t),
//   g_t = -exp(A_log_h) * softplus(a_t + dt_bias_h)
//   S_t = exp(g_t) S_{t-1} (I - beta_t k_t k_t^T) + beta_t k_t v_t^T,   o_t = S_t^T q_t
//
// Three kernels, with cum = the cumulative sum of g inside the chunk:
// - `delta_chunk` (one block per chunk per head, parallel): the within-chunk quantities that don't depend on
//   the state. A = (k k^T) ⊙ beta_i exp(cum_i - cum_j), strictly lower; u = (I + A)^-1 (v beta) and
//   w = (I + A)^-1 (k beta exp(cum)) by forward substitution in fp32; the masked intra-chunk attention
//   at = (q k^T) ⊙ exp(cum_i - cum_j) / sqrt(128) for j <= i.
// - `delta_state` (four blocks per sequence per head, each a quarter of the state's rows, sequential over the
//   chunks): keeps S^T (128 x 128) in
//   registers as mma accumulators, stores it per chunk, forms v_new = u - w S and advances
//   S = exp(cum_last) S + (k exp(cum_last - cum))^T v_new.
// - `delta_out` (four blocks per chunk per head, a quarter of the value dims each, parallel):
//   o = (q exp(cum) / sqrt(128)) S_chunk + at v_new.
//
// Products run in bf16 with fp32 accumulation, as fla's tl.dot does; the state and v_new are stored in bf16
// between kernels, as fla stores them.
//
// Inputs come straight from the conv output [N, 10240] bf16 (q | k | v) and the a/b projection output [N, 96]
// bf16 (a | b). Padding tokens of a chunk (beyond the sequence) act as zeros: q = k = v = 0, beta = 0, g = 0.
//
// Fragment layouts (g = lane / 4, t = lane % 4): see attention.cu. An operand stored [rows][k] row-major in
// shared memory serves as A (rows = m) or as B (rows = n), so every product below stores its B operand
// transposed, [n][k].

#include "common.cuh"

#define CHUNK 64
#define DK 128
#define DV 128
#define V_HEADS 48
#define QKV_STRIDE 10240
#define AB_STRIDE 96
// Shared-memory row strides (bf16 elements), padded so fragment loads are bank-conflict free.
#define QS 136
#define TS 72

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

// Load a token's q (or k) vector of key head kh, l2-normalized, into `v4`: lane l holds dims 4l..4l+3.
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

// Lane l's 4 values of a row, scaled, as 4 bf16 at `row + 4l`.
__device__ __forceinline__ void store_lane4(bf16* row, const float* v4, float scale) {
    uint2 packed;
    packed.x = pack_bf16(v4[0] * scale, v4[1] * scale);
    packed.y = pack_bf16(v4[2] * scale, v4[3] * scale);
    *reinterpret_cast<uint2*>(row + (threadIdx.x & 31) * 4) = packed;
}

// Copy `rows` rows of `width` bf16 (a multiple of 8) from a global row-major buffer into shared memory with
// row stride `stride`; rows at or beyond `valid` are zero.
__device__ __forceinline__ void stage_rows(bf16* smem, int stride, const bf16* __restrict__ src, size_t src_stride,
                                           int rows, int width, int valid) {
    const int per_row = width / 8;
    for (int i = threadIdx.x; i < rows * per_row; i += blockDim.x) {
        const int row = i / per_row, c8 = (i % per_row) * 8;
        uint4 v = make_uint4(0u, 0u, 0u, 0u);
        if (row < valid) v = *reinterpret_cast<const uint4*>(src + (size_t)row * src_stride + c8);
        *reinterpret_cast<uint4*>(smem + row * stride + c8) = v;
    }
}

// ---------------------------------------------------------------------------------------------------------
// delta_chunk: block (chunk, head) of 256 threads = 8 warps.
// Shared: q [64][QS] bf16, k [64][QS] bf16; A [64][64] f32 reuses q's space once the products are formed, so two
// blocks fit on an SM.
#define DELTA_CHUNK_SMEM (2 * CHUNK * QS * 2)

// Normalize the 64 rows of a staged [64][QS] tile in place (warp per row, lane l owning dims 4l..4l+3), each
// scaled by `scale(row)`; rows at or beyond `valid` become zero.
template <class Scale>
__device__ __forceinline__ void l2norm_rows(bf16* tile, int valid, const Scale& scale) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int row = warp; row < CHUNK; row += blockDim.x / 32) {
        bf16* p = tile + row * QS + lane * 4;
        const uint2 raw = *reinterpret_cast<const uint2*>(p);
        const bf16* pb = reinterpret_cast<const bf16*>(&raw);
        float v4[4], ss = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) { v4[i] = bf2f(pb[i]); ss += v4[i] * v4[i]; }
        ss = warp_sum(ss);
        const float f = row < valid ? rsqrtf(ss + 1e-6f) * scale(row) : 0.f;
        store_lane4(tile + row * QS, v4, f);
    }
}

extern "C" __global__ void delta_chunk(const DeltaChunk* __restrict__ chunks, const bf16* __restrict__ qkv,
                                       const bf16* __restrict__ ab, const float* __restrict__ a_log_exp,
                                       const float* __restrict__ dt_bias, bf16* __restrict__ u_out,
                                       bf16* __restrict__ w_out, bf16* __restrict__ intra,
                                       float* __restrict__ cum_decay, bf16* __restrict__ kt) {
    extern __shared__ __align__(16) unsigned char dsmem[];
    bf16* qs = reinterpret_cast<bf16*>(dsmem);
    bf16* ks = qs + CHUNK * QS;
    float* ut = reinterpret_cast<float*>(qs); // after the products: q is no longer needed
    __shared__ float cum[CHUNK];
    __shared__ float beta[CHUNK];

    const DeltaChunk c = chunks[blockIdx.x];
    if (c.seq_len == 0) return; // a padding entry of a fixed-size chunk table
    const uint32_t h = blockIdx.y, kh = h / 3;
    const uint32_t remaining = c.seq_len - c.chunk_start; // tokens of this chunk inside the sequence
    const uint32_t tok0 = c.seq_start + c.chunk_start;
    const int len = (int)min(remaining, (uint32_t)CHUNK);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;

    // Raw q and k rows of the chunk into shared memory (16-byte loads, all threads), then normalized in place.
    stage_rows(qs, QS, qkv + (size_t)tok0 * QKV_STRIDE + kh * DK, QKV_STRIDE, CHUNK, DK, len);
    stage_rows(ks, QS, qkv + (size_t)tok0 * QKV_STRIDE + 2048 + kh * DK, QKV_STRIDE, CHUNK, DK, len);
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
    __syncthreads();
    l2norm_rows(qs, len, [](int) { return 1.f; });
    l2norm_rows(ks, len, [](int) { return 1.f; });
    __syncthreads();
    for (int i = threadIdx.x; i < CHUNK; i += blockDim.x)
        if ((uint32_t)i < remaining) cum_decay[(size_t)(tok0 + i) * V_HEADS + h] = cum[i];

    // kk = k k^T and qk = q k^T (64 x 64 each): warp w takes row tile w / 2 and column tiles 4 (w % 2) .. + 4.
    const int mt = warp >> 1, nb = (warp & 1) * 4;
    float kk[4][4], qk[4][4];
#pragma unroll
    for (int nt = 0; nt < 4; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) { kk[nt][e] = 0.f; qk[nt][e] = 0.f; }
#pragma unroll
    for (int kstep = 0; kstep < DK / 16; kstep++) {
        uint32_t aq[4], ak[4];
        load_a_frag(aq, qs, QS, mt * 16, kstep * 16, g, t);
        load_a_frag(ak, ks, QS, mt * 16, kstep * 16, g, t);
#pragma unroll
        for (int nt = 0; nt < 4; nt++) {
            uint32_t b[2];
            load_b_frag(b, ks, QS, (nb + nt) * 8, kstep * 16, g, t);
            mma_bf16(kk[nt], ak, b);
            mma_bf16(qk[nt], aq, b);
        }
    }
    __syncthreads(); // every warp is done reading q: its space becomes A
    // A (strictly lower, f32, for the solves) and at (lower incl. diagonal, bf16, for delta_out).
    const float qscale = rsqrtf((float)DK);
    bf16* at_out = intra + ((size_t)blockIdx.x * V_HEADS + h) * CHUNK * CHUNK;
#pragma unroll
    for (int nt = 0; nt < 4; nt++) {
#pragma unroll
        for (int half = 0; half < 2; half++) {
            const int i = mt * 16 + g + half * 8, j0 = (nb + nt) * 8 + 2 * t;
            float a2[2];
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int j = j0 + e;
                const float dec = (j <= i) ? __expf(cum[i] - cum[j]) : 0.f;
                ut[i * CHUNK + j] = (j < i) ? beta[i] * kk[nt][half * 2 + e] * dec : 0.f;
                a2[e] = (j <= i) ? qk[nt][half * 2 + e] * qscale * dec : 0.f;
            }
            *reinterpret_cast<uint32_t*>(at_out + i * CHUNK + j0) = pack_bf16(a2[0], a2[1]);
        }
    }
    __syncthreads();

    // k~^T for delta_state: (k exp(cum_last - cum_j))^T as [128][64] bf16 per (chunk, head), written once here
    // rather than by each of the four state blocks.
    {
        const float cum_last = cum[len - 1];
        bf16* kt_out = kt + ((size_t)blockIdx.x * V_HEADS + h) * DK * CHUNK;
        for (int i = threadIdx.x; i < DK * (CHUNK / 8); i += blockDim.x) {
            const int d = i >> 3, j0 = (i & 7) * 8;
            uint4 packed;
            bf16* pp = reinterpret_cast<bf16*>(&packed);
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const int j = j0 + e;
                pp[e] = f2bf(bf2f(ks[j * QS + d]) * __expf(cum_last - cum[j]));
            }
            *reinterpret_cast<uint4*>(kt_out + d * CHUNK + j0) = packed;
        }
    }

    // Forward substitution with the unit lower triangular (I + A), fully unrolled so x stays in registers:
    // threads 0..127 solve u (column of v beta), threads 128..255 solve w (column of k beta exp(cum)).
    {
        const int col = threadIdx.x & 127;
        const bool kc = threadIdx.x >= 128;
        float x[CHUNK];
#pragma unroll
        for (int i = 0; i < CHUNK; i++) {
            float rhs = 0.f;
            if ((uint32_t)i < remaining) {
                if (kc) rhs = bf2f(ks[i * QS + col]) * beta[i] * __expf(cum[i]);
                else rhs = bf2f(qkv[(size_t)(tok0 + i) * QKV_STRIDE + 4096 + h * DV + col]) * beta[i];
            }
            x[i] = rhs;
        }
#pragma unroll
        for (int i = 1; i < CHUNK; i++) {
            float acc = x[i];
#pragma unroll
            for (int j = 0; j < i; j++) acc -= ut[i * CHUNK + j] * x[j];
            x[i] = acc;
        }
        bf16* dst = kc ? w_out : u_out;
#pragma unroll
        for (int i = 0; i < CHUNK; i++)
            if ((uint32_t)i < remaining) dst[((size_t)(tok0 + i) * V_HEADS + h) * DV + col] = f2bf(x[i]);
    }
}

// ---------------------------------------------------------------------------------------------------------
// delta_state: block (sequence, head, quarter of the state rows) of 64 threads = 2 warps; warp w holds rows
// 32 quarter + 16w .. + 15 of S^T (S^T[n][d]: n = value dim, d = key dim) as 16 mma accumulators. Four blocks
// per head (192 per sequence) rather than one, since the walk is sequential and the GPU has 170 SMs; each
// block re-reads the chunk's w and k from L2.
// Shared: w [64][QS] bf16, v_new^T [32][TS] bf16 (this block's columns), k~^T [128][TS] bf16, cum [64] f32.
#define STATE_SPLIT 4
#define STATE_ROWS (DV / STATE_SPLIT)
#define DELTA_STATE_SMEM (CHUNK * QS * 2 + STATE_ROWS * TS * 2 + DK * TS * 2 + CHUNK * 4)

// CARRY: the walk may start from a saved state (`carry_in`) and may save its final state (`carry_out`), both in
// `carry` [48][128][128] f32. Locate's isolated questions use it: the context's walk saves its state, and every
// question block starts from it.
template <bool CARRY>
__device__ void delta_state_impl(const DeltaSeq* __restrict__ seqs, const DeltaChunk* __restrict__ chunks,
                                 const bf16* __restrict__ kt_in, const bf16* __restrict__ u_in,
                                 const bf16* __restrict__ w_in, const float* __restrict__ cum_decay,
                                 bf16* __restrict__ states, bf16* __restrict__ vnew_out, float* carry,
                                 uint32_t carry_in, uint32_t carry_out) {
    extern __shared__ __align__(16) unsigned char state_smem[];
    bf16* ws = reinterpret_cast<bf16*>(state_smem);  // [64][QS]
    bf16* vT = ws + CHUNK * QS;                  // [32][TS]: v_new^T, this block's 32 value dims
    bf16* kT = vT + STATE_ROWS * TS;             // [128][TS]: (k exp(cum_last - cum))^T
    float* cum = reinterpret_cast<float*>(kT + DK * TS);
    const DeltaSeq s = seqs[blockIdx.x];
    if (s.chunks == 0) return; // a padding entry of a fixed-size sequence table
    const uint32_t h = blockIdx.y;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int nbase = blockIdx.z * STATE_ROWS;   // first value dim of this block
    const int wrow = warp * 16;                  // this warp's first row within the block's 32

    float hT[DK / 8][4];
#pragma unroll
    for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) hT[nt][e] = 0.f;
    // This thread's entries of S^T: rows nbase + wrow + g (and + 8), columns nt * 8 + 2t (and + 1).
    float* carried = CARRY ? carry + ((size_t)h * DV + nbase + wrow + g) * DK + 2 * t : nullptr;
    if (CARRY && carry_in) {
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++) {
            hT[nt][0] = carried[nt * 8];
            hT[nt][1] = carried[nt * 8 + 1];
            hT[nt][2] = carried[8 * DK + nt * 8];
            hT[nt][3] = carried[8 * DK + nt * 8 + 1];
        }
    }

    for (uint32_t ci = 0; ci < s.chunks; ci++) {
        const uint32_t chunk = s.first_chunk + ci;
        const DeltaChunk c = chunks[chunk];
        const uint32_t tok0 = c.seq_start + c.chunk_start, remaining = c.seq_len - c.chunk_start;
        const int len = (int)min(remaining, (uint32_t)CHUNK);
        __syncthreads(); // the previous chunk is done with the shared buffers
        cum[threadIdx.x] = (int)threadIdx.x < len ? cum_decay[(size_t)(tok0 + threadIdx.x) * V_HEADS + h] : 0.f;
        // The state entering this chunk, for delta_out.
        bf16* st = states + ((size_t)chunk * V_HEADS + h) * DV * DK;
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++) {
            const int n = nbase + wrow + g, d = nt * 8 + 2 * t;
            *reinterpret_cast<uint32_t*>(st + n * DK + d) = pack_bf16(hT[nt][0], hT[nt][1]);
            *reinterpret_cast<uint32_t*>(st + (n + 8) * DK + d) = pack_bf16(hT[nt][2], hT[nt][3]);
        }
        stage_rows(ws, QS, w_in + ((size_t)tok0 * V_HEADS + h) * DV, (size_t)V_HEADS * DV, CHUNK, DV, len);
        stage_rows(kT, TS, kt_in + ((size_t)chunk * V_HEADS + h) * DK * CHUNK, CHUNK, DK, CHUNK, DK);
        __syncthreads(); // cum, ws, kT
        const float cum_last = cum[len - 1];

        // v_new = u - w S for this warp's 16 columns, one 8-column tile at a time; S as B fragments from hT
        // (rows g for tile 0, rows g + 8 for tile 1).
#pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            float cf[4][4];
#pragma unroll
            for (int m = 0; m < 4; m++)
#pragma unroll
                for (int e = 0; e < 4; e++) cf[m][e] = 0.f;
#pragma unroll
            for (int kstep = 0; kstep < DK / 16; kstep++) {
                uint32_t b[2];
                b[0] = pack_bf16(hT[2 * kstep][nt * 2], hT[2 * kstep][nt * 2 + 1]);
                b[1] = pack_bf16(hT[2 * kstep + 1][nt * 2], hT[2 * kstep + 1][nt * 2 + 1]);
#pragma unroll
                for (int m = 0; m < 4; m++) {
                    uint32_t a[4];
                    load_a_frag(a, ws, QS, m * 16, kstep * 16, g, t);
                    mma_bf16(cf[m], a, b);
                }
            }
            const int nl = wrow + nt * 8 + 2 * t; // local value dim (within the block's 32)
            const int n0 = nbase + nl;
#pragma unroll
            for (int m = 0; m < 4; m++)
#pragma unroll
                for (int rr = 0; rr < 2; rr++) {
                    const int j = m * 16 + g + rr * 8;
                    float u0 = 0.f, u1 = 0.f;
                    if (j < len) {
                        const uint32_t raw = *reinterpret_cast<const uint32_t*>(u_in + ((size_t)(tok0 + j) * V_HEADS + h) * DV + n0);
                        const bf16* up = reinterpret_cast<const bf16*>(&raw);
                        u0 = bf2f(up[0]);
                        u1 = bf2f(up[1]);
                    }
                    vT[nl * TS + j] = f2bf(u0 - cf[m][rr * 2]);
                    vT[(nl + 1) * TS + j] = f2bf(u1 - cf[m][rr * 2 + 1]);
                }
        }
        __syncthreads(); // vT
        // v_new^T out for delta_out: [chunk][head][128][64], this block's 32 rows.
        bf16* vo = vnew_out + (((size_t)chunk * V_HEADS + h) * DV + nbase) * CHUNK;
        for (int i = threadIdx.x; i < STATE_ROWS * (CHUNK / 8); i += blockDim.x) {
            const int n = i >> 3, c8 = (i & 7) * 8;
            *reinterpret_cast<uint4*>(vo + n * CHUNK + c8) = *reinterpret_cast<const uint4*>(vT + n * TS + c8);
        }
        // S^T = exp(cum_last) S^T + v_new^T k~.
        const float decay = __expf(cum_last);
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) hT[nt][e] *= decay;
#pragma unroll
        for (int kstep = 0; kstep < CHUNK / 16; kstep++) {
            uint32_t a[4];
            load_a_frag(a, vT, TS, wrow, kstep * 16, g, t);
#pragma unroll
            for (int nt = 0; nt < DK / 8; nt++) {
                uint32_t b[2];
                load_b_frag(b, kT, TS, nt * 8, kstep * 16, g, t);
                mma_bf16(hT[nt], a, b);
            }
        }
    }
    if (CARRY && carry_out) {
#pragma unroll
        for (int nt = 0; nt < DK / 8; nt++) {
            carried[nt * 8] = hT[nt][0];
            carried[nt * 8 + 1] = hT[nt][1];
            carried[8 * DK + nt * 8] = hT[nt][2];
            carried[8 * DK + nt * 8 + 1] = hT[nt][3];
        }
    }
}

extern "C" __global__ void delta_state(const DeltaSeq* __restrict__ seqs, const DeltaChunk* __restrict__ chunks,
                                       const bf16* __restrict__ kt_in, const bf16* __restrict__ u_in,
                                       const bf16* __restrict__ w_in, const float* __restrict__ cum_decay,
                                       bf16* __restrict__ states, bf16* __restrict__ vnew_out) {
    delta_state_impl<false>(seqs, chunks, kt_in, u_in, w_in, cum_decay, states, vnew_out, nullptr, 0, 0);
}

// The same walk with a carried state (see delta_state_impl); the flags apply to every sequence of the launch.
extern "C" __global__ void delta_state_carry(const DeltaSeq* __restrict__ seqs, const DeltaChunk* __restrict__ chunks,
                                             const bf16* __restrict__ kt_in, const bf16* __restrict__ u_in,
                                             const bf16* __restrict__ w_in, const float* __restrict__ cum_decay,
                                             bf16* __restrict__ states, bf16* __restrict__ vnew_out, float* carry,
                                             uint32_t carry_in, uint32_t carry_out) {
    delta_state_impl<true>(seqs, chunks, kt_in, u_in, w_in, cum_decay, states, vnew_out, carry, carry_in, carry_out);
}

// ---------------------------------------------------------------------------------------------------------
// delta_out: block (chunk, head, quarter of the value dims) of 128 threads = 4 warps; warp w produces rows
// 16w..16w+15 of the chunk for the block's 32 value dims.
// Shared: S^T rows [32][QS] bf16, v_new^T rows [32][TS] bf16, q~ [64][QS] bf16, at [64][TS] bf16, cum [64] f32.
#define OUT_SPLIT 4
#define OUT_COLS (DV / OUT_SPLIT)
#define DELTA_OUT_SMEM (OUT_COLS * QS * 2 + OUT_COLS * TS * 2 + CHUNK * QS * 2 + CHUNK * TS * 2 + CHUNK * 4)

extern "C" __global__ void delta_out(const DeltaChunk* __restrict__ chunks, const bf16* __restrict__ qkv,
                                     const float* __restrict__ cum_decay, const bf16* __restrict__ intra,
                                     const bf16* __restrict__ states, const bf16* __restrict__ vnew_in,
                                     bf16* __restrict__ out) {
    extern __shared__ __align__(16) unsigned char dsmem[];
    bf16* hs = reinterpret_cast<bf16*>(dsmem); // [32][QS]: this block's rows of S^T
    bf16* vT = hs + OUT_COLS * QS;              // [32][TS]
    bf16* qs = vT + OUT_COLS * TS;              // [64][QS]
    bf16* at = qs + CHUNK * QS;                 // [64][TS]
    float* cum = reinterpret_cast<float*>(at + CHUNK * TS);
    const uint32_t chunk = blockIdx.x;
    const DeltaChunk c = chunks[chunk];
    if (c.seq_len == 0) return; // a padding entry of a fixed-size chunk table
    const uint32_t h = blockIdx.y, kh = h / 3;
    const int nbase = blockIdx.z * OUT_COLS;
    const uint32_t tok0 = c.seq_start + c.chunk_start, remaining = c.seq_len - c.chunk_start;
    const int len = (int)min(remaining, (uint32_t)CHUNK);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;

    if (threadIdx.x < CHUNK) cum[threadIdx.x] = threadIdx.x < len ? cum_decay[(size_t)(tok0 + threadIdx.x) * V_HEADS + h] : 0.f;
    stage_rows(hs, QS, states + (((size_t)chunk * V_HEADS + h) * DV + nbase) * DK, DK, OUT_COLS, DK, OUT_COLS);
    stage_rows(vT, TS, vnew_in + (((size_t)chunk * V_HEADS + h) * DV + nbase) * CHUNK, CHUNK, OUT_COLS, CHUNK, OUT_COLS);
    stage_rows(at, TS, intra + ((size_t)chunk * V_HEADS + h) * CHUNK * CHUNK, CHUNK, CHUNK, CHUNK, CHUNK);
    stage_rows(qs, QS, qkv + (size_t)tok0 * QKV_STRIDE + kh * DK, QKV_STRIDE, CHUNK, DK, len);
    __syncthreads(); // cum, raw q
    const float qscale = rsqrtf((float)DK);
    l2norm_rows(qs, len, [&](int row) { return qscale * __expf(cum[row]); });
    __syncthreads();

    float o[OUT_COLS / 8][4];
#pragma unroll
    for (int nt = 0; nt < OUT_COLS / 8; nt++)
#pragma unroll
        for (int e = 0; e < 4; e++) o[nt][e] = 0.f;
    // o = q~ S: A from q~, B from S^T[n][d].
#pragma unroll
    for (int kstep = 0; kstep < DK / 16; kstep++) {
        uint32_t a[4];
        load_a_frag(a, qs, QS, warp * 16, kstep * 16, g, t);
#pragma unroll
        for (int nt = 0; nt < OUT_COLS / 8; nt++) {
            uint32_t b[2];
            load_b_frag(b, hs, QS, nt * 8, kstep * 16, g, t);
            mma_bf16(o[nt], a, b);
        }
    }
    // o += at v_new: A from at [j][i], B from v_new^T [n][i].
#pragma unroll
    for (int kstep = 0; kstep < CHUNK / 16; kstep++) {
        uint32_t a[4];
        load_a_frag(a, at, TS, warp * 16, kstep * 16, g, t);
#pragma unroll
        for (int nt = 0; nt < OUT_COLS / 8; nt++) {
            uint32_t b[2];
            load_b_frag(b, vT, TS, nt * 8, kstep * 16, g, t);
            mma_bf16(o[nt], a, b);
        }
    }
#pragma unroll
    for (int rr = 0; rr < 2; rr++) {
        const int j = warp * 16 + g + rr * 8;
        if (j >= len) continue;
        bf16* orow = out + ((size_t)(tok0 + j) * V_HEADS + h) * DV + nbase;
#pragma unroll
        for (int nt = 0; nt < OUT_COLS / 8; nt++)
            *reinterpret_cast<uint32_t*>(orow + nt * 8 + 2 * t) = pack_bf16(o[nt][rr * 2], o[nt][rr * 2 + 1]);
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

// The same gated norm with z read from the FP8 matmul's f32 accumulator [N, 6144] and its row and column
// scales (z = bf16(acc * row_scale * col_scale)).
extern "C" __global__ void gated_norm_scaled(const bf16* __restrict__ o, const float* __restrict__ z_acc,
                                             const float* __restrict__ row_scale, const float* __restrict__ col_scale,
                                             const bf16* __restrict__ w, bf16* __restrict__ y, uint32_t rows, float eps) {
    const uint32_t item = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    if (item >= rows * V_HEADS) return;
    const int lane = threadIdx.x & 31;
    const size_t base = (size_t)item * DV + lane * 4;
    const uint32_t tok = item / V_HEADS, ch = (item % V_HEADS) * DV + lane * 4;
    const uint2 raw = *reinterpret_cast<const uint2*>(o + base);
    const uint2 wraw = *reinterpret_cast<const uint2*>(w + lane * 4);
    const float4 za = *reinterpret_cast<const float4*>(z_acc + base);
    const float4 cs = *reinterpret_cast<const float4*>(col_scale + ch);
    const float rs = row_scale[tok];
    const float zv[4] = {round_bf16(za.x * rs * cs.x), round_bf16(za.y * rs * cs.y), round_bf16(za.z * rs * cs.z), round_bf16(za.w * rs * cs.w)};
    const bf16* op = reinterpret_cast<const bf16*>(&raw);
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
        const float nv = round_bf16(v[i] * inv);
        const float gv = round_bf16(bf2f(wp[i]) * nv);
        outp[i] = f2bf(gv * silu(zv[i]));
    }
    *reinterpret_cast<uint2*>(y + base) = outraw;
}
