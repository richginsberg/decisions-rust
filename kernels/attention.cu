// Non-causal attention over packed sequences, on the tensor cores (bf16 mma.sync, f32 accumulation), with
// the online softmax of FlashAttention-2.
//
// One block = 4 warps = 64 queries of one sequence for one head; each warp owns 16 query rows. Keys and values
// of the same sequence stream through shared memory 32 at a time; scores never leave registers: the C-fragment
// layout of S = Q K^T is exactly the A-fragment layout needed for P V, so P is re-packed in place.
//
// Fragment layouts of mma.sync.m16n8k16 (row.col, bf16 -> f32), with g = lane / 4 and t = lane % 4:
//   A (16x16, row-major): a0 = A[g][2t..2t+1], a1 = A[g+8][2t..], a2 = A[g][2t+8..], a3 = A[g+8][2t+8..]
//   B (16x8, "col"):      b0 = B[2t..2t+1][g], b1 = B[2t+8..2t+9][g]
//   C (16x8, f32):        c0,c1 = C[g][2t, 2t+1], c2,c3 = C[g+8][2t, 2t+1]
//
// Text: head_dim 256, 24 query heads over 4 key/value heads (GQA), the output gated by sigmoid(gate).
// Vision: head_dim 72 padded to 80 (zeros beyond 72), 16 heads, no gate.
//
// Buffers are row-major per token with the strides given: q at `q + token * q_stride + head * D`, etc.

#include "common.cuh"

#define KEYS 32
#define QUERIES 64
#define THREADS 128

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

__device__ __forceinline__ void mma_bf16(float* c, const uint32_t* a, const uint32_t* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A sequence's slice of the pack and which 64-query tile this block handles.
struct AttnTile {
    uint32_t seq_start; // first token of the sequence in the pack
    uint32_t seq_len;
    uint32_t q_start;   // first query of the tile, relative to seq_start
};

template <int D, int DREAL, bool GATED>
__device__ void attention_impl(const AttnTile* __restrict__ tiles, const bf16* __restrict__ q, const bf16* __restrict__ k,
                               const bf16* __restrict__ v, const bf16* __restrict__ gate, bf16* __restrict__ out,
                               uint32_t q_stride, uint32_t kv_stride, uint32_t gate_stride, uint32_t gate_head_stride,
                               uint32_t out_stride, uint32_t heads, uint32_t kv_heads, float scale_log2) {
    constexpr int KSTEPS = D / 16;    // k-steps over the head dim for Q K^T
    constexpr int NTILES = D / 8;     // n-tiles over the head dim for P V
    constexpr int KPAD = D + 8;       // smem row stride (elements) of the K tile
    constexpr int VPAD = KEYS + 8;    // smem row stride of the transposed V tile
    __shared__ __align__(16) bf16 ks[KEYS * KPAD];
    __shared__ __align__(16) bf16 vt[D * VPAD];

    const AttnTile tile = tiles[blockIdx.x];
    const uint32_t head = blockIdx.y;
    const uint32_t kv_head = head / (heads / kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t row0 = tile.q_start + warp * 16;
    // A warp whose 16 rows all lie beyond the sequence still runs the loop (it shares the block's barriers);
    // its rows read row 0 and store nothing.

    // Q fragments for this warp's 16 rows, straight from global memory (rows beyond the sequence read row 0:
    // their outputs are never stored).
    uint32_t qa[KSTEPS][4];
    {
        const uint32_t r_lo = min(row0 + g, tile.seq_len - 1), r_hi = min(row0 + g + 8, tile.seq_len - 1);
        const bf16* q_lo = q + (size_t)(tile.seq_start + r_lo) * q_stride + head * DREAL;
        const bf16* q_hi = q + (size_t)(tile.seq_start + r_hi) * q_stride + head * DREAL;
#pragma unroll
        for (int s = 0; s < KSTEPS; s++) {
            const int c0 = s * 16 + 2 * t, c1 = c0 + 8;
            qa[s][0] = c0 < DREAL ? *reinterpret_cast<const uint32_t*>(q_lo + c0) : 0u;
            qa[s][1] = c0 < DREAL ? *reinterpret_cast<const uint32_t*>(q_hi + c0) : 0u;
            qa[s][2] = c1 < DREAL ? *reinterpret_cast<const uint32_t*>(q_lo + c1) : 0u;
            qa[s][3] = c1 < DREAL ? *reinterpret_cast<const uint32_t*>(q_hi + c1) : 0u;
        }
    }

    float o[NTILES][4];
#pragma unroll
    for (int n = 0; n < NTILES; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;

    for (uint32_t key0 = 0; key0 < tile.seq_len; key0 += KEYS) {
        // Load K [KEYS][D] and V^T [D][KEYS] for keys key0.. (zero beyond the sequence).
        __syncthreads();
        for (int i = threadIdx.x; i < KEYS * (D / 8); i += THREADS) {
            const int key = i / (D / 8), d = (i % (D / 8)) * 8;
            uint4 kval = make_uint4(0, 0, 0, 0), vval = make_uint4(0, 0, 0, 0);
            if (key0 + key < tile.seq_len && d < DREAL) {
                const size_t base = (size_t)(tile.seq_start + key0 + key) * kv_stride + kv_head * DREAL + d;
                kval = *reinterpret_cast<const uint4*>(k + base);
                vval = *reinterpret_cast<const uint4*>(v + base);
            }
            *reinterpret_cast<uint4*>(ks + key * KPAD + d) = kval;
            const bf16* vp = reinterpret_cast<const bf16*>(&vval);
#pragma unroll
            for (int j = 0; j < 8; j++) vt[(d + j) * VPAD + key] = vp[j];
        }
        __syncthreads();

        // S = Q K^T for this warp's 16 rows x 32 keys: 4 n-tiles of 8 keys.
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
#pragma unroll
        for (int st = 0; st < KSTEPS; st++) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; n++) {
                uint32_t b[2];
                const bf16* kp = ks + (n * 8 + g) * KPAD + st * 16 + 2 * t;
                b[0] = *reinterpret_cast<const uint32_t*>(kp);
                b[1] = *reinterpret_cast<const uint32_t*>(kp + 8);
                mma_bf16(s[n], qa[st], b);
            }
        }

        // Online softmax over the 32 keys, per row (rows g and g+8 of the warp's 16).
        const uint32_t valid = tile.seq_len - key0; // keys with index < valid are real
        float rmax_lo = -INFINITY, rmax_hi = -INFINITY;
#pragma unroll
        for (int n = 0; n < KEYS / 8; n++) {
            const uint32_t c0 = n * 8 + 2 * t, c1 = c0 + 1;
            s[n][0] = c0 < valid ? s[n][0] * scale_log2 : -INFINITY;
            s[n][1] = c1 < valid ? s[n][1] * scale_log2 : -INFINITY;
            s[n][2] = c0 < valid ? s[n][2] * scale_log2 : -INFINITY;
            s[n][3] = c1 < valid ? s[n][3] * scale_log2 : -INFINITY;
            rmax_lo = fmaxf(rmax_lo, fmaxf(s[n][0], s[n][1]));
            rmax_hi = fmaxf(rmax_hi, fmaxf(s[n][2], s[n][3]));
        }
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffffu, rmax_lo, 1));
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xffffffffu, rmax_lo, 2));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffffu, rmax_hi, 1));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xffffffffu, rmax_hi, 2));
        const float mn_lo = fmaxf(m_lo, rmax_lo), mn_hi = fmaxf(m_hi, rmax_hi);
        const float alpha_lo = exp2f(m_lo - mn_lo), alpha_hi = exp2f(m_hi - mn_hi);
        float sum_lo = 0.f, sum_hi = 0.f;
#pragma unroll
        for (int n = 0; n < KEYS / 8; n++) {
            s[n][0] = exp2f(s[n][0] - mn_lo);
            s[n][1] = exp2f(s[n][1] - mn_lo);
            s[n][2] = exp2f(s[n][2] - mn_hi);
            s[n][3] = exp2f(s[n][3] - mn_hi);
            sum_lo += s[n][0] + s[n][1];
            sum_hi += s[n][2] + s[n][3];
        }
        sum_lo += __shfl_xor_sync(0xffffffffu, sum_lo, 1);
        sum_lo += __shfl_xor_sync(0xffffffffu, sum_lo, 2);
        sum_hi += __shfl_xor_sync(0xffffffffu, sum_hi, 1);
        sum_hi += __shfl_xor_sync(0xffffffffu, sum_hi, 2);
        l_lo = l_lo * alpha_lo + sum_lo;
        l_hi = l_hi * alpha_hi + sum_hi;
        m_lo = mn_lo;
        m_hi = mn_hi;
#pragma unroll
        for (int n = 0; n < NTILES; n++) {
            o[n][0] *= alpha_lo; o[n][1] *= alpha_lo;
            o[n][2] *= alpha_hi; o[n][3] *= alpha_hi;
        }

        // O += P V: P as A fragments (2 k-steps of 16 keys), V^T from smem as B fragments.
        uint32_t pa[2][4];
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ks2++) {
            pa[ks2][0] = pack_bf16(s[2 * ks2][0], s[2 * ks2][1]);
            pa[ks2][1] = pack_bf16(s[2 * ks2][2], s[2 * ks2][3]);
            pa[ks2][2] = pack_bf16(s[2 * ks2 + 1][0], s[2 * ks2 + 1][1]);
            pa[ks2][3] = pack_bf16(s[2 * ks2 + 1][2], s[2 * ks2 + 1][3]);
        }
#pragma unroll
        for (int ks2 = 0; ks2 < 2; ks2++) {
#pragma unroll
            for (int n = 0; n < NTILES; n++) {
                uint32_t b[2];
                const bf16* vp = vt + (n * 8 + g) * VPAD + ks2 * 16 + 2 * t;
                b[0] = *reinterpret_cast<const uint32_t*>(vp);
                b[1] = *reinterpret_cast<const uint32_t*>(vp + 8);
                mma_bf16(o[n], pa[ks2], b);
            }
        }
    }

    // Normalize, gate, store rows g and g+8 (if inside the sequence).
    const float inv_lo = 1.f / l_lo, inv_hi = 1.f / l_hi;
    const uint32_t r_lo = row0 + g, r_hi = row0 + g + 8;
#pragma unroll
    for (int n = 0; n < NTILES; n++) {
        const int d = n * 8 + 2 * t;
        if (d >= DREAL) continue;
        if (r_lo < tile.seq_len) {
            const size_t tok = tile.seq_start + r_lo;
            float a = o[n][0] * inv_lo, b = o[n][1] * inv_lo;
            if (GATED) {
                const bf16* gp = gate + tok * gate_stride + head * gate_head_stride + d;
                a = round_bf16(a) * round_bf16(sigmoid(bf2f(gp[0])));
                b = round_bf16(b) * round_bf16(sigmoid(bf2f(gp[1])));
            }
            *reinterpret_cast<uint32_t*>(out + tok * out_stride + head * DREAL + d) = pack_bf16(a, b);
        }
        if (r_hi < tile.seq_len) {
            const size_t tok = tile.seq_start + r_hi;
            float a = o[n][2] * inv_hi, b = o[n][3] * inv_hi;
            if (GATED) {
                const bf16* gp = gate + tok * gate_stride + head * gate_head_stride + d;
                a = round_bf16(a) * round_bf16(sigmoid(bf2f(gp[0])));
                b = round_bf16(b) * round_bf16(sigmoid(bf2f(gp[1])));
            }
            *reinterpret_cast<uint32_t*>(out + tok * out_stride + head * DREAL + d) = pack_bf16(a, b);
        }
    }
}

// Text attention: q [N, 24, 256] (stride q_stride), k/v [N, 4, 256] (stride kv_stride), gate inside the q_proj
// output [N, 24, 512] at offset 256 of each head (pass gate = q_proj output + 256, gate_stride 12288,
// gate_head_stride 512), out [N, 24, 256].
extern "C" __global__ void attention(const AttnTile* __restrict__ tiles, const bf16* __restrict__ q, const bf16* __restrict__ k,
                                     const bf16* __restrict__ v, const bf16* __restrict__ gate, bf16* __restrict__ out,
                                     uint32_t q_stride, uint32_t kv_stride, uint32_t gate_stride, uint32_t gate_head_stride,
                                     uint32_t out_stride, uint32_t heads, uint32_t kv_heads, float scale_log2) {
    attention_impl<256, 256, true>(tiles, q, k, v, gate, out, q_stride, kv_stride, gate_stride, gate_head_stride, out_stride, heads, kv_heads, scale_log2);
}

// Vision attention: head_dim 72, q/k/v inside the qkv matmul output [N, 3, 16, 72] (q_stride = kv_stride = 3456).
extern "C" __global__ void vision_attention(const AttnTile* __restrict__ tiles, const bf16* __restrict__ q, const bf16* __restrict__ k,
                                            const bf16* __restrict__ v, bf16* __restrict__ out, uint32_t qkv_stride,
                                            uint32_t out_stride, uint32_t heads, float scale_log2) {
    attention_impl<80, 72, false>(tiles, q, k, v, nullptr, out, qkv_stride, qkv_stride, 0, 0, out_stride, heads, heads, scale_log2);
}
