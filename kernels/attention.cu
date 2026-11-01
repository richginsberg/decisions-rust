// Non-causal attention over packed sequences, on the tensor cores (bf16 mma.sync, f32 accumulation), with
// the online softmax of FlashAttention-2.
//
// One block = 8 warps = 128 queries of one sequence for one head; each warp owns 16 query rows. Keys and
// values of the same sequence stream through a double-buffered shared-memory tile of 32 keys, filled with
// cp.async while the previous tile is computed. Scores never leave registers: the C-fragment layout of
// S = Q K^T is exactly the A-fragment layout needed for P V, so P is re-packed in place. K and V stay
// row-major in shared memory; ldmatrix (and ldmatrix.trans for V) forms the B fragments.
//
// Fragment layouts of mma.sync.m16n8k16 (row.col, bf16 -> f32): see common.cuh.
//
// Text: head_dim 256, 24 query heads over 4 key/value heads (GQA), the output gated by sigmoid(gate).
// Vision: head_dim 72 padded to 80 (zeros beyond 72), 16 heads, no gate.
//
// Buffers are row-major per token with the strides given: q at `q + token * q_stride + head * D`, etc.

#include "common.cuh"

#define KEYS 32
#define QUERIES 128
#define THREADS 256

// A sequence's slice of the pack and which 128-query tile this block handles.
struct AttnTile {
    uint32_t seq_start; // first token of the sequence in the pack
    uint32_t seq_len;
    uint32_t q_start;   // first query of the tile, relative to seq_start
};

__device__ __forceinline__ void cp_async16(void* smem, const void* gmem, int src_bytes) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(s), "l"(gmem), "r"(src_bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

// Two 8x8 b16 matrices whose row addresses lanes 0..15 give: r[i] = matrix i's (lane / 4, 2 (lane % 4)..+1).
__device__ __forceinline__ void ldmatrix_x2(uint32_t* r, const bf16* p) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}
// The same, transposed: r[i] = matrix i's (2 (lane % 4)..+1, lane / 4).
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t* r, const bf16* p) {
    const uint32_t s = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n" : "=r"(r[0]), "=r"(r[1]) : "r"(s));
}

template <int D, int DREAL, bool GATED>
__device__ void attention_impl(const AttnTile* __restrict__ tiles, const bf16* __restrict__ q, const bf16* __restrict__ k,
                               const bf16* __restrict__ v, const bf16* __restrict__ gate, bf16* __restrict__ out,
                               uint32_t q_stride, uint32_t kv_stride, uint32_t gate_stride, uint32_t gate_head_stride,
                               uint32_t out_stride, uint32_t heads, uint32_t kv_heads, float scale_log2) {
    constexpr int KSTEPS = D / 16;      // k-steps over the head dim for Q K^T
    constexpr int NTILES = D / 8;       // n-tiles over the head dim for P V
    constexpr int PAD = D + 8;          // smem row stride (elements) of the K and V tiles
    constexpr int STAGE = 2 * KEYS * PAD; // elements per stage: K tile, then V tile
    extern __shared__ __align__(16) unsigned char attn_smem[];
    bf16* smem = reinterpret_cast<bf16*>(attn_smem);

    const AttnTile tile = tiles[blockIdx.x];
    if (tile.seq_len == 0) return; // a padding entry of a fixed-size tile table (the whole block leaves)
    const uint32_t head = blockIdx.y;
    const uint32_t kv_head = head / (heads / kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const uint32_t row0 = tile.q_start + warp * 16;
    // A warp whose 16 rows all lie beyond the sequence still runs the loop (it shares the block's barriers);
    // its rows read row 0 and store nothing.

    // Q fragments for this warp's 16 rows, straight from global memory.
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

    // Issue the loads of key tile `it` into stage `it & 1` (16 bytes per cp.async; zero-filled beyond the
    // sequence or the real head dim, reading a valid address with no bytes).
    auto issue = [&](uint32_t it) {
        const uint32_t key0 = it * KEYS;
        bf16* kd = smem + (it & 1) * STAGE;
        bf16* vd = kd + KEYS * PAD;
        for (int i = threadIdx.x; i < KEYS * (D / 8); i += THREADS) {
            const int key = i / (D / 8), d = (i % (D / 8)) * 8;
            const bool ok = key0 + key < tile.seq_len && d < DREAL;
            const size_t base = ok ? (size_t)(tile.seq_start + key0 + key) * kv_stride + kv_head * DREAL + d : (size_t)tile.seq_start * kv_stride;
            cp_async16(kd + key * PAD + d, k + base, ok ? 16 : 0);
            cp_async16(vd + key * PAD + d, v + base, ok ? 16 : 0);
        }
    };

    float o[NTILES][4];
#pragma unroll
    for (int n = 0; n < NTILES; n++) { o[n][0] = o[n][1] = o[n][2] = o[n][3] = 0.f; }
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.f, l_hi = 0.f;

    const uint32_t iters = (tile.seq_len + KEYS - 1) / KEYS;
    issue(0);
    cp_async_commit();
    for (uint32_t it = 0; it < iters; it++) {
        if (it + 1 < iters) {
            issue(it + 1); // into the stage the previous iteration finished reading (its trailing barrier)
            cp_async_commit();
            cp_async_wait<1>();
        } else {
            cp_async_wait<0>();
        }
        __syncthreads(); // tile `it` is visible to every warp
        const bf16* kd = smem + (it & 1) * STAGE;
        const bf16* vd = kd + KEYS * PAD;
        const uint32_t key0 = it * KEYS;

        // S = Q K^T for this warp's 16 rows x 32 keys: 4 n-tiles of 8 keys; K rows as B fragments by ldmatrix.
        float s[KEYS / 8][4];
#pragma unroll
        for (int n = 0; n < KEYS / 8; n++) { s[n][0] = s[n][1] = s[n][2] = s[n][3] = 0.f; }
#pragma unroll
        for (int st = 0; st < KSTEPS; st++) {
#pragma unroll
            for (int n = 0; n < KEYS / 8; n++) {
                uint32_t b[2];
                // lanes 0..7: rows n*8.. at columns st*16; lanes 8..15: the same rows at columns st*16 + 8
                ldmatrix_x2(b, kd + (n * 8 + (lane & 7)) * PAD + st * 16 + ((lane >> 3) & 1) * 8);
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

        // O += P V: P as A fragments (2 k-steps of 16 keys), V rows as B fragments by ldmatrix.trans.
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
                // lanes 0..15: V rows ks2*16 + lane at columns n*8 (matrix 0 = keys ..+8, matrix 1 = keys +8..+16)
                ldmatrix_x2_trans(b, vd + (ks2 * 16 + (lane & 15)) * PAD + n * 8);
                mma_bf16(o[n], pa[ks2], b);
            }
        }
        __syncthreads(); // everyone is done with this stage before the next iteration refills it
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

// Dynamic shared memory of each instantiation: 2 stages x (K + V) x KEYS x (D + 8) bf16.
#define ATTENTION_SMEM (2 * 2 * KEYS * (256 + 8) * 2)
#define VISION_ATTENTION_SMEM (2 * 2 * KEYS * (80 + 8) * 2)

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
