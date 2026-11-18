// Activation quantization for the cuBLASLt matmuls, as the Python server does it (nvfp4_matmul.py,
// fp8_matmul.py), fused with the elementwise work that produces the activations:
//
// - NVFP4: per 16-value block, scale = e4m3(amax / 6 * encode), value = round_to_e2m1(x * encode / scale), ties
//   rounding down to the nearest magnitude; block scales written straight into the 128x4 tiled layout. Rows of
//   the scale buffer beyond `rows` (padding to a multiple of 128) are zeroed by `fp4_zero_scale_padding`.
// - FP8: per row, scale = max(amax / 448, 1e-12), value = e4m3(x / scale).
//
// Variants fuse the producer: the plain quantizer reads a bf16 matrix; `rms_norm_fp4` applies the (1+w)
// RMSNorm (with an optional residual add written back in bf16) before quantizing; `silu_mul_fp4` applies
// silu(gate) * up to the merged gate/up matmul output; `rms_norm_fp8` norms then row-quantizes.
//
// All take row-major inputs; one block of 256 threads handles one row of up to 8192 values per thread
// iteration (k <= 256 * 16 * 8 = 32768), with each thread owning whole 16-value blocks.

#include "common.cuh"

#define FP4_MAX 6.0f
#define FP8_MAX 448.0f

// E2M1 magnitude by code, and the midpoints between magnitudes (ties go down, as torch.bucketize does).
__device__ __forceinline__ uint32_t e2m1_code(float v) {
    const float a = fabsf(v);
    uint32_t m;
    if (a <= 0.25f) m = 0;
    else if (a <= 0.75f) m = 1;
    else if (a <= 1.25f) m = 2;
    else if (a <= 1.75f) m = 3;
    else if (a <= 2.5f) m = 4;
    else if (a <= 3.5f) m = 5;
    else if (a <= 5.0f) m = 6;
    else m = 7;
    return m | ((v < 0.f) ? 8u : 0u);
}

// Quantize 16 consecutive f32 values (already the final activations) of row `row`, block `blk` (of k/16).
__device__ __forceinline__ void quantize_block16(const float* v, float encode, uint32_t row, uint32_t blk,
                                                 uint32_t col_blocks, uint8_t* packed, uint8_t* scales, uint32_t k) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    const float raw = fminf(amax / FP4_MAX * encode, FP8_MAX);
    const uint8_t s = f2e4m3(raw);
    const float decoded = e4m3tof(s);
    const float factor = decoded > 0.f ? encode / decoded : 0.f;
    uint8_t out[8];
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const uint32_t lo = e2m1_code(v[2 * i] * factor), hi = e2m1_code(v[2 * i + 1] * factor);
        out[i] = static_cast<uint8_t>(lo | (hi << 4));
    }
    *reinterpret_cast<uint2*>(packed + (static_cast<size_t>(row) * k + blk * 16) / 2) = *reinterpret_cast<uint2*>(out);
    scales[swizzled_scale_index(row, blk, col_blocks)] = s;
}

// Load 16 bf16 values at `p` (16-byte aligned) into f32.
__device__ __forceinline__ void load16(const bf16* p, float* v) {
    const uint4 a = *reinterpret_cast<const uint4*>(p);
    const uint4 b = *reinterpret_cast<const uint4*>(p + 8);
    const bf16* pa = reinterpret_cast<const bf16*>(&a);
    const bf16* pb = reinterpret_cast<const bf16*>(&b);
#pragma unroll
    for (int i = 0; i < 8; i++) { v[i] = bf2f(pa[i]); v[8 + i] = bf2f(pb[i]); }
}

__device__ __forceinline__ void store16(bf16* p, const float* v) {
    uint4 a, b;
    bf16* pa = reinterpret_cast<bf16*>(&a);
    bf16* pb = reinterpret_cast<bf16*>(&b);
#pragma unroll
    for (int i = 0; i < 8; i++) { pa[i] = f2bf(v[i]); pb[i] = f2bf(v[8 + i]); }
    *reinterpret_cast<uint4*>(p) = a;
    *reinterpret_cast<uint4*>(p + 8) = b;
}

// x [rows, k] bf16 -> packed [rows, k/2], scales (swizzled).
extern "C" __global__ void quantize_fp4(const bf16* __restrict__ x, uint8_t* __restrict__ packed,
                                        uint8_t* __restrict__ scales, uint32_t k, float encode) {
    const uint32_t row = blockIdx.x, blocks = k / 16, col_blocks = (blocks + 3) / 4;
    for (uint32_t blk = threadIdx.x; blk < blocks; blk += blockDim.x) {
        float v[16];
        load16(x + static_cast<size_t>(row) * k + blk * 16, v);
        quantize_block16(v, encode, row, blk, col_blocks, packed, scales, k);
    }
}

// Zero the scale rows from `rows` up to the next multiple of 128 (cuBLASLt reads whole tiles).
extern "C" __global__ void fp4_zero_scale_padding(uint8_t* __restrict__ scales, uint32_t rows, uint32_t k) {
    const uint32_t blocks = k / 16, col_blocks = (blocks + 3) / 4;
    const uint32_t padded = (rows + 127) / 128 * 128;
    const uint32_t total = (padded - rows) * col_blocks * 4;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < total; i += gridDim.x * blockDim.x) {
        const uint32_t row = rows + i / (col_blocks * 4), col = i % (col_blocks * 4);
        scales[swizzled_scale_index(row, col, col_blocks)] = 0;
    }
}

// The (1 + w) RMSNorm of Qwen3.5 in f32, as the compiled norm does it: y = x * rsqrt(mean(x^2) + eps) * (1 + w).
// With `residual`, the input is first replaced by residual + x, rounded to bf16 and written to `residual`
// (the fused add_rms_norm: the residual stream stays bf16-rounded as in eager PyTorch).
// One block per row; `n` <= 32768 and a multiple of 16; blockDim 256.
// Loaders of one 16-value block of the norm's input: a bf16 row, or an FP8 matmul's f32 accumulator row with
// its row and column scales (bf16(acc * row_scale * col_scale), as `scale_rows_cols` would have written).
struct LoadBf16 {
    const bf16* x;
    __device__ __forceinline__ void operator()(uint32_t blk, float* v) const { load16(x + blk * 16, v); }
};
struct LoadScaled {
    const float* acc;
    const float* col_scale;
    float rs;
    __device__ __forceinline__ void operator()(uint32_t blk, float* v) const {
        const float4* a = reinterpret_cast<const float4*>(acc + blk * 16);
        const float4* c = reinterpret_cast<const float4*>(col_scale + blk * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 av = a[i], cv = c[i];
            v[i * 4] = round_bf16(av.x * rs * cv.x);
            v[i * 4 + 1] = round_bf16(av.y * rs * cv.y);
            v[i * 4 + 2] = round_bf16(av.z * rs * cv.z);
            v[i * 4 + 3] = round_bf16(av.w * rs * cv.w);
        }
    }
};

template <class Load>
__device__ __forceinline__ void rms_norm_row(const Load& load, bf16* __restrict__ residual,
                                             const bf16* __restrict__ w, float eps, uint32_t n, float* out_values,
                                             uint32_t* out_blocks, float* scratch, uint32_t max_blocks) {
    const uint32_t blocks = n / 16;
    float sum = 0.f;
    uint32_t count = 0;
    for (uint32_t blk = threadIdx.x; blk < blocks && count < max_blocks; blk += blockDim.x, count++) {
        float* v = out_values + count * 16;
        load(blk, v);
        if (residual) {
            float r[16];
            load16(residual + blk * 16, r);
#pragma unroll
            for (int i = 0; i < 16; i++) v[i] = round_bf16(r[i] + v[i]);
            store16(residual + blk * 16, v);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) sum += v[i] * v[i];
        out_blocks[count] = blk;
    }
    const float total = block_sum(sum, scratch);
    const float inv = rsqrtf(total / static_cast<float>(n) + eps);
    for (uint32_t c = 0; c < count; c++) {
        float* v = out_values + c * 16;
        float wv[16];
        load16(w + out_blocks[c] * 16, wv);
#pragma unroll
        for (int i = 0; i < 16; i++) v[i] = round_bf16(v[i] * inv * (1.f + wv[i]));
    }
}

// Each thread holds up to 8 blocks of 16 values: n <= 256 * 8 * 16 = 32768.
#define NORM_MAX_BLOCKS 8

// y = rms_norm(x (+ residual)) as bf16.
extern "C" __global__ void rms_norm(const bf16* __restrict__ x, bf16* __restrict__ residual, const bf16* __restrict__ w,
                                    bf16* __restrict__ y, uint32_t n, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float values[NORM_MAX_BLOCKS * 16];
    uint32_t blocks[NORM_MAX_BLOCKS];
    rms_norm_row(LoadBf16{x + row * n}, residual ? residual + row * n : nullptr, w, eps, n, values, blocks, scratch, NORM_MAX_BLOCKS);
    const uint32_t count = threadIdx.x < n / 16 ? min((n / 16 - threadIdx.x + blockDim.x - 1) / blockDim.x, (uint32_t)NORM_MAX_BLOCKS) : 0u;
    for (uint32_t c = 0; c < count; c++) store16(y + row * n + blocks[c] * 16, values + c * 16);
}

// rms_norm(x (+ residual)) quantized to NVFP4 for the next matmul.
extern "C" __global__ void rms_norm_fp4(const bf16* __restrict__ x, bf16* __restrict__ residual, const bf16* __restrict__ w,
                                        uint8_t* __restrict__ packed, uint8_t* __restrict__ scales, uint32_t n, float eps,
                                        float encode) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float values[NORM_MAX_BLOCKS * 16];
    uint32_t blocks[NORM_MAX_BLOCKS];
    rms_norm_row(LoadBf16{x + row * n}, residual ? residual + row * n : nullptr, w, eps, n, values, blocks, scratch, NORM_MAX_BLOCKS);
    const uint32_t count = threadIdx.x < n / 16 ? min((n / 16 - threadIdx.x + blockDim.x - 1) / blockDim.x, (uint32_t)NORM_MAX_BLOCKS) : 0u;
    const uint32_t col_blocks = (n / 16 + 3) / 4;
    for (uint32_t c = 0; c < count; c++) quantize_block16(values + c * 16, encode, row, blocks[c], col_blocks, packed, scales, n);
}

// rms_norm_fp4 over an FP8 matmul's f32 accumulator with its row and column scales (the out_proj of the
// linear-attention layers), with the residual add.
extern "C" __global__ void rms_norm_fp4_scaled(const float* __restrict__ acc, const float* __restrict__ row_scale,
                                               const float* __restrict__ col_scale, bf16* __restrict__ residual,
                                               const bf16* __restrict__ w, uint8_t* __restrict__ packed,
                                               uint8_t* __restrict__ scales, uint32_t n, float eps, float encode) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float values[NORM_MAX_BLOCKS * 16];
    uint32_t blocks[NORM_MAX_BLOCKS];
    rms_norm_row(LoadScaled{acc + row * n, col_scale, row_scale[row]}, residual ? residual + row * n : nullptr, w, eps, n, values, blocks, scratch, NORM_MAX_BLOCKS);
    const uint32_t count = threadIdx.x < n / 16 ? min((n / 16 - threadIdx.x + blockDim.x - 1) / blockDim.x, (uint32_t)NORM_MAX_BLOCKS) : 0u;
    const uint32_t col_blocks = (n / 16 + 3) / 4;
    for (uint32_t c = 0; c < count; c++) quantize_block16(values + c * 16, encode, row, blocks[c], col_blocks, packed, scales, n);
}

// rms_norm(x (+ residual)) quantized to FP8 with one scale per row (the linear-attention projections).
extern "C" __global__ void rms_norm_fp8(const bf16* __restrict__ x, bf16* __restrict__ residual, const bf16* __restrict__ w,
                                        uint8_t* __restrict__ values_out, float* __restrict__ row_scale, uint32_t n, float eps) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float values[NORM_MAX_BLOCKS * 16];
    uint32_t blocks[NORM_MAX_BLOCKS];
    rms_norm_row(LoadBf16{x + row * n}, residual ? residual + row * n : nullptr, w, eps, n, values, blocks, scratch, NORM_MAX_BLOCKS);
    const uint32_t count = threadIdx.x < n / 16 ? min((n / 16 - threadIdx.x + blockDim.x - 1) / blockDim.x, (uint32_t)NORM_MAX_BLOCKS) : 0u;
    float amax = 0.f;
    for (uint32_t c = 0; c < count; c++)
#pragma unroll
        for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(values[c * 16 + i]));
    amax = block_max(amax, scratch);
    const float scale = fmaxf(amax / FP8_MAX, 1e-12f);
    if (threadIdx.x == 0) row_scale[row] = scale;
    const float inv = 1.f / scale;
    for (uint32_t c = 0; c < count; c++) {
        uint8_t out[16];
#pragma unroll
        for (int i = 0; i < 16; i++) out[i] = f2e4m3(fminf(fmaxf(values[c * 16 + i] * inv, -FP8_MAX), FP8_MAX));
        *reinterpret_cast<uint4*>(values_out + row * n + blocks[c] * 16) = *reinterpret_cast<uint4*>(out);
    }
}

// x [rows, k] bf16 -> e4m3 values and one scale per row (the FP8 projections whose input is not a norm output).
extern "C" __global__ void quantize_fp8(const bf16* __restrict__ x, uint8_t* __restrict__ values_out,
                                        float* __restrict__ row_scale, uint32_t k) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    const bf16* xr = x + row * k;
    float amax = 0.f;
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x) amax = fmaxf(amax, fabsf(bf2f(xr[i])));
    amax = block_max(amax, scratch);
    const float scale = fmaxf(amax / FP8_MAX, 1e-12f);
    if (threadIdx.x == 0) row_scale[row] = scale;
    const float inv = 1.f / scale;
    for (uint32_t i = threadIdx.x; i < k; i += blockDim.x)
        values_out[row * k + i] = f2e4m3(fminf(fmaxf(bf2f(xr[i]) * inv, -FP8_MAX), FP8_MAX));
}

// The merged gate/up output [rows, 2n] bf16 -> silu(gate) * up (bf16-rounded, as eager does) quantized to NVFP4
// [rows, n]. The Python path rounds act_fn(gate) to bf16, then the product to bf16.
extern "C" __global__ void silu_mul_fp4(const bf16* __restrict__ gate_up, uint8_t* __restrict__ packed,
                                        uint8_t* __restrict__ scales, uint32_t n, float encode) {
    const uint32_t row = blockIdx.x, blocks = n / 16, col_blocks = (blocks + 3) / 4;
    const bf16* g = gate_up + static_cast<size_t>(row) * 2 * n;
    const bf16* u = g + n;
    for (uint32_t blk = threadIdx.x; blk < blocks; blk += blockDim.x) {
        float gv[16], uv[16];
        load16(g + blk * 16, gv);
        load16(u + blk * 16, uv);
#pragma unroll
        for (int i = 0; i < 16; i++) gv[i] = round_bf16(round_bf16(silu(gv[i])) * uv[i]);
        quantize_block16(gv, encode, row, blk, col_blocks, packed, scales, n);
    }
}

// The merged gate/up output [rows, 2n] bf16 -> silu(gate) * up [rows, n] in bf16 (rounded as `silu_mul_fp4`
// rounds before it quantizes). Locate needs the product itself: the down projection's LoRA reads it.
extern "C" __global__ void silu_mul(const bf16* __restrict__ gate_up, bf16* __restrict__ out, uint32_t n, uint32_t rows) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= (size_t)rows * n) return;
    const size_t row = i / n, col = i % n;
    const bf16* g = gate_up + row * 2 * n + col;
    const uint4 graw = *reinterpret_cast<const uint4*>(g);
    const uint4 uraw = *reinterpret_cast<const uint4*>(g + n);
    const bf16* gp = reinterpret_cast<const bf16*>(&graw);
    const bf16* up = reinterpret_cast<const bf16*>(&uraw);
    uint4 packed;
    bf16* op = reinterpret_cast<bf16*>(&packed);
#pragma unroll
    for (int e = 0; e < 8; e++) op[e] = f2bf(round_bf16(silu(bf2f(gp[e]))) * bf2f(up[e]));
    *reinterpret_cast<uint4*>(out + i) = packed;
}

// The per-tensor FP8 matmul's f32 accumulator [rows, n] times row_scale[row] * col_scale[col], rounded once to
// bf16: what torch._scaled_mm's rowwise mode produces. (cuBLASLt applies outer-vector FP8 scales on Hopper only;
// Blackwell runs the scalar-scaled matmul and this kernel finishes it.)
extern "C" __global__ void scale_rows_cols(const float* __restrict__ acc, const float* __restrict__ row_scale,
                                           const float* __restrict__ col_scale, bf16* __restrict__ out, uint32_t n, uint32_t total) {
    const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= total) return;
    const uint32_t row = i / n, col = i - row * n;
    const float rs = row_scale[row];
    const float4 a0 = *reinterpret_cast<const float4*>(acc + i);
    const float4 a1 = *reinterpret_cast<const float4*>(acc + i + 4);
    const float4 c0 = *reinterpret_cast<const float4*>(col_scale + col);
    const float4 c1 = *reinterpret_cast<const float4*>(col_scale + col + 4);
    bf16 o[8] = {f2bf(a0.x * rs * c0.x), f2bf(a0.y * rs * c0.y), f2bf(a0.z * rs * c0.z), f2bf(a0.w * rs * c0.w),
                 f2bf(a1.x * rs * c1.x), f2bf(a1.y * rs * c1.y), f2bf(a1.z * rs * c1.z), f2bf(a1.w * rs * c1.w)};
    *reinterpret_cast<uint4*>(out + i) = *reinterpret_cast<uint4*>(o);
}

// y[rows, n] bf16 += x[rows, n] bf16 (the residual add after the MLP), rounded to bf16.
extern "C" __global__ void add_inplace(bf16* __restrict__ y, const bf16* __restrict__ x, uint32_t total) {
    const uint32_t i = (blockIdx.x * blockDim.x + threadIdx.x) * 8;
    if (i >= total) return;
    float a[8], b[8];
    const uint4 ya = *reinterpret_cast<const uint4*>(y + i);
    const uint4 xa = *reinterpret_cast<const uint4*>(x + i);
    const bf16* yp = reinterpret_cast<const bf16*>(&ya);
    const bf16* xp = reinterpret_cast<const bf16*>(&xa);
#pragma unroll
    for (int j = 0; j < 8; j++) { a[j] = bf2f(yp[j]); b[j] = bf2f(xp[j]); a[j] += b[j]; }
    uint4 out;
    bf16* op = reinterpret_cast<bf16*>(&out);
#pragma unroll
    for (int j = 0; j < 8; j++) op[j] = f2bf(a[j]);
    *reinterpret_cast<uint4*>(y + i) = out;
}
