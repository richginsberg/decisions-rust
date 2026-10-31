// Shared helpers for the kernels: types, warp reductions, bf16 and e4m3 conversions.
#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

typedef __nv_bfloat16 bf16;

__device__ __forceinline__ float bf2f(bf16 v) { return __bfloat162float(v); }
__device__ __forceinline__ bf16 f2bf(float v) { return __float2bfloat16(v); }

// Round-trip through bf16: what PyTorch does when a bf16 tensor receives an f32 result.
__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16(v)); }

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v += __shfl_xor_sync(0xffffffffu, v, offset);
    return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, offset));
    return v;
}

// Sum across a block of up to 1024 threads; `scratch` holds 32 floats. Every thread gets the total.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
    v = warp_sum(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    if (lane < warps) total = scratch[lane];
    total = warp_sum(total);
    return total;
}

__device__ __forceinline__ float block_max(float v, float* scratch) {
    v = warp_max(v);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = (blockDim.x + 31) >> 5;
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = -INFINITY;
    if (lane < warps) total = scratch[lane];
    total = warp_max(total);
    return total;
}

// float8 e4m3fn, round to nearest even, saturating at 448 (what torch's .to(float8_e4m3fn) does).
__device__ __forceinline__ uint8_t f2e4m3(float v) {
    return static_cast<uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
}

__device__ __forceinline__ float e4m3tof(uint8_t v) {
    return __half2float(__nv_cvt_fp8_to_halfraw(v, __NV_E4M3));
}

__device__ __forceinline__ float silu(float v) { return v / (1.f + __expf(-v)); }

__device__ __forceinline__ float sigmoid(float v) { return 1.f / (1.f + __expf(-v)); }

// PyTorch's gelu(approximate="tanh").
__device__ __forceinline__ float gelu_tanh(float v) {
    const float c = 0.7978845608028654f; // sqrt(2/pi)
    return 0.5f * v * (1.f + tanhf(c * (v + 0.044715f * v * v * v)));
}

// Where the e4m3 scale of block `col` of row `row` goes in cuBLASLt's 128x4 tiled layout (blas.rs's
// swizzled_scale_index). `col_blocks` = ceil(k/16 / 4).
__device__ __forceinline__ uint32_t swizzled_scale_index(uint32_t row, uint32_t col, uint32_t col_blocks) {
    const uint32_t tile = (row >> 7) * col_blocks + (col >> 2);
    const uint32_t in_row = row & 127, hi = in_row >> 5, lo = in_row & 31;
    return tile * 512u + lo * 16u + hi * 4u + (col & 3);
}
