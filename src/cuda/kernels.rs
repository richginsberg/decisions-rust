//! The kernels in `kernels/*.cu`, built by build.rs into one fatbin, loaded with the driver API, and
//! wrapped as typed launchers. Every launcher takes row-major device buffers and the shapes the kernel
//! needs; the argument order here must match the `extern "C"` signatures in the .cu files.

use std::sync::Arc;

use cudarc::driver::sys::CUfunction_attribute;
use cudarc::driver::{CudaFunction, CudaModule, CudaSlice, CudaStream, DevicePtr, DevicePtrMut, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

use super::CudaError;

const FATBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/kernels.fatbin"));

/// Threads per block for the row kernels (one block per row).
const ROW_THREADS: u32 = 256;

/// The delta-rule kernels' dynamic shared memory (deltanet.cu's DELTA_*_SMEM; QS = 136, TS = 72).
pub const DELTA_CHUNK: usize = 64;
const DELTA_CHUNK_SMEM: u32 = (2 * 64 * 136 * 2 + 64 * 64 * 4) as u32;
const DELTA_STATE_SMEM: u32 = (64 * 136 * 2 + 2 * 128 * 72 * 2 + 64 * 4) as u32;
const DELTA_OUT_SMEM: u32 = (128 * 136 * 2 + 128 * 72 * 2 + 64 * 136 * 2 + 64 * 72 * 2 + 64 * 4) as u32;

/// Queries per attention block (attention.cu's QUERIES).
pub const ATTENTION_QUERIES: usize = 64;

pub struct Kernels {
    _module: Arc<CudaModule>,
    stream: Arc<CudaStream>,
    quantize_fp4: CudaFunction,
    fp4_zero_scale_padding: CudaFunction,
    rms_norm: CudaFunction,
    rms_norm_fp4: CudaFunction,
    rms_norm_fp8: CudaFunction,
    quantize_fp8: CudaFunction,
    silu_mul_fp4: CudaFunction,
    add_inplace: CudaFunction,
    scale_rows_cols: CudaFunction,
    attention: CudaFunction,
    vision_attention: CudaFunction,
    head_norm_rotary: CudaFunction,
    conv1d_silu: CudaFunction,
    delta_chunk: CudaFunction,
    delta_state: CudaFunction,
    delta_out: CudaFunction,
    gated_norm: CudaFunction,
    gather_rows: CudaFunction,
    scatter_rows: CudaFunction,
    layer_norm: CudaFunction,
    gelu: CudaFunction,
    vision_rotary: CudaFunction,
    vision_pos_embed: CudaFunction,
}

/// A device address passed to a kernel: a buffer's start, an offset into one, or 0 for an absent optional buffer.
#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Ptr(pub u64);
unsafe impl cudarc::driver::DeviceRepr for Ptr {}

impl Ptr {
    pub fn null() -> Self {
        Self(0)
    }
    /// The address of `buffer` plus `offset` elements.
    pub fn of<T>(buffer: &CudaSlice<T>, offset: usize, stream: &CudaStream) -> Self {
        let (p, _r) = buffer.device_ptr(stream);
        Self(p + (offset * size_of::<T>()) as u64)
    }
    pub fn of_mut<T>(buffer: &mut CudaSlice<T>, offset: usize, stream: &CudaStream) -> Self {
        let (p, _r) = buffer.device_ptr_mut(stream);
        Self(p + (offset * size_of::<T>()) as u64)
    }
}

impl Kernels {
    pub fn load(stream: Arc<CudaStream>) -> Result<Self, CudaError> {
        let module = stream.context().load_module(Ptx::from_binary(FATBIN.to_vec()))?;
        let f = |name: &str| module.load_function(name);
        let delta_chunk = f("delta_chunk")?;
        let delta_state = f("delta_state")?;
        let delta_out = f("delta_out")?;
        delta_chunk.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DELTA_CHUNK_SMEM as i32)?;
        delta_state.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DELTA_STATE_SMEM as i32)?;
        delta_out.set_attribute(CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, DELTA_OUT_SMEM as i32)?;
        Ok(Self {
            quantize_fp4: f("quantize_fp4")?,
            fp4_zero_scale_padding: f("fp4_zero_scale_padding")?,
            rms_norm: f("rms_norm")?,
            rms_norm_fp4: f("rms_norm_fp4")?,
            rms_norm_fp8: f("rms_norm_fp8")?,
            quantize_fp8: f("quantize_fp8")?,
            silu_mul_fp4: f("silu_mul_fp4")?,
            add_inplace: f("add_inplace")?,
            scale_rows_cols: f("scale_rows_cols")?,
            attention: f("attention")?,
            vision_attention: f("vision_attention")?,
            head_norm_rotary: f("head_norm_rotary")?,
            conv1d_silu: f("conv1d_silu")?,
            delta_chunk,
            delta_state,
            delta_out,
            gated_norm: f("gated_norm")?,
            gather_rows: f("gather_rows")?,
            scatter_rows: f("scatter_rows")?,
            layer_norm: f("layer_norm")?,
            gelu: f("gelu")?,
            vision_rotary: f("vision_rotary")?,
            vision_pos_embed: f("vision_pos_embed")?,
            _module: module,
            stream,
        })
    }

    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream
    }

    fn rows(rows: usize) -> LaunchConfig {
        LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (ROW_THREADS, 1, 1), shared_mem_bytes: 0 }
    }

    fn elements(total: usize, per_thread: usize) -> LaunchConfig {
        let threads = total.div_ceil(per_thread) as u32;
        LaunchConfig { grid_dim: (threads.div_ceil(256).max(1), 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 }
    }

    fn ptr<T>(&self, buffer: &CudaSlice<T>) -> Ptr {
        Ptr::of(buffer, 0, &self.stream)
    }

    fn ptr_mut<T>(&self, buffer: &mut CudaSlice<T>) -> Ptr {
        Ptr::of_mut(buffer, 0, &self.stream)
    }

    fn opt(&self, buffer: Option<&mut CudaSlice<u16>>) -> Ptr {
        buffer.map_or(Ptr::null(), |b| self.ptr_mut(b))
    }

    /// x [rows, k] bf16 -> NVFP4 codes and swizzled scales (padding rows of the scale buffer zeroed).
    pub fn quantize_fp4(&self, x: Ptr, packed: &mut CudaSlice<u8>, scales: &mut CudaSlice<u8>, rows: usize, k: usize, encode: f32) -> Result<(), CudaError> {
        let k32 = k as u32;
        unsafe {
            self.stream.launch_builder(&self.quantize_fp4).arg(&x).arg(&mut *packed).arg(&mut *scales).arg(&k32).arg(&encode).launch(Self::rows(rows))?;
        }
        self.zero_scale_padding(scales, rows, k)
    }

    fn zero_scale_padding(&self, scales: &mut CudaSlice<u8>, rows: usize, k: usize) -> Result<(), CudaError> {
        if rows.is_multiple_of(128) {
            return Ok(());
        }
        let (rows32, k32) = (rows as u32, k as u32);
        let padding = (128 - rows % 128) * (k / 16).div_ceil(4) * 4;
        unsafe {
            self.stream.launch_builder(&self.fp4_zero_scale_padding).arg(&mut *scales).arg(&rows32).arg(&k32).launch(Self::elements(padding, 1))?;
        }
        Ok(())
    }

    /// y = rms_norm(x (+ residual, written back)) in bf16, the (1 + w) form.
    pub fn rms_norm(&self, x: &CudaSlice<u16>, residual: Option<&mut CudaSlice<u16>>, w: &CudaSlice<u16>, y: &mut CudaSlice<u16>, rows: usize, n: usize, eps: f32) -> Result<(), CudaError> {
        let residual = self.opt(residual);
        let n32 = n as u32;
        unsafe {
            self.stream.launch_builder(&self.rms_norm).arg(x).arg(&residual).arg(w).arg(y).arg(&n32).arg(&eps).launch(Self::rows(rows))?;
        }
        Ok(())
    }

    /// rms_norm(x (+ residual)) quantized to NVFP4.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_fp4(&self, x: &CudaSlice<u16>, residual: Option<&mut CudaSlice<u16>>, w: &CudaSlice<u16>, packed: &mut CudaSlice<u8>, scales: &mut CudaSlice<u8>, rows: usize, n: usize, eps: f32, encode: f32) -> Result<(), CudaError> {
        let residual = self.opt(residual);
        let n32 = n as u32;
        unsafe {
            self.stream.launch_builder(&self.rms_norm_fp4).arg(x).arg(&residual).arg(w).arg(&mut *packed).arg(&mut *scales).arg(&n32).arg(&eps).arg(&encode).launch(Self::rows(rows))?;
        }
        self.zero_scale_padding(scales, rows, n)
    }

    /// rms_norm(x (+ residual)) quantized to FP8 with one scale per row.
    #[allow(clippy::too_many_arguments)]
    pub fn rms_norm_fp8(&self, x: &CudaSlice<u16>, residual: Option<&mut CudaSlice<u16>>, w: &CudaSlice<u16>, values: &mut CudaSlice<u8>, row_scale: &mut CudaSlice<f32>, rows: usize, n: usize, eps: f32) -> Result<(), CudaError> {
        let residual = self.opt(residual);
        let n32 = n as u32;
        unsafe {
            self.stream.launch_builder(&self.rms_norm_fp8).arg(x).arg(&residual).arg(w).arg(values).arg(row_scale).arg(&n32).arg(&eps).launch(Self::rows(rows))?;
        }
        Ok(())
    }

    pub fn quantize_fp8(&self, x: &CudaSlice<u16>, values: &mut CudaSlice<u8>, row_scale: &mut CudaSlice<f32>, rows: usize, k: usize) -> Result<(), CudaError> {
        let k32 = k as u32;
        unsafe {
            self.stream.launch_builder(&self.quantize_fp8).arg(x).arg(values).arg(row_scale).arg(&k32).launch(Self::rows(rows))?;
        }
        Ok(())
    }

    /// silu(gate) * up from the merged [rows, 2n] matmul output, quantized to NVFP4 [rows, n].
    pub fn silu_mul_fp4(&self, gate_up: &CudaSlice<u16>, packed: &mut CudaSlice<u8>, scales: &mut CudaSlice<u8>, rows: usize, n: usize, encode: f32) -> Result<(), CudaError> {
        let n32 = n as u32;
        unsafe {
            self.stream.launch_builder(&self.silu_mul_fp4).arg(gate_up).arg(&mut *packed).arg(&mut *scales).arg(&n32).arg(&encode).launch(Self::rows(rows))?;
        }
        self.zero_scale_padding(scales, rows, n)
    }

    /// out[rows, n] bf16 = acc[rows, n] f32 * row_scale[row] * col_scale[col] (n a multiple of 8).
    pub fn scale_rows_cols(&self, acc: &CudaSlice<f32>, row_scale: &CudaSlice<f32>, col_scale: &CudaSlice<f32>, out: &mut CudaSlice<u16>, rows: usize, n: usize) -> Result<(), CudaError> {
        let total = rows * n;
        let (n32, total32) = (n as u32, total as u32);
        unsafe {
            self.stream.launch_builder(&self.scale_rows_cols).arg(acc).arg(row_scale).arg(col_scale).arg(out).arg(&n32).arg(&total32).launch(Self::elements(total, 8))?;
        }
        Ok(())
    }

    /// y += x over `total` bf16 values (a multiple of 8).
    pub fn add_inplace(&self, y: &mut CudaSlice<u16>, x: &CudaSlice<u16>, total: usize) -> Result<(), CudaError> {
        let total32 = total as u32;
        unsafe {
            self.stream.launch_builder(&self.add_inplace).arg(y).arg(x).arg(&total32).launch(Self::elements(total, 8))?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// Text attention over the tiles (3 u32 each): q [N, heads, 256], k/v [N, kv_heads, 256], gate inside the q_proj
    /// output (+256 per head, head stride 512), out [N, heads, 256].
    #[allow(clippy::too_many_arguments)]
    pub fn attention(&self, tiles: &CudaSlice<u32>, n_tiles: usize, q: &CudaSlice<u16>, k: &CudaSlice<u16>, v: &CudaSlice<u16>, gate: Ptr, out: &mut CudaSlice<u16>, heads: usize, kv_heads: usize, scale: f32) -> Result<(), CudaError> {
        let (q_stride, kv_stride, gate_stride, gate_head_stride, out_stride) = ((heads * 256) as u32, (kv_heads * 256) as u32, (heads * 512) as u32, 512u32, (heads * 256) as u32);
        let (heads32, kv32) = (heads as u32, kv_heads as u32);
        let scale_log2 = scale * std::f32::consts::LOG2_E;
        let cfg = LaunchConfig { grid_dim: (n_tiles as u32, heads32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream
                .launch_builder(&self.attention)
                .arg(tiles).arg(q).arg(k).arg(v).arg(&gate).arg(out)
                .arg(&q_stride).arg(&kv_stride).arg(&gate_stride).arg(&gate_head_stride).arg(&out_stride).arg(&heads32).arg(&kv32).arg(&scale_log2)
                .launch(cfg)?;
        }
        Ok(())
    }

    /// Vision attention: q/k/v inside the qkv buffer [N, 3, heads, 72]; out [N, heads * 72].
    #[allow(clippy::too_many_arguments)]
    pub fn vision_attention(&self, tiles: &CudaSlice<u32>, n_tiles: usize, qkv: &CudaSlice<u16>, out: &mut CudaSlice<u16>, heads: usize, scale: f32) -> Result<(), CudaError> {
        let stride = (3 * heads * 72) as u32;
        let out_stride = (heads * 72) as u32;
        let heads32 = heads as u32;
        let scale_log2 = scale * std::f32::consts::LOG2_E;
        let q = self.ptr(qkv);
        let k = Ptr::of(qkv, heads * 72, &self.stream);
        let v = Ptr::of(qkv, 2 * heads * 72, &self.stream);
        let cfg = LaunchConfig { grid_dim: (n_tiles as u32, heads32, 1), block_dim: (128, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.vision_attention).arg(tiles).arg(&q).arg(&k).arg(&v).arg(out).arg(&stride).arg(&out_stride).arg(&heads32).arg(&scale_log2).launch(cfg)?;
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    /// q_norm / k_norm and the partial multimodal rotary: q_proj output [N, heads, 512], k_proj output [N, kv_heads, 256]
    /// -> q [N, heads, 256], k [N, kv_heads, 256]. positions [N][3] i32.
    #[allow(clippy::too_many_arguments)]
    pub fn head_norm_rotary(&self, qg: &CudaSlice<u16>, kin: &CudaSlice<u16>, qw: &CudaSlice<u16>, kw: &CudaSlice<u16>, positions: &CudaSlice<i32>, q: &mut CudaSlice<u16>, k: &mut CudaSlice<u16>, rows: usize, heads: usize, kv_heads: usize, rot: usize, theta: f32, eps: f32) -> Result<(), CudaError> {
        let (heads32, kv32, rot32) = (heads as u32, kv_heads as u32, rot as u32);
        unsafe {
            self.stream.launch_builder(&self.head_norm_rotary).arg(qg).arg(kin).arg(qw).arg(kw).arg(positions).arg(q).arg(k).arg(&heads32).arg(&kv32).arg(&rot32).arg(&theta).arg(&eps).launch(Self::rows(rows))?;
        }
        Ok(())
    }

    /// Depthwise causal conv (kernel 4) + SiLU over packed sequences: x, y [N, c]; w [c, 4]; seq_of [N] sequence starts.
    pub fn conv1d_silu(&self, x: &CudaSlice<u16>, w: &CudaSlice<u16>, seq_of: &CudaSlice<i32>, y: &mut CudaSlice<u16>, rows: usize, c: usize) -> Result<(), CudaError> {
        let (n32, c32) = (rows as u32, c as u32);
        unsafe {
            self.stream.launch_builder(&self.conv1d_silu).arg(x).arg(w).arg(seq_of).arg(y).arg(&n32).arg(&c32).launch(Self::elements(rows * c, 8))?;
        }
        Ok(())
    }

    /// The within-chunk part of the delta rule: one block per (chunk, value head). Writes u, w [N, 48, 128] bf16,
    /// the intra-chunk attention [chunks, 48, 64, 64] bf16 and the cumulative decays [N, 48] f32.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_chunk(&self, chunks: &CudaSlice<u32>, n_chunks: usize, qkv: &CudaSlice<u16>, ab: &CudaSlice<u16>, a_log_exp: &CudaSlice<f32>, dt_bias: &CudaSlice<f32>, u: &mut CudaSlice<u16>, w: &mut CudaSlice<u16>, intra: &mut CudaSlice<u16>, cum_decay: &mut CudaSlice<f32>) -> Result<(), CudaError> {
        let cfg = LaunchConfig { grid_dim: (n_chunks as u32, 48, 1), block_dim: (256, 1, 1), shared_mem_bytes: DELTA_CHUNK_SMEM };
        unsafe {
            self.stream.launch_builder(&self.delta_chunk).arg(chunks).arg(qkv).arg(ab).arg(a_log_exp).arg(dt_bias).arg(u).arg(w).arg(intra).arg(cum_decay).launch(cfg)?;
        }
        Ok(())
    }

    /// The sequential part: one block per (sequence, value head) walks its chunks, storing each chunk's incoming
    /// state [chunks, 48, 128, 128] bf16 and v_new^T [chunks, 48, 128, 64] bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_state(&self, seqs: &CudaSlice<u32>, n_seqs: usize, chunks: &CudaSlice<u32>, qkv: &CudaSlice<u16>, u: &CudaSlice<u16>, w: &CudaSlice<u16>, cum_decay: &CudaSlice<f32>, states: &mut CudaSlice<u16>, vnew: &mut CudaSlice<u16>) -> Result<(), CudaError> {
        let cfg = LaunchConfig { grid_dim: (n_seqs as u32, 48, 1), block_dim: (128, 1, 1), shared_mem_bytes: DELTA_STATE_SMEM };
        unsafe {
            self.stream.launch_builder(&self.delta_state).arg(seqs).arg(chunks).arg(qkv).arg(u).arg(w).arg(cum_decay).arg(states).arg(vnew).launch(cfg)?;
        }
        Ok(())
    }

    /// The outputs: one block per (chunk, value head); out [N, 48, 128] bf16.
    #[allow(clippy::too_many_arguments)]
    pub fn delta_out(&self, chunks: &CudaSlice<u32>, n_chunks: usize, qkv: &CudaSlice<u16>, cum_decay: &CudaSlice<f32>, intra: &CudaSlice<u16>, states: &CudaSlice<u16>, vnew: &CudaSlice<u16>, out: &mut CudaSlice<u16>) -> Result<(), CudaError> {
        let cfg = LaunchConfig { grid_dim: (n_chunks as u32, 48, 1), block_dim: (128, 1, 1), shared_mem_bytes: DELTA_OUT_SMEM };
        unsafe {
            self.stream.launch_builder(&self.delta_out).arg(chunks).arg(qkv).arg(cum_decay).arg(intra).arg(states).arg(vnew).arg(out).launch(cfg)?;
        }
        Ok(())
    }

    /// The gated RMSNorm after the delta rule: o [N, 48, 128], z [N, 6144], w [128] -> y [N, 6144].
    pub fn gated_norm(&self, o: &CudaSlice<u16>, z: &CudaSlice<u16>, w: &CudaSlice<u16>, y: &mut CudaSlice<u16>, rows: usize, eps: f32) -> Result<(), CudaError> {
        let rows32 = rows as u32;
        let items = rows * 48;
        let cfg = LaunchConfig { grid_dim: ((items * 32).div_ceil(256) as u32, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.gated_norm).arg(o).arg(z).arg(w).arg(y).arg(&rows32).arg(&eps).launch(cfg)?;
        }
        Ok(())
    }

    /// dst[i] = src[idx[i]] over rows of `width` bf16.
    pub fn gather_rows(&self, src: &CudaSlice<u16>, idx: &CudaSlice<i32>, dst: &mut CudaSlice<u16>, rows: usize, width: usize) -> Result<(), CudaError> {
        let (rows32, width32) = (rows as u32, width as u32);
        unsafe {
            self.stream.launch_builder(&self.gather_rows).arg(src).arg(idx).arg(dst).arg(&rows32).arg(&width32).launch(Self::elements(rows * width, 8))?;
        }
        Ok(())
    }

    /// dst[idx[i]] = src[i].
    pub fn scatter_rows(&self, src: &CudaSlice<u16>, idx: &CudaSlice<i32>, dst: &mut CudaSlice<u16>, rows: usize, width: usize) -> Result<(), CudaError> {
        let (rows32, width32) = (rows as u32, width as u32);
        unsafe {
            self.stream.launch_builder(&self.scatter_rows).arg(src).arg(idx).arg(dst).arg(&rows32).arg(&width32).launch(Self::elements(rows * width, 8))?;
        }
        Ok(())
    }

    /// LayerNorm with weight and bias (the vision tower).
    pub fn layer_norm(&self, x: &CudaSlice<u16>, w: &CudaSlice<u16>, b: &CudaSlice<u16>, y: &mut CudaSlice<u16>, rows: usize, n: usize, eps: f32) -> Result<(), CudaError> {
        let n32 = n as u32;
        unsafe {
            self.stream.launch_builder(&self.layer_norm).arg(x).arg(w).arg(b).arg(y).arg(&n32).arg(&eps).launch(Self::rows(rows))?;
        }
        Ok(())
    }

    /// In-place GELU over `total` bf16 values: the tanh approximation or the exact form.
    pub fn gelu(&self, x: &mut CudaSlice<u16>, total: usize, tanh_form: bool) -> Result<(), CudaError> {
        let total32 = total as u32;
        let form = tanh_form as u32;
        unsafe {
            self.stream.launch_builder(&self.gelu).arg(x).arg(&total32).arg(&form).launch(Self::elements(total, 8))?;
        }
        Ok(())
    }

    /// Axial rotary on q and k inside the vision qkv buffer, in place; positions [N][2] (row, col).
    pub fn vision_rotary(&self, qkv: &mut CudaSlice<u16>, positions: &CudaSlice<i32>, rows: usize, heads: usize, theta: f32) -> Result<(), CudaError> {
        let heads32 = heads as u32;
        let cfg = LaunchConfig { grid_dim: (rows as u32, 1, 1), block_dim: (64, 1, 1), shared_mem_bytes: 0 };
        unsafe {
            self.stream.launch_builder(&self.vision_rotary).arg(qkv).arg(positions).arg(&heads32).arg(&theta).launch(cfg)?;
        }
        Ok(())
    }

    /// h [rows, n] += the 4-tap bilinear resample of the learned position embedding.
    pub fn vision_pos_embed(&self, h: &mut CudaSlice<u16>, pos: &CudaSlice<u16>, taps: &CudaSlice<i32>, weights: &CudaSlice<f32>, rows: usize, n: usize) -> Result<(), CudaError> {
        let (rows32, n32) = (rows as u32, n as u32);
        unsafe {
            self.stream.launch_builder(&self.vision_pos_embed).arg(h).arg(pos).arg(taps).arg(weights).arg(&rows32).arg(&n32).launch(Self::elements(rows * n, 8))?;
        }
        Ok(())
    }
}
