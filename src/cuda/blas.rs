//! cuBLASLt matrix multiplies: NVFP4 block-scaled (the quantized linear layers), FP8 row-scaled (the
//! linear-attention projections) and bf16 (the vision tower, the small projections, the readout).
//!
//! Every call computes `out[M, N] = x[M, K] · w[N, K]^T` with row-major buffers. In cuBLASLt's column-major
//! terms that is `D^T[N, M] = W[N, K] · X^T[K, M]`: A = the weight with `TRANSA = T`, B = the activations
//! with `TRANSB = N`, the "TN" layout the narrow types require.
//!
//! Scales follow cuBLASLt's conventions for each type:
//! - NVFP4: `VEC16_UE4M3` block scales in the 128x4 tiled ("swizzled") layout for both operands, with the two
//!   per-tensor decode scales (1 / input_global_scale, 1 / weight_global_scale) folded into alpha.
//! - FP8: per-tensor (scale 1) into an f32 accumulator; the per-row activation and per-column weight scales are
//!   applied by a kernel afterwards, since `OUTER_VEC_32F` is Hopper-only.
//!
//! Each distinct call (types, shape, scale and bias pointers) keeps its descriptor, layouts and the heuristic's
//! algorithm after the first time: a pass makes a few hundred calls, and the heuristic query alone is tens of
//! microseconds each, which is most of a short pass.

use std::collections::{BTreeMap, HashMap};
use std::ffi::c_void;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use cudarc::cublaslt::result as lt;
use cudarc::cublaslt::sys;
use cudarc::driver::{CudaSlice, CudaStream, DevicePtr, DevicePtrMut};
use cudarc::driver::sys::CUevent_flags;
use parking_lot::Mutex;

use super::CudaError;

/// Rows of the activation matmuls are rounded up to a multiple of this: each distinct row count is a new plan
/// (heuristic query and kernel load, tens of milliseconds for a pass's worth), so a request's row count is
/// bucketed and the buckets are warmed at startup. Workspace buffers hold a multiple of this many rows, and
/// the rows beyond the real ones hold stale values whose outputs nothing reads.
pub const ROW_BUCKET: usize = 64;
/// Above this many rows the bucket is `ROW_BUCKET_LARGE`: long requests rarely repeat a row count, and a bucket
/// seen for the first time costs a plan (and, at load, its tuning).
pub const ROW_BUCKET_FROM: usize = 2048;
pub const ROW_BUCKET_LARGE: usize = 256;

/// The row count a matmul of `m` rows is run with.
pub fn bucket_rows(m: usize) -> usize {
    if m <= ROW_BUCKET_FROM { m.div_ceil(ROW_BUCKET) * ROW_BUCKET } else { m.div_ceil(ROW_BUCKET_LARGE) * ROW_BUCKET_LARGE }
}

/// The cuBLASLt workspace, in bytes: cuBLASLt's own recommendation for Blackwell.
pub const WORKSPACE_BYTES: usize = 32 << 20;

/// Rows of a block-scale tile; scale buffers are padded to a multiple of these rows and of 4 scale columns.
pub const SCALE_TILE_ROWS: usize = 128;
pub const SCALE_TILE_COLS: usize = 4;

/// Bytes of a swizzled block-scale buffer for `rows` x `k` values (one e4m3 scale per 16 values).
pub fn swizzled_scale_bytes(rows: usize, k: usize) -> usize {
    rows.div_ceil(SCALE_TILE_ROWS) * SCALE_TILE_ROWS * (k / 16).div_ceil(SCALE_TILE_COLS) * SCALE_TILE_COLS
}

/// Where the e4m3 scale of block `col` (of k/16) of row `row` sits in the swizzled buffer. The CPU-side
/// layout of the weights at load time; the quantization kernels compute the same on the GPU.
pub fn swizzled_scale_index(row: usize, col: usize, k: usize) -> usize {
    let col_blocks = (k / 16).div_ceil(SCALE_TILE_COLS);
    let (row_block, row_in_block) = (row / SCALE_TILE_ROWS, row % SCALE_TILE_ROWS);
    let (col_block, col_in_block) = (col / SCALE_TILE_COLS, col % SCALE_TILE_COLS);
    let tile = row_block * col_blocks + col_block;
    // Inside a tile: 32 chunks of 16 bytes; chunk r_lo holds the 4 columns of rows r_lo + 32 * r_hi.
    let (r_hi, r_lo) = (row_in_block / 32, row_in_block % 32);
    tile * (SCALE_TILE_ROWS * SCALE_TILE_COLS) + r_lo * 16 + r_hi * 4 + col_in_block
}

pub struct Lt {
    handle: sys::cublasLtHandle_t,
    /// Kept alive for `workspace_ptr`, which cuBLASLt writes on the stream.
    _workspace: CudaSlice<u8>,
    workspace_ptr: *mut c_void,
    stream: Arc<CudaStream>,
    plans: Mutex<HashMap<Key, Plan>>,
    /// Time the heuristic's candidates on first use of each shape and keep the fastest (DECISION_AUTOTUNE,
    /// default on), but only while `tuning` is set: during the warm-up passes at load. A shape first seen on a
    /// live request takes the heuristic's first choice, since timing 8 candidates costs over a second.
    autotune: bool,
    tuning: AtomicBool,
    /// Tuned algorithms by shape, persisted on the volume (per GPU and cuBLASLt version) so a restart skips
    /// the timing: `tuned` holds what was loaded plus what this run timed, `tuned_dirty` whether to save.
    tuned: Mutex<HashMap<String, [u64; 8]>>,
    tuned_dirty: AtomicBool,
    tuning_file: Option<PathBuf>,
}

/// The shape of a matmul as a cache key: types, dimensions, whether block-scaled and whether biased (the
/// pointers themselves differ between runs).
fn shape_key(a_type: i32, out_type: i32, m: usize, n: usize, k: usize, scaled: bool, bias: bool) -> String {
    format!("{a_type}:{out_type}:{m}x{n}x{k}:{}{}", scaled as u8, bias as u8)
}

/// Heuristic candidates considered per shape, and the row count from which timing them is worth it (below
/// it the matmuls are launch-bound and the heuristic's first choice is as good as any).
const CANDIDATES: usize = 8;
const AUTOTUNE_MIN_ROWS: usize = 512;

/// What distinguishes one prepared matmul from another: everything but the A, B and D data pointers.
#[derive(Hash, PartialEq, Eq, Clone, Copy)]
struct Key {
    a_type: i32,
    out_type: i32,
    m: usize,
    n: usize,
    k: usize,
    scale_mode: i32,
    a_scale: usize,
    b_scale: usize,
    bias: usize,
    /// The output's row stride in elements (n unless the result goes into a wider buffer).
    d_ld: usize,
    /// Whether the product is added to the output (beta = 1) rather than replacing it.
    accumulate: bool,
}

/// A prepared matmul: its descriptor, layouts and the algorithm the heuristic chose.
struct Plan {
    desc: Desc,
    a: Layout,
    b: Layout,
    d: Layout,
    algo: sys::cublasLtMatmulAlgo_t,
}

unsafe impl Send for Lt {}
unsafe impl Sync for Lt {}

impl Drop for Lt {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_handle(self.handle);
        }
    }
}

/// An NVFP4 weight on the device, laid out for cuBLASLt.
pub struct Fp4Weight {
    /// [N, K/2] packed codes.
    pub packed: CudaSlice<u8>,
    /// Swizzled e4m3 block scales.
    pub scale: CudaSlice<u8>,
    /// 1 / weight_global_scale.
    pub decode: f32,
    /// input_global_scale: the activation encode scale.
    pub input_encode: f32,
    pub n: usize,
    pub k: usize,
}

/// An FP8 (e4m3) weight with one f32 scale per output row.
pub struct Fp8Weight {
    pub values: CudaSlice<u8>,
    pub scale: CudaSlice<f32>,
    pub n: usize,
    pub k: usize,
}

/// A bf16 weight [N, K] with an optional bias [N].
pub struct Bf16Weight {
    pub values: CudaSlice<u16>,
    pub bias: Option<CudaSlice<u16>>,
    pub n: usize,
    pub k: usize,
}

struct Desc(sys::cublasLtMatmulDesc_t);
impl Drop for Desc {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matmul_desc(self.0);
        }
    }
}
struct Layout(sys::cublasLtMatrixLayout_t);
impl Drop for Layout {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matrix_layout(self.0);
        }
    }
}
struct Pref(sys::cublasLtMatmulPreference_t);
impl Drop for Pref {
    fn drop(&mut self) {
        unsafe {
            let _ = lt::destroy_matmul_pref(self.0);
        }
    }
}

unsafe fn set_desc<T>(desc: &Desc, attr: sys::cublasLtMatmulDescAttributes_t, value: &T) -> Result<(), CudaError> {
    unsafe { lt::set_matmul_desc_attribute(desc.0, attr, value as *const T as *const c_void, size_of::<T>()) }
        .map_err(|e| CudaError::Blas(format!("setting {attr:?}: {e:?}")))
}

impl Lt {
    /// `tuning_dir`: where tuned algorithms are kept between runs (DECISION_TUNING_CACHE overrides it; an
    /// empty value disables the file).
    pub fn new(stream: Arc<CudaStream>, tuning_dir: Option<&Path>) -> Result<Self, CudaError> {
        let handle = lt::create_handle().map_err(|e| CudaError::Blas(format!("cublasLtCreate: {e:?}")))?;
        let tuning_dir = match std::env::var("DECISION_TUNING_CACHE") {
            Ok(v) if v.is_empty() => None,
            Ok(v) => Some(PathBuf::from(v)),
            Err(_) => tuning_dir.map(Path::to_path_buf),
        };
        let tuning_file = match tuning_dir {
            Some(dir) => {
                let device = stream.context().name().unwrap_or_else(|_| "gpu".into());
                let device: String = device.chars().map(|c| if c.is_ascii_alphanumeric() { c } else { '-' }).collect();
                let version = unsafe { sys::cublasLtGetVersion() };
                Some(dir.join(format!("cublaslt-tuning-{device}-v{version}.json")))
            }
            None => None,
        };
        let mut tuned = HashMap::new();
        if let Some(file) = &tuning_file
            && let Ok(text) = std::fs::read_to_string(file)
        {
            match serde_json::from_str::<BTreeMap<String, Vec<u64>>>(&text) {
                Ok(saved) => {
                    tuned.extend(saved.into_iter().filter_map(|(key, data)| <[u64; 8]>::try_from(data).ok().map(|d| (key, d))));
                    tracing::info!("{} tuned cuBLASLt algorithms from {}", tuned.len(), file.display());
                }
                Err(error) => tracing::warn!("ignoring {}: {error}", file.display()),
            }
        }
        let workspace = stream.alloc_zeros::<u8>(WORKSPACE_BYTES)?;
        let workspace_ptr = {
            let (ptr, _record) = workspace.device_ptr(&stream);
            ptr as *mut c_void
        };
        let autotune = std::env::var("DECISION_AUTOTUNE").map_or(true, |v| v != "0");
        Ok(Self {
            handle,
            _workspace: workspace,
            workspace_ptr,
            stream,
            plans: Mutex::new(HashMap::new()),
            autotune,
            tuning: AtomicBool::new(false),
            tuned: Mutex::new(tuned),
            tuned_dirty: AtomicBool::new(false),
            tuning_file,
        })
    }

    /// Write the tuned algorithms to the volume if this run timed any.
    pub fn save_tuning(&self) {
        let Some(file) = &self.tuning_file else { return };
        if !self.tuned_dirty.swap(false, Ordering::Relaxed) {
            return;
        }
        let tuned = self.tuned.lock();
        let ordered: BTreeMap<&String, Vec<u64>> = tuned.iter().map(|(k, v)| (k, v.to_vec())).collect();
        let written = file
            .parent()
            .map_or(Ok(()), std::fs::create_dir_all)
            .and_then(|_| std::fs::write(file, serde_json::to_string_pretty(&ordered).unwrap_or_default()));
        match written {
            Ok(()) => tracing::info!("{} tuned cuBLASLt algorithms saved to {}", ordered.len(), file.display()),
            Err(error) => tracing::warn!("could not save the tuned algorithms to {}: {error}", file.display()),
        }
    }

    /// Whether plans made from now on are tuned (see `autotune`).
    pub fn set_tuning(&self, on: bool) {
        self.tuning.store(on, Ordering::Relaxed);
    }

    /// The common part of every call: `D[N x M] = alpha · A^T[N x K] · B[K x M]` in column-major terms. `scales`
    /// is the block-scale mode with the A and B scale pointers, for the narrow types.
    #[allow(clippy::too_many_arguments)]
    unsafe fn run(
        &self,
        a: (*const c_void, sys::cudaDataType, i64),
        b: (*const c_void, sys::cudaDataType, i64),
        out: (*mut c_void, sys::cudaDataType),
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        scales: Option<(sys::cublasLtMatmulMatrixScale_t, *const c_void, *const c_void)>,
        bias: Option<*const c_void>,
        d_ld: usize,
        beta: f32,
    ) -> Result<(), CudaError> {
        let blas = |e: cudarc::cublaslt::result::CublasError, what: &str| CudaError::Blas(format!("{what}: {e:?}"));
        let key = Key {
            d_ld,
            accumulate: beta != 0.0,
            a_type: a.1 as i32,
            out_type: out.1 as i32,
            m,
            n,
            k,
            scale_mode: scales.map_or(-1, |(mode, _, _)| mode as i32),
            a_scale: scales.map_or(0, |(_, p, _)| p as usize),
            b_scale: scales.map_or(0, |(_, _, p)| p as usize),
            bias: bias.map_or(0, |p| p as usize),
        };
        let mut plans = self.plans.lock();
        let plan = match plans.get(&key) {
            Some(plan) => plan,
            None => {
                let plan = unsafe { self.prepare(a, b, out, m, n, k, alpha, scales, bias, d_ld, beta) }?;
                plans.entry(key).or_insert(plan)
            }
        };
        unsafe { self.execute(&plan.desc, &plan.a, &plan.b, &plan.d, &plan.algo, a.0, b.0, out.0, alpha, beta) }
            .map_err(|e| blas(e, &format!("cublasLtMatmul {m}x{n}x{k} {:?}", a.1)))
    }

    /// One matmul launch with prepared descriptors and a given algorithm.
    #[allow(clippy::too_many_arguments)]
    unsafe fn execute(
        &self,
        desc: &Desc,
        a_layout: &Layout,
        b_layout: &Layout,
        d_layout: &Layout,
        algo: &sys::cublasLtMatmulAlgo_t,
        a: *const c_void,
        b: *const c_void,
        out: *mut c_void,
        alpha: f32,
        beta: f32,
    ) -> Result<(), cudarc::cublaslt::result::CublasError> {
        unsafe {
            lt::matmul(
                self.handle,
                desc.0,
                &alpha as *const f32 as *const c_void,
                &beta as *const f32 as *const c_void,
                a,
                a_layout.0,
                b,
                b_layout.0,
                out as *const c_void,
                d_layout.0,
                out,
                d_layout.0,
                algo,
                self.workspace_ptr,
                WORKSPACE_BYTES,
                self.stream.cu_stream() as sys::cudaStream_t,
            )
        }
    }

    /// The descriptor, layouts and algorithm for one distinct call.
    #[allow(clippy::too_many_arguments)]
    unsafe fn prepare(
        &self,
        a: (*const c_void, sys::cudaDataType, i64),
        b: (*const c_void, sys::cudaDataType, i64),
        out: (*mut c_void, sys::cudaDataType),
        m: usize,
        n: usize,
        k: usize,
        alpha: f32,
        scales: Option<(sys::cublasLtMatmulMatrixScale_t, *const c_void, *const c_void)>,
        bias: Option<*const c_void>,
        d_ld: usize,
        beta: f32,
    ) -> Result<Plan, CudaError> {
        let blas = |e: cudarc::cublaslt::result::CublasError, what: &str| CudaError::Blas(format!("{what}: {e:?}"));
        unsafe {
            let desc = Desc(
                lt::create_matmul_desc(sys::cublasComputeType_t::CUBLAS_COMPUTE_32F, sys::cudaDataType_t::CUDA_R_32F)
                    .map_err(|e| blas(e, "matmul desc"))?,
            );
            // cublasOperation_t: CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 (a C enum, 4 bytes).
            let trans_t: i32 = 1;
            let trans_n: i32 = 0;
            set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSA, &trans_t)?;
            set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_TRANSB, &trans_n)?;
            if let Some((mode, a_scale, b_scale)) = scales {
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &mode)?;
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &mode)?;
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &a_scale)?;
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &b_scale)?;
            }
            if let Some(bias) = bias {
                let epilogue = sys::cublasLtEpilogue_t::CUBLASLT_EPILOGUE_BIAS;
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_EPILOGUE, &epilogue)?;
                set_desc(&desc, sys::cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias)?;
            }
            // A: the weight, stored [N, K] row-major = column-major [K, N] with ld = K (in elements of its type).
            let a_layout = Layout(lt::create_matrix_layout(a.1, k as u64, n as u64, a.2).map_err(|e| blas(e, "A layout"))?);
            // B: the activations, [M, K] row-major = column-major [K, M] with ld = K.
            let b_layout = Layout(lt::create_matrix_layout(b.1, k as u64, m as u64, b.2).map_err(|e| blas(e, "B layout"))?);
            // D: [M, N] row-major = column-major [N, M] with ld = N.
            let d_layout = Layout(lt::create_matrix_layout(out.1, n as u64, m as u64, d_ld as i64).map_err(|e| blas(e, "D layout"))?);
            let pref = Pref(lt::create_matmul_pref().map_err(|e| blas(e, "preference"))?);
            let workspace_bytes = WORKSPACE_BYTES;
            lt::set_matmul_pref_attribute(
                pref.0,
                sys::cublasLtMatmulPreferenceAttributes_t::CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                &workspace_bytes as *const usize as *const c_void,
                size_of::<usize>(),
            )
            .map_err(|e| blas(e, "workspace preference"))?;
            // The heuristic's candidates, best first by its own estimate; timed on the real operands when the
            // shape is large enough for the choice to matter.
            let mut results: [std::mem::MaybeUninit<sys::cublasLtMatmulHeuristicResult_t>; CANDIDATES] =
                [const { std::mem::MaybeUninit::uninit() }; CANDIDATES];
            let mut count = 0i32;
            let status = sys::cublasLtMatmulAlgoGetHeuristic(
                self.handle,
                desc.0,
                a_layout.0,
                b_layout.0,
                d_layout.0,
                d_layout.0,
                pref.0,
                CANDIDATES as i32,
                results.as_mut_ptr() as *mut sys::cublasLtMatmulHeuristicResult_t,
                &mut count,
            );
            if status != sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                return Err(CudaError::Blas(format!("no cuBLASLt algorithm for {m}x{n}x{k} {:?}: {status:?}", a.1)));
            }
            let candidates: Vec<sys::cublasLtMatmulAlgo_t> = results[..count.max(0) as usize]
                .iter()
                .map(|r| r.assume_init())
                .filter(|r| r.state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                .map(|r| r.algo)
                .collect();
            let Some(&first) = candidates.first() else {
                return Err(CudaError::Blas(format!("no cuBLASLt algorithm for {m}x{n}x{k} {:?}", a.1)));
            };
            let mut algo = first;
            let key = shape_key(a.1 as i32, out.1 as i32, m, n, k, scales.is_some(), bias.is_some());
            // A saved choice, if cuBLASLt still accepts it for this configuration.
            let saved = self.tuned.lock().get(&key).copied();
            let mut from_cache = false;
            if let Some(data) = saved {
                let candidate = sys::cublasLtMatmulAlgo_t { data };
                let mut result = std::mem::MaybeUninit::<sys::cublasLtMatmulHeuristicResult_t>::uninit();
                let status = sys::cublasLtMatmulAlgoCheck(self.handle, desc.0, a_layout.0, b_layout.0, d_layout.0, d_layout.0, &candidate, result.as_mut_ptr());
                if status == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS && result.assume_init().state == sys::cublasStatus_t::CUBLAS_STATUS_SUCCESS {
                    algo = candidate;
                    from_cache = true;
                } else {
                    tracing::debug!("saved cuBLASLt algorithm for {key} no longer accepted; retuning");
                }
            }
            // (An accumulating call is never timed: each run would add to its output.)
            if !from_cache && beta == 0.0 && self.autotune && self.tuning.load(Ordering::Relaxed) && m >= AUTOTUNE_MIN_ROWS && candidates.len() > 1 {
                let ctx = self.stream.context();
                let start = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                let end = ctx.new_event(Some(CUevent_flags::CU_EVENT_DEFAULT))?;
                let (mut best, mut first_ms) = (f32::INFINITY, 0.0f32);
                for (i, candidate) in candidates.iter().enumerate() {
                    if self.execute(&desc, &a_layout, &b_layout, &d_layout, candidate, a.0, b.0, out.0, alpha, beta).is_err() {
                        continue; // a candidate the heuristic offered but the call rejects
                    }
                    start.record(&self.stream)?;
                    for _ in 0..2 {
                        self.execute(&desc, &a_layout, &b_layout, &d_layout, candidate, a.0, b.0, out.0, alpha, beta)
                            .map_err(|e| blas(e, "timing a candidate"))?;
                    }
                    end.record(&self.stream)?;
                    let ms = start.elapsed_ms(&end)? / 2.0;
                    if i == 0 {
                        first_ms = ms;
                    }
                    if ms < best {
                        best = ms;
                        algo = *candidate;
                    }
                }
                tracing::debug!("cuBLASLt {m}x{n}x{k} {:?}: {} candidates, best {best:.3} ms, heuristic's first {first_ms:.3} ms", a.1, candidates.len());
                self.tuned.lock().insert(key, algo.data);
                self.tuned_dirty.store(true, Ordering::Relaxed);
            }
            Ok(Plan { desc, a: a_layout, b: b_layout, d: d_layout, algo })
        }
    }

    /// `out[M, N] bf16 = (x · w^T)` with x as NVFP4 codes [M, K/2] and swizzled block scales.
    pub fn fp4(
        &self,
        x_packed: &CudaSlice<u8>,
        x_scale: &CudaSlice<u8>,
        w: &Fp4Weight,
        out: &mut CudaSlice<u16>,
        m: usize,
    ) -> Result<(), CudaError> {
        let (n, k) = (w.n, w.k);
        let m = bucket_rows(m);
        debug_assert!(x_packed.len() >= m * k / 2 && out.len() >= m * n);
        unsafe {
            let mode = sys::cublasLtMatmulMatrixScale_t::CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
            let (w_scale, _r1) = w.scale.device_ptr(&self.stream);
            let (x_scale_ptr, _r2) = x_scale.device_ptr(&self.stream);
            let (w_ptr, _r3) = w.packed.device_ptr(&self.stream);
            let (x_ptr, _r4) = x_packed.device_ptr(&self.stream);
            let (out_ptr, _r5) = out.device_ptr_mut(&self.stream);
            // Leading dimensions of FP4 layouts count elements (two per byte).
            self.run(
                (w_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_4F_E2M1, k as i64),
                (x_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_4F_E2M1, k as i64),
                (out_ptr as *mut c_void, sys::cudaDataType_t::CUDA_R_16BF),
                m,
                n,
                k,
                w.decode / w.input_encode,
                Some((mode, w_scale as *const c_void, x_scale_ptr as *const c_void)),
                None,
                n,
                0.0,
            )
        }
    }

    /// `acc[M, N] f32 = xq · wq^T` for e4m3 operands with their per-tensor scales left at 1: the row and column
    /// scales are applied by `Kernels::scale_rows_cols` afterwards. cuBLASLt's outer-vector FP8 scaling
    /// (`OUTER_VEC_32F`) runs on Hopper only and returns NOT_SUPPORTED on Blackwell, so the rowwise matmul is
    /// this scalar one plus a rescale; the products are exact in f32 either way and the result is rounded to bf16
    /// once, as `torch._scaled_mm` does.
    pub fn fp8(
        &self,
        x_values: &CudaSlice<u8>,
        w: &Fp8Weight,
        acc: &mut CudaSlice<f32>,
        m: usize,
    ) -> Result<(), CudaError> {
        let (n, k) = (w.n, w.k);
        let m = bucket_rows(m);
        debug_assert!(x_values.len() >= m * k && acc.len() >= m * n);
        unsafe {
            let (w_ptr, _r3) = w.values.device_ptr(&self.stream);
            let (x_ptr, _r4) = x_values.device_ptr(&self.stream);
            let (acc_ptr, _r5) = acc.device_ptr_mut(&self.stream);
            self.run(
                (w_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_8F_E4M3, k as i64),
                (x_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_8F_E4M3, k as i64),
                (acc_ptr as *mut c_void, sys::cudaDataType_t::CUDA_R_32F),
                m,
                n,
                k,
                1.0,
                None,
                None,
                n,
                0.0,
            )
        }
    }

    /// `out[M, N] bf16 = x[M, K] bf16 · w^T (+ bias)` with M rounded up to the row bucket: for workspace buffers,
    /// whose rows are a multiple of it.
    pub fn bf16_padded(&self, x: &CudaSlice<u16>, w: &Bf16Weight, out: &mut CudaSlice<u16>, m: usize) -> Result<(), CudaError> {
        self.bf16(x, w, out, bucket_rows(m))
    }

    /// `out[M, N] bf16 = x[M, K] bf16 · w^T (+ bias)` with exactly M rows.
    pub fn bf16(&self, x: &CudaSlice<u16>, w: &Bf16Weight, out: &mut CudaSlice<u16>, m: usize) -> Result<(), CudaError> {
        let (n, k) = (w.n, w.k);
        debug_assert!(x.len() >= m * k && out.len() >= m * n);
        unsafe {
            let (w_ptr, _r1) = w.values.device_ptr(&self.stream);
            let (x_ptr, _r2) = x.device_ptr(&self.stream);
            let (out_ptr, _r3) = out.device_ptr_mut(&self.stream);
            let bias = w.bias.as_ref().map(|b| {
                let (p, _r) = b.device_ptr(&self.stream);
                p as *const c_void
            });
            self.run(
                (w_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (x_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (out_ptr as *mut c_void, sys::cudaDataType_t::CUDA_R_16BF),
                m,
                n,
                k,
                1.0,
                None,
                bias,
                n,
                0.0,
            )
        }
    }

    /// `out[m, n] (rows `out_ld` apart) = alpha · x[m, k] · w[n, k]^T`, added to what `out` holds when
    /// `accumulate`, all bf16 with f32 accumulation and one rounding. `x` and `out` are raw addresses, so the
    /// operands may be parts of wider buffers: Locate's low-rank updates read a layer's input and add to its
    /// output in place.
    #[allow(clippy::too_many_arguments)]
    pub fn bf16_into(&self, x: u64, w: &CudaSlice<u16>, n: usize, k: usize, out: u64, out_ld: usize, m: usize, alpha: f32, accumulate: bool) -> Result<(), CudaError> {
        debug_assert!(w.len() >= n * k && out_ld >= n);
        unsafe {
            let (w_ptr, _r1) = w.device_ptr(&self.stream);
            self.run(
                (w_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (x as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (out as *mut c_void, sys::cudaDataType_t::CUDA_R_16BF),
                m,
                n,
                k,
                alpha,
                None,
                None,
                out_ld,
                if accumulate { 1.0 } else { 0.0 },
            )
        }
    }

    /// `out[M, N] f32 = x[M, K] bf16 · w^T`: the readout logits, kept in f32 for the softmax.
    pub fn bf16_to_f32(&self, x: &CudaSlice<u16>, w: &Bf16Weight, out: &mut CudaSlice<f32>, m: usize) -> Result<(), CudaError> {
        let (n, k) = (w.n, w.k);
        debug_assert!(x.len() >= m * k && out.len() >= m * n && w.bias.is_none());
        unsafe {
            let (w_ptr, _r1) = w.values.device_ptr(&self.stream);
            let (x_ptr, _r2) = x.device_ptr(&self.stream);
            let (out_ptr, _r3) = out.device_ptr_mut(&self.stream);
            self.run(
                (w_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (x_ptr as *const c_void, sys::cudaDataType_t::CUDA_R_16BF, k as i64),
                (out_ptr as *mut c_void, sys::cudaDataType_t::CUDA_R_32F),
                m,
                n,
                k,
                1.0,
                None,
                None,
                n,
                0.0,
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn swizzle_matches_the_reference_layout() {
        // The reference: pad to (128r, 4c), view (rb, 128, cb, 4) -> permute (rb, cb, 128, 4) -> view (-1, 4, 32, 4)
        // -> transpose(1, 2) -> flatten. Checked by building that permutation explicitly.
        let (rows, k): (usize, usize) = (300, 1024 * 2); // 300 rows, 128 scale columns
        let cols = k / 16;
        let (rb, cb) = (rows.div_ceil(128), cols.div_ceil(4));
        let mut reference = vec![usize::MAX; rb * 128 * cb * 4];
        for r in 0..rows {
            for c in 0..cols {
                let (tile_r, in_r) = (r / 128, r % 128);
                let (tile_c, in_c) = (c / 4, c % 4);
                let tile = tile_r * cb + tile_c;
                // inside the tile, (in_r, in_c) viewed as (4, 32, 4) = (hi, lo, c), transposed to (lo, hi, c)
                let (hi, lo) = (in_r / 32, in_r % 32);
                reference[tile * 512 + lo * 16 + hi * 4 + in_c] = r * cols + c;
            }
        }
        for r in 0..rows {
            for c in 0..cols {
                assert_eq!(reference[swizzled_scale_index(r, c, k)], r * cols + c);
            }
        }
        assert_eq!(swizzled_scale_bytes(rows, k), reference.len());
    }
}
