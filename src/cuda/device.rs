//! The checkpoint on the device: every weight in the form its matmul wants, the workspace a pass runs in, and
//! the `Model` implementation the server drives.
//!
//! Loading mirrors what the Python server does at startup: NVFP4 layers keep their packed codes with the block
//! scales re-laid into cuBLASLt's tiled layout; the gate and up projections are merged into one matmul; the
//! linear-attention projections (bf16 in the checkpoint) are quantized to FP8 with one scale per output row;
//! everything else stays bf16. `A_log` is stored as exp(A_log), the only form the recurrence uses.

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaContext, CudaSlice, CudaStream};
use parking_lot::Mutex;

use super::blas::{Bf16Weight, Fp4Weight, Fp8Weight, Lt, swizzled_scale_bytes, swizzled_scale_index};
use super::kernels::Kernels;
use super::{CudaError, forward};
use crate::config::Checkpoint;
use crate::images::ImageSettings;
use crate::model::{Budget, Model, PassError, RowAnswers};
use crate::protocol::{Encoded, Encoder, Tokenize};
use crate::weights::Weights;

/// Threads used for the CPU-side weight conversions at load.
const LOAD_THREADS: usize = 8;

/// GPU memory to keep free beyond the weights before loading the embedding table and vision tower on the device.
const LOW_MEMORY_HEADROOM: usize = 4 << 30;

/// The vision tower's largest input: images per request x patches of the largest image.
const MAX_IMAGES: usize = 4;

pub struct FullAttention {
    pub q: Fp4Weight,
    pub k: Fp4Weight,
    pub v: Fp4Weight,
    pub o: Fp4Weight,
    pub q_norm: CudaSlice<u16>,
    pub k_norm: CudaSlice<u16>,
}

pub struct LinearAttention {
    pub qkv: Fp8Weight,
    pub z: Fp8Weight,
    /// in_proj_a and in_proj_b stacked: [96, 5120].
    pub ab: Bf16Weight,
    /// conv1d.weight as [10240, 4].
    pub conv: CudaSlice<u16>,
    pub a_log_exp: CudaSlice<f32>,
    pub dt_bias: CudaSlice<f32>,
    pub norm: CudaSlice<u16>,
    pub out: Fp8Weight,
}

pub enum Mixer {
    Full(FullAttention),
    Linear(LinearAttention),
}

pub struct TextLayer {
    pub input_norm: CudaSlice<u16>,
    pub post_norm: CudaSlice<u16>,
    pub mixer: Mixer,
    /// gate_proj and up_proj stacked: [2 * intermediate, hidden].
    pub gate_up: Fp4Weight,
    pub down: Fp4Weight,
}

pub struct VisionBlock {
    pub norm1: (CudaSlice<u16>, CudaSlice<u16>),
    pub norm2: (CudaSlice<u16>, CudaSlice<u16>),
    pub qkv: Bf16Weight,
    pub proj: Bf16Weight,
    pub fc1: Bf16Weight,
    pub fc2: Bf16Weight,
}

pub struct Vision {
    pub patch_embed: Bf16Weight,
    pub pos_embed: CudaSlice<u16>,
    pub blocks: Vec<VisionBlock>,
    pub merger_norm: (CudaSlice<u16>, CudaSlice<u16>),
    pub merger_fc1: Bf16Weight,
    pub merger_fc2: Bf16Weight,
}

/// Where the embedding table lives.
pub enum Embeddings {
    Device(CudaSlice<u16>),
    /// Low-memory mode: rows are gathered on the CPU from the memory-mapped checkpoint.
    Host { weights: Arc<Weights>, name: String },
}

pub struct DeviceWeights {
    pub embeddings: Embeddings,
    pub layers: Vec<TextLayer>,
    pub final_norm: CudaSlice<u16>,
    pub readout: Bf16Weight,
    pub vision: Option<Vision>,
}

/// Buffers for one pass of up to `max_tokens` tokens, allocated once.
pub struct Workspace {
    pub max_tokens: usize,
    pub h: CudaSlice<u16>,
    pub normed: CudaSlice<u16>,
    pub fp4_packed: CudaSlice<u8>,
    pub fp4_scales: CudaSlice<u8>,
    pub fp8_values: CudaSlice<u8>,
    pub fp8_scales: CudaSlice<f32>,
    /// [N, 2 * intermediate]: the merged gate/up output; also the q_proj and linear qkv outputs.
    pub wide: CudaSlice<u16>,
    /// [N, 10240]: the conv output; also the k_proj / v_proj outputs and the attention output.
    pub mid: CudaSlice<u16>,
    /// [N, 6144]: z; also the rotated q.
    pub z: CudaSlice<u16>,
    /// [N, 6144]: the gated-norm output / attention output before o_proj.
    pub mixed: CudaSlice<u16>,
    /// [N, 1024] x2: rotated k, and v.
    pub k_rot: CudaSlice<u16>,
    pub v: CudaSlice<u16>,
    /// [N, 96]: a | b.
    pub ab: CudaSlice<u16>,
    /// [N, 5120]: the mixer's projected output / the MLP output.
    pub out: CudaSlice<u16>,
    pub new_values: CudaSlice<f32>,
    pub k_cumdecay: CudaSlice<f32>,
    pub cum_decay: CudaSlice<f32>,
    pub intra: CudaSlice<f32>,
    pub max_chunks: usize,
    pub positions: CudaSlice<i32>,
    pub seq_of: CudaSlice<i32>,
    pub ids: CudaSlice<i32>,
    /// Vision: patches x hidden and the wider intermediates.
    pub max_patches: usize,
    pub vx: CudaSlice<u16>,
    pub vnormed: CudaSlice<u16>,
    pub vqkv: CudaSlice<u16>,
    pub vattn: CudaSlice<u16>,
    pub vmlp: CudaSlice<u16>,
    pub vout: CudaSlice<u16>,
    pub vmerged: CudaSlice<u16>,
    pub vembeds: CudaSlice<u16>,
    pub vpixels: CudaSlice<u16>,
}

pub struct Engine {
    pub stream: Arc<CudaStream>,
    pub lt: Lt,
    pub kernels: Kernels,
    pub weights: DeviceWeights,
    pub workspace: Workspace,
}

pub struct GpuModel {
    pub checkpoint: Checkpoint,
    pub low_memory: bool,
    encoder: Encoder,
    budget: Budget,
    max_length: usize,
    engine: Mutex<Engine>,
}

// --- CPU-side conversions -------------------------------------------------------------------------------------

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits((bits as u32) << 16)
}

/// float8 e4m3fn, round to nearest even, saturating at 448 (the callers clamp first, as the Python side does).
pub fn f32_to_e4m3(v: f32) -> u8 {
    if v.is_nan() {
        return 0x7f;
    }
    let sign = if v.is_sign_negative() { 0x80u8 } else { 0 };
    let a = v.abs();
    if a < 2f32.powi(-6) {
        // Subnormal: multiples of 2^-9.
        let q = (a * 2f32.powi(9)).round_ties_even() as u32;
        return if q >= 8 { sign | 0x08 } else { sign | q as u8 };
    }
    let mut e = a.log2().floor() as i32;
    // log2 rounding can misplace the exponent by one at powers of two; fix by checking the mantissa range.
    if a < 2f32.powi(e) {
        e -= 1;
    } else if a >= 2f32.powi(e + 1) {
        e += 1;
    }
    let mut q = ((a / 2f32.powi(e) - 1.0) * 8.0).round_ties_even() as u32;
    if q == 8 {
        e += 1;
        q = 0;
    }
    if e > 8 || (e == 8 && q > 7) {
        return sign | 0x7e; // 448
    }
    sign | (((e + 7) as u8) << 3) | q as u8
}

/// Quantize bf16 rows [n, k] to e4m3 with one f32 scale per row, as fp8_matmul.py's `quantize(rowwise=True)`.
fn quantize_fp8_rows(bits: &[u8], n: usize, k: usize) -> (Vec<u8>, Vec<f32>) {
    let mut values = vec![0u8; n * k];
    let mut scales = vec![0f32; n];
    let rows_per = n.div_ceil(LOAD_THREADS);
    std::thread::scope(|scope| {
        for ((vchunk, schunk), r0) in values.chunks_mut(rows_per * k).zip(scales.chunks_mut(rows_per)).zip((0..n).step_by(rows_per)) {
            let bits = &bits[r0 * k * 2..];
            scope.spawn(move || {
                for (i, (vrow, scale)) in vchunk.chunks_mut(k).zip(schunk.iter_mut()).enumerate() {
                    let src = &bits[i * k * 2..(i + 1) * k * 2];
                    let row: Vec<f32> = src.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect();
                    let amax = row.iter().fold(0f32, |m, v| m.max(v.abs()));
                    let s = (amax / 448.0).max(1e-12);
                    *scale = s;
                    for (dst, v) in vrow.iter_mut().zip(&row) {
                        *dst = f32_to_e4m3((v / s).clamp(-448.0, 448.0));
                    }
                }
            });
        }
    });
    (values, scales)
}

fn u16s(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
}

fn f32s_of_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(2).map(|b| bf16_to_f32(u16::from_le_bytes([b[0], b[1]]))).collect()
}

struct Loader<'a> {
    stream: &'a Arc<CudaStream>,
    weights: &'a Weights,
}

impl Loader<'_> {
    fn vec_bf16(&self, name: &str, len: usize) -> Result<CudaSlice<u16>, CudaError> {
        let (bytes, _) = self.weights.bf16(name, &[Some(len)])?;
        Ok(self.stream.clone_htod(&u16s(bytes))?)
    }

    fn vec_f32_of_bf16(&self, name: &str, len: usize, map: impl Fn(f32) -> f32) -> Result<CudaSlice<f32>, CudaError> {
        let (bytes, _) = self.weights.bf16(name, &[Some(len)])?;
        let values: Vec<f32> = f32s_of_bf16(bytes).into_iter().map(map).collect();
        Ok(self.stream.clone_htod(&values)?)
    }

    fn bf16_matrix(&self, name: &str, n: usize, k: usize, bias: Option<&str>) -> Result<Bf16Weight, CudaError> {
        let (bytes, _) = self.weights.bf16(name, &[Some(n), Some(k)])?;
        let values = self.stream.clone_htod(&u16s(bytes))?;
        let bias = match bias {
            Some(name) => Some(self.vec_bf16(name, n)?),
            None => None,
        };
        Ok(Bf16Weight { values, bias, n, k })
    }

    /// Several bf16 [n_i, k] matrices stacked along the rows.
    fn bf16_stacked(&self, names: &[(&str, usize)], k: usize) -> Result<Bf16Weight, CudaError> {
        let mut all = Vec::new();
        for (name, n) in names {
            let (bytes, _) = self.weights.bf16(name, &[Some(*n), Some(k)])?;
            all.extend(u16s(bytes));
        }
        let n: usize = names.iter().map(|(_, n)| n).sum();
        Ok(Bf16Weight { values: self.stream.clone_htod(&all)?, bias: None, n, k })
    }

    fn fp8(&self, name: &str, n: usize, k: usize) -> Result<Fp8Weight, CudaError> {
        let (bytes, _) = self.weights.bf16(name, &[Some(n), Some(k)])?;
        let (values, scales) = quantize_fp8_rows(bytes, n, k);
        Ok(Fp8Weight { values: self.stream.clone_htod(&values)?, scale: self.stream.clone_htod(&scales)?, n, k })
    }

    /// NVFP4 layers stacked along the rows (one, or gate and up); their input scales must agree.
    fn fp4(&self, prefixes: &[&str], n_each: usize, k: usize) -> Result<Fp4Weight, CudaError> {
        let n = n_each * prefixes.len();
        let mut packed = self.stream.alloc_zeros::<u8>(n * k / 2)?;
        let mut scales = vec![0u8; swizzled_scale_bytes(n, k)];
        let mut decode = None;
        let mut input_encode = None;
        for (i, prefix) in prefixes.iter().enumerate() {
            let layer = self.weights.nvfp4(prefix, n_each, k)?;
            let mut view = packed.slice_mut(i * n_each * k / 2..(i + 1) * n_each * k / 2);
            self.stream.memcpy_htod(layer.packed, &mut view)?;
            let cols = k / 16;
            for row in 0..n_each {
                let src = &layer.scale[row * cols..(row + 1) * cols];
                for (col, &s) in src.iter().enumerate() {
                    scales[swizzled_scale_index(i * n_each + row, col, k)] = s;
                }
            }
            let d = 1.0 / layer.weight_global_scale;
            if let Some(previous) = decode.replace(d)
                && previous != d
            {
                return Err(CudaError::Invalid(format!("{prefix}: weight_global_scale differs from the layer it is merged with")));
            }
            if let Some(previous) = input_encode.replace(layer.input_global_scale)
                && previous != layer.input_global_scale
            {
                return Err(CudaError::Invalid(format!("{prefix}: input_global_scale differs from the layer it is merged with")));
            }
        }
        Ok(Fp4Weight {
            packed,
            scale: self.stream.clone_htod(&scales)?,
            decode: decode.unwrap(),
            input_encode: input_encode.unwrap(),
            n,
            k,
        })
    }

    fn layer_norm(&self, prefix: &str, n: usize) -> Result<(CudaSlice<u16>, CudaSlice<u16>), CudaError> {
        Ok((self.vec_bf16(&format!("{prefix}.weight"), n)?, self.vec_bf16(&format!("{prefix}.bias"), n)?))
    }
}

/// Bytes of every tensor of the text layers, for the memory check.
fn tensor_bytes(weights: &Weights, filter: impl Fn(&str) -> bool) -> usize {
    weights
        .names()
        .filter(|name| filter(name))
        .filter_map(|name| weights.get(name).ok())
        .map(|view| view.data().len())
        .sum()
}

impl GpuModel {
    pub fn load(path: &Path, budget: Budget) -> Result<Self, CudaError> {
        let checkpoint = Checkpoint::load(path)?;
        let text = &checkpoint.model.text_config;
        let vision_cfg = &checkpoint.model.vision_config;
        let weights = Arc::new(Weights::open(path)?);
        let tokenizer = tokenizers::Tokenizer::from_file(path.join("tokenizer.json")).map_err(|e| CudaError::Invalid(format!("tokenizer.json: {e}")))?;
        let pad_id = tokenizer.token_to_id("<|endoftext|>").ok_or_else(|| CudaError::Invalid("tokenizer has no <|endoftext|>".into()))?;
        let encoder = Encoder::new(Box::new(tokenizer), checkpoint.decision.codes.clone(), pad_id, checkpoint.image_processor.merge_size)
            .map_err(|e| CudaError::Invalid(e.to_string()))?;
        if encoder.image_pad_id != checkpoint.model.image_token_id {
            return Err(CudaError::Invalid("the tokenizer's <|image_pad|> is not config.json's image_token_id".into()));
        }
        let max_length: usize = std::env::var("DECISION_MAX_LENGTH").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);

        let ctx = CudaContext::new(0)?;
        let stream = ctx.default_stream();
        let (free, total) = ctx.mem_get_info()?;
        tracing::info!("GPU 0: {:.1} GiB free of {:.1} GiB", free as f64 / (1u64 << 30) as f64, total as f64 / (1u64 << 30) as f64);
        let kernels = Kernels::load(stream.clone())?;
        let lt = Lt::new(stream.clone())?;
        let loader = Loader { stream: &stream, weights: &weights };

        // Memory: the text layers roughly as they will sit on the device (FP4 as stored, FP8 at one byte per bf16
        // pair's half), plus the embedding table and vision tower if they fit with headroom.
        let text_bytes = tensor_bytes(&weights, |n| n.starts_with("language_model.layers."));
        let embed_bytes = tensor_bytes(&weights, |n| n == "language_model.embed_tokens.weight");
        let vision_bytes = tensor_bytes(&weights, |n| n.starts_with("visual."));
        let fp8_saving = tensor_bytes(&weights, |n| n.contains(".linear_attn.in_proj_qkv.") || n.contains(".linear_attn.in_proj_z.") || n.contains(".linear_attn.out_proj.")) / 2;
        let needed = text_bytes - fp8_saving + embed_bytes + vision_bytes;
        let low_memory = match std::env::var("DECISION_LOW_MEMORY").as_deref() {
            Ok("1") => true,
            Ok("0") => false,
            _ => free < needed + LOW_MEMORY_HEADROOM,
        };
        if low_memory {
            tracing::warn!("Low memory mode: embeddings on the CPU, no vision tower ({:.1} GiB needed, {:.1} GiB free)", needed as f64 / (1u64 << 30) as f64, free as f64 / (1u64 << 30) as f64);
        }

        let (hidden, inter) = (text.hidden_size, text.intermediate_size);
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for i in 0..text.num_hidden_layers {
            let p = format!("language_model.layers.{i}");
            let mixer = if text.is_full_attention(i) {
                let q_dim = text.num_attention_heads * text.head_dim;
                let kv_dim = text.num_key_value_heads * text.head_dim;
                Mixer::Full(FullAttention {
                    q: loader.fp4(&[&format!("{p}.self_attn.q_proj")], 2 * q_dim, hidden)?,
                    k: loader.fp4(&[&format!("{p}.self_attn.k_proj")], kv_dim, hidden)?,
                    v: loader.fp4(&[&format!("{p}.self_attn.v_proj")], kv_dim, hidden)?,
                    o: loader.fp4(&[&format!("{p}.self_attn.o_proj")], hidden, q_dim)?,
                    q_norm: loader.vec_bf16(&format!("{p}.self_attn.q_norm.weight"), text.head_dim)?,
                    k_norm: loader.vec_bf16(&format!("{p}.self_attn.k_norm.weight"), text.head_dim)?,
                })
            } else {
                let qkv_dim = text.linear_qkv_dim();
                let v_dim = text.linear_value_dim();
                let heads = text.linear_num_value_heads;
                let (conv_bytes, _) = weights.bf16(&format!("{p}.linear_attn.conv1d.weight"), &[Some(qkv_dim), Some(1), Some(text.linear_conv_kernel_dim)])?;
                Mixer::Linear(LinearAttention {
                    qkv: loader.fp8(&format!("{p}.linear_attn.in_proj_qkv.weight"), qkv_dim, hidden)?,
                    z: loader.fp8(&format!("{p}.linear_attn.in_proj_z.weight"), v_dim, hidden)?,
                    ab: loader.bf16_stacked(&[(&format!("{p}.linear_attn.in_proj_a.weight"), heads), (&format!("{p}.linear_attn.in_proj_b.weight"), heads)], hidden)?,
                    conv: stream.clone_htod(&u16s(conv_bytes))?,
                    a_log_exp: loader.vec_f32_of_bf16(&format!("{p}.linear_attn.A_log"), heads, f32::exp)?,
                    dt_bias: loader.vec_f32_of_bf16(&format!("{p}.linear_attn.dt_bias"), heads, |v| v)?,
                    norm: loader.vec_bf16(&format!("{p}.linear_attn.norm.weight"), text.linear_value_head_dim)?,
                    out: loader.fp8(&format!("{p}.linear_attn.out_proj.weight"), hidden, v_dim)?,
                })
            };
            layers.push(TextLayer {
                input_norm: loader.vec_bf16(&format!("{p}.input_layernorm.weight"), hidden)?,
                post_norm: loader.vec_bf16(&format!("{p}.post_attention_layernorm.weight"), hidden)?,
                mixer,
                gate_up: loader.fp4(&[&format!("{p}.mlp.gate_proj"), &format!("{p}.mlp.up_proj")], inter, hidden)?,
                down: loader.fp4(&[&format!("{p}.mlp.down_proj")], hidden, inter)?,
            });
            if i % 8 == 7 {
                tracing::info!("Loaded {} of {} layers", i + 1, text.num_hidden_layers);
            }
        }
        let final_norm = loader.vec_bf16("language_model.norm.weight", hidden)?;
        let readout = loader.bf16_matrix("weight", crate::config::MAX_OPTIONS, hidden, None)?;
        let embeddings = if low_memory {
            Embeddings::Host { weights: Arc::clone(&weights), name: "language_model.embed_tokens.weight".into() }
        } else {
            let (bytes, _) = weights.bf16("language_model.embed_tokens.weight", &[Some(text.vocab_size), Some(hidden)])?;
            Embeddings::Device(stream.clone_htod(&u16s(bytes))?)
        };
        let vision = if low_memory {
            None
        } else {
            let vh = vision_cfg.hidden_size;
            let patch_len = vision_cfg.in_channels * vision_cfg.temporal_patch_size * vision_cfg.patch_size * vision_cfg.patch_size;
            let (pe_bytes, _) = weights.bf16("visual.patch_embed.proj.weight", &[Some(vh), Some(vision_cfg.in_channels), Some(vision_cfg.temporal_patch_size), Some(vision_cfg.patch_size), Some(vision_cfg.patch_size)])?;
            let patch_embed = Bf16Weight { values: stream.clone_htod(&u16s(pe_bytes))?, bias: Some(loader.vec_bf16("visual.patch_embed.proj.bias", vh)?), n: vh, k: patch_len };
            let mut blocks = Vec::with_capacity(vision_cfg.depth);
            for b in 0..vision_cfg.depth {
                let p = format!("visual.blocks.{b}");
                blocks.push(VisionBlock {
                    norm1: loader.layer_norm(&format!("{p}.norm1"), vh)?,
                    norm2: loader.layer_norm(&format!("{p}.norm2"), vh)?,
                    qkv: loader.bf16_matrix(&format!("{p}.attn.qkv.weight"), 3 * vh, vh, Some(&format!("{p}.attn.qkv.bias")))?,
                    proj: loader.bf16_matrix(&format!("{p}.attn.proj.weight"), vh, vh, Some(&format!("{p}.attn.proj.bias")))?,
                    fc1: loader.bf16_matrix(&format!("{p}.mlp.linear_fc1.weight"), vision_cfg.intermediate_size, vh, Some(&format!("{p}.mlp.linear_fc1.bias")))?,
                    fc2: loader.bf16_matrix(&format!("{p}.mlp.linear_fc2.weight"), vh, vision_cfg.intermediate_size, Some(&format!("{p}.mlp.linear_fc2.bias")))?,
                });
            }
            let merged = vh * vision_cfg.spatial_merge_size * vision_cfg.spatial_merge_size;
            Some(Vision {
                patch_embed,
                pos_embed: {
                    let (bytes, _) = weights.bf16("visual.pos_embed.weight", &[Some(vision_cfg.num_position_embeddings), Some(vh)])?;
                    stream.clone_htod(&u16s(bytes))?
                },
                blocks,
                merger_norm: loader.layer_norm("visual.merger.norm", vh)?,
                merger_fc1: loader.bf16_matrix("visual.merger.linear_fc1.weight", merged, merged, Some("visual.merger.linear_fc1.bias"))?,
                merger_fc2: loader.bf16_matrix("visual.merger.linear_fc2.weight", vision_cfg.out_hidden_size, merged, Some("visual.merger.linear_fc2.bias"))?,
            })
        };
        let weights_on_device = DeviceWeights { embeddings, layers, final_norm, readout, vision };

        // A multiple of 16: the narrow-type matmuls round their row count up to one.
        let max_tokens = budget.max_tokens.max(64).div_ceil(16) * 16;
        let workspace = Workspace::allocate(&stream, &checkpoint, max_tokens, !low_memory)?;
        let (free_after, _) = ctx.mem_get_info()?;
        tracing::info!("Weights and workspace on the device; {:.1} GiB free", free_after as f64 / (1u64 << 30) as f64);
        let engine = Engine { stream, lt, kernels, weights: weights_on_device, workspace };
        Ok(Self { checkpoint, low_memory, encoder, budget, max_length, engine: Mutex::new(engine) })
    }
}

impl Workspace {
    fn allocate(stream: &Arc<CudaStream>, checkpoint: &Checkpoint, n: usize, vision: bool) -> Result<Self, CudaError> {
        let text = &checkpoint.model.text_config;
        let vc = &checkpoint.model.vision_config;
        let hidden = text.hidden_size;
        let wide = 2 * text.intermediate_size;
        let qkv_dim = text.linear_qkv_dim();
        let v_dim = text.linear_value_dim();
        let heads = text.linear_num_value_heads;
        let max_chunks = n.div_ceil(super::kernels::DELTA_CHUNK) + 64;
        let max_patches = if vision { MAX_IMAGES * (checkpoint.image_processor.size.longest_edge as usize / (vc.patch_size * vc.patch_size)) } else { 64 };
        let vh = vc.hidden_size;
        let patch_len = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size;
        let merged = vh * vc.spatial_merge_size * vc.spatial_merge_size;
        Ok(Self {
            max_tokens: n,
            h: stream.alloc_zeros(n * hidden)?,
            normed: stream.alloc_zeros(n * hidden)?,
            fp4_packed: stream.alloc_zeros(n * text.intermediate_size / 2)?,
            fp4_scales: stream.alloc_zeros(swizzled_scale_bytes(n, text.intermediate_size))?,
            fp8_values: stream.alloc_zeros(n * v_dim.max(hidden))?,
            fp8_scales: stream.alloc_zeros(n)?,
            wide: stream.alloc_zeros(n * wide)?,
            mid: stream.alloc_zeros(n * qkv_dim)?,
            z: stream.alloc_zeros(n * v_dim)?,
            mixed: stream.alloc_zeros(n * v_dim)?,
            k_rot: stream.alloc_zeros(n * text.num_key_value_heads * text.head_dim)?,
            v: stream.alloc_zeros(n * text.num_key_value_heads * text.head_dim)?,
            ab: stream.alloc_zeros(n * 2 * heads)?,
            out: stream.alloc_zeros(n * hidden)?,
            new_values: stream.alloc_zeros(n * heads * text.linear_value_head_dim)?,
            k_cumdecay: stream.alloc_zeros(n * heads * text.linear_value_head_dim)?,
            cum_decay: stream.alloc_zeros(n * heads)?,
            intra: stream.alloc_zeros(max_chunks * heads * 64 * 64)?,
            max_chunks,
            positions: stream.alloc_zeros(n * 3)?,
            seq_of: stream.alloc_zeros(n)?,
            ids: stream.alloc_zeros(n)?,
            max_patches,
            vx: stream.alloc_zeros(max_patches * vh)?,
            vnormed: stream.alloc_zeros(max_patches * vh)?,
            vqkv: stream.alloc_zeros(max_patches * 3 * vh)?,
            vattn: stream.alloc_zeros(max_patches * vh)?,
            vmlp: stream.alloc_zeros(max_patches * vc.intermediate_size)?,
            vout: stream.alloc_zeros(max_patches * vh)?,
            vmerged: stream.alloc_zeros(max_patches / 4 * merged)?,
            vembeds: stream.alloc_zeros(max_patches / 4 * vc.out_hidden_size)?,
            vpixels: stream.alloc_zeros(max_patches * patch_len)?,
        })
    }
}

impl Model for GpuModel {
    fn encoder(&self) -> &Encoder {
        &self.encoder
    }
    fn max_length(&self) -> usize {
        self.max_length
    }
    fn budget(&self) -> Budget {
        self.budget
    }
    fn image_settings(&self) -> ImageSettings {
        let p = &self.checkpoint.image_processor;
        ImageSettings {
            patch_size: p.patch_size,
            merge_size: p.merge_size,
            temporal_patch_size: p.temporal_patch_size,
            min_pixels: p.size.shortest_edge,
            max_pixels: self.checkpoint.decision.max_pixels.unwrap_or(p.size.longest_edge),
            mean: p.image_mean,
            std: p.image_std,
        }
    }
    fn run_pass(&self, rows: &[&Encoded]) -> Result<Vec<RowAnswers>, PassError> {
        let mut engine = self.engine.lock();
        forward::run(&mut engine, &self.checkpoint, rows).map_err(|error| match error {
            CudaError::OutOfMemory => PassError::OutOfMemory,
            CudaError::Invalid(message) => PassError::Request(message),
            other => PassError::Internal(other.to_string()),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::e4m3_to_f32;

    #[test]
    fn e4m3_round_trips() {
        for byte in 0..=0xffu8 {
            let value = e4m3_to_f32(byte);
            if value.is_nan() {
                continue;
            }
            assert_eq!(f32_to_e4m3(value), if value == 0.0 && byte == 0x80 { 0x80 } else { byte }, "byte {byte:#x} -> {value}");
        }
        assert_eq!(f32_to_e4m3(448.0), 0x7e);
        assert_eq!(f32_to_e4m3(1000.0), 0x7e);
        assert_eq!(f32_to_e4m3(-1.0), 0xb8);
        // Ties to even: 1.0625 is halfway between 1.0 (m=0) and 1.125 (m=1) -> 1.0; 1.1875 -> 1.25 (m=2).
        assert_eq!(f32_to_e4m3(1.0625), 0x38);
        assert_eq!(f32_to_e4m3(1.1875), 0x3a);
        assert_eq!(f32_to_e4m3(2f32.powi(-10)), 0x00); // half of the smallest subnormal: ties to even -> 0
        assert_eq!(f32_to_e4m3(3.0 * 2f32.powi(-10)), 0x02); // 1.5 subnormal steps -> ties to even -> 2
    }
}
