//! One forward pass of the text backbone over packed sequences, and the readout.
//!
//! The rows of a pass are laid end to end as one token stream; every kernel that looks across tokens
//! (attention, the causal conv, the delta rule) gets the sequence boundaries and stays inside them, so each
//! row gets the answers it would get alone. Positions restart at 0 per row, as pplx_decider's fast path does.
//!
//! Per layer, with h the bf16 residual stream [N, 5120]:
//!   linear attention: x = rms_norm(h) -> fp8 -> qkv, z (FP8 matmuls), a|b (bf16) -> conv + silu -> delta rule
//!                     -> gated norm -> fp8 -> out_proj;  h += out
//!   full attention:   x = rms_norm(h) -> fp4 -> q|gate, k, v (FP4) -> head norms + rotary -> attention
//!                     (gated) -> fp4 -> o_proj;  h += out
//!   then              h, x = h + out, rms_norm(h + out) -> fp4 -> gate|up (FP4) -> silu * up -> fp4 -> down;  h += mlp
//! Finally the marker rows are gathered, normed, and multiplied by the readout; the softmax with the saved
//! temperature happens on the CPU over each question's own options.

use cudarc::driver::CudaStream;
use super::CudaError;
use super::device::{Embeddings, Engine, Mixer};
use super::kernels::{ATTENTION_QUERIES, DELTA_CHUNK, Ptr};
use super::vision;
use crate::config::Checkpoint;
use crate::model::RowAnswers;
use crate::protocol::Encoded;

/// Everything about a pass that is decided on the CPU before any kernel runs.
pub struct Pack {
    pub n: usize,
    pub ids: Vec<i32>,
    /// (start, len) per row.
    pub seqs: Vec<(usize, usize)>,
    pub positions: Vec<i32>,
    pub seq_of: Vec<i32>,
    /// attention.cu's AttnTile: (seq_start, seq_len, q_start).
    pub attn_tiles: Vec<u32>,
    /// deltanet.cu's DeltaChunk: (seq_start, seq_len, chunk_start, seq_index).
    pub delta_chunks: Vec<u32>,
    /// deltanet.cu's DeltaSeq: (seq_start, seq_len, first_chunk, chunks).
    pub delta_seqs: Vec<u32>,
    /// Global token index of each readout marker, row by row.
    pub markers: Vec<i32>,
    /// Global token indices of the image pad tokens, image by image in pack order.
    pub image_tokens: Vec<i32>,
    /// (row, image index within the row) of each image in pack order.
    pub images: Vec<(usize, usize)>,
}

impl Pack {
    pub fn build(rows: &[&Encoded], image_pad_id: u32, merge_size: usize) -> Result<Self, CudaError> {
        let n: usize = rows.iter().map(|r| r.length()).sum();
        let mut pack = Pack {
            n,
            ids: Vec::with_capacity(n),
            seqs: Vec::with_capacity(rows.len()),
            positions: Vec::with_capacity(n * 3),
            seq_of: Vec::with_capacity(n),
            attn_tiles: Vec::new(),
            delta_chunks: Vec::new(),
            delta_seqs: Vec::new(),
            markers: Vec::new(),
            image_tokens: Vec::new(),
            images: Vec::new(),
        };
        for (row_index, row) in rows.iter().enumerate() {
            let start = pack.ids.len();
            let len = row.length();
            pack.seqs.push((start, len));
            pack.ids.extend(row.input_ids.iter().map(|&id| id as i32));
            pack.seq_of.extend(std::iter::repeat_n(start as i32, len));
            pack.markers.extend(row.markers.iter().map(|&m| (start + m) as i32));
            // Positions: text counts up; an image's tokens take (t, h, w) = (cur, cur + i, cur + j) and advance by max(h, w).
            let mut cur: i32 = 0;
            let mut t = 0;
            let mut image = 0;
            while t < len {
                if row.input_ids[t] == image_pad_id {
                    let grid = *row.images.get(image).ok_or_else(|| CudaError::Invalid("more image tokens than images".into()))?;
                    if row.pixels.len() != row.images.len() {
                        return Err(CudaError::Invalid("a row with images but no pixels".into()));
                    }
                    let (lh, lw) = (grid.h / merge_size, grid.w / merge_size);
                    let count = grid.tokens(merge_size);
                    if t + count > len || row.input_ids[t..t + count].iter().any(|&id| id != image_pad_id) {
                        return Err(CudaError::Invalid("image tokens do not match the image grid".into()));
                    }
                    for i in 0..lh {
                        for j in 0..lw {
                            pack.positions.extend([cur, cur + i as i32, cur + j as i32]);
                        }
                    }
                    pack.image_tokens.extend((start + t..start + t + count).map(|g| g as i32));
                    pack.images.push((row_index, image));
                    cur += lh.max(lw) as i32;
                    t += count;
                    image += 1;
                } else {
                    pack.positions.extend([cur, cur, cur]);
                    cur += 1;
                    t += 1;
                }
            }
            if image != row.images.len() {
                return Err(CudaError::Invalid("fewer image tokens than images".into()));
            }
            for q_start in (0..len).step_by(ATTENTION_QUERIES) {
                pack.attn_tiles.extend([start as u32, len as u32, q_start as u32]);
            }
            let first_chunk = pack.delta_chunks.len() / 4;
            let chunks = len.div_ceil(DELTA_CHUNK);
            for c in 0..chunks {
                pack.delta_chunks.extend([start as u32, len as u32, (c * DELTA_CHUNK) as u32, row_index as u32]);
            }
            pack.delta_seqs.extend([start as u32, len as u32, first_chunk as u32, chunks as u32]);
        }
        Ok(pack)
    }
}

/// The pass, from encoded rows to each question's probabilities.
pub fn run(engine: &mut Engine, checkpoint: &Checkpoint, rows: &[&Encoded]) -> Result<Vec<RowAnswers>, CudaError> {
    let text = &checkpoint.model.text_config;
    let pad_id = engine_image_pad(checkpoint);
    let pack = Pack::build(rows, pad_id, checkpoint.image_processor.merge_size)?;
    let n = pack.n;
    if n == 0 {
        return Ok(rows.iter().map(|_| Vec::new()).collect());
    }
    if n > engine.workspace.max_tokens {
        return Err(CudaError::Invalid(format!("a pass of {n} tokens exceeds the workspace of {}", engine.workspace.max_tokens)));
    }
    let stream = engine.stream.clone();
    let mut timer = Timer::new(engine.profile, &stream);
    let eps = text.rms_norm_eps as f32;
    let hidden = text.hidden_size;

    // Inputs.
    let ids = stream.clone_htod(&pack.ids)?;
    let positions = stream.clone_htod(&pack.positions)?;
    let seq_of = stream.clone_htod(&pack.seq_of)?;
    let attn_tiles = stream.clone_htod(&pack.attn_tiles)?;
    let delta_chunks = stream.clone_htod(&pack.delta_chunks)?;
    let delta_seqs = stream.clone_htod(&pack.delta_seqs)?;
    let n_tiles = pack.attn_tiles.len() / 3;
    let n_chunks = pack.delta_chunks.len() / 4;
    if n_chunks > engine.workspace.max_chunks {
        return Err(CudaError::Invalid("too many sequences for the delta-rule workspace".into()));
    }

    timer.tick("inputs");

    // Embeddings.
    let Engine { stream: _, lt, kernels, weights, workspace: ws, profile: _, last_profile } = engine;
    match &weights.embeddings {
        Embeddings::Device(table) => kernels.gather_rows(table, &ids, &mut ws.h, n, hidden)?,
        Embeddings::Host { weights: host, name } => {
            let (bytes, _) = host.bf16(name, &[Some(text.vocab_size), Some(hidden)])?;
            let mut gathered = vec![0u16; n * hidden];
            for (i, &id) in pack.ids.iter().enumerate() {
                let src = &bytes[id as usize * hidden * 2..(id as usize + 1) * hidden * 2];
                for (j, pair) in src.chunks_exact(2).enumerate() {
                    gathered[i * hidden + j] = u16::from_le_bytes([pair[0], pair[1]]);
                }
            }
            let mut view = ws.h.slice_mut(0..n * hidden);
            stream.memcpy_htod(&gathered, &mut view)?;
        }
    }
    if !pack.images.is_empty() {
        let Some(vision_weights) = &weights.vision else {
            return Err(CudaError::Invalid("This server runs text-only (low memory mode), so it can't read images.".into()));
        };
        let images: Vec<&crate::images::PreparedImage> = pack.images.iter().map(|&(row, i)| rows[row].pixels[i].as_ref()).collect();
        let merged = vision::run(&stream, lt, kernels, vision_weights, ws, checkpoint, &images)?;
        if merged != pack.image_tokens.len() {
            return Err(CudaError::Invalid("the vision tower produced a different number of tokens than the prompt holds".into()));
        }
        let image_tokens = stream.clone_htod(&pack.image_tokens)?;
        kernels.scatter_rows(&ws.vembeds, &image_tokens, &mut ws.h, merged, hidden)?;
        timer.tick("vision");
    }
    timer.tick("embed");

    // The layers.
    for layer in &weights.layers {
        match &layer.mixer {
            Mixer::Linear(lin) => {
                let qkv_dim = lin.qkv.n;
                let v_dim = lin.z.n;
                kernels.rms_norm(&ws.h, None, &layer.input_norm, &mut ws.normed, n, hidden, eps)?;
                timer.tick("norm");
                kernels.quantize_fp8(&ws.normed, &mut ws.fp8_values, &mut ws.fp8_scales, n, hidden)?;
                timer.tick("quant8");
                lt.fp8(&ws.fp8_values, &lin.qkv, &mut ws.fp8_acc, n)?;
                timer.tick("fp8mm");
                kernels.scale_rows_cols(&ws.fp8_acc, &ws.fp8_scales, &lin.qkv.scale, &mut ws.wide, n, qkv_dim)?;
                timer.tick("rescale");
                lt.fp8(&ws.fp8_values, &lin.z, &mut ws.fp8_acc, n)?;
                timer.tick("fp8mm");
                kernels.scale_rows_cols(&ws.fp8_acc, &ws.fp8_scales, &lin.z.scale, &mut ws.z, n, v_dim)?;
                timer.tick("rescale");
                lt.bf16(&ws.normed, &lin.ab, &mut ws.ab, n)?;
                timer.tick("bf16mm");
                kernels.conv1d_silu(&ws.wide, &lin.conv, &seq_of, &mut ws.mid, n, qkv_dim)?;
                timer.tick("conv");
                kernels.delta_chunk(&delta_chunks, n_chunks, &ws.mid, &ws.ab, &lin.a_log_exp, &lin.dt_bias, &mut ws.delta_u, &mut ws.delta_w, &mut ws.delta_intra, &mut ws.cum_decay)?;
                timer.tick("delta.chunk");
                kernels.delta_state(&delta_seqs, pack.seqs.len(), &delta_chunks, &ws.mid, &ws.delta_u, &ws.delta_w, &ws.cum_decay, &mut ws.delta_states, &mut ws.delta_vnew)?;
                timer.tick("delta.state");
                kernels.delta_out(&delta_chunks, n_chunks, &ws.mid, &ws.cum_decay, &ws.delta_intra, &ws.delta_states, &ws.delta_vnew, &mut ws.mixed)?;
                timer.tick("delta.out");
                // mixed holds the delta-rule output [N, 48, 128]; the gated norm writes out_proj's input (into
                // `wide`, whose qkv the conv has consumed).
                kernels.gated_norm(&ws.mixed, &ws.z, &lin.norm, &mut ws.wide, n, eps)?;
                timer.tick("gnorm");
                kernels.quantize_fp8(&ws.wide, &mut ws.fp8_values, &mut ws.fp8_scales, n, v_dim)?;
                timer.tick("quant8");
                lt.fp8(&ws.fp8_values, &lin.out, &mut ws.fp8_acc, n)?;
                timer.tick("fp8mm");
                kernels.scale_rows_cols(&ws.fp8_acc, &ws.fp8_scales, &lin.out.scale, &mut ws.out, n, hidden)?;
                timer.tick("rescale");
            }
            Mixer::Full(full) => {
                let heads = text.num_attention_heads;
                let kv_heads = text.num_key_value_heads;
                let head_dim = text.head_dim;
                kernels.rms_norm(&ws.h, None, &layer.input_norm, &mut ws.normed, n, hidden, eps)?;
                timer.tick("norm");
                // q, k and v share the normed input; it is quantized once per distinct input scale.
                let mut quantized_for = f32::NAN;
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.q, &mut ws.wide, n, hidden)?;
                timer.tick("fp4mm");
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.k, &mut ws.mid, n, hidden)?;
                timer.tick("fp4mm");
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.v, &mut ws.v, n, hidden)?;
                timer.tick("fp4mm");
                let rot = text.rotary_dim();
                kernels.head_norm_rotary(&ws.wide, &ws.mid, &full.q_norm, &full.k_norm, &positions, &mut ws.z, &mut ws.k_rot, n, heads, kv_heads, rot, text.rope_parameters.rope_theta as f32, eps)?;
                timer.tick("rope");
                let gate = Ptr::of(&ws.wide, head_dim, &stream);
                let scale = 1.0 / (head_dim as f32).sqrt();
                kernels.attention(&attn_tiles, n_tiles, &ws.z, &ws.k_rot, &ws.v, gate, &mut ws.mixed, heads, kv_heads, scale)?;
                timer.tick("attention");
                kernels.quantize_fp4(Ptr::of(&ws.mixed, 0, &stream), &mut ws.fp4_packed, &mut ws.fp4_scales, n, heads * head_dim, full.o.input_encode)?;
                timer.tick("quant4");
                lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &full.o, &mut ws.out, n)?;
                timer.tick("fp4mm");
            }
        }
        // h += out; x = rms_norm(h) -> fp4; gate|up; silu * up -> fp4; down; h += mlp.
        kernels.rms_norm_fp4(&ws.out, Some(&mut ws.h), &layer.post_norm, &mut ws.fp4_packed, &mut ws.fp4_scales, n, hidden, eps, layer.gate_up.input_encode)?;
        timer.tick("norm");
        lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &layer.gate_up, &mut ws.wide, n)?;
        timer.tick("fp4mm");
        kernels.silu_mul_fp4(&ws.wide, &mut ws.fp4_packed, &mut ws.fp4_scales, n, text.intermediate_size, layer.down.input_encode)?;
        timer.tick("silu");
        lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &layer.down, &mut ws.out, n)?;
        timer.tick("fp4mm");
        kernels.add_inplace(&mut ws.h, &ws.out, n * hidden)?;
        timer.tick("add");
    }

    // Readout at the markers.
    let m = pack.markers.len();
    let markers = stream.clone_htod(&pack.markers)?;
    kernels.gather_rows(&ws.h, &markers, &mut ws.out, m, hidden)?;
    kernels.rms_norm(&ws.out, None, &weights.final_norm, &mut ws.normed, m, hidden, eps)?;
    let options = weights.readout.n;
    let mut logits = stream.alloc_zeros::<f32>(m * options)?;
    lt.bf16_to_f32(&ws.normed, &weights.readout, &mut logits, m)?;
    let mut host = vec![0f32; m * options];
    stream.memcpy_dtoh(&logits, &mut host)?;
    stream.synchronize()?;
    timer.tick("readout");
    if let Some(report) = timer.report(n) {
        tracing::info!("{report}");
        *last_profile = Some(report);
    }

    let temperature = checkpoint.decision.temperature as f32;
    let mut cursor = 0;
    let mut answers = Vec::with_capacity(rows.len());
    for row in rows {
        let mut row_answers = Vec::with_capacity(row.counts.len());
        for &count in &row.counts {
            let logits = &host[cursor * options..cursor * options + count];
            row_answers.push(softmax(logits, temperature));
            cursor += 1;
        }
        answers.push(row_answers);
    }
    Ok(answers)
}

/// Per-stage GPU time of a pass (DECISION_PROFILE=1): a stream sync after every stage, so only for profiling.
struct Timer<'a> {
    stream: Option<&'a CudaStream>,
    last: std::time::Instant,
    stages: Vec<(&'static str, f64)>,
}

impl<'a> Timer<'a> {
    fn new(enabled: bool, stream: &'a CudaStream) -> Self {
        Self { stream: enabled.then_some(stream), last: std::time::Instant::now(), stages: Vec::new() }
    }

    fn tick(&mut self, stage: &'static str) {
        let Some(stream) = self.stream else { return };
        let _ = stream.synchronize();
        let now = std::time::Instant::now();
        let ms = (now - self.last).as_secs_f64() * 1000.0;
        self.last = now;
        match self.stages.iter_mut().find(|(name, _)| *name == stage) {
            Some((_, total)) => *total += ms,
            None => self.stages.push((stage, ms)),
        }
    }

    /// `server-timing` entries, largest first: `fp4mm;dur=123.4, ...`.
    fn report(&self, tokens: usize) -> Option<String> {
        self.stream?;
        let mut stages = self.stages.clone();
        stages.sort_by(|a, b| b.1.total_cmp(&a.1));
        let total: f64 = stages.iter().map(|(_, ms)| ms).sum();
        let mut parts = vec![format!("pass;dur={total:.1};desc=\"{tokens} tokens\"")];
        parts.extend(stages.iter().map(|(name, ms)| format!("{name};dur={ms:.1}")));
        Some(parts.join(", "))
    }
}

fn engine_image_pad(checkpoint: &Checkpoint) -> u32 {
    checkpoint.model.image_token_id
}

/// One NVFP4 projection of the normed input, quantizing it first unless the previous projection already did
/// with the same input scale.
#[allow(clippy::too_many_arguments)]
fn fp4_project(
    kernels: &super::kernels::Kernels,
    lt: &super::blas::Lt,
    stream: &std::sync::Arc<cudarc::driver::CudaStream>,
    normed: &cudarc::driver::CudaSlice<u16>,
    packed: &mut cudarc::driver::CudaSlice<u8>,
    scales: &mut cudarc::driver::CudaSlice<u8>,
    quantized_for: &mut f32,
    w: &super::blas::Fp4Weight,
    out: &mut cudarc::driver::CudaSlice<u16>,
    n: usize,
    k: usize,
) -> Result<(), CudaError> {
    if *quantized_for != w.input_encode {
        kernels.quantize_fp4(Ptr::of(normed, 0, stream), packed, scales, n, k, w.input_encode)?;
        *quantized_for = w.input_encode;
    }
    lt.fp4(packed, scales, w, out, n)
}

/// softmax(logits / temperature) in f32, as the Python side does on the CPU.
pub fn softmax(logits: &[f32], temperature: f32) -> Vec<f32> {
    let scaled: Vec<f32> = logits.iter().map(|l| l / temperature).collect();
    let max = scaled.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = scaled.iter().map(|v| (v - max).exp()).collect();
    let total: f32 = exps.iter().sum();
    exps.iter().map(|e| e / total).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::{Encoded, ImageGrid};

    #[test]
    fn pack_lays_rows_end_to_end_with_image_positions() {
        let pad = 7;
        let a = Encoded { keys: vec!["q".into()], counts: vec![2], input_ids: vec![1, 2, 3], markers: vec![2], images: vec![], pixels: vec![] };
        // 4x4 patch grid = 2x2 merged = 4 image tokens between two text tokens.
        let image = crate::images::PreparedImage { grid: ImageGrid { t: 1, h: 4, w: 4 }, patches: vec![] };
        let b = Encoded {
            keys: vec!["q".into()],
            counts: vec![3],
            input_ids: vec![5, pad, pad, pad, pad, 6, 9],
            markers: vec![6],
            images: vec![image.grid],
            pixels: vec![std::sync::Arc::new(image)],
        };
        let pack = Pack::build(&[&a, &b], pad, 2).unwrap();
        assert_eq!(pack.n, 10);
        assert_eq!(pack.seqs, vec![(0, 3), (3, 7)]);
        assert_eq!(pack.markers, vec![2, 9]);
        assert_eq!(pack.seq_of, vec![0, 0, 0, 3, 3, 3, 3, 3, 3, 3]);
        // Row b positions: text 0; image (1, 1+i, 1+j); text resumes at 1 + max(2, 2) = 3, then 4.
        let b_positions = &pack.positions[9..];
        assert_eq!(b_positions, &[0, 0, 0, 1, 1, 1, 1, 1, 2, 1, 2, 1, 1, 2, 2, 3, 3, 3, 4, 4, 4]);
        assert_eq!(pack.image_tokens, vec![4, 5, 6, 7]);
        assert_eq!(pack.attn_tiles, vec![0, 3, 0, 3, 7, 0]);
        assert_eq!(pack.delta_chunks, vec![0, 3, 0, 0, 3, 7, 0, 1]);
        assert_eq!(pack.delta_seqs, vec![0, 3, 0, 1, 3, 7, 1, 1]);
    }

    #[test]
    fn softmax_applies_the_temperature() {
        let p = softmax(&[1.0, 2.0], 1.0);
        assert!((p[1] - 1.0 / (1.0 + (-1f32).exp())).abs() < 1e-6);
        let flat = softmax(&[1.0, 2.0], 1e6);
        assert!((flat[0] - 0.5).abs() < 1e-5);
    }
}
