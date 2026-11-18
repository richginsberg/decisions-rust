//! The Locate adapter on the GPU: its weights, and the forward pass of a locate sequence.
//!
//! A locate pass is the text backbone with three differences (pplx_decider's `locate.py` and the isolated
//! layout of its `model.py`):
//!
//! - **The adapter is on.** Every linear layer of the language model gets a low-rank update,
//!   `y = W x + B (A x) * alpha / rank`, computed in bf16 beside the quantized matmul and added to its output;
//!   and the `<|locate|>` marker reads its own trained embedding.
//! - **Questions are isolated.** The sequence is the context, then one block per question. In full attention
//!   the context attends everything and a block attends the context and itself (`attention_split`). In linear
//!   attention every block continues from the context's final state: the causal conv reads the context's last
//!   tokens before a block's first (`conv1d_silu_indexed`), and the delta rule's state walk runs the context
//!   once, saves its state, and starts every block from it (`delta_state_carry`). Positions restart after the
//!   context's last in every block.
//! - **The span head replaces the readout.** Its projections of the final hidden states are computed here for
//!   every token in one matmul; `crate::locate::predict` scores the spans on the CPU.
//!
//! The fused kernels of the decision path keep some activations only in quantized form; the updates need them
//! in bf16, so this pass uses the unfused ones. One sequence runs per pass, eagerly (no CUDA graph).

use std::path::Path;
use std::sync::Arc;

use cudarc::driver::{CudaSlice, CudaStream};

use super::CudaError;
use super::blas::{Bf16Weight, Lt, bucket_rows};
use super::device::{DeviceWeights, Embeddings, Engine, Mixer};
use super::forward::fp4_project;
use super::kernels::{ATTENTION_QUERIES, DELTA_CHUNK, Ptr};
use crate::config::Checkpoint;
use crate::locate::{FORMAT, HeadParams, LOCATE_MARKER, LocateConfig, MAX_QUESTIONS, Projections, TokenSpan, f32_to_bf16, predict};
use crate::protocol::Encoded;
use crate::weights::Weights;

/// One linear layer's update: A [rank, k] and B [n, rank], bf16.
pub struct Lora {
    a: CudaSlice<u16>,
    b: CudaSlice<u16>,
    n: usize,
    k: usize,
}

pub enum LocateMixer {
    Linear { qkv: Lora, z: Lora, a: Lora, b: Lora, out: Lora },
    Full { q: Lora, k: Lora, v: Lora, o: Lora },
}

pub struct LocateLayer {
    mixer: LocateMixer,
    gate: Lora,
    up: Lora,
    down: Lora,
}

pub struct LocateWeights {
    layers: Vec<LocateLayer>,
    rank: usize,
    /// alpha / rank.
    scale: f32,
    /// The marker's trained embedding, repeated for every question of a pass: [MAX_QUESTIONS, hidden].
    marker_rows: CudaSlice<u16>,
    /// The span head's linear layers stacked (ts, te, qs, qe, qp, thresholds), with their biases.
    head: Bf16Weight,
    pub params: HeadParams,
    pub marker_id: u32,
    /// [max_tokens, rank]: A x of the update being applied.
    t: CudaSlice<u16>,
    /// [48, 128, 128] f32: the delta rule's state at the end of the context.
    carry: CudaSlice<f32>,
}

fn bf16_of_f32(bytes: &[u8]) -> Vec<u16> {
    bytes.chunks_exact(4).map(|b| f32_to_bf16(f32::from_le_bytes([b[0], b[1], b[2], b[3]]))).collect()
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes.chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
}

impl LocateWeights {
    /// Load `locate-span/` beside a version 4 checkpoint. `max_tokens` sizes the pass buffers.
    pub fn load(stream: &Arc<CudaStream>, checkpoint: &Checkpoint, directory: &Path, config: &LocateConfig, max_tokens: usize) -> Result<Self, CudaError> {
        let invalid = |message: String| CudaError::Invalid(format!("locate-span: {message}"));
        if config.format != FORMAT || config.marker != LOCATE_MARKER {
            return Err(invalid(format!("expected a {FORMAT} adapter")));
        }
        if config.lora.layers != 0 {
            return Err(invalid("only all-layer adapters are supported".into()));
        }
        let text = &checkpoint.model.text_config;
        if config.marker_id as usize >= text.vocab_size {
            return Err(invalid("the Locate marker has no embedding row".into()));
        }
        let hidden = text.hidden_size;
        let rank = config.lora.rank;
        let adapter = Weights::open_files(&[directory.join("adapter.safetensors")])?;
        let lora = |prefix: &str, n: usize, k: usize| -> Result<Lora, CudaError> {
            let (a, _) = adapter.f32(&format!("{prefix}.lora_a"), &[Some(rank), Some(k)])?;
            let (b, _) = adapter.f32(&format!("{prefix}.lora_b"), &[Some(n), Some(rank)])?;
            Ok(Lora { a: stream.clone_htod(&bf16_of_f32(a))?, b: stream.clone_htod(&bf16_of_f32(b))?, n, k })
        };
        let intermediate = text.intermediate_size;
        let (qkv_dim, v_dim, value_heads) = (text.linear_qkv_dim(), text.linear_value_dim(), text.linear_num_value_heads);
        let attention = text.num_attention_heads * text.head_dim;
        let kv = text.num_key_value_heads * text.head_dim;
        let mut layers = Vec::with_capacity(text.num_hidden_layers);
        for (i, kind) in text.layer_types.iter().enumerate() {
            let p = format!("language_model.layers.{i}");
            let mixer = match kind.as_str() {
                "linear_attention" => LocateMixer::Linear {
                    qkv: lora(&format!("{p}.linear_attn.in_proj_qkv"), qkv_dim, hidden)?,
                    z: lora(&format!("{p}.linear_attn.in_proj_z"), v_dim, hidden)?,
                    a: lora(&format!("{p}.linear_attn.in_proj_a"), value_heads, hidden)?,
                    b: lora(&format!("{p}.linear_attn.in_proj_b"), value_heads, hidden)?,
                    out: lora(&format!("{p}.linear_attn.out_proj"), hidden, v_dim)?,
                },
                "full_attention" => LocateMixer::Full {
                    // q_proj also produces the attention output gate: 2 * heads * head_dim rows.
                    q: lora(&format!("{p}.self_attn.q_proj"), 2 * attention, hidden)?,
                    k: lora(&format!("{p}.self_attn.k_proj"), kv, hidden)?,
                    v: lora(&format!("{p}.self_attn.v_proj"), kv, hidden)?,
                    o: lora(&format!("{p}.self_attn.o_proj"), hidden, attention)?,
                },
                other => return Err(invalid(format!("unexpected layer type {other}"))),
            };
            layers.push(LocateLayer {
                mixer,
                gate: lora(&format!("{p}.mlp.gate_proj"), intermediate, hidden)?,
                up: lora(&format!("{p}.mlp.up_proj"), intermediate, hidden)?,
                down: lora(&format!("{p}.mlp.down_proj"), hidden, intermediate)?,
            });
        }
        let (marker, _) = adapter.f32("language_model.embed_tokens.locate", &[Some(hidden)])?;
        let marker_rows: Vec<u16> = bf16_of_f32(marker).repeat(MAX_QUESTIONS);

        // The head: five projections of `width` and the three thresholds, stacked into one biased matmul.
        let head = Weights::open_files(&[directory.join("head.safetensors")])?;
        let width = config.head.width;
        let rows = head_rows(width);
        let mut weight = vec![0u16; rows * hidden];
        let mut bias = vec![0u16; rows];
        let mut at = 0;
        for (name, n) in [("ts", width), ("te", width), ("qs", width), ("qe", width), ("qp", width), ("thresholds", 3)] {
            let (w, _) = head.f32(&format!("{name}.weight"), &[Some(n), Some(hidden)])?;
            let (b, _) = head.f32(&format!("{name}.bias"), &[Some(n)])?;
            weight[at * hidden..(at + n) * hidden].copy_from_slice(&bf16_of_f32(w));
            bias[at..at + n].copy_from_slice(&bf16_of_f32(b));
            at += n;
        }
        let (width_bias, _) = head.f32("width_bias.weight", &[Some(32), Some(1)])?;
        let params = HeadParams {
            width,
            max_span: config.head.max_span,
            proposals: config.head.proposals,
            cutoff: config.thresholds.default as f32,
            start_bias: head.scalar_f32("start_bias")?,
            end_bias: head.scalar_f32("end_bias")?,
            width_bias: f32s(width_bias).try_into().map_err(|_| invalid("width_bias must have 32 buckets".into()))?,
        };
        Ok(Self {
            layers,
            rank,
            scale: (config.lora.alpha / rank as f64) as f32,
            marker_rows: stream.clone_htod(&marker_rows)?,
            head: Bf16Weight { values: stream.clone_htod(&weight)?, bias: Some(stream.clone_htod(&bias)?), n: rows, k: hidden },
            params,
            marker_id: config.marker_id,
            t: stream.alloc_zeros(max_tokens * rank)?,
            carry: stream.alloc_zeros(text.linear_num_value_heads * text.linear_value_head_dim * text.linear_key_head_dim)?,
        })
    }
}

/// Rows of the stacked head matmul: five projections and the thresholds, padded to a multiple of 8.
fn head_rows(width: usize) -> usize {
    (5 * width + 3).div_ceil(8) * 8
}

/// The tables of a locate sequence's isolated layout, decided on the CPU.
pub struct Layout {
    pub positions: Vec<i32>,
    /// [tokens, 3]: each token's predecessors at -3, -2, -1 in its own stream (context, or context + block).
    pub conv_sources: Vec<i32>,
    /// attention.cu's AttnSplitTile, 6 u32 each.
    pub tiles: Vec<u32>,
    /// deltanet.cu's DeltaChunk, 4 u32 each: the context's chunks, then each block's.
    pub chunks: Vec<u32>,
    /// deltanet.cu's DeltaSeq, 4 u32 each: the context, then each block.
    pub seqs: Vec<u32>,
}

impl Layout {
    /// `segments`: the context's length, then each question block's; `rows`: the padded token count.
    pub fn build(segments: &[usize], rows: usize) -> Self {
        let context = segments[0];
        let total: usize = segments.iter().sum();
        let mut layout = Self { positions: vec![0; 3 * rows], conv_sources: vec![-1; 3 * rows], tiles: Vec::new(), chunks: Vec::new(), seqs: Vec::new() };
        let chunk_walk = |layout: &mut Self, start: usize, len: usize, index: usize| {
            let first = layout.chunks.len() / 4;
            let count = len.div_ceil(DELTA_CHUNK);
            for c in 0..count {
                layout.chunks.extend([start as u32, len as u32, (c * DELTA_CHUNK) as u32, index as u32]);
            }
            layout.seqs.extend([start as u32, len as u32, first as u32, count as u32]);
        };
        // The context: positions 0.., its own causal stream, queries attending the whole sequence.
        for t in 0..context {
            layout.positions[3 * t..3 * t + 3].fill(t as i32);
            for j in 0..3 {
                layout.conv_sources[3 * t + j] = t as i32 - 3 + j as i32; // negative: before the sequence
            }
        }
        for q in (0..context).step_by(ATTENTION_QUERIES) {
            layout.tiles.extend([q as u32, ATTENTION_QUERIES.min(context - q) as u32, 0, total as u32, 0, 0]);
        }
        chunk_walk(&mut layout, 0, context, 0);
        // Each question block: positions continue from the context's last; its stream is context + block.
        let mut start = context;
        for (index, &len) in segments[1..].iter().enumerate() {
            for j in 0..len {
                let t = start + j;
                layout.positions[3 * t..3 * t + 3].fill((context + j) as i32);
                for back in 1..=3usize {
                    // The token `back` places earlier in context ++ block.
                    let source = if j >= back { (t - back) as i32 } else { context as i32 - (back - j) as i32 };
                    layout.conv_sources[3 * t + 3 - back] = source.max(-1);
                }
            }
            for q in (0..len).step_by(ATTENTION_QUERIES) {
                layout.tiles.extend([(start + q) as u32, ATTENTION_QUERIES.min(len - q) as u32, 0, context as u32, start as u32, len as u32]);
            }
            chunk_walk(&mut layout, start, len, index + 1);
            start += len;
        }
        layout
    }
}

/// One low-rank update: `out[m, n] (rows out_ld apart) += scale * (x[m, k] A^T) B^T`.
#[allow(clippy::too_many_arguments)]
fn update(lt: &Lt, stream: &Arc<CudaStream>, t: &mut CudaSlice<u16>, rank: usize, scale: f32, lora: &Lora, x: Ptr, out: Ptr, out_ld: usize, m: usize) -> Result<(), CudaError> {
    let t = Ptr::of_mut(t, 0, stream);
    lt.bf16_into(x.0, &lora.a, rank, lora.k, t.0, rank, m, 1.0, false)?;
    lt.bf16_into(t.0, &lora.b, lora.n, rank, out.0, out_ld, m, scale, true)
}

/// The locate pass of one sequence: each question's token spans (indices into the row's `positions`).
pub fn run(engine: &mut Engine, checkpoint: &Checkpoint, row: &Encoded) -> Result<Vec<Vec<TokenSpan>>, CudaError> {
    let Some(plan) = row.locate.as_ref() else {
        return Err(CudaError::Invalid("not a locate sequence".into()));
    };
    let questions = row.markers.len();
    if plan.positions.is_empty() {
        return Ok(vec![Vec::new(); questions]);
    }
    let text = &checkpoint.model.text_config;
    let n = row.length();
    let nb = bucket_rows(n);
    let stream = engine.stream.clone();
    let Engine { lt, kernels, weights, workspace: ws, locate, .. } = engine;
    let Some(loc) = locate.as_mut() else {
        return Err(CudaError::Invalid("'locate' questions need a checkpoint with a locate-span/ adapter".into()));
    };
    let DeviceWeights { embeddings: Embeddings::Device(table), layers, final_norm, .. } = &*weights else {
        return Err(CudaError::Invalid("Locate is not available in low memory mode".into()));
    };
    if nb > ws.max_tokens {
        return Err(CudaError::Invalid(format!("a locate pass of {n} tokens exceeds the workspace of {}", ws.max_tokens)));
    }
    if questions > MAX_QUESTIONS || plan.segments.len() != questions + 1 || plan.segments.iter().sum::<usize>() != n {
        return Err(CudaError::Invalid("the locate sequence's segments do not match its questions".into()));
    }
    let layout = Layout::build(&plan.segments, nb);
    let (n_tiles, n_chunks, n_blocks) = (layout.tiles.len() / 6, layout.chunks.len() / 4, questions);
    if n_chunks > ws.max_chunks {
        return Err(CudaError::Invalid("too many questions for the delta-rule workspace".into()));
    }
    let eps = text.rms_norm_eps as f32;
    let hidden = text.hidden_size;
    let (rank, scale) = (loc.rank, loc.scale);

    // Inputs.
    let mut ids: Vec<i32> = row.input_ids.iter().map(|&id| id as i32).collect();
    ids.resize(nb, 0);
    stream.memcpy_htod(&ids, &mut ws.ids.slice_mut(0..nb))?;
    stream.memcpy_htod(&layout.positions, &mut ws.positions.slice_mut(0..3 * nb))?;
    let markers: Vec<i32> = row.markers.iter().map(|&m| m as i32).collect();
    stream.memcpy_htod(&markers, &mut ws.markers.slice_mut(0..questions))?;
    let conv_sources = stream.clone_htod(&layout.conv_sources)?;
    let tiles = stream.clone_htod(&layout.tiles)?;
    let chunks = stream.clone_htod(&layout.chunks)?;
    let seqs = stream.clone_htod(&layout.seqs)?;

    // Embeddings, with the marker's own vector at every `<|locate|>`.
    kernels.gather_rows(table, &ws.ids, &mut ws.h, nb, hidden)?;
    kernels.scatter_rows(&loc.marker_rows, &ws.markers, &mut ws.h, questions, hidden)?;

    for (layer, adapter) in layers.iter().zip(&loc.layers) {
        kernels.rms_norm(&ws.h, None, &layer.input_norm, &mut ws.normed, nb, hidden, eps)?;
        let normed = Ptr::of(&ws.normed, 0, &stream);
        match (&layer.mixer, &adapter.mixer) {
            (Mixer::Linear(lin), LocateMixer::Linear { qkv, z, a, b, out }) => {
                let (qkv_dim, v_dim, value_heads) = (lin.qkv.n, lin.z.n, lin.ab.n / 2);
                kernels.quantize_fp8(&ws.normed, &mut ws.fp8_values, &mut ws.fp8_scales, nb, hidden)?;
                lt.fp8(&ws.fp8_values, &lin.qkv, &mut ws.fp8_acc, nb)?;
                kernels.scale_rows_cols(&ws.fp8_acc, &ws.fp8_scales, &lin.qkv.scale, &mut ws.wide, nb, qkv_dim)?;
                update(lt, &stream, &mut loc.t, rank, scale, qkv, normed, Ptr::of_mut(&mut ws.wide, 0, &stream), qkv_dim, nb)?;
                lt.fp8(&ws.fp8_values, &lin.z, &mut ws.z_acc, nb)?;
                kernels.scale_rows_cols(&ws.z_acc, &ws.fp8_scales, &lin.z.scale, &mut ws.z, nb, v_dim)?;
                update(lt, &stream, &mut loc.t, rank, scale, z, normed, Ptr::of_mut(&mut ws.z, 0, &stream), v_dim, nb)?;
                lt.bf16_padded(&ws.normed, &lin.ab, &mut ws.ab, nb)?;
                update(lt, &stream, &mut loc.t, rank, scale, a, normed, Ptr::of_mut(&mut ws.ab, 0, &stream), 2 * value_heads, nb)?;
                update(lt, &stream, &mut loc.t, rank, scale, b, normed, Ptr::of_mut(&mut ws.ab, value_heads, &stream), 2 * value_heads, nb)?;
                kernels.conv1d_silu_indexed(&ws.wide, &lin.conv, &conv_sources, &mut ws.mid, nb, qkv_dim)?;
                kernels.delta_chunk(&chunks, n_chunks, &ws.mid, &ws.ab, &lin.a_log_exp, &lin.dt_bias, &mut ws.delta_u, &mut ws.delta_w, &mut ws.delta_intra, &mut ws.cum_decay, &mut ws.delta_kt)?;
                // The context's walk saves its final state; every block's walk starts from it.
                kernels.delta_state_carry(Ptr::of(&seqs, 0, &stream), 1, &chunks, &ws.delta_kt, &ws.delta_u, &ws.delta_w, &ws.cum_decay, &mut ws.delta_states, &mut ws.delta_vnew, &mut loc.carry, false, true)?;
                kernels.delta_state_carry(Ptr::of(&seqs, 4, &stream), n_blocks, &chunks, &ws.delta_kt, &ws.delta_u, &ws.delta_w, &ws.cum_decay, &mut ws.delta_states, &mut ws.delta_vnew, &mut loc.carry, true, false)?;
                kernels.delta_out(&chunks, n_chunks, &ws.mid, &ws.cum_decay, &ws.delta_intra, &ws.delta_states, &ws.delta_vnew, &mut ws.mixed)?;
                kernels.gated_norm(&ws.mixed, &ws.z, &lin.norm, &mut ws.wide, nb, eps)?;
                kernels.quantize_fp8(&ws.wide, &mut ws.fp8_values, &mut ws.fp8_scales, nb, v_dim)?;
                lt.fp8(&ws.fp8_values, &lin.out, &mut ws.fp8_acc, nb)?;
                kernels.scale_rows_cols(&ws.fp8_acc, &ws.fp8_scales, &lin.out.scale, &mut ws.out, nb, hidden)?;
                update(lt, &stream, &mut loc.t, rank, scale, out, Ptr::of(&ws.wide, 0, &stream), Ptr::of_mut(&mut ws.out, 0, &stream), hidden, nb)?;
            }
            (Mixer::Full(full), LocateMixer::Full { q, k, v, o }) => {
                let (heads, kv_heads, head_dim) = (text.num_attention_heads, text.num_key_value_heads, text.head_dim);
                let mut quantized_for = f32::NAN;
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.q, &mut ws.wide, nb, hidden)?;
                update(lt, &stream, &mut loc.t, rank, scale, q, normed, Ptr::of_mut(&mut ws.wide, 0, &stream), full.q.n, nb)?;
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.k, &mut ws.mid, nb, hidden)?;
                update(lt, &stream, &mut loc.t, rank, scale, k, normed, Ptr::of_mut(&mut ws.mid, 0, &stream), full.k.n, nb)?;
                fp4_project(kernels, lt, &stream, &ws.normed, &mut ws.fp4_packed, &mut ws.fp4_scales, &mut quantized_for, &full.v, &mut ws.v, nb, hidden)?;
                update(lt, &stream, &mut loc.t, rank, scale, v, normed, Ptr::of_mut(&mut ws.v, 0, &stream), full.v.n, nb)?;
                kernels.head_norm_rotary(&ws.wide, &ws.mid, &full.q_norm, &full.k_norm, &ws.positions, &mut ws.z, &mut ws.k_rot, nb, heads, kv_heads, text.rotary_dim(), text.rope_parameters.rope_theta as f32, eps)?;
                let gate = Ptr::of(&ws.wide, head_dim, &stream);
                kernels.attention_split(&tiles, n_tiles, &ws.z, &ws.k_rot, &ws.v, gate, &mut ws.mixed, heads, kv_heads, 1.0 / (head_dim as f32).sqrt())?;
                kernels.quantize_fp4(Ptr::of(&ws.mixed, 0, &stream), &mut ws.fp4_packed, &mut ws.fp4_scales, nb, heads * head_dim, full.o.input_encode)?;
                lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &full.o, &mut ws.out, nb)?;
                update(lt, &stream, &mut loc.t, rank, scale, o, Ptr::of(&ws.mixed, 0, &stream), Ptr::of_mut(&mut ws.out, 0, &stream), hidden, nb)?;
            }
            _ => return Err(CudaError::Invalid("the Locate adapter's layers do not match the model's".into())),
        }
        // h += out; x = rms_norm(h); gate|up (+ updates); silu(gate) * up; down (+ update); h += mlp.
        let intermediate = text.intermediate_size;
        kernels.rms_norm(&ws.out, Some(&mut ws.h), &layer.post_norm, &mut ws.normed, nb, hidden, eps)?;
        kernels.quantize_fp4(normed, &mut ws.fp4_packed, &mut ws.fp4_scales, nb, hidden, layer.gate_up.input_encode)?;
        lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &layer.gate_up, &mut ws.wide, nb)?;
        update(lt, &stream, &mut loc.t, rank, scale, &adapter.gate, normed, Ptr::of_mut(&mut ws.wide, 0, &stream), 2 * intermediate, nb)?;
        update(lt, &stream, &mut loc.t, rank, scale, &adapter.up, normed, Ptr::of_mut(&mut ws.wide, intermediate, &stream), 2 * intermediate, nb)?;
        // The product in bf16, in the FP8 accumulator's memory (free here, and large enough: 4 bytes x 10240
        // per token against 2 x 17408).
        debug_assert!(size_of::<f32>() * ws.fp8_acc.len() >= size_of::<u16>() * ws.max_tokens * intermediate);
        let product = Ptr::of_mut(&mut ws.fp8_acc, 0, &stream);
        kernels.silu_mul(&ws.wide, product, nb, intermediate)?;
        kernels.quantize_fp4(product, &mut ws.fp4_packed, &mut ws.fp4_scales, nb, intermediate, layer.down.input_encode)?;
        lt.fp4(&ws.fp4_packed, &ws.fp4_scales, &layer.down, &mut ws.out, nb)?;
        update(lt, &stream, &mut loc.t, rank, scale, &adapter.down, product, Ptr::of_mut(&mut ws.out, 0, &stream), hidden, nb)?;
        kernels.add_inplace(&mut ws.h, &ws.out, nb * hidden)?;
    }

    // The final norm, then the head's projections of every token in one matmul.
    kernels.rms_norm(&ws.h, None, final_norm, &mut ws.normed, nb, hidden, eps)?;
    let rows = loc.head.n;
    lt.bf16(&ws.normed, &loc.head, &mut ws.mid, nb)?;
    let mut host = vec![0u16; n * rows];
    stream.memcpy_dtoh(&ws.mid.slice(0..n * rows), &mut host)?;
    stream.synchronize()?;

    let width = loc.params.width;
    let value = |token: usize, column: usize| f32::from_bits((host[token * rows + column] as u32) << 16);
    let gather = |tokens: &[usize], from: usize, count: usize| -> Vec<f32> {
        tokens.iter().flat_map(|&token| (from..from + count).map(move |column| value(token, column))).collect()
    };
    let (ts, te) = (gather(&plan.positions, 0, width), gather(&plan.positions, width, width));
    let (qs, qe, qp) = (gather(&row.markers, 2 * width, width), gather(&row.markers, 3 * width, width), gather(&row.markers, 4 * width, width));
    let thresholds = gather(&row.markers, 5 * width, 3);
    Ok(predict(&Projections { ts: &ts, te: &te, qs: &qs, qe: &qe, qp: &qp, thresholds: &thresholds }, &loc.params, true))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_isolates_question_blocks() {
        // A context of 5 tokens, then blocks of 2 and 4, padded to 16 rows.
        let layout = Layout::build(&[5, 2, 4], 16);
        let position = |t: usize| layout.positions[3 * t];
        assert_eq!((0..11).map(position).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5, 6, 5, 6, 7, 8]);
        let sources = |t: usize| layout.conv_sources[3 * t..3 * t + 3].to_vec();
        assert_eq!(sources(0), vec![-3, -2, -1]);
        assert_eq!(sources(4), vec![1, 2, 3]);
        // The first block follows the context; so does the second, not the first block.
        assert_eq!(sources(5), vec![2, 3, 4]);
        assert_eq!(sources(6), vec![3, 4, 5]);
        assert_eq!(sources(7), vec![2, 3, 4]);
        assert_eq!(sources(8), vec![3, 4, 7]);
        assert_eq!(sources(10), vec![7, 8, 9]);
        assert_eq!(sources(11), vec![-1, -1, -1]);
        // The context attends the whole sequence; a block attends the context and itself.
        assert_eq!(layout.tiles, vec![0, 5, 0, 11, 0, 0, 5, 2, 0, 5, 5, 2, 7, 4, 0, 5, 7, 4]);
        // One delta chunk each (all under 64 tokens); the context's walk first.
        assert_eq!(layout.chunks, vec![0, 5, 0, 0, 5, 2, 0, 1, 7, 4, 0, 2]);
        assert_eq!(layout.seqs, vec![0, 5, 0, 1, 5, 2, 1, 1, 7, 4, 2, 1]);
    }

    #[test]
    fn layout_tiles_and_chunks_long_segments() {
        let layout = Layout::build(&[300, 130], 448);
        assert_eq!(layout.tiles.len() / 6, 3 + 2);
        assert_eq!(&layout.tiles[12..18], &[256, 44, 0, 430, 0, 0]);
        assert_eq!(&layout.tiles[24..30], &[428, 2, 0, 300, 300, 130]);
        assert_eq!(layout.chunks.len() / 4, 5 + 3);
        assert_eq!(layout.seqs, vec![0, 300, 0, 5, 300, 130, 5, 3]);
        assert_eq!(layout.positions[3 * 300], 300);
        assert_eq!(layout.positions[3 * 429], 429);
    }
}
