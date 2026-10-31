//! The vision tower: an image's patches to one text-space embedding per merged 2x2 block, as
//! `Qwen3_5VisionModel` computes them (bf16 matmuls on cuBLASLt, LayerNorms, axial rotary, non-causal
//! attention within each image, the GELU MLPs, and the patch merger).
//!
//! All images of a pass go through together as packed sequences of patches, in pack order; the merger's output
//! rows are the image tokens in the same order, which forward.rs scatters into the text sequence.

use std::sync::Arc;

use cudarc::driver::CudaStream;

use super::CudaError;
use super::blas::Lt;
use super::device::{Vision, Workspace};
use super::kernels::{ATTENTION_QUERIES, Kernels};
use crate::config::Checkpoint;
use crate::images::PreparedImage;

/// Run the tower over `images`; the merged embeddings land in `ws.vembeds`. Returns how many rows it produced.
pub fn run(stream: &Arc<CudaStream>, lt: &Lt, kernels: &Kernels, vision: &Vision, ws: &mut Workspace, checkpoint: &Checkpoint, images: &[&PreparedImage]) -> Result<usize, CudaError> {
    let cfg = &checkpoint.model.vision_config;
    let (vh, heads, merge) = (cfg.hidden_size, cfg.num_heads, cfg.spatial_merge_size);
    let head_dim = vh / heads;
    let side = (cfg.num_position_embeddings as f64).sqrt() as usize;
    let patch_len = vision.patch_embed.k;
    let total: usize = images.iter().map(|i| i.grid.t * i.grid.h * i.grid.w).sum();
    if total > ws.max_patches {
        return Err(CudaError::Invalid("The request's images have more patches than the vision workspace holds.".into()));
    }
    if total == 0 {
        return Ok(0);
    }

    // Host-side tables: pixels as bf16, patch (row, col) positions, position-embedding taps, attention tiles.
    let mut pixels = Vec::with_capacity(total * patch_len);
    let mut positions = Vec::with_capacity(total * 2);
    let mut taps = Vec::with_capacity(total * 4);
    let mut weights = Vec::with_capacity(total * 4);
    let mut tiles = Vec::new();
    let mut start = 0usize;
    for image in images {
        let grid = image.grid;
        if image.patches.len() != grid.t * grid.h * grid.w * patch_len {
            return Err(CudaError::Invalid("an image's patches do not match its grid".into()));
        }
        pixels.extend(image.patches.iter().map(|&v| (half::bf16::from_f32(v)).to_bits()));
        let (hb, wb) = (grid.h / merge, grid.w / merge);
        for _ in 0..grid.t {
            for bh in 0..hb {
                for bw in 0..wb {
                    for mh in 0..merge {
                        for mw in 0..merge {
                            let (row, col) = (bh * merge + mh, bw * merge + mw);
                            positions.extend([row as i32, col as i32]);
                            // Bilinear taps on the side x side learned grid, align_corners style.
                            let src_h = row as f64 * (side - 1) as f64 / (grid.h.max(2) - 1) as f64;
                            let src_w = col as f64 * (side - 1) as f64 / (grid.w.max(2) - 1) as f64;
                            let (h0, w0) = (src_h.floor() as usize, src_w.floor() as usize);
                            let (h1, w1) = ((h0 + 1).min(side - 1), (w0 + 1).min(side - 1));
                            let (dh, dw) = ((src_h - h0 as f64) as f32, (src_w - w0 as f64) as f32);
                            taps.extend([(h0 * side + w0) as i32, (h0 * side + w1) as i32, (h1 * side + w0) as i32, (h1 * side + w1) as i32]);
                            weights.extend([(1.0 - dh) * (1.0 - dw), (1.0 - dh) * dw, dh * (1.0 - dw), dh * dw]);
                        }
                    }
                }
            }
        }
        let len = grid.t * grid.h * grid.w;
        for q_start in (0..len).step_by(ATTENTION_QUERIES) {
            tiles.extend([start as u32, len as u32, q_start as u32]);
        }
        start += len;
    }
    let n = total;
    let n_tiles = tiles.len() / 3;
    {
        let mut view = ws.vpixels.slice_mut(0..n * patch_len);
        stream.memcpy_htod(&pixels, &mut view)?;
    }
    let positions = stream.clone_htod(&positions)?;
    let taps = stream.clone_htod(&taps)?;
    let weights = stream.clone_htod(&weights)?;
    let tiles = stream.clone_htod(&tiles)?;
    let eps = 1e-6f32;
    let theta = cfg_rope_theta(checkpoint);

    // Patch embedding + position embedding.
    lt.bf16(&ws.vpixels, &vision.patch_embed, &mut ws.vx, n)?;
    kernels.vision_pos_embed(&mut ws.vx, &vision.pos_embed, &taps, &weights, n, vh)?;

    // The blocks.
    let scale = 1.0 / (head_dim as f32).sqrt();
    for block in &vision.blocks {
        kernels.layer_norm(&ws.vx, &block.norm1.0, &block.norm1.1, &mut ws.vnormed, n, vh, eps)?;
        lt.bf16(&ws.vnormed, &block.qkv, &mut ws.vqkv, n)?;
        kernels.vision_rotary(&mut ws.vqkv, &positions, n, heads, theta)?;
        kernels.vision_attention(&tiles, n_tiles, &ws.vqkv, &mut ws.vattn, heads, scale)?;
        lt.bf16(&ws.vattn, &block.proj, &mut ws.vout, n)?;
        kernels.add_inplace(&mut ws.vx, &ws.vout, n * vh)?;
        kernels.layer_norm(&ws.vx, &block.norm2.0, &block.norm2.1, &mut ws.vnormed, n, vh, eps)?;
        lt.bf16(&ws.vnormed, &block.fc1, &mut ws.vmlp, n)?;
        kernels.gelu(&mut ws.vmlp, n * block.fc1.n, true)?;
        lt.bf16(&ws.vmlp, &block.fc2, &mut ws.vout, n)?;
        kernels.add_inplace(&mut ws.vx, &ws.vout, n * vh)?;
    }

    // The merger: LayerNorm per patch, 2x2 blocks concatenated (consecutive patches), fc1, exact GELU, fc2.
    kernels.layer_norm(&ws.vx, &vision.merger_norm.0, &vision.merger_norm.1, &mut ws.vnormed, n, vh, eps)?;
    let merged_rows = n / (merge * merge);
    lt.bf16(&ws.vnormed, &vision.merger_fc1, &mut ws.vmerged, merged_rows)?;
    kernels.gelu(&mut ws.vmerged, merged_rows * vision.merger_fc1.n, false)?;
    lt.bf16(&ws.vmerged, &vision.merger_fc2, &mut ws.vembeds, merged_rows)?;
    Ok(merged_rows)
}

fn cfg_rope_theta(checkpoint: &Checkpoint) -> f32 {
    checkpoint.model.vision_config.rope_parameters.as_ref().map_or(10000.0, |r| r.rope_theta) as f32
}
