//! What the server needs from a model: the encoder, and forward passes over encoded rows.
//!
//! The GPU model lives in `crate::cuda` (feature `cuda`); `FakeModel` answers deterministically on the CPU
//! for the batcher and server tests.

use std::sync::Arc;

use crate::images::ImageSettings;
use crate::protocol::{Encoded, Encoder, ProtocolError};

#[derive(Debug, thiserror::Error)]
pub enum PassError {
    /// pplx_decider's input checks (422).
    #[error("{0}")]
    Request(String),
    /// The pass needs more GPU memory than is free (413). The server recovers and keeps serving.
    #[error("The request needs more GPU memory than is free. Send fewer questions or a shorter state.")]
    OutOfMemory,
    #[error("{0}")]
    Internal(String),
}

impl From<ProtocolError> for PassError {
    fn from(error: ProtocolError) -> Self {
        match error {
            ProtocolError::Request(m) | ProtocolError::Length(m) => PassError::Request(m),
            ProtocolError::Internal(m) => PassError::Internal(m),
        }
    }
}

/// Per row of a pass: each question's probabilities over its own options (temperature applied, softmaxed).
pub type RowAnswers = Vec<Vec<f32>>;

/// How many sequences fit in one forward pass.
#[derive(Debug, Clone, Copy)]
pub struct Budget {
    /// Tokens through the model in one pass. Padded: rows x the longest row. Packed: the sum of the rows.
    pub max_tokens: usize,
    pub max_rows: usize,
    pub packed: bool,
}

impl Budget {
    /// Whether rows of these lengths fit in one pass.
    pub fn fits(&self, lengths: &[usize]) -> bool {
        if lengths.is_empty() || lengths.len() > self.max_rows {
            return lengths.is_empty();
        }
        let cost = if self.packed {
            lengths.iter().sum::<usize>()
        } else {
            lengths.len() * lengths.iter().copied().max().unwrap_or(0)
        };
        cost <= self.max_tokens
    }
}

pub trait Model: Send + Sync {
    fn encoder(&self) -> &Encoder;
    /// Per-sequence token limit; longer requests are split into question chunks.
    fn max_length(&self) -> usize;
    fn budget(&self) -> Budget;
    /// How images are resized and patched for this checkpoint.
    fn image_settings(&self) -> ImageSettings;
    /// One forward pass over sequences, one result per row in order. Rows are independent, so each gets
    /// the answers it would get alone.
    fn run_pass(&self, rows: &[&Encoded]) -> Result<Vec<RowAnswers>, PassError>;
}

pub type SharedModel = Arc<dyn Model>;

/// A CPU stand-in: a question's answer depends only on its row's tokens, and a pass takes time in
/// proportion to its tokens, like the real model.
pub struct FakeModel {
    pub encoder: Encoder,
    pub budget: Budget,
    pub seconds_per_token: f64,
    pub passes: parking_lot::Mutex<Vec<Vec<Vec<u32>>>>,
}

impl FakeModel {
    pub fn answers(row: &Encoded) -> RowAnswers {
        row.counts
            .iter()
            .enumerate()
            .map(|(question, &count)| {
                // A hash of the row's tokens and the question index, spread into a distribution.
                let mut seed: u64 = 0xcbf2_9ce4_8422_2325 ^ question as u64;
                for &id in &row.input_ids {
                    seed = (seed ^ id as u64).wrapping_mul(0x0000_0100_0000_01b3);
                }
                let mut values: Vec<f32> = (0..count)
                    .map(|i| {
                        seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407 + i as u64);
                        ((seed >> 40) as f32 / (1u64 << 24) as f32) + 0.05
                    })
                    .collect();
                let total: f32 = values.iter().sum();
                values.iter_mut().for_each(|v| *v /= total);
                values
            })
            .collect()
    }
}

impl Model for FakeModel {
    fn encoder(&self) -> &Encoder {
        &self.encoder
    }
    fn max_length(&self) -> usize {
        32768
    }
    fn budget(&self) -> Budget {
        self.budget
    }
    fn image_settings(&self) -> ImageSettings {
        ImageSettings {
            patch_size: 16,
            merge_size: 2,
            temporal_patch_size: 2,
            min_pixels: 65_536,
            max_pixels: 1_638_400,
            mean: [0.5; 3],
            std: [0.5; 3],
        }
    }
    fn run_pass(&self, rows: &[&Encoded]) -> Result<Vec<RowAnswers>, PassError> {
        for row in rows {
            // The fixture tokenizer gives a known id to "bad", standing in for an input the model rejects.
            if row.input_ids.contains(&u32::MAX) {
                return Err(PassError::Request("an invalid question".into()));
            }
        }
        let tokens: usize =
            if self.budget.packed { rows.iter().map(|r| r.length()).sum() } else { rows.len() * rows.iter().map(|r| r.length()).max().unwrap_or(0) };
        std::thread::sleep(std::time::Duration::from_secs_f64(self.seconds_per_token * tokens as f64));
        self.passes.lock().push(rows.iter().map(|r| r.input_ids.clone()).collect());
        Ok(rows.iter().map(|row| Self::answers(row)).collect())
    }
}
