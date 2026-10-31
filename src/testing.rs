//! Test doubles: a word-splitting tokenizer and a service around the fake model, used by the integration tests
//! and by `--fake` (the server end to end without a GPU).

use std::sync::Arc;

use crate::api::Service;
use crate::batcher::Batcher;
use crate::model::{Budget, FakeModel, SharedModel};
use crate::protocol::{Encoder, ProtocolError, Tokenize};

/// Tokens are words and `<|...|>` special tokens; an id is a hash of the token. "bad" gets `u32::MAX`, which the
/// fake model rejects, standing in for an input the real model refuses.
pub struct WordTokenizer;

pub const WORD_PAD: u32 = 0;
pub const WORD_MARKER: u32 = 1;
pub const WORD_IMAGE_PAD: u32 = 2;

impl WordTokenizer {
    pub fn id_of(token: &str) -> u32 {
        match token {
            "<|endoftext|>" => WORD_PAD,
            "<|fim_suffix|>" => WORD_MARKER,
            "<|image_pad|>" => WORD_IMAGE_PAD,
            "bad" => u32::MAX,
            _ => {
                let mut hash: u32 = 2166136261;
                for byte in token.bytes() {
                    hash = (hash ^ byte as u32).wrapping_mul(16777619);
                }
                3 + hash % 1_000_000
            }
        }
    }

    /// Split into words, with each `<|...|>` as its own token even when glued to its neighbours.
    pub fn split(text: &str) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(start) = rest.find("<|") {
                if let Some(end) = rest[start..].find("|>") {
                    tokens.extend(rest[..start].split_whitespace().map(str::to_string));
                    tokens.push(rest[start..start + end + 2].to_string());
                    rest = &rest[start + end + 2..];
                    continue;
                }
            }
            tokens.extend(rest.split_whitespace().map(str::to_string));
            break;
        }
        tokens
    }
}

impl Tokenize for WordTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, ProtocolError> {
        Ok(Self::split(text).iter().map(|t| Self::id_of(t)).collect())
    }
    fn token_to_id(&self, token: &str) -> Option<u32> {
        Some(Self::id_of(token))
    }
}

/// A word encoder with every two-letter code in order. The real checkpoint's code list skips some pairs (whatever
/// is not one token in its tokenizer); tests that compare against the Python fixture use `word_encoder_with` and its codes.
pub fn word_encoder() -> Encoder {
    let codes: Vec<String> = (0..255).map(|i| if i < 26 { ((b'A' + i as u8) as char).to_string() } else { format!("{}{}", (b'A' + (i / 26 - 1) as u8) as char, (b'A' + (i % 26) as u8) as char) }).collect();
    word_encoder_with(codes)
}

pub fn word_encoder_with(codes: Vec<String>) -> Encoder {
    Encoder::new(Box::new(WordTokenizer), codes, WORD_PAD, 2).expect("word encoder")
}

pub fn fake_model(budget: Budget, seconds_per_token: f64) -> Arc<FakeModel> {
    Arc::new(FakeModel { encoder: word_encoder(), budget, seconds_per_token, passes: parking_lot::Mutex::new(Vec::new()) })
}

/// A ready service around the fake model, its batcher started.
pub fn fake_service(model: SharedModel, api_key: Option<&str>) -> Arc<Service> {
    let batcher = Batcher::new(Arc::clone(&model), 30.0);
    batcher.start();
    Arc::new(Service {
        model: Some(model),
        batcher: Some(batcher),
        name: "pplx-decider-v1.2".into(),
        checkpoint: "fake".into(),
        release_date: "2026-01-01".into(),
        api_key: api_key.map(str::to_string),
        low_memory: false,
    })
}
