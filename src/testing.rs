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
pub const WORD_SEPARATOR: u32 = 3;
/// The Locate marker's id in the word tokenizer's fake model (never produced from text).
pub const WORD_LOCATE: u32 = 4;

impl WordTokenizer {
    pub fn id_of(token: &str) -> u32 {
        match token {
            "<|endoftext|>" => WORD_PAD,
            "<|fim_suffix|>" => WORD_MARKER,
            "<|image_pad|>" => WORD_IMAGE_PAD,
            "<|fim_middle|>" => WORD_SEPARATOR,
            "bad" => u32::MAX,
            _ => {
                let mut hash: u32 = 2166136261;
                for byte in token.bytes() {
                    hash = (hash ^ byte as u32).wrapping_mul(16777619);
                }
                5 + hash % 1_000_000
            }
        }
    }

    /// Split into words, with each `<|...|>` as its own token even when glued to its neighbours.
    pub fn split(text: &str) -> Vec<String> {
        Self::split_offsets(text).into_iter().map(|(token, _)| token).collect()
    }

    /// The same tokens with the characters (start, end exclusive) each covers.
    pub fn split_offsets(text: &str) -> Vec<(String, (usize, usize))> {
        let chars: Vec<char> = text.chars().collect();
        let mut tokens = Vec::new();
        let mut i = 0;
        while i < chars.len() {
            if chars[i].is_whitespace() {
                i += 1;
                continue;
            }
            let special = |at: usize| -> Option<usize> {
                if chars[at] != '<' || chars.get(at + 1) != Some(&'|') {
                    return None;
                }
                (at + 2..chars.len().saturating_sub(1)).find(|&j| chars[j] == '|' && chars[j + 1] == '>').map(|j| j + 2)
            };
            let end = match special(i) {
                Some(end) => end,
                None => {
                    let mut j = i;
                    while j < chars.len() && !chars[j].is_whitespace() && (j == i || special(j).is_none()) {
                        j += 1;
                    }
                    j
                }
            };
            tokens.push((chars[i..end].iter().collect(), (i, end)));
            i = end;
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
    fn encode_offsets(&self, text: &str) -> Result<Vec<(u32, (usize, usize))>, ProtocolError> {
        Ok(Self::split_offsets(text).into_iter().map(|(token, span)| (Self::id_of(&token), span)).collect())
    }
}

/// A word encoder with every two-letter code in order. The real checkpoint's code list skips some pairs (whatever
/// is not one token in its tokenizer); tests that compare against the Python fixture use `word_encoder_with` and its codes.
pub fn word_encoder() -> Encoder {
    let codes: Vec<String> = (0..255).map(|i| if i < 26 { ((b'A' + i as u8) as char).to_string() } else { format!("{}{}", (b'A' + (i / 26 - 1) as u8) as char, (b'A' + (i % 26) as u8) as char) }).collect();
    word_encoder_with(codes)
}

/// A word encoder in a given protocol version.
pub fn word_encoder_for(protocol: crate::config::Protocol, codes: Vec<String>) -> Encoder {
    Encoder::with_protocol(protocol, Box::new(WordTokenizer), codes, WORD_PAD, 2).expect("word encoder")
}

pub fn word_encoder_with(codes: Vec<String>) -> Encoder {
    Encoder::new(Box::new(WordTokenizer), codes, WORD_PAD, 2).expect("word encoder")
}

pub fn fake_model(budget: Budget, seconds_per_token: f64) -> Arc<FakeModel> {
    let locate = crate::locate::LocateEncoder::new(Box::new(WordTokenizer), word_encoder().codes().to_vec(), WORD_PAD, 2, WORD_LOCATE).expect("locate encoder");
    Arc::new(FakeModel { encoder: word_encoder(), budget, seconds_per_token, passes: parking_lot::Mutex::new(Vec::new()), locate: Some(locate) })
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
