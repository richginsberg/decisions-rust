//! Locate: verbatim span extraction, the `"locate"` question type (pplx_decider's `locate.py`, v1.3).
//!
//! A locate question asks for the spans of the state that match a label (its instructions, or its key). It is
//! answered by an adapter that ships beside the checkpoint in `locate-span/`: a LoRA on every linear layer of the
//! language model and a trained embedding for the `<|locate|>` marker, both enabled only for locate passes, and
//! a span head over the final hidden states. This module is everything about it that runs on the CPU:
//!
//! - the request rules and the prompt: the version 4 context (system prompt, `State:\n`, the state, the
//!   separator), then one chunk per question, `Question:\nLocate verbatim spans:\n` + label + `<|locate|>`,
//!   with every question of a request in one sequence (split in halves only when over the length limit);
//! - which tokens are the state's, and the characters each covers, so token spans map back to the state;
//! - the span head's scoring from its projections (which the GPU computes), and the clean-up of its spans.
//!
//! The forward pass itself (the isolated-question layout, the LoRA updates) is in `crate::cuda::locate`.

use std::collections::BTreeMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::config::Protocol;
use crate::protocol::{Encoded, Encoder, ProtocolError, Question, Request, Tokenize, python_isspace, python_repr, python_strip};

pub const LOCATE_MARKER: &str = "<|locate|>";
pub const FORMAT: &str = "pplx-decider-locate-span-v1";
pub const PROMPT: &str = "Locate verbatim spans:\n";
pub const MAX_QUESTIONS: usize = 128;

/// `locate-span/locate_config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct LocateConfig {
    pub format: String,
    pub marker: String,
    pub marker_id: u32,
    pub head: HeadConfig,
    pub lora: LoraConfig,
    #[serde(default)]
    pub thresholds: Thresholds,
    #[serde(default)]
    pub base: BaseConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HeadConfig {
    pub width: usize,
    pub max_span: usize,
    pub proposals: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LoraConfig {
    pub rank: usize,
    pub alpha: f64,
    #[serde(default)]
    pub layers: u64,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Thresholds {
    #[serde(default = "default_cutoff")]
    pub default: f64,
}

fn default_cutoff() -> f64 {
    0.5
}

impl Default for Thresholds {
    fn default() -> Self {
        Self { default: default_cutoff() }
    }
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct BaseConfig {
    /// The checkpoint the adapter was trained on.
    #[serde(default)]
    pub repo: Option<String>,
    #[serde(default)]
    pub revision: Option<String>,
    #[serde(default)]
    pub sha256: BTreeMap<String, String>,
}

/// What a locate sequence carries besides its tokens: where the state's tokens are and the characters each
/// covers, and the state itself, to turn token spans into the answer.
#[derive(Debug, Clone, PartialEq)]
pub struct LocateRow {
    /// Chunk lengths: the context (with its separator), then each question (with its marker).
    pub segments: Vec<usize>,
    /// Indices of the tokens that cover characters of the state, in order.
    pub positions: Vec<usize>,
    /// Characters of the state (start, end exclusive) each of those tokens covers.
    pub offsets: Vec<(usize, usize)>,
    /// The state, by character (offsets are Unicode code points, as in Python).
    pub state: Vec<char>,
}

/// A span of the state, as the head reports it: the first and last (inclusive) of `positions`, and a probability.
pub type TokenSpan = (usize, usize, f32);

#[derive(Debug, Clone, Serialize, PartialEq)]
pub struct Span {
    /// Character offsets into `state` (`end` exclusive).
    pub start: usize,
    pub end: usize,
    pub text: String,
    pub score: f64,
}

/// Builds locate sequences. It has its own escape list: user text can produce neither the decision markers nor
/// `<|locate|>`.
pub struct LocateEncoder {
    encoder: Encoder,
    pub marker_id: u32,
}

impl LocateEncoder {
    /// `marker_id` is the id training gave `<|locate|>` (the checkpoint's tokenizer does not hold the token; it
    /// is appended as an id, and never produced from text).
    pub fn new(tokenizer: Box<dyn Tokenize>, codes: Vec<String>, pad_id: u32, merge_size: usize, marker_id: u32) -> Result<Self, ProtocolError> {
        let encoder = Encoder::with_protocol(Protocol::ChunkedV4, tokenizer, codes, pad_id, merge_size)?.with_escape(LOCATE_MARKER);
        if marker_id == encoder.marker_id || Some(marker_id) == encoder.separator_id {
            return Err(ProtocolError::Internal("The Locate marker id collides with a decision token".into()));
        }
        Ok(Self { encoder, marker_id })
    }

    /// The labels of a request's locate questions, or why the request is invalid (locate.py's `validate`).
    /// `total_questions` counts every question of the request, of any type.
    pub fn validate(&self, state: &Value, images: usize, total_questions: usize, questions: &[(String, Question)]) -> Result<Vec<String>, ProtocolError> {
        let text = match state {
            Value::String(text) if images == 0 && total_questions <= MAX_QUESTIONS => text,
            _ => return Err(ProtocolError::Request(format!("Locate requires text state, no images, and at most {MAX_QUESTIONS} questions"))),
        };
        if text.is_empty() {
            return Err(ProtocolError::Request("Locate state must contain text".into()));
        }
        let mut labels = Vec::with_capacity(questions.len());
        for (key, question) in questions {
            if key.is_empty() {
                return Err(ProtocolError::Request("Question IDs must be nonempty strings".into()));
            }
            match question.instructions() {
                None | Some(Value::Null) => labels.push(key.clone()),
                Some(Value::String(text)) => labels.push(if python_strip(text).is_empty() { key.clone() } else { text.clone() }),
                Some(_) => return Err(ProtocolError::Request(format!("Question {}: Locate instructions must be text or null", python_repr(key)))),
            }
        }
        Ok(labels)
    }

    /// One sequence holding the state and the given questions, or `ProtocolError::Length` if it exceeds
    /// `max_length`.
    pub fn encode(&self, state: &str, keys: &[String], labels: &[String], max_length: Option<usize>) -> Result<Encoded, ProtocolError> {
        let encoder = &self.encoder;
        let separator_id = encoder.separator_id.expect("a version 4 encoder");
        let request = Request { state: Value::String(state.to_string()), questions: Vec::new(), images: Vec::new() };
        let context = encoder.context_text(&request);
        // Escaping keeps the character count, so offsets into the escaped state are offsets into the state.
        let escaped = encoder.text_of(&request.state);
        let (context_chars, escaped_chars) = (context.chars().count(), escaped.chars().count());
        let begin = context_chars - escaped_chars;
        let tokens = encoder.tokenizer().encode_offsets(&context)?;
        let mut input_ids: Vec<u32> = tokens.iter().map(|(id, _)| *id).collect();
        input_ids.push(separator_id);
        let mut segments = vec![input_ids.len()];
        let mut markers = Vec::with_capacity(labels.len());
        for label in labels {
            let text = format!("Question:\n{PROMPT}{}", encoder.text_of(&Value::String(label.clone())));
            let chunk = encoder.tokenizer().encode(&text)?;
            input_ids.extend_from_slice(&chunk);
            input_ids.push(self.marker_id);
            segments.push(chunk.len() + 1);
            markers.push(input_ids.len() - 1);
        }
        if let Some(limit) = max_length
            && input_ids.len() > limit
        {
            return Err(ProtocolError::Length(format!("Locate input exceeds max_length ({} > {limit})", input_ids.len())));
        }
        let mut positions = Vec::new();
        let mut offsets = Vec::new();
        for (i, &(_, (a, b))) in tokens.iter().enumerate() {
            if a < context_chars && b > begin && a < b {
                positions.push(i);
                offsets.push((a.saturating_sub(begin), escaped_chars.min(b - begin)));
            }
        }
        Ok(Encoded {
            keys: keys.to_vec(),
            counts: vec![0; keys.len()],
            input_ids,
            markers,
            images: Vec::new(),
            pixels: Vec::new(),
            locate: Some(Arc::new(LocateRow { segments, positions, offsets, state: state.chars().collect() })),
        })
    }

    /// The sequences of a request's locate questions: one, or halves of the questions when it is over
    /// `max_length`; every group keeps the whole state, and everything is prepared before any forward pass.
    pub fn plan(&self, state: &str, keys: &[String], labels: &[String], max_length: Option<usize>) -> Result<Vec<Encoded>, ProtocolError> {
        match self.encode(state, keys, labels, max_length) {
            Err(ProtocolError::Length(_)) if labels.len() > 1 => {
                let middle = labels.len() / 2;
                let mut out = self.plan(state, &keys[..middle], &labels[..middle], max_length)?;
                out.extend(self.plan(state, &keys[middle..], &labels[middle..], max_length)?);
                Ok(out)
            }
            other => other.map(|encoded| vec![encoded]),
        }
    }
}

// ------------------------------------------------------------------------------------------- the span head

/// The span head's scalars (the rest of it, the projections, runs on the GPU).
#[derive(Debug, Clone)]
pub struct HeadParams {
    pub width: usize,
    pub max_span: usize,
    pub proposals: usize,
    pub cutoff: f32,
    pub start_bias: f32,
    pub end_bias: f32,
    pub width_bias: [f32; 32],
}

/// The head's projections of one sequence: of the state's tokens (`ts`, `te`: [tokens, width]) and of each
/// question's marker (`qs`, `qe`, `qp`: [questions, width]; `thresholds`: [questions, 3]), row-major.
pub struct Projections<'a> {
    pub ts: &'a [f32],
    pub te: &'a [f32],
    pub qs: &'a [f32],
    pub qe: &'a [f32],
    pub qp: &'a [f32],
    pub thresholds: &'a [f32],
}

/// Round to bf16 and back (round to nearest even), as a bf16 tensor would hold the value.
pub fn round_bf16(value: f32) -> f32 {
    f32::from_bits((f32_to_bf16(value) as u32) << 16)
}

pub fn f32_to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return ((bits >> 16) | 0x0040) as u16;
    }
    ((bits + 0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn row(width: usize, matrix: &[f32], index: usize) -> &[f32] {
    &matrix[index * width..(index + 1) * width]
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// The `k` largest values with their indices, largest first (the lower index first among equals).
fn topk(values: &[f32], k: usize) -> Vec<(f32, usize)> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
    order.truncate(k);
    order.into_iter().map(|i| (values[i], i)).collect()
}

/// SpanHead.predict: for each question, the token spans whose probability reaches the cutoff. Start and end
/// scores pick `proposals` candidates each; every (start, end) pair of them with a width of 1 to `max_span`
/// tokens is scored. `bf16` rounds where the evaluated arithmetic (BF16 autocast on CUDA) holds bf16 values;
/// without it this is the head's FP32 arithmetic.
pub fn predict(p: &Projections, params: &HeadParams, bf16: bool) -> Vec<Vec<TokenSpan>> {
    let width = params.width;
    let tokens = p.ts.len() / width;
    let questions = p.qs.len() / width;
    let r = |v: f32| if bf16 { round_bf16(v) } else { v };
    let scale = (width as f32).sqrt();
    let k = params.proposals.min(tokens);
    let mut result = Vec::with_capacity(questions);
    for q in 0..questions {
        let start: Vec<f32> = (0..tokens).map(|d| r(r(r(dot(row(width, p.qs, q), row(width, p.ts, d))) / scale) + params.start_bias)).collect();
        let finish: Vec<f32> = (0..tokens).map(|d| r(r(r(dot(row(width, p.qe, q), row(width, p.te, d))) / scale) + params.end_bias)).collect();
        let starts = topk(&start, k);
        let ends = topk(&finish, k);
        let gate: Vec<f32> = row(width, p.qp, q).iter().map(|v| r(v.tanh())).collect();
        let threshold = p.thresholds[q * 3 + 2];
        let mut spans = Vec::new();
        let mut gated = vec![0f32; width];
        for &(start_value, si) in &starts {
            for (g, (t, w)) in gated.iter_mut().zip(row(width, p.ts, si).iter().zip(&gate)) {
                *g = r(t * w);
            }
            for &(end_value, ei) in &ends {
                let span = ei as i64 - si as i64 + 1;
                if span <= 0 || span > params.max_span as i64 {
                    continue;
                }
                let bucket = ((span as f32).log2().floor() as usize).min(31);
                let pair = r(r(dot(&gated, row(width, p.te, ei))) / scale);
                let score = r(r(pair + start_value) + end_value) + params.width_bias[bucket];
                let probability = 1.0 / (1.0 + (-(score - threshold)).exp());
                if probability >= params.cutoff {
                    spans.push((si, ei, probability));
                }
            }
        }
        result.push(spans);
    }
    result
}

/// One question's answer from its token spans: character offsets into the state with surrounding whitespace
/// trimmed, each distinct span once with its best score, sorted by position.
pub fn spans(row: &LocateRow, found: &[TokenSpan]) -> Vec<Span> {
    let text = &row.state;
    let mut unique: BTreeMap<(usize, usize), f32> = BTreeMap::new();
    for &(first, last, score) in found {
        let (mut a, mut b) = (row.offsets[first].0, row.offsets[last].1);
        while a < b && python_isspace(text[a]) {
            a += 1;
        }
        while b > a && python_isspace(text[b - 1]) {
            b -= 1;
        }
        if a < b {
            let best = unique.entry((a, b)).or_insert(0.0);
            *best = best.max(score);
        }
    }
    unique.into_iter().map(|((start, end), score)| Span { start, end, text: text[start..end].iter().collect(), score: score as f64 }).collect()
}
