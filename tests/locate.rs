//! The Rust Locate port against what v1.3's own `locate.py` produced (fixtures/locate.json, written by
//! scripts/make_locate_fixture.py): the request rules, the prompt and its token offsets, the span head's
//! scoring and the clean-up of its spans.
//!
//! The fixture drives the reference's `LocateAdapter` with the real tokenizer and the real `SpanHead` class;
//! only the model is replaced, by recorded hidden states of a small width. The token-level checks need the
//! tokenizer and run when `DECISION_CHECKPOINT_V4` names a directory holding the checkpoint's `tokenizer.json`.

use std::collections::BTreeMap;

use decisions_rust_nvfp4::locate::{HeadParams, LocateEncoder, LocateRow, Projections, Span, predict, spans};
use decisions_rust_nvfp4::protocol::{ProtocolError, Question};
use decisions_rust_nvfp4::testing::{WORD_PAD, WordTokenizer};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    marker_id: u32,
    pad_id: u32,
    codes: Vec<String>,
    head: Head,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Head {
    hidden: usize,
    width: usize,
    max_span: usize,
    proposals: usize,
    cutoff: f32,
    state: BTreeMap<String, Vec<f32>>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    request: Value,
    max_length: Option<usize>,
    request_error: Option<String>,
    length_error: Option<String>,
    labels: Option<Vec<String>>,
    plans: Option<Vec<Plan>>,
    answers: Option<BTreeMap<String, ExpectedAnswer>>,
    usage: Option<Value>,
}

#[derive(Deserialize)]
struct Plan {
    keys_from: usize,
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    segments: Vec<usize>,
    positions: Vec<usize>,
    offsets: Vec<(usize, usize)>,
    hidden: Vec<f32>,
}

#[derive(Deserialize)]
struct ExpectedAnswer {
    spans: Vec<ExpectedSpan>,
}

#[derive(Deserialize)]
struct ExpectedSpan {
    start: usize,
    end: usize,
    text: String,
    score: f64,
}

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/locate.json")).expect("fixture");
    serde_json::from_str(&text).expect("fixture json")
}

struct Parsed {
    state: Value,
    images: usize,
    total: usize,
    /// The request's locate questions, in order.
    locate: Vec<(String, Question)>,
}

fn parse(request: &Value) -> Parsed {
    let object = request.as_object().expect("request");
    let questions = object["questions"].as_object().expect("questions");
    let locate = questions
        .iter()
        .filter(|(_, q)| q["type"] == "locate")
        .map(|(key, q)| (key.clone(), serde_json::from_value::<Question>(q.clone()).expect("question")))
        .collect();
    Parsed { state: object["state"].clone(), images: object.get("images").and_then(Value::as_array).map_or(0, Vec::len), total: questions.len(), locate }
}

fn word_locate(fixture: &Fixture) -> LocateEncoder {
    LocateEncoder::new(Box::new(WordTokenizer), fixture.codes.clone(), WORD_PAD, 2, 999_999_999).expect("encoder")
}

#[test]
fn request_rules_match_python() {
    let fixture = fixture();
    let encoder = word_locate(&fixture);
    let mut errors = 0;
    for case in &fixture.cases {
        let parsed = parse(&case.request);
        match (encoder.validate(&parsed.state, parsed.images, parsed.total, &parsed.locate), &case.request_error) {
            (Err(ProtocolError::Request(message)), Some(expected)) => {
                assert_eq!(&message, expected, "case {}", case.id);
                errors += 1;
            }
            (Ok(labels), None) => {
                if let Some(expected) = &case.labels {
                    assert_eq!(&labels, expected, "case {}", case.id);
                }
            }
            (got, expected) => panic!("case {}: {got:?} where Python gave {expected:?}", case.id),
        }
    }
    assert!(errors >= 7);
}

fn linear(weights: &BTreeMap<String, Vec<f32>>, name: &str, x: &[f32], hidden: usize) -> Vec<f32> {
    let (weight, bias) = (&weights[&format!("{name}.weight")], &weights[&format!("{name}.bias")]);
    let out = bias.len();
    let mut y = Vec::with_capacity(x.len() / hidden * out);
    for row in x.chunks_exact(hidden) {
        for o in 0..out {
            y.push(weight[o * hidden..(o + 1) * hidden].iter().zip(row).map(|(w, v)| w * v).sum::<f32>() + bias[o]);
        }
    }
    y
}

fn gather(hidden: &[f32], rows: &[usize], width: usize) -> Vec<f32> {
    rows.iter().flat_map(|&r| hidden[r * width..(r + 1) * width].iter().copied()).collect()
}

/// The head's FP32 scoring and the span clean-up, from the recorded hidden states to the recorded answers.
#[test]
fn head_and_spans_match_python() {
    let fixture = fixture();
    let head = &fixture.head;
    let params = HeadParams {
        width: head.width,
        max_span: head.max_span,
        proposals: head.proposals,
        cutoff: head.cutoff,
        start_bias: head.state["start_bias"][0],
        end_bias: head.state["end_bias"][0],
        width_bias: head.state["width_bias.weight"].clone().try_into().expect("32 width buckets"),
    };
    let (mut compared, mut borderline) = (0, 0);
    for case in &fixture.cases {
        let (Some(plans), Some(answers)) = (&case.plans, &case.answers) else { continue };
        let parsed = parse(&case.request);
        let state: Vec<char> = parsed.state.as_str().expect("text state").chars().collect();
        for plan in plans {
            let row = LocateRow { segments: plan.segments.clone(), positions: plan.positions.clone(), offsets: plan.offsets.clone(), state: state.clone() };
            let questions = plan.markers.len();
            let keys = &parsed.locate[plan.keys_from..plan.keys_from + questions];
            if plan.positions.is_empty() {
                for (key, _) in keys {
                    assert!(answers[key].spans.is_empty(), "case {} question {key}", case.id);
                }
                continue;
            }
            let document = gather(&plan.hidden, &plan.positions, head.hidden);
            let labels = gather(&plan.hidden, &plan.markers, head.hidden);
            let (ts, te) = (linear(&head.state, "ts", &document, head.hidden), linear(&head.state, "te", &document, head.hidden));
            let (qs, qe, qp) = (linear(&head.state, "qs", &labels, head.hidden), linear(&head.state, "qe", &labels, head.hidden), linear(&head.state, "qp", &labels, head.hidden));
            let thresholds = linear(&head.state, "thresholds", &labels, head.hidden);
            let found = predict(&Projections { ts: &ts, te: &te, qs: &qs, qe: &qe, qp: &qp, thresholds: &thresholds }, &params, false);
            assert_eq!(found.len(), questions);
            for ((key, _), token_spans) in keys.iter().zip(&found) {
                let got: Vec<Span> = spans(&row, token_spans);
                let expected = &answers[key].spans;
                // A span whose probability sits on the cutoff may fall either side in another summation order.
                let near = |score: f64| (score - head.cutoff as f64).abs() < 2e-4;
                let got_sure: Vec<&Span> = got.iter().filter(|s| !near(s.score)).collect();
                let expected_sure: Vec<&ExpectedSpan> = expected.iter().filter(|s| !near(s.score)).collect();
                borderline += got.len() - got_sure.len();
                assert_eq!(got_sure.len(), expected_sure.len(), "case {} question {key}", case.id);
                for (g, e) in got_sure.iter().zip(&expected_sure) {
                    assert_eq!((g.start, g.end, g.text.as_str()), (e.start, e.end, e.text.as_str()), "case {} question {key}", case.id);
                    assert!((g.score - e.score).abs() < 2e-4, "case {} question {key}: {} vs {}", case.id, g.score, e.score);
                    assert_eq!(g.text, state[g.start..g.end].iter().collect::<String>());
                    compared += 1;
                }
            }
        }
    }
    assert!(compared > 1000, "{compared} spans compared");
    assert!(borderline < 5, "{borderline} spans on the cutoff");
}

/// The prompt, its tokens and the state's token offsets, with the checkpoint's tokenizer.
#[test]
fn encoding_matches_python() {
    let fixture = fixture();
    let Ok(checkpoint) = std::env::var("DECISION_CHECKPOINT_V4") else {
        eprintln!("DECISION_CHECKPOINT_V4 not set: skipping the Locate token check");
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{checkpoint}/tokenizer.json")).expect("tokenizer.json");
    let encoder = LocateEncoder::new(Box::new(tokenizer), fixture.codes.clone(), fixture.pad_id, 2, fixture.marker_id).expect("encoder");
    let (mut sequences, mut length_errors) = (0, 0);
    for case in &fixture.cases {
        if case.request_error.is_some() {
            continue;
        }
        let parsed = parse(&case.request);
        let labels = encoder.validate(&parsed.state, parsed.images, parsed.total, &parsed.locate).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        let keys: Vec<String> = parsed.locate.iter().map(|(key, _)| key.clone()).collect();
        let state = parsed.state.as_str().expect("text state");
        match (encoder.plan(state, &keys, &labels, case.max_length), &case.length_error) {
            (Err(ProtocolError::Length(message)), Some(expected)) => {
                assert_eq!(&message, expected, "case {}", case.id);
                length_errors += 1;
            }
            (Ok(plan), None) => {
                let expected = case.plans.as_ref().expect("plans");
                assert_eq!(plan.len(), expected.len(), "case {}", case.id);
                let mut tokens = 0;
                for (encoded, want) in plan.iter().zip(expected) {
                    let row = encoded.locate.as_ref().expect("a locate row");
                    assert_eq!(encoded.input_ids, want.input_ids, "case {}", case.id);
                    assert_eq!(encoded.markers, want.markers, "case {}", case.id);
                    assert_eq!(row.segments, want.segments, "case {}", case.id);
                    assert_eq!(row.positions, want.positions, "case {}", case.id);
                    assert_eq!(row.offsets, want.offsets, "case {}", case.id);
                    assert_eq!(encoded.keys, keys[want.keys_from..want.keys_from + want.markers.len()], "case {}", case.id);
                    assert!(encoded.markers.iter().all(|&m| encoded.input_ids[m] == fixture.marker_id));
                    tokens += encoded.length();
                    sequences += 1;
                }
                let usage = case.usage.as_ref().expect("usage");
                assert_eq!((usage["input_tokens"].as_u64(), usage["sequences"].as_u64()), (Some(tokens as u64), Some(plan.len() as u64)), "case {}", case.id);
            }
            (got, expected) => panic!("case {}: {:?} where Python gave {expected:?}", case.id, got.map(|p| p.len())),
        }
    }
    assert!(sequences >= 14 && length_errors >= 1);
}
