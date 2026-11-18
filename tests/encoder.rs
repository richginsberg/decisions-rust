//! The Rust encoder against what pplx_decider's encoders produced, in both protocol versions:
//! fixtures/encoder.json (version 3, written by scripts/make_encoder_fixture.py) and fixtures/encoder_v4.json
//! (version 4, written by scripts/make_encoder_fixture_v4.py from the checkpoint's own protocol.py).
//!
//! The prompt text is checked for every case with no tokenizer involved. The token ids are checked when
//! `DECISION_CHECKPOINT` (version 3) or `DECISION_CHECKPOINT_V4` names a directory with that checkpoint's
//! `tokenizer.json` and `decision_config.json`.

use std::collections::BTreeSet;

use decisions_rust_nvfp4::config::Protocol;
use decisions_rust_nvfp4::protocol::{Encoder, ImageGrid, Question, Request, SPECIAL_TOKENS, SYSTEM_PROMPT, SYSTEM_PROMPT_V4};
use decisions_rust_nvfp4::testing::{word_encoder, word_encoder_for, word_encoder_with};
use serde::Deserialize;
use serde_json::Value;

#[derive(Deserialize)]
struct Fixture {
    system_prompt: String,
    escapes: Vec<String>,
    codes: Vec<String>,
    marker_id: u32,
    pad_id: u32,
    image_pad_id: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    id: String,
    request: Value,
    text: String,
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    counts: Vec<usize>,
}

fn fixture() -> Fixture {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/encoder.json")).expect("fixture");
    serde_json::from_str(&text).expect("fixture json")
}

/// A fixture request (questions as a JSON object, in order) as the protocol's `Request`.
fn request_of(value: &Value) -> Request {
    let object = value.as_object().expect("request object");
    let questions = object["questions"]
        .as_object()
        .expect("questions")
        .iter()
        .map(|(key, question)| (key.clone(), serde_json::from_value::<Question>(question.clone()).expect("question")))
        .collect();
    let images = object.get("images").and_then(Value::as_array).map(|a| a.iter().map(|v| v.as_str().unwrap().to_string()).collect()).unwrap_or_default();
    Request { state: object["state"].clone(), questions, images }
}

#[test]
fn constants_match_the_fixture() {
    let fixture = fixture();
    assert_eq!(fixture.system_prompt, SYSTEM_PROMPT);
    let mut expected: BTreeSet<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
    expected.insert("<|fim_suffix|>".into());
    assert_eq!(fixture.escapes.iter().cloned().collect::<BTreeSet<_>>(), expected);
}

#[test]
fn prompt_text_matches_python() {
    let fixture = fixture();
    let encoder = word_encoder_with(fixture.codes.clone());
    for case in &fixture.cases {
        let request = request_of(&case.request);
        encoder.validate(&request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(encoder.text(&request), case.text, "case {}", case.id);
        let counts: Vec<usize> = request.questions.iter().map(|(_, q)| decisions_rust_nvfp4::protocol::options(q).0.len()).collect();
        assert_eq!(counts, case.counts, "case {}", case.id);
    }
    assert!(fixture.cases.len() > 100);
}

fn real_encoder(fixture: &Fixture) -> Option<Encoder> {
    let checkpoint = std::env::var("DECISION_CHECKPOINT").ok()?;
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{checkpoint}/tokenizer.json")).expect("tokenizer.json");
    let decision: Value = serde_json::from_str(&std::fs::read_to_string(format!("{checkpoint}/decision_config.json")).expect("decision_config")).unwrap();
    let codes = decision["codes"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    let encoder = Encoder::new(Box::new(tokenizer), codes, fixture.pad_id, 2).expect("encoder");
    assert_eq!(encoder.marker_id, fixture.marker_id);
    assert_eq!(encoder.image_pad_id, fixture.image_pad_id);
    Some(encoder)
}

#[test]
fn token_ids_match_python() {
    let fixture = fixture();
    let Some(encoder) = real_encoder(&fixture) else {
        eprintln!("DECISION_CHECKPOINT not set: skipping the token id check");
        return;
    };
    for case in &fixture.cases {
        let request = request_of(&case.request);
        // The fixture holds one unexpanded <|image_pad|> per image: a 2x2 grid merges to exactly one token.
        let grids = vec![ImageGrid { t: 1, h: 2, w: 2 }; request.images.len()];
        let encoded = encoder.encode(&request, &grids).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(encoded.input_ids, case.input_ids, "case {}", case.id);
        assert_eq!(encoded.markers, case.markers, "case {}", case.id);
        assert_eq!(encoded.counts, case.counts, "case {}", case.id);
    }
}

#[test]
fn plan_bisects_over_length_requests() {
    let encoder = word_encoder();
    let request = Request {
        state: Value::String("word ".repeat(100)),
        questions: (0..8).map(|i| (format!("q{i}"), Question::Noul { criteria: None, instructions: Some(Value::String("x ".repeat(50))) })).collect(),
        images: vec![],
    };
    let whole = encoder.plan(&request, &[], None).unwrap();
    assert_eq!(whole.len(), 1);
    let chunks = encoder.plan(&request, &[], Some(whole[0].length() / 2)).unwrap();
    assert!(chunks.len() >= 2);
    let keys: Vec<&str> = chunks.iter().flat_map(|c| c.keys.iter().map(String::as_str)).collect();
    assert_eq!(keys, (0..8).map(|i| format!("q{i}")).collect::<Vec<_>>());
    assert!(chunks.iter().all(|c| c.length() <= whole[0].length() / 2));
    let error = encoder.plan(&request, &[], Some(120)).unwrap_err();
    assert!(error.to_string().starts_with("Question 'q0' needs"), "{error}");
}

// ------------------------------------------------------------------------------------------- version 4

#[derive(Deserialize)]
struct FixtureV4 {
    system_prompt: String,
    escapes: Vec<String>,
    codes: Vec<String>,
    marker_id: u32,
    separator_id: u32,
    pad_id: u32,
    image_pad_id: u32,
    merge_size: usize,
    cases: Vec<CaseV4>,
}

#[derive(Deserialize)]
struct CaseV4 {
    id: String,
    request: Value,
    context_text: String,
    question_texts: Vec<String>,
    /// (t, h, w) patches of each image, as the processor reported them.
    grids: Vec<[usize; 3]>,
    sequences: Vec<SequenceV4>,
}

#[derive(Deserialize)]
struct SequenceV4 {
    keys: Vec<String>,
    counts: Vec<usize>,
    input_ids: Vec<u32>,
    markers: Vec<usize>,
    segments: Vec<usize>,
}

fn fixture_v4() -> FixtureV4 {
    let text = std::fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/fixtures/encoder_v4.json")).expect("fixture");
    serde_json::from_str(&text).expect("fixture json")
}

#[test]
fn v4_constants_match_the_fixture() {
    let fixture = fixture_v4();
    assert_eq!(fixture.system_prompt, SYSTEM_PROMPT_V4);
    let mut expected: BTreeSet<String> = SPECIAL_TOKENS.iter().map(|s| s.to_string()).collect();
    expected.insert("<|fim_suffix|>".into());
    expected.insert("<|fim_middle|>".into());
    assert_eq!(fixture.escapes.iter().cloned().collect::<BTreeSet<_>>(), expected);
}

#[test]
fn v4_prompt_text_matches_python() {
    let fixture = fixture_v4();
    let encoder = word_encoder_for(Protocol::ChunkedV4, fixture.codes.clone());
    for case in &fixture.cases {
        let request = request_of(&case.request);
        encoder.validate(&request).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(encoder.context_text(&request), case.context_text, "case {}", case.id);
        let questions: Vec<String> = request.questions.iter().map(|(_, q)| encoder.question_text(q)).collect();
        assert_eq!(questions, case.question_texts, "case {}", case.id);
    }
    assert!(fixture.cases.len() > 100);
}

#[test]
fn v4_token_ids_match_python() {
    let fixture = fixture_v4();
    let Ok(checkpoint) = std::env::var("DECISION_CHECKPOINT_V4") else {
        eprintln!("DECISION_CHECKPOINT_V4 not set: skipping the version 4 token id check");
        return;
    };
    let tokenizer = tokenizers::Tokenizer::from_file(format!("{checkpoint}/tokenizer.json")).expect("tokenizer.json");
    let decision: Value = serde_json::from_str(&std::fs::read_to_string(format!("{checkpoint}/decision_config.json")).expect("decision_config")).unwrap();
    let codes: Vec<String> = decision["codes"].as_array().unwrap().iter().map(|c| c.as_str().unwrap().to_string()).collect();
    assert_eq!(codes, fixture.codes);
    let encoder = Encoder::with_protocol(Protocol::ChunkedV4, Box::new(tokenizer), codes, fixture.pad_id, fixture.merge_size).expect("encoder");
    assert_eq!(encoder.marker_id, fixture.marker_id);
    assert_eq!(encoder.separator_id, Some(fixture.separator_id));
    assert_eq!(encoder.image_pad_id, fixture.image_pad_id);
    let mut sequences = 0;
    for case in &fixture.cases {
        let request = request_of(&case.request);
        let grids: Vec<ImageGrid> = case.grids.iter().map(|&[t, h, w]| ImageGrid { t, h, w }).collect();
        let plan = encoder.plan(&request, &grids, None).unwrap_or_else(|e| panic!("{}: {e}", case.id));
        assert_eq!(plan.len(), case.sequences.len(), "case {}", case.id);
        for (encoded, expected) in plan.iter().zip(&case.sequences) {
            assert_eq!(encoded.keys, expected.keys, "case {}", case.id);
            assert_eq!(encoded.counts, expected.counts, "case {}", case.id);
            assert_eq!(encoded.input_ids, expected.input_ids, "case {} question {:?}", case.id, expected.keys);
            assert_eq!(encoded.markers, expected.markers, "case {}", case.id);
            // The context (with its separator) is the first segment; the marker is the sequence's last token.
            assert_eq!(encoded.input_ids[expected.segments[0] - 1], fixture.separator_id, "case {}", case.id);
            assert_eq!(encoded.markers, vec![encoded.length() - 1], "case {}", case.id);
            sequences += 1;
        }
    }
    assert!(sequences > 300);
}

#[test]
fn v4_plans_one_sequence_per_question() {
    let encoder = word_encoder_for(Protocol::ChunkedV4, word_encoder().codes().to_vec());
    let request = Request {
        state: Value::String("word ".repeat(100)),
        questions: (0..8).map(|i| (format!("q{i}"), Question::Noul { criteria: None, instructions: Some(Value::String(format!("number {i} ")) ) })).collect(),
        images: vec![],
    };
    let plan = encoder.plan(&request, &[], None).unwrap();
    assert_eq!(plan.len(), 8);
    let context = encoder.context_ids(&request, &[]).unwrap();
    for (i, encoded) in plan.iter().enumerate() {
        assert_eq!(encoded.keys, vec![format!("q{i}")]);
        assert_eq!(encoded.counts, vec![2]);
        assert_eq!(&encoded.input_ids[..context.len()], &context[..]);
        assert_eq!(encoded.markers, vec![encoded.length() - 1]);
    }
    // Nothing is bisected or truncated: a question that does not fit fails the request.
    let error = encoder.plan(&request, &[], Some(plan[0].length() - 1)).unwrap_err();
    assert!(error.to_string().starts_with("Question 'q0' needs"), "{error}");
}

/// The real config files of both releases load, each as its own protocol (when the checkpoints are given).
#[test]
fn checkpoints_declare_their_protocols() {
    use decisions_rust_nvfp4::config::Checkpoint;
    for (variable, protocol, name) in [("DECISION_CHECKPOINT", Protocol::InlineV3, "v1.2"), ("DECISION_CHECKPOINT_V4", Protocol::ChunkedV4, "v1.3")] {
        let Ok(path) = std::env::var(variable) else { continue };
        let checkpoint = Checkpoint::load(std::path::Path::new(&path)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(checkpoint.protocol, protocol, "{name}");
    }
}
