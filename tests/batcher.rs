//! The batcher's continuous batching, on the CPU, with the fake model (decisions-endpoint-nvfp4's
//! test_continuous_batching.py, ported).
//!
//! Only the scheduling is under test: which rows share a pass, when each request is answered, and how
//! failures, disconnects and backpressure are handled. The fake model answers each question from its row's
//! tokens alone and takes time in proportion to a pass's tokens. Requests over the pass budget are split
//! into question chunks, so a long request spans several passes.

use std::sync::Arc;
use std::time::{Duration, Instant};

use decisions_rust_nvfp4::batcher::{Batcher, Outcome};
use decisions_rust_nvfp4::model::{Budget, FakeModel, Model};
use decisions_rust_nvfp4::protocol::{Answer, Question, Request, answer};
use decisions_rust_nvfp4::testing::{WordTokenizer, fake_model};
use serde_json::{Value, json};

const BUDGET: Budget = Budget { max_tokens: 4000, max_rows: 8, packed: true };

fn request(name: &str, questions: usize, words: usize, bad: Option<usize>) -> Request {
    Request {
        state: Value::String(format!("{name} {}", "word ".repeat(words))),
        questions: (0..questions)
            .map(|i| {
                let instructions = format!("{name} question {i} {}{}", "x ".repeat(300), if bad == Some(i) { " bad" } else { "" });
                (format!("q{i}"), Question::Choice { criteria: json!({"a": "x", "b": "y", "c": "z"}).as_object().unwrap().clone(), instructions: Some(Value::String(instructions)) })
            })
            .collect(),
        images: vec![],
    }
}

/// Answer a request through the batcher; returns the outcome and when it started and finished.
async fn submit(batcher: &Batcher, model: &FakeModel, body: &Request, delay: Duration) -> (Outcome, Instant, Instant) {
    tokio::time::sleep(delay).await;
    let started = Instant::now();
    let encoder = model.encoder();
    let rows = encoder.plan(body, &[], Some(BUDGET.max_tokens)).expect("plan");
    let estimate = encoder.count_tokens(body.state.as_str().unwrap()).unwrap() + 80 * body.questions.len();
    let outcome = batcher.submit(body.questions.clone(), rows, estimate).await;
    (outcome, started, Instant::now())
}

/// The answers a request gets from the fake model on its own.
fn alone(model: &FakeModel, body: &Request) -> Vec<(String, Answer)> {
    let rows = model.encoder().plan(body, &[], Some(BUDGET.max_tokens)).unwrap();
    let mut out = Vec::new();
    for row in &rows {
        let answers = FakeModel::answers(row);
        for ((key, question), values) in body.questions.iter().filter(|(k, _)| row.keys.contains(k)).zip(answers) {
            out.push((key.clone(), answer(question, &values).unwrap()));
        }
    }
    out
}

fn answered(outcome: &Outcome) -> &Vec<(String, Answer)> {
    match outcome {
        Outcome::Answered(decision) => &decision.answers,
        Outcome::Failed(failure) => panic!("failed: {} {}", failure.status, failure.message),
    }
}

fn passes_with(model: &FakeModel, name: &str) -> Vec<usize> {
    let id = WordTokenizer::id_of(name);
    // The state's first word follows the fixed prompt prefix: "...user\nState:\n{name} word ..."; "State:" splits off.
    model.passes.lock().iter().enumerate().filter(|(_, rows)| rows.iter().any(|ids| ids.contains(&id))).map(|(i, _)| i).collect()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn continuous_batching() {
    let model = fake_model(BUDGET, 2e-5);
    let batcher = Batcher::new(model.clone(), 30.0);
    batcher.start();

    // 1. Concurrent requests get the answers they would get alone.
    let bodies: Vec<Request> = (0..10).map(|i| request(&format!("r{i}"), 3 + i % 4, 50 + 40 * i, None)).collect();
    let results = futures_join_all(bodies.iter().map(|b| submit(&batcher, &model, b, Duration::ZERO))).await;
    for (body, (outcome, _, _)) in bodies.iter().zip(&results) {
        assert_eq!(answered(outcome), &alone(&model, body), "{}", body.state.as_str().unwrap().split(' ').next().unwrap());
    }
    println!("1. ten concurrent requests: same answers as alone");

    // 2. Continuous: a short request that arrives while a long one runs joins a later pass and is answered
    //    long before the long request finishes. (40 questions x ~300 tokens split into chunks of ~3,000
    //    tokens, one per pass under the 4,000-token budget, so the long request needs at least 4 passes.)
    model.passes.lock().clear();
    let long_body = request("long", 40, 450, None);
    let short_body = request("short", 2, 20, None);
    assert!(model.encoder().plan(&long_body, &[], Some(BUDGET.max_tokens)).unwrap().len() >= 4);
    let ((long_result, _, long_done), (short_result, short_start, short_done)) =
        tokio::join!(submit(&batcher, &model, &long_body, Duration::ZERO), submit(&batcher, &model, &short_body, Duration::from_millis(20)));
    let long_passes = passes_with(&model, "long");
    let short_passes = passes_with(&model, "short");
    assert!(short_done < long_done);
    assert!(long_passes.len() >= 4, "{long_passes:?}");
    assert!(short_passes[0] < *long_passes.last().unwrap(), "short waited for the whole long request: {short_passes:?} vs {long_passes:?}");
    assert_eq!(answered(&short_result), &alone(&model, &short_body));
    assert_eq!(answered(&long_result), &alone(&model, &long_body));
    println!(
        "2. continuous: short answered {} ms after arriving, {} ms before the long request ({} passes) finished",
        (short_done - short_start).as_millis(),
        (long_done - short_done).as_millis(),
        long_passes.len()
    );

    // 3. A failing question fails only its own request; requests sharing its pass are answered.
    let good = request("good", 3, 30, None);
    let broken = request("broken", 3, 30, Some(1));
    let ((good_result, _, _), (broken_result, _, _)) =
        tokio::join!(submit(&batcher, &model, &good, Duration::ZERO), submit(&batcher, &model, &broken, Duration::ZERO));
    assert_eq!(answered(&good_result), &alone(&model, &good));
    match broken_result {
        Outcome::Failed(failure) => assert_eq!((failure.status, failure.message.as_str()), (422, "an invalid question")),
        Outcome::Answered(_) => panic!("the broken request was answered"),
    }
    println!("3. a bad question: 422 for its request only");

    // 4. A client that disconnects: its remaining rows are dropped and its tokens released.
    model.passes.lock().clear();
    let gone = request("gone", 40, 450, None);
    let rows = model.encoder().plan(&gone, &[], Some(BUDGET.max_tokens)).unwrap().len();
    let task = {
        let (batcher, model, gone) = (Arc::clone(&batcher), Arc::clone(&model), gone.clone());
        tokio::spawn(async move { submit(&batcher, &model, &gone, Duration::ZERO).await })
    };
    tokio::time::sleep(Duration::from_millis(50)).await;
    task.abort();
    tokio::time::sleep(Duration::from_millis(500)).await;
    let gone_rows: usize = model.passes.lock().iter().flatten().filter(|ids| ids.contains(&WordTokenizer::id_of("gone"))).count();
    assert!(gone_rows < rows, "{gone_rows} of {rows}");
    println!("4. disconnect: {gone_rows} of {rows} rows run before it was dropped");

    // 5. Backpressure: a request over the queue limit gets 529 with Retry-After.
    let strict = Batcher::new(model.clone(), 0.1); // 0.1 s of queue at 1,000 tokens/s: 100 tokens
    strict.start();
    strict.set_rate(1000.0);
    let flood: Vec<Request> = (0..4).map(|i| request(&format!("f{i}"), 40, 400, None)).collect();
    let out = futures_join_all(flood.iter().map(|b| submit(&strict, &model, b, Duration::ZERO))).await;
    let refused: Vec<&Outcome> = out.iter().map(|(o, _, _)| o).filter(|o| matches!(o, Outcome::Failed(f) if f.status == 529)).collect();
    assert!(!refused.is_empty());
    assert!(refused.iter().all(|o| matches!(o, Outcome::Failed(f) if f.retry_after.is_some())));
    println!("5. backpressure: {} of 4 large requests refused with 529", refused.len());

    // 6. Everything answered or dropped: nothing left counted against the queue.
    tokio::time::sleep(Duration::from_millis(200)).await;
    assert_eq!(batcher.waiting_tokens(), 0);
    assert_eq!(strict.waiting_tokens(), 0);
    println!("6. queue empty: waiting tokens back to 0");

    // 7. Every pass respects the budget.
    for pass in model.passes.lock().iter() {
        let lengths: Vec<usize> = pass.iter().map(Vec::len).collect();
        assert!(BUDGET.fits(&lengths), "{lengths:?}");
    }
    println!("7. passes within the budget");
}

async fn futures_join_all<F: std::future::Future>(futures: impl Iterator<Item = F>) -> Vec<F::Output> {
    let mut handles = Vec::new();
    for future in futures {
        handles.push(Box::pin(future));
    }
    let mut out = Vec::with_capacity(handles.len());
    // Poll them together: a JoinSet needs 'static futures, so drive them with a simple select loop.
    let mut pending: Vec<Option<std::pin::Pin<Box<F>>>> = handles.into_iter().map(Some).collect();
    let mut results: Vec<Option<F::Output>> = (0..pending.len()).map(|_| None).collect();
    std::future::poll_fn(|cx| {
        let mut all_done = true;
        for (i, slot) in pending.iter_mut().enumerate() {
            if let Some(future) = slot {
                match future.as_mut().poll(cx) {
                    std::task::Poll::Ready(value) => {
                        results[i] = Some(value);
                        *slot = None;
                    }
                    std::task::Poll::Pending => all_done = false,
                }
            }
        }
        if all_done { std::task::Poll::Ready(()) } else { std::task::Poll::Pending }
    })
    .await;
    for result in results {
        out.push(result.expect("completed"));
    }
    out
}

#[test]
fn budget_fits() {
    let padded = Budget { max_tokens: 1000, max_rows: 3, packed: false };
    assert!(padded.fits(&[400, 100]));
    assert!(!padded.fits(&[400, 100, 300])); // 3 x 400 > 1000
    assert!(!padded.fits(&[1, 1, 1, 1]));
    let packed = Budget { max_tokens: 1000, max_rows: 8, packed: true };
    assert!(packed.fits(&[400, 100, 300]));
    assert!(!packed.fits(&[400, 100, 600]));
}
