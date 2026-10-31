//! Continuous batching: a queue of requests in front of one GPU thread, scheduled pass by pass.
//!
//! A request is split into its rows: its sequences (the whole request, or chunks of its questions). Before
//! every pass the GPU thread admits the requests that have arrived since the last one, and plans the pass
//! from all rows still waiting. The requests take turns: the pass starts with a row of the request that has
//! waited longest since a pass last answered one of its rows (a new request first, then the oldest), and
//! fills up with the waiting rows closest to that row's length, from any request, while the pass fits the
//! model's budget. A request is answered as soon as its last row is, not when a round ends: a short request
//! that arrives while a long one is being answered gets a row into the next pass. Rows are independent, so
//! each request gets the answers it would get alone. If a pass fails, its rows are retried one at a time, so
//! only the request at fault gets the error. A request is refused with 529 only when the work ahead of it
//! would take longer than `max_queue_seconds`, estimated from the throughput of recent passes.

use std::cell::RefCell;
use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;
use std::time::Instant;

use parking_lot::{Condvar, Mutex};
use tokio::sync::oneshot;

use crate::model::{PassError, RowAnswers, SharedModel};
use crate::protocol::{Answer, Encoded, ProtocolError, Question, answer};

/// How a request ends: the answers, or an HTTP error.
#[derive(Debug)]
pub enum Outcome {
    Answered(Decision),
    Failed(Failure),
}

#[derive(Debug, Clone)]
pub struct Decision {
    pub answers: Vec<(String, Answer)>,
    pub input_tokens: usize,
    pub sequences: usize,
}

#[derive(Debug, Clone)]
pub struct Failure {
    pub status: u16,
    pub message: String,
    pub retry_after: Option<u64>,
}

impl Failure {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self { status, message: message.into(), retry_after: None }
    }
    fn from_pass(error: PassError) -> Self {
        match error {
            PassError::Request(m) => Self::new(422, m),
            PassError::OutOfMemory => Self::new(413, error_text(PassError::OutOfMemory)),
            PassError::Internal(m) => Self::new(500, m),
        }
    }
    pub fn from_protocol(error: ProtocolError) -> Self {
        match error {
            ProtocolError::Request(m) => Self::new(422, m),
            ProtocolError::Length(m) => Self::new(413, m),
            ProtocolError::Internal(m) => Self::new(500, m),
        }
    }
}

fn error_text(error: PassError) -> String {
    error.to_string()
}

/// A request in the batcher: its rows, the answers so far, and where to send the response.
struct Job {
    questions: Vec<(String, Question)>,
    rows: Vec<Encoded>,
    /// Estimated tokens, for backpressure (counted until `finish`).
    tokens: usize,
    results: Vec<Option<RowAnswers>>,
    remaining: usize,
    failed: bool,
    counted: bool,
    /// The last pass that answered one of its rows (-1: none yet).
    served: i64,
    reply: Option<oneshot::Sender<Outcome>>,
}

impl Job {
    fn gone(&self) -> bool {
        self.failed || self.reply.as_ref().is_none_or(|reply| reply.is_closed())
    }
}

type Shared = Rc<RefCell<Job>>;

struct Inner {
    /// Jobs submitted since the last pass.
    arrivals: VecDeque<Job>,
    /// Estimated tokens of everything not yet answered.
    waiting_tokens: usize,
    /// Tokens per second over recent passes.
    rate: Option<f64>,
}

pub struct Batcher {
    model: SharedModel,
    inner: Mutex<Inner>,
    wake: Condvar,
    pub max_queue_seconds: f64,
}

impl Batcher {
    pub fn new(model: SharedModel, max_queue_seconds: f64) -> Arc<Self> {
        Arc::new(Self {
            model,
            inner: Mutex::new(Inner { arrivals: VecDeque::new(), waiting_tokens: 0, rate: None }),
            wake: Condvar::new(),
            max_queue_seconds,
        })
    }

    /// Start the GPU thread.
    pub fn start(self: &Arc<Self>) {
        let batcher = Arc::clone(self);
        std::thread::Builder::new().name("gpu".into()).spawn(move || Worker::new(batcher).run()).expect("gpu thread");
    }

    pub fn waiting_tokens(&self) -> usize {
        self.inner.lock().waiting_tokens
    }

    /// Set the throughput estimate (tests).
    pub fn set_rate(&self, rate: f64) {
        self.inner.lock().rate = Some(rate);
    }

    /// Queue a request: its questions and its encoded rows. `tokens` is the backpressure estimate.
    pub async fn submit(&self, questions: Vec<(String, Question)>, rows: Vec<Encoded>, tokens: usize) -> Outcome {
        let (sender, receiver) = oneshot::channel();
        {
            let mut inner = self.inner.lock();
            if inner.waiting_tokens > 0 {
                let budget = self.model.budget().max_tokens;
                let limit = inner.rate.map_or(4.0 * budget as f64, |rate| self.max_queue_seconds * rate);
                if (inner.waiting_tokens + tokens) as f64 > limit {
                    let wait = inner.rate.map_or(1.0, |rate| inner.waiting_tokens as f64 / rate);
                    return Outcome::Failed(Failure {
                        status: 529,
                        message: "The model is busy. Retry shortly.".into(),
                        retry_after: Some(wait.ceil().max(1.0) as u64),
                    });
                }
            }
            let remaining = rows.len();
            inner.arrivals.push_back(Job {
                questions,
                results: vec![None; rows.len()],
                rows,
                tokens,
                remaining,
                failed: false,
                counted: true,
                served: -1,
                reply: Some(sender),
            });
            inner.waiting_tokens += tokens;
        }
        self.wake.notify_one();
        match receiver.await {
            Ok(outcome) => outcome,
            Err(_) => Outcome::Failed(Failure::new(500, "The batcher dropped the request.")),
        }
    }
}

/// The GPU thread's state: admitted jobs and their rows still waiting.
struct Worker {
    batcher: Arc<Batcher>,
    jobs: Vec<Shared>,
    /// (job, row index) not yet answered, in arrival order.
    pending: Vec<(Shared, usize)>,
    passes: i64,
}

impl Worker {
    fn new(batcher: Arc<Batcher>) -> Self {
        Self { batcher, jobs: Vec::new(), pending: Vec::new(), passes: 0 }
    }

    fn run(mut self) {
        loop {
            let arrivals: Vec<Job> = {
                let mut inner = self.batcher.inner.lock();
                while inner.arrivals.is_empty() && self.pending.is_empty() {
                    self.batcher.wake.wait(&mut inner);
                }
                inner.arrivals.drain(..).collect()
            };
            for job in arrivals {
                self.admit(job);
            }
            self.step();
        }
    }

    fn admit(&mut self, job: Job) {
        let shared = Rc::new(RefCell::new(job));
        let rows = shared.borrow().rows.len();
        self.jobs.push(Rc::clone(&shared));
        self.pending.extend((0..rows).map(|index| (Rc::clone(&shared), index)));
    }

    /// The next pass: a row of the request whose turn it is, then the rows nearest its length that fit.
    fn plan(&mut self) -> Vec<(Shared, usize)> {
        self.prune();
        self.pending.retain(|(job, _)| !job.borrow().gone());
        if self.pending.is_empty() {
            return Vec::new();
        }
        let budget = self.batcher.model.budget();
        let (first, _) = self
            .pending
            .iter()
            .enumerate()
            .min_by_key(|(_, (job, _))| job.borrow().served)
            .map(|(i, item)| (i, item.clone()))
            .expect("pending is not empty");
        let (anchor_job, anchor_index) = self.pending.remove(first);
        let anchor = anchor_job.borrow().rows[anchor_index].length();
        let mut chosen = vec![(anchor_job, anchor_index)];
        let mut lengths = vec![anchor];
        let mut rest: Vec<(usize, usize)> =
            self.pending.iter().enumerate().map(|(i, (job, index))| (i, job.borrow().rows[*index].length())).collect();
        rest.sort_by_key(|&(i, length)| (length.abs_diff(anchor), i));
        let mut taken = Vec::new();
        for (i, length) in rest {
            if chosen.len() == budget.max_rows {
                break;
            }
            lengths.push(length);
            if budget.fits(&lengths) {
                chosen.push(self.pending[i].clone());
                taken.push(i);
            } else {
                lengths.pop();
            }
        }
        taken.sort_unstable_by(|a, b| b.cmp(a));
        for i in taken {
            self.pending.remove(i);
        }
        self.passes += 1;
        for (job, _) in &chosen {
            job.borrow_mut().served = self.passes;
        }
        chosen
    }

    fn step(&mut self) {
        let chosen = self.plan();
        if chosen.is_empty() {
            self.prune();
            return;
        }
        let started = Instant::now();
        let outcome = {
            let borrowed: Vec<std::cell::Ref<Job>> = chosen.iter().map(|(job, _)| job.borrow()).collect();
            let rows: Vec<&Encoded> = borrowed.iter().zip(&chosen).map(|(job, (_, index))| &job.rows[*index]).collect();
            self.batcher.model.run_pass(&rows)
        };
        match outcome {
            Ok(results) => {
                for (item, result) in chosen.iter().zip(results) {
                    self.record(item, result);
                }
            }
            Err(error) => {
                tracing::error!("Pass of {} row(s) failed: {error}", chosen.len());
                if chosen.len() == 1 {
                    Self::fail(&chosen[0].0, Failure::from_pass(error));
                } else {
                    // Find the row at fault by answering each alone.
                    for item in &chosen {
                        self.answer_row(item);
                    }
                }
            }
        }
        let elapsed = started.elapsed().as_secs_f64();
        let tokens: usize = chosen.iter().map(|(job, index)| job.borrow().rows[*index].length()).sum();
        if elapsed > 0.05 {
            let rate = tokens as f64 / elapsed;
            let mut inner = self.batcher.inner.lock();
            inner.rate = Some(inner.rate.map_or(rate, |old| 0.7 * old + 0.3 * rate));
        }
        let requests = {
            let mut ids: Vec<*const RefCell<Job>> = chosen.iter().map(|(job, _)| Rc::as_ptr(job)).collect();
            ids.sort_unstable();
            ids.dedup();
            ids.len()
        };
        tracing::info!(
            "Pass: {requests} request(s), {} row(s), {tokens} tokens in {:.0} ms ({:.0} tokens/s)",
            chosen.len(),
            elapsed * 1000.0,
            if elapsed > 0.0 { tokens as f64 / elapsed } else { 0.0 }
        );
        self.prune();
    }

    /// Drop answered, failed and disconnected requests, releasing their tokens.
    fn prune(&mut self) {
        for job in &self.jobs {
            if job.borrow().gone() {
                self.finish(job);
            }
        }
        self.jobs.retain(|job| {
            let job = job.borrow();
            job.remaining > 0 && !job.gone()
        });
    }

    fn record(&self, (job, index): &(Shared, usize), result: RowAnswers) {
        let mut borrowed = job.borrow_mut();
        if borrowed.gone() {
            return;
        }
        borrowed.results[*index] = Some(result);
        borrowed.remaining -= 1;
        if borrowed.remaining == 0 {
            drop(borrowed);
            self.finish(job);
            let mut borrowed = job.borrow_mut();
            let outcome = match Self::decision(&borrowed) {
                Ok(decision) => Outcome::Answered(decision),
                Err(failure) => Outcome::Failed(failure),
            };
            if let Some(reply) = borrowed.reply.take() {
                let _ = reply.send(outcome);
            }
        }
    }

    fn decision(job: &Job) -> Result<Decision, Failure> {
        let mut by_key: std::collections::HashMap<&str, &Vec<f32>> = std::collections::HashMap::new();
        for (row, result) in job.rows.iter().zip(&job.results) {
            let result = result.as_ref().expect("every row answered");
            for (key, values) in row.keys.iter().zip(result) {
                by_key.insert(key.as_str(), values);
            }
        }
        let mut answers = Vec::with_capacity(job.questions.len());
        for (key, question) in &job.questions {
            let values = by_key.get(key.as_str()).ok_or_else(|| Failure::new(500, format!("no answer for question {key}")))?;
            answers.push((key.clone(), answer(question, values).map_err(Failure::from_protocol)?));
        }
        Ok(Decision { answers, input_tokens: job.rows.iter().map(|r| r.length()).sum(), sequences: job.rows.len() })
    }

    fn answer_row(&self, item: &(Shared, usize)) {
        let (job, index) = item;
        if job.borrow().gone() {
            return;
        }
        let outcome = {
            let borrowed = job.borrow();
            self.batcher.model.run_pass(&[&borrowed.rows[*index]])
        };
        match outcome {
            Ok(mut results) => self.record(item, results.remove(0)),
            Err(error) => {
                tracing::error!("Row of {} tokens failed alone: {error}", job.borrow().rows[*index].length());
                Self::fail(job, Failure::from_pass(error));
            }
        }
    }

    /// Release a request's tokens from the backpressure estimate, once.
    fn finish(&self, job: &Shared) {
        let mut borrowed = job.borrow_mut();
        if borrowed.counted {
            borrowed.counted = false;
            self.batcher.inner.lock().waiting_tokens -= borrowed.tokens;
        }
    }

    fn fail(job: &Shared, failure: Failure) {
        let mut borrowed = job.borrow_mut();
        if borrowed.failed {
            return;
        }
        borrowed.failed = true;
        if let Some(reply) = borrowed.reply.take() {
            let _ = reply.send(Outcome::Failed(failure));
        }
        // `finish` runs at the next prune, which sees the job gone.
    }
}
