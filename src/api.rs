//! The HTTP layer: autojev's contract, as decider_api.py keeps it.
//!
//! Routes: `GET /health` (open), `GET /v1/models`, `POST /v1/systemone`, `POST /v1/decisions` (an alias). Bearer
//! auth with `AUTOJEV_API_KEY`. Errors: 401 (key), 413 (too long, or out of GPU memory), 422 (invalid request),
//! 503 (loading), 529 with `Retry-After` (queue full).

use std::sync::Arc;
use std::time::Instant;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router, middleware};
use base64::Engine as _;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::batcher::{Batcher, Failure, Outcome};
use crate::config::MAX_OPTIONS;
use crate::images;
use crate::model::SharedModel;
use crate::protocol::{Question, Request};

pub const DEFAULT_MODEL: &str = "pplx-decider-v1.2";
/// Names clients may send in "model": the served one, and v1.1's (autojev's), so existing clients keep working.
pub const ALIASES: [&str; 7] =
    ["pplx-decider", "jev-latest", "jev-preview", "autojev", "autojev-qwen3.8-27b", "pplx-decider-v1.2-27b", "pplx-decider-v1.2"];

pub struct Service {
    pub model: Option<SharedModel>,
    pub batcher: Option<Arc<Batcher>>,
    pub name: String,
    pub checkpoint: String,
    pub release_date: String,
    pub api_key: Option<String>,
    pub low_memory: bool,
}

pub type AppState = Arc<Service>;

// ---------------------------------------------------------------------------------------------- the request schema

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Body {
    #[serde(default)]
    model: Option<String>,
    state: Value,
    questions: serde_json::Map<String, Value>,
    #[serde(default)]
    images: Vec<String>,
}

pub struct ApiError {
    pub status: StatusCode,
    pub detail: Value,
    pub headers: Vec<(&'static str, String)>,
}

impl ApiError {
    pub fn new(status: u16, message: impl Into<String>) -> Self {
        Self { status: StatusCode::from_u16(status).unwrap_or(StatusCode::INTERNAL_SERVER_ERROR), detail: json!(message.into()), headers: vec![] }
    }
    /// pydantic's 422 body: a list of {loc, msg, type}.
    fn validation(loc: &[&str], message: impl Into<String>) -> Self {
        let loc: Vec<Value> = std::iter::once(json!("body")).chain(loc.iter().map(|l| json!(l))).collect();
        Self {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: json!([{"loc": loc, "msg": message.into(), "type": "value_error"}]),
            headers: vec![],
        }
    }
    fn from_failure(failure: Failure) -> Self {
        let mut error = Self::new(failure.status, failure.message);
        if let Some(seconds) = failure.retry_after {
            error.headers.push(("retry-after", seconds.to_string()));
        }
        error
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        if self.status.is_server_error() {
            tracing::error!("{}: {}", self.status, self.detail);
        } else {
            tracing::debug!("{}: {}", self.status, self.detail);
        }
        let mut response = (self.status, Json(json!({"detail": self.detail}))).into_response();
        for (name, value) in self.headers {
            if let Ok(value) = HeaderValue::from_str(&value) {
                response.headers_mut().insert(name, value);
            }
        }
        response
    }
}

fn content_ok(value: &Value) -> bool {
    matches!(value, Value::String(_) | Value::Object(_) | Value::Array(_))
}

/// Parse and validate a request body the way decider_api.py's pydantic models do.
pub fn parse_request(bytes: &[u8], service: &Service) -> Result<(Option<String>, Request), ApiError> {
    let body: Body = serde_json::from_slice(bytes).map_err(|error| ApiError::validation(&[], error.to_string()))?;
    if let Some(model) = &body.model
        && !ALIASES.contains(&model.as_str())
        && *model != service.name
    {
        return Err(ApiError::validation(&["model"], format!("Value error, Unknown model. Use {}, or leave model out.", service.name)));
    }
    if !content_ok(&body.state) {
        return Err(ApiError::validation(&["state"], "Input should be a valid string, object or array"));
    }
    if body.questions.is_empty() {
        return Err(ApiError::validation(&["questions"], "Dictionary should have at least 1 item after validation, not 0"));
    }
    let mut questions = Vec::with_capacity(body.questions.len());
    for (key, value) in body.questions {
        let question: Question =
            serde_json::from_value(value).map_err(|error| ApiError::validation(&["questions", &key], error.to_string()))?;
        match &question {
            Question::Choice { criteria, .. } if criteria.is_empty() || criteria.len() > MAX_OPTIONS => {
                return Err(ApiError::validation(&["questions", &key, "criteria"], format!("Dictionary should have 1 to {MAX_OPTIONS} items")));
            }
            Question::Score { criteria, .. } if criteria.len() < 2 || criteria.len() > MAX_OPTIONS => {
                return Err(ApiError::validation(&["questions", &key, "criteria"], format!("List should have 2 to {MAX_OPTIONS} items")));
            }
            Question::Noul { criteria: Some(criteria), .. } if criteria.keys().any(|k| k != "true" && k != "false") => {
                return Err(ApiError::validation(&["questions", &key, "criteria"], "Keys must be 'true' or 'false'"));
            }
            _ => {}
        }
        if let Some(instructions) = question.instructions()
            && !content_ok(instructions)
        {
            return Err(ApiError::validation(&["questions", &key, "instructions"], "Input should be a valid string, object or array"));
        }
        if let Question::Choice { criteria, .. } = &question
            && let Some((option, _)) = criteria.iter().find(|(_, v)| !v.is_null() && !content_ok(v))
        {
            return Err(ApiError::validation(&["questions", &key, "criteria", option], "Input should be a valid string, object or array"));
        }
        if let Question::Score { criteria, .. } = &question
            && criteria.iter().any(|v| !content_ok(v))
        {
            return Err(ApiError::validation(&["questions", &key, "criteria"], "Input should be a valid string, object or array"));
        }
        questions.push((key, question));
    }
    if body.images.len() > 4 {
        return Err(ApiError::validation(&["images"], "List should have at most 4 items after validation"));
    }
    for value in &body.images {
        validate_image(value).map_err(|message| ApiError::validation(&["images"], format!("Value error, {message}")))?;
    }
    Ok((body.model, Request { state: body.state, questions, images: body.images }))
}

const IMAGE_HEADERS: [&str; 3] = ["data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64"];

fn validate_image(value: &str) -> Result<(), String> {
    if value.len() > 12_000_000 {
        return Err("Each image must be at most 8 MB before base64 encoding.".into());
    }
    let Some((header, encoded)) = value.split_once(',') else {
        return Err("Images must be base64 PNG, JPEG, or WebP data URLs.".into());
    };
    if !IMAGE_HEADERS.contains(&header) {
        return Err("Images must be base64 PNG, JPEG, or WebP data URLs.".into());
    }
    let content = base64::engine::general_purpose::STANDARD.decode(encoded).map_err(|_| "Invalid image data.".to_string())?;
    if content.len() > 8_000_000 {
        return Err("Each image must be at most 8 MB.".into());
    }
    images::check(&content)
}

// ---------------------------------------------------------------------------------------------- routes

fn authenticate(service: &Service, headers: &HeaderMap) -> Result<(), ApiError> {
    let Some(key) = &service.api_key else { return Ok(()) };
    let expected = format!("Bearer {key}");
    let given = headers.get(header::AUTHORIZATION).and_then(|v| v.to_str().ok()).unwrap_or("");
    let same = given.len() == expected.len() && bool::from(subtle::ConstantTimeEq::ct_eq(given.as_bytes(), expected.as_bytes()));
    if same {
        Ok(())
    } else {
        let mut error = ApiError::new(401, "Missing or invalid API key.");
        error.headers.push(("www-authenticate", "Bearer".into()));
        Err(error)
    }
}

async fn index(State(service): State<AppState>) -> Json<Value> {
    Json(json!({"service": "decisions", "model": service.name,
                "routes": ["GET /health", "GET /v1/models", "POST /v1/systemone", "POST /v1/decisions"]}))
}

async fn health(State(service): State<AppState>) -> Json<Value> {
    Json(json!({"status": if service.model.is_some() { "ready" } else { "loading" }, "model": service.name,
                "checkpoint": service.checkpoint, "authentication": service.api_key.is_some(),
                "modalities": if service.low_memory { json!(["text"]) } else { json!(["text", "image"]) }}))
}

async fn models(State(service): State<AppState>, headers: HeaderMap) -> Result<Json<Value>, ApiError> {
    authenticate(&service, &headers)?;
    let mut names: Vec<&str> = ALIASES.to_vec();
    names.push(&service.name);
    names.sort_unstable();
    names.dedup();
    Ok(Json(json!({"models": names.iter().map(|name| json!({"name": name,
        "description": "Local pplx-decider text and image decisions.", "release_date": service.release_date})).collect::<Vec<_>>()})))
}

async fn decide(State(service): State<AppState>, headers: HeaderMap, body: Bytes) -> Result<Json<Value>, ApiError> {
    authenticate(&service, &headers)?;
    let (Some(model), Some(batcher)) = (&service.model, &service.batcher) else {
        return Err(ApiError::new(503, "The model is not ready."));
    };
    let (_, request) = parse_request(&body, &service)?;
    if !request.images.is_empty() && service.low_memory {
        return Err(ApiError::new(
            400,
            "This server runs text-only (low memory mode left out the vision encoder to fit this GPU), so it can't read \
             images. Send the request without images.",
        ));
    }
    let encoder = model.encoder();
    let prepared = images::prepare_all(&request.images, model.as_ref()).map_err(|m| ApiError::new(422, m))?;
    let grids: Vec<_> = prepared.iter().map(|image| image.grid).collect();
    let mut rows = encoder.plan(&request, &grids, Some(limit_of(model.as_ref()))).map_err(|error| ApiError::from_failure(Failure::from_protocol(error)))?;
    for row in &mut rows {
        row.pixels = prepared.clone();
    }
    // About how many tokens the request will take, for backpressure: its state once, plus each question.
    let state_text = crate::protocol::describe(&request.state);
    let estimate = encoder.count_tokens(&state_text).unwrap_or(0) + 80 * request.questions.len();
    let (questions, images, started) = (request.questions.len(), request.images.len(), Instant::now());
    match batcher.submit(request.questions, rows, estimate).await {
        Outcome::Answered(decision) => {
            tracing::info!(
                "Decision: {questions} question(s), {images} image(s), {} sequence(s), {} tokens in {:.0} ms",
                decision.sequences,
                decision.input_tokens,
                started.elapsed().as_secs_f64() * 1000.0
            );
            let answers: serde_json::Map<String, Value> =
                decision.answers.into_iter().map(|(key, answer)| (key, serde_json::to_value(answer).unwrap_or(Value::Null))).collect();
            Ok(Json(json!({"model": service.name, "answers": answers,
                "usage": {"input_tokens": decision.input_tokens, "output_tokens": 0, "sequences": decision.sequences}})))
        }
        Outcome::Failed(failure) => Err(ApiError::from_failure(failure)),
    }
}

fn limit_of(model: &dyn crate::model::Model) -> usize {
    model.max_length().min(model.budget().max_tokens)
}

/// `x-request-id` and `server-timing` on every response, as decider_api.py's middleware adds them.
async fn request_metadata(request: axum::extract::Request, next: middleware::Next) -> Response {
    let started = Instant::now();
    let id = uuid::Uuid::new_v4().simple().to_string();
    let (method, path) = (request.method().clone(), request.uri().path().to_string());
    let mut response = next.run(request).await;
    let status = response.status().as_u16();
    let ms = started.elapsed().as_secs_f64() * 1000.0;
    if status >= 400 {
        tracing::warn!("{method} {path} -> {status} in {ms:.0} ms [{id}]");
    } else if path.starts_with("/v1/") && method == axum::http::Method::POST {
        tracing::info!("{method} {path} -> {status} in {ms:.0} ms [{id}]");
    } else {
        tracing::debug!("{method} {path} -> {status} in {ms:.0} ms [{id}]");
    }
    let headers = response.headers_mut();
    if let Ok(value) = HeaderValue::from_str(&id) {
        headers.insert("x-typesafe-request-id", value.clone());
        headers.insert("x-request-id", value);
    }
    if let Ok(value) = HeaderValue::from_str(&format!("total;dur={:.1}", started.elapsed().as_secs_f64() * 1000.0)) {
        headers.insert("server-timing", value);
    }
    response
}

pub fn router(service: AppState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/v1/models", get(models))
        .route("/v1/systemone", post(decide))
        .route("/v1/decisions", post(decide))
        .layer(middleware::from_fn(request_metadata))
        .with_state(service)
}
