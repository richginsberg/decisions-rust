//! The HTTP contract, through the router with the fake model: routes, auth, the request and response shape,
//! and the error codes decider_api.py returns.

use axum::body::Body;
use axum::http::{Request, StatusCode, header};
use decisions_rust_nvfp4::api::router;
use decisions_rust_nvfp4::model::Budget;
use decisions_rust_nvfp4::testing::{fake_model, fake_service};
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tower::ServiceExt;

const KEY: &str = "test-key";

fn app() -> axum::Router {
    let model = fake_model(Budget { max_tokens: 4000, max_rows: 8, packed: true }, 0.0);
    router(fake_service(model, Some(KEY)))
}

async fn call(app: &axum::Router, method: &str, path: &str, key: Option<&str>, body: Option<Value>) -> (StatusCode, axum::http::HeaderMap, Value) {
    let mut request = Request::builder().method(method).uri(path);
    if let Some(key) = key {
        request = request.header(header::AUTHORIZATION, format!("Bearer {key}"));
    }
    let request = match body {
        Some(body) => request.header(header::CONTENT_TYPE, "application/json").body(Body::from(body.to_string())).unwrap(),
        None => request.body(Body::empty()).unwrap(),
    };
    let response = app.clone().oneshot(request).await.unwrap();
    let status = response.status();
    let headers = response.headers().clone();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let value = serde_json::from_slice(&bytes).unwrap_or(Value::Null);
    (status, headers, value)
}

fn decision() -> Value {
    json!({"state": "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan.",
           "questions": {
             "department": {"type": "choice", "instructions": "Which department should handle this?",
                            "criteria": {"billing": "invoices, payments, refunds", "technical": "bugs, outages", "other": "everything else"}},
             "urgency": {"type": "score", "instructions": "How urgent is this?", "criteria": ["not urgent", "soon", "blocking"]},
             "churn_risk": {"type": "noul", "instructions": "Does the customer threaten to cancel or leave?"}}})
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn routes_and_auth() {
    let app = app();
    let (status, headers, body) = call(&app, "GET", "/health", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["status"], "ready");
    assert_eq!(body["authentication"], true);
    assert!(headers.contains_key("x-request-id") && headers.contains_key("server-timing"));

    let (status, headers, _) = call(&app, "GET", "/v1/models", None, None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    assert_eq!(headers["www-authenticate"], "Bearer");
    let (status, _, _) = call(&app, "GET", "/v1/models", Some("wrong"), None).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);
    let (status, _, body) = call(&app, "GET", "/v1/models", Some(KEY), None).await;
    assert_eq!(status, StatusCode::OK);
    let names: Vec<&str> = body["models"].as_array().unwrap().iter().map(|m| m["name"].as_str().unwrap()).collect();
    assert!(names.contains(&"pplx-decider-v1.2") && names.contains(&"jev-latest"));

    let (status, _, body) = call(&app, "GET", "/", None, None).await;
    assert_eq!(status, StatusCode::OK);
    assert_eq!(body["service"], "decisions");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn decisions() {
    let app = app();
    for path in ["/v1/decisions", "/v1/systemone"] {
        let (status, _, body) = call(&app, "POST", path, Some(KEY), Some(decision())).await;
        assert_eq!(status, StatusCode::OK, "{body}");
        assert_eq!(body["model"], "pplx-decider-v1.2");
        assert_eq!(body["usage"]["output_tokens"], 0);
        assert_eq!(body["usage"]["sequences"], 1);
        assert!(body["usage"]["input_tokens"].as_u64().unwrap() > 20);
        let department = &body["answers"]["department"];
        assert_eq!(department["type"], "choice");
        assert!(["billing", "technical", "other"].contains(&department["choice"].as_str().unwrap()));
        let total: f64 = department["probabilities"].as_object().unwrap().values().map(|v| v.as_f64().unwrap()).sum();
        assert!((total - 1.0).abs() < 1e-6);
        assert_eq!(body["answers"]["urgency"]["type"], "score");
        assert_eq!(body["answers"]["urgency"]["legend"]["0"], "not urgent");
        assert!(body["answers"]["churn_risk"]["noul"].as_f64().unwrap() <= 1.0);
    }
    // The served name and v1.1's aliases are accepted in "model"; anything else is 422.
    let mut with_model = decision();
    with_model["model"] = json!("jev-latest");
    assert_eq!(call(&app, "POST", "/v1/decisions", Some(KEY), Some(with_model.clone())).await.0, StatusCode::OK);
    with_model["model"] = json!("gpt-4");
    let (status, _, body) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(with_model)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"][0]["msg"].as_str().unwrap().contains("Unknown model"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn invalid_requests() {
    let app = app();
    let (status, _, _) = call(&app, "POST", "/v1/decisions", None, Some(decision())).await;
    assert_eq!(status, StatusCode::UNAUTHORIZED);

    let mut extra = decision();
    extra["min_confidence"] = json!(0.5);
    let (status, _, body) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(extra)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"].is_array(), "{body}");

    let (status, _, _) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(json!({"state": "x", "questions": {}}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _, _) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(json!({"state": 5, "questions": {"q": {"type": "noul"}}}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _, _) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(json!({"state": "x", "questions": {"q": {"type": "essay"}}}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    let (status, _, _) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(json!({"state": "x", "questions": {"q": {"type": "score", "criteria": ["one"]}}}))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);

    let mut images = decision();
    images["images"] = json!(["data:image/png;base64,AAAA"]);
    let (status, _, body) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(images)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"][0]["msg"].as_str().unwrap().contains("Invalid image data"));
    let mut images = decision();
    images["images"] = json!(["data:image/gif;base64,AAAA"]);
    let (status, _, body) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(images)).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
    assert!(body["detail"][0]["msg"].as_str().unwrap().contains("data URLs"));
    let mut images = decision();
    images["images"] = json!(vec!["data:image/png;base64,AAAA"; 5]);
    assert_eq!(call(&app, "POST", "/v1/decisions", Some(KEY), Some(images)).await.0, StatusCode::UNPROCESSABLE_ENTITY);

    let (status, _, _) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(json!("not an object"))).await;
    assert_eq!(status, StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn images_are_accepted_when_valid() {
    let app = app();
    // A 64x64 red PNG.
    let mut png = Vec::new();
    let image = image::RgbImage::from_pixel(64, 64, image::Rgb([200, 30, 30]));
    image::DynamicImage::ImageRgb8(image).write_to(&mut std::io::Cursor::new(&mut png), image::ImageFormat::Png).unwrap();
    let url = format!("data:image/png;base64,{}", base64::Engine::encode(&base64::engine::general_purpose::STANDARD, &png));
    let mut body = decision();
    body["images"] = json!([url]);
    let (status, _, response) = call(&app, "POST", "/v1/decisions", Some(KEY), Some(body)).await;
    assert_eq!(status, StatusCode::OK, "{response}");
    // 64x64 is scaled up to the minimum 65,536 pixels: 256x256 = 16x16 patches, merged 2x2 = 64 image tokens.
    assert!(response["usage"]["input_tokens"].as_u64().unwrap() > 64);
}
