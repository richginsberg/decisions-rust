//! The server: load the checkpoint (or the fake model with `--fake`), start the batcher, serve HTTP on `PORT`.
//!
//! Environment (decisions-endpoint-nvfp4's names, so a Pod's settings carry over):
//! - `AUTOJEV_API_KEY`: required (the Pod's port is public); `--fake` runs without it.
//! - `AUTOJEV_CHECKPOINT` or `DECISION_CHECKPOINT`: the checkpoint directory.
//! - `DECISION_MAX_LENGTH` (32768), `DECISION_MAX_BATCH_TOKENS` (16384), `DECISION_MAX_BATCH_ROWS` (64),
//!   `DECISION_MAX_QUEUE_SECONDS` (30), `DECISION_PACKED` (1), `PORT` (8000).

use std::sync::Arc;

use decisions_rust_nvfp4::api::{self, Service};
use decisions_rust_nvfp4::batcher::Batcher;
use decisions_rust_nvfp4::model::{Budget, SharedModel};

fn env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    std::env::var(name).ok().and_then(|v| v.parse().ok()).unwrap_or(default)
}

/// The loaded model and what the service reports about it.
pub struct Loaded {
    pub model: SharedModel,
    pub name: String,
    pub checkpoint: String,
    pub release_date: String,
    pub low_memory: bool,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()))
        .init();
    let fake = std::env::args().any(|arg| arg == "--fake");
    let api_key = std::env::var("AUTOJEV_API_KEY").ok().filter(|k| !k.is_empty());
    if api_key.is_none() && !fake {
        anyhow::bail!("AUTOJEV_API_KEY must be set: the server's port is reachable by anyone with its URL");
    }
    let budget = Budget {
        max_tokens: env_or("DECISION_MAX_BATCH_TOKENS", 16384),
        max_rows: env_or("DECISION_MAX_BATCH_ROWS", 64),
        packed: env_or::<u8>("DECISION_PACKED", 1) == 1,
    };
    let max_queue_seconds: f64 = env_or("DECISION_MAX_QUEUE_SECONDS", 30.0);

    let loaded = if fake {
        Loaded {
            model: decisions_rust_nvfp4::testing::fake_model(budget, 2e-5),
            name: api::DEFAULT_MODEL.into(),
            checkpoint: "fake".into(),
            release_date: "2026-01-01".into(),
            low_memory: false,
        }
    } else {
        let checkpoint = std::env::var("AUTOJEV_CHECKPOINT")
            .or_else(|_| std::env::var("DECISION_CHECKPOINT"))
            .unwrap_or_else(|_| "checkpoints/selected".into());
        load_gpu(&checkpoint, budget)?
    };

    let batcher = Batcher::new(Arc::clone(&loaded.model), max_queue_seconds);
    batcher.start();
    let service = Arc::new(Service {
        model: Some(loaded.model),
        batcher: Some(batcher),
        name: loaded.name,
        checkpoint: loaded.checkpoint,
        release_date: loaded.release_date,
        api_key,
        low_memory: loaded.low_memory,
    });
    let port: u16 = env_or("PORT", 8000);
    let listener = tokio::net::TcpListener::bind(("0.0.0.0", port)).await?;
    tracing::info!("Serving {} on port {port}", service.name);
    axum::serve(listener, api::router(service))
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

#[cfg(feature = "cuda")]
fn load_gpu(checkpoint: &str, budget: Budget) -> anyhow::Result<Loaded> {
    let gpu = decisions_rust_nvfp4::cuda::load(std::path::Path::new(checkpoint), budget)?;
    Ok(Loaded { model: gpu.model, name: gpu.name, checkpoint: checkpoint.into(), release_date: gpu.release_date, low_memory: gpu.low_memory })
}

#[cfg(not(feature = "cuda"))]
fn load_gpu(_checkpoint: &str, _budget: Budget) -> anyhow::Result<Loaded> {
    anyhow::bail!("built without the `cuda` feature: only --fake can run")
}
