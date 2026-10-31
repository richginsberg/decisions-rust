//! The GPU model: the checkpoint's weights on one Blackwell device, the CUDA kernels, and the forward pass.
//!
//! - [`blas`]: cuBLASLt matmuls (NVFP4 block-scaled, FP8 row-scaled, bf16).
//! - [`kernels`]: the fatbin built from `kernels/*.cu` and typed launchers for each kernel.
//! - [`device`]: the weights on the device and the workspace a pass runs in.
//! - [`forward`]: the text backbone's forward pass over packed sequences and the readout.
//! - [`vision`]: the vision tower: image patches to text-space embeddings.

pub mod blas;
pub mod device;
pub mod forward;
pub mod kernels;
pub mod vision;

use std::path::Path;
use std::sync::Arc;

use crate::model::{Budget, SharedModel};

#[derive(Debug, thiserror::Error)]
pub enum CudaError {
    #[error("CUDA driver: {0}")]
    Driver(#[from] cudarc::driver::DriverError),
    #[error("cuBLASLt: {0}")]
    Blas(String),
    #[error("{0}")]
    Config(#[from] crate::config::ConfigError),
    #[error("{0}")]
    Weights(#[from] crate::weights::WeightsError),
    #[error("{0}")]
    Invalid(String),
    #[error("out of GPU memory")]
    OutOfMemory,
}

/// A loaded model and what the server reports about it.
pub struct Loaded {
    pub model: SharedModel,
    pub name: String,
    pub release_date: String,
    pub low_memory: bool,
}

/// Load the checkpoint onto GPU 0 and wrap it as the server's model.
pub fn load(checkpoint: &Path, budget: Budget) -> anyhow::Result<Loaded> {
    let model = device::GpuModel::load(checkpoint, budget)?;
    let name = model.checkpoint.served_name();
    let low_memory = model.low_memory;
    Ok(Loaded { model: Arc::new(model), name, release_date: "2026-01-01".into(), low_memory })
}
