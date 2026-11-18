//! Serves pplx-decider v1.2 (the joint inline-readout protocol) from an NVFP4 checkpoint on one Blackwell GPU.
//!
//! - [`protocol`]: the prompt, tokenization and answers, token for token as pplx_decider's protocol.py.
//! - [`api`]: autojev's HTTP contract (routes, Bearer auth, request and response shape).
//! - [`batcher`]: continuous batching in front of the GPU thread.
//! - [`model`]: what the server needs from a model; a CPU fake for tests.
//! - [`cuda`] (feature `cuda`): the real model: weights, kernels, the forward pass.

pub mod api;
pub mod batcher;
pub mod config;
pub mod images;
pub mod locate;
pub mod model;
pub mod protocol;
pub mod testing;
pub mod weights;

#[cfg(feature = "cuda")]
pub mod cuda;
