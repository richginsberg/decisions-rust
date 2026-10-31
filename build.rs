//! Compiles the CUDA kernels in kernels/ to a fatbin for the `cuda` feature.
//!
//! Every kernels/*.cu becomes one cubin per architecture in `DECISION_CUDA_ARCHS` (default: sm_100 for
//! B200-class and sm_120 for RTX 50 / RTX PRO Blackwell), bundled into `$OUT_DIR/kernels.fatbin`, which
//! src/cuda/kernels.rs embeds with `include_bytes!` and loads with the driver API at startup. Without
//! the feature nothing is compiled, so the crate builds on machines without a CUDA toolkit.

use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rerun-if-changed=kernels");
    println!("cargo:rerun-if-env-changed=DECISION_CUDA_ARCHS");
    println!("cargo:rerun-if-env-changed=NVCC");
    if env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    let archs = env::var("DECISION_CUDA_ARCHS").unwrap_or_else(|_| "100,120".to_string());
    let nvcc = env::var("NVCC").unwrap_or_else(|_| "nvcc".to_string());
    let mut sources: Vec<PathBuf> = std::fs::read_dir("kernels")
        .expect("kernels/ directory")
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "cu"))
        .collect();
    sources.sort();
    let mut command = Command::new(&nvcc);
    command
        .arg("--fatbin")
        .arg("-std=c++17")
        .arg("-O3")
        .arg("--use_fast_math")
        .arg("--expt-relaxed-constexpr")
        .arg("-Ikernels")
        .arg("-o")
        .arg(out.join("kernels.fatbin"));
    for arch in archs.split(',') {
        let arch = arch.trim();
        command.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
    command.args(&sources);
    let status = command.status().unwrap_or_else(|error| panic!("running {nvcc}: {error}"));
    assert!(status.success(), "nvcc failed on {sources:?}");
}
