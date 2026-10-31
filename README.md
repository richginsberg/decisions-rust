# decisions-rust-nvfp4

A Rust GPU container that serves **pplx-decider v1.2** (an NVFP4 quantization of Perplexity's Qwen3.5-27B decision model) 
on one Blackwell GPU, with the same HTTP contract, and the model's forward pass written as CUDA kernels called from
Rust instead of PyTorch, Triton and Python.

Why: the Python server spends its time launching ~1,500 small kernels per pass, padding mixed-length batches,
and re-tuning Triton kernels for minutes at every start. Here the launch overhead is a few milliseconds, passes
are packed without padding, the small kernels around each matrix multiply are fused (norm + residual + FP4
quantization, SiLU-gate + FP4 quantization, gated norm + FP8 quantization), and the image starts in seconds
with no JIT. The matrix multiplies themselves (62% of a long pass) run on the same cuBLASLt NVFP4 and FP8
tensor-core paths as before.

## Layout

| Path | What |
|---|---|
| `src/protocol.rs` | The joint prompt, tokenization and answers, token for token as pplx_decider's `protocol.py` |
| `src/api.rs` | autojev's HTTP contract: routes, Bearer auth, request and response shape, error codes |
| `src/batcher.rs` | Continuous batching in front of the GPU thread |
| `src/model.rs` | What the server needs from a model; a CPU fake for the tests |
| `src/images.rs` | Image checks and the Qwen2-VL preprocessing (`smart_resize`, patches) |
| `src/weights.rs` | The checkpoint's safetensors: NVFP4 layouts, memory-mapped shards |
| `src/config.rs` | `config.json`, `decision_config.json`, `joint_config.json` |
| `src/cuda/` | (feature `cuda`) the GPU model: weights on the device, kernels, the forward pass |
| `kernels/*.cu` | CUDA kernels, compiled by `build.rs` into a fatbin for sm_100 and sm_120 |
| `fixtures/encoder.json` | What the Python encoder produced for 156 requests (`scripts/make_encoder_fixture.py`) |

## Building and testing without a GPU

```bash
cargo test
```

Runs the encoder (prompt text), batcher, schema and route tests with the fake model. With a checkpoint directory
(at least `tokenizer.json` and `decision_config.json`) the token ids are checked against the Python encoder too:

```bash
DECISION_CHECKPOINT=/path/to/pplx-decider-v1.2-NVFP4 cargo test --test encoder
```

The server also runs end to end with the fake model: `cargo run -- --fake`.

## Building with the GPU

Needs a CUDA 13 toolkit (`nvcc`) and a Blackwell GPU at runtime:

```bash
cargo build --release --features cuda
```

## Environment

The Pod's settings carry over: `AUTOJEV_API_KEY` (required),
`AUTOJEV_CHECKPOINT`, `DECISION_MAX_LENGTH` (32768), `DECISION_MAX_BATCH_TOKENS` (16384),
`DECISION_MAX_BATCH_ROWS` (64), `DECISION_MAX_QUEUE_SECONDS` (30), `DECISION_MODEL_NAME`, `PORT` (8000).
`DECISION_PACKED` (1) packs a pass's sequences end to end instead of padding them.

## Status

Phase 1 (no GPU): protocol, API, batcher, weights loader, image preprocessing. Phase 2: the kernels and the
forward pass, checked against logits recorded from the Python server. Phase 3: packing, CUDA graphs, tuning,
the Docker image and deployment.
