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
| `src/protocol.rs` | The joint prompt, tokenization and answers, token for token as pplx_decider's `protocol.py`, in both protocol versions (3: v1.2, 4: v1.3) |
| `src/api.rs` | autojev's HTTP contract: routes, Bearer auth, request and response shape, error codes |
| `src/batcher.rs` | Continuous batching in front of the GPU thread |
| `src/model.rs` | What the server needs from a model; a CPU fake for the tests |
| `src/images.rs` | Image checks and the Qwen2-VL preprocessing (`smart_resize`, patches) |
| `src/weights.rs` | The checkpoint's safetensors: NVFP4 layouts, memory-mapped shards |
| `src/config.rs` | `config.json`, `decision_config.json`, `joint_config.json` |
| `src/cuda/blas.rs` | (feature `cuda`) cuBLASLt matmuls: NVFP4 block-scaled, FP8 row-scaled, bf16 |
| `src/cuda/device.rs` | Weights onto the device (FP4 scales re-tiled, gate/up merged, linear-attention projections to FP8), the workspace, the `Model` |
| `src/cuda/forward.rs` | The text backbone over packed sequences and the readout |
| `src/cuda/vision.rs` | The vision tower: patches to text-space embeddings |
| `src/cuda/kernels.rs` | The fatbin and typed launchers for every kernel |
| `kernels/quant.cu` | NVFP4 / FP8 activation quantization fused with the RMSNorm, residual add and SiLU-gate that feed it |
| `kernels/attention.cu` | Non-causal attention on the tensor cores (bf16 `mma.sync`, online softmax), text and vision |
| `kernels/deltanet.cu` | The gated delta rule (chunked, fp32) and its gated RMSNorm |
| `kernels/layers.cu` | Head norms + multimodal rotary, the causal conv, gathers, LayerNorm, GELU, vision rotary and position embedding |
| `Dockerfile`, `scripts/start.sh`, `scripts/fetch_model.py` | The image: build with the CUDA toolkit, run on the CUDA runtime, fetch the checkpoint to the volume |
| `scripts/compare_servers.py` | Send the same decisions to the Python and Rust servers and compare their probabilities |
| `fixtures/encoder.json` | What the Python encoder produced for 156 requests (`scripts/make_encoder_fixture.py`) |
| `fixtures/encoder_v4.json` | What v1.3's own `protocol.py` produced for 162 requests, 353 sequences, with real images (`scripts/make_encoder_fixture_v4.py`) |

## Building and testing without a GPU

```bash
cargo test
```

Runs the encoder (prompt text), batcher, schema and route tests with the fake model. With a checkpoint directory
(at least `tokenizer.json` and `decision_config.json`) the token ids are checked against the Python encoder too:

```bash
DECISION_CHECKPOINT=/path/to/pplx-decider-v1.2-NVFP4 cargo test --test encoder
```

`DECISION_CHECKPOINT_V4` does the same for a version 4 checkpoint (pplx-decider v1.3).

The server also runs end to end with the fake model: `cargo run -- --fake`.

## Checkpoints and protocol versions

The server reads the protocol a checkpoint declares in `joint_config.json` and encodes requests accordingly; the
model, kernels and HTTP contract are the same for both. Anything else is refused at startup.

| Checkpoint | Protocol | A request becomes |
|---|---|---|
| pplx-decider v1.2 | 3, `joint-question-inline-readout-v1` | one sequence: the state and every numbered question |
| pplx-decider v1.3 | 4, `joint-chunked-readout-v1` (`separate` layout) | one sequence per question: the context, `<|fim_middle|>`, that question |

Under version 4 every answer is independent of the request's other questions, and a request of Q questions
runs its state Q times, so many-question requests over long states cost more than under version 3. Not
implemented from v1.3: the `isolated` multi-question layout and the `locate-span` adapter. Switch checkpoints
with `DECISION_MODEL_ID` and `DECISION_MODEL_REVISION`; the tuned-algorithm cache is per checkpoint directory.

## Building with the GPU

Needs a CUDA 13.1+ toolkit (`nvcc`, for the kernels; the `cudarc` bindings are pinned to 13.4) and a Blackwell GPU
with a CUDA 13 driver at runtime:

```bash
NVCC=/usr/local/cuda/bin/nvcc cargo build --release --features cuda
```

`DECISION_CUDA_ARCHS` picks the cubins (default `100,120`; `120` alone halves the kernel build for a 5090).
The toolkit is only needed at build time: the binary loads the driver, cuBLASLt and the embedded fatbin at startup.

## Deploying on RunPod

```bash
docker build -t decision-rust .            # DECISION_CUDA_ARCHS=120 as --build-arg for a 5090-only image
docker push <registry>/decision-rust
```

Start a Pod on an RTX 5090 (or any Blackwell GPU) with a network volume at `/workspace`, HTTP port 8000 exposed,
and the same environment as the Python server: `AUTOJEV_API_KEY` (required), optionally `DECISION_MODEL_ID` /
`DECISION_MODEL_REVISION`, `DECISION_MAX_BATCH_TOKENS` and the other limits below. The first start downloads the
checkpoint (~25 GB) to the volume; later starts load it in about a minute. Clients keep their URL scheme: the
routes, auth and response shape are the Python server's.

To check the port against the Python server before switching clients:

```bash
python3 scripts/compare_servers.py https://<python-pod>:8000 https://<rust-pod>:8000 --key $AUTOJEV_API_KEY
```

## Environment

The Pod's settings carry over: `AUTOJEV_API_KEY` (required),
`AUTOJEV_CHECKPOINT`, `DECISION_MAX_LENGTH` (32768), `DECISION_MAX_BATCH_TOKENS` (16384),
`DECISION_MAX_BATCH_ROWS` (64), `DECISION_MAX_QUEUE_SECONDS` (30), `DECISION_MODEL_NAME`, `PORT` (8000).
`DECISION_PACKED` (1) packs a pass's sequences end to end instead of padding them.
`DECISION_AUTOTUNE` (1): during the warm-up passes at load (every row bucket to 4096 tokens), time cuBLASLt's
candidate algorithms for each matmul shape of 512 rows or more and keep the fastest; shapes first met on a live
request take the heuristic's first choice (0 does so always).
`DECISION_GRAPHS` (1): replay a captured CUDA graph of the text layers for requests without images in the
buckets warmed at load (to 4096 tokens), instead of some 700 kernel launches; 0 launches eagerly.
`DECISION_PREFETCH_THREADS` (8): readers that pull each weight shard into the page cache in parallel while it
loads, for cold volumes; 0 disables.
`DECISION_TUNING_CACHE` (`<checkpoint>/.decision-tuning`): where the tuned cuBLASLt algorithms are kept between
runs, per GPU and cuBLASLt version, so only the first start on a volume pays the tuning; empty disables.
`DECISION_PROFILE` (0): with 1, every pass stage is timed (a stream sync after each) and reported in the
`server-timing` response header and the log; for profiling only, since the syncs slow passes down.

## Status

Phase 1 (no GPU): protocol, API, batcher, weights loader, image preprocessing: done and tested.

Phase 2: the kernels, the forward pass (text and vision) and the Docker image: done and verified on an RTX 5090
(2026-10-31) against the Python server on the same checkpoint: 316 labeled decisions all agree, accuracy 96.5% on
both; `scripts/compare_servers.py` sees probabilities within 0.07 (0.02 on average), where app.py accepts 0.2.
On consumer Blackwell cuBLASLt has no rowwise FP8 scaling, so the FP8 projections run with scalar scales and a
rescale kernel (`scale_rows_cols`).

Phase 3 (2026-11-01): server-side medians on the 5090, proxy excluded, against the Python server on the same
checkpoint: a 3,400-token pass 266 ms vs about 540; 1,000 tokens 79 vs 143; 150 tokens 30 vs 40; many-question
medium requests about 2x. 315 of 316 labeled decisions agree (the one is a 0.50 vs 0.52 coin flip), accuracy
96.2% vs 96.5%. Done: cuBLASLt plans cached per row bucket and their algorithms tuned at load; the delta rule on
the tensor cores in fla's three-kernel structure (it was 73% of a long pass on CUDA cores); attention with
cp.async double buffering and ldmatrix; the FP8 rescale fused into its consumers; warm-up passes at load;
CUDA graphs of the text layers per bucket.
Short passes 22 ms at 135 tokens with the layers replayed as a CUDA graph. Startup is about 80 s from a warm
volume (weights 43 s, warm-up, tuning and graph capture 32 s); a cold volume adds about a minute. Where a long
pass goes now:
NVFP4 matmuls 37%, FP8 matmuls 22%, both near the GPU's peak; the delta rule 15%; attention 10%; elementwise
kernels 16%. Not done: the last layer computing every token rather than only the marker rows (about 1%).

## Environment

The Pod's settings carry over: `AUTOJEV_API_KEY` (required),
`AUTOJEV_CHECKPOINT`, `DECISION_MAX_LENGTH` (32768), `DECISION_MAX_BATCH_TOKENS` (16384),
`DECISION_MAX_BATCH_ROWS` (64), `DECISION_MAX_QUEUE_SECONDS` (30), `DECISION_MODEL_NAME`, `PORT` (8000).
`DECISION_PACKED` (1) packs a pass's sequences end to end instead of padding them.
`DECISION_AUTOTUNE` (1): during the warm-up passes at load (every row bucket to 4096 tokens), time cuBLASLt's
candidate algorithms for each matmul shape of 512 rows or more and keep the fastest; shapes first met on a live
request take the heuristic's first choice (0 does so always).
`DECISION_PROFILE` (0): with 1, every pass stage is timed (a stream sync after each) and reported in the
`server-timing` response header and the log; for profiling only, since the syncs slow passes down.

## Status

Phase 1 (no GPU): protocol, API, batcher, weights loader, image preprocessing: done and tested.

Phase 2: the kernels, the forward pass (text and vision) and the Docker image: done and verified on an RTX 5090
(2026-10-31) against the Python server on the same checkpoint: 316 labeled decisions all agree, accuracy 96.5% on
both; `scripts/compare_servers.py` sees probabilities within 0.07 (0.02 on average), where app.py accepts 0.2.
On consumer Blackwell cuBLASLt has no rowwise FP8 scaling, so the FP8 projections run with scalar scales and a
rescale kernel (`scale_rows_cols`).

Phase 3 (in progress): startup 43 s from a warm volume. Server-side medians on the 5090, proxy excluded, against
the Python server: 1 question x long input 415 vs 542 ms, 16 x medium 166 vs 317 ms, 16 x short 80 vs 129 ms;
long inputs 20-25% faster, many-question requests up to 2x. Done: cuBLASLt plans cached per shape; the delta
rule on the tensor cores in fla's three-kernel structure (it was 73% of a long pass on CUDA cores: 704 of 963
ms at 3,400 tokens, now 78 of 337); warm-up passes at startup; `DECISION_PROFILE=1` stage timing. Still to do:
the last layer computing every token rather than only the marker rows; no CUDA graphs.

