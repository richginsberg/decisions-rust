# The decision server for one Blackwell GPU: a Rust binary over cuBLASLt and the kernels in kernels/.
#
# Two stages: the build stage has the CUDA 13 toolkit (nvcc for the kernels, the headers), the runtime stage only
# the libraries the binary loads at startup (cuBLASLt and its dependencies; the driver comes from the host).
# The checkpoint is not in the image: start.sh downloads it to the Pod's volume on the first start.

ARG CUDA_VERSION=13.4.3
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS build

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates build-essential pkg-config && \
    rm -rf /var/lib/apt/lists/*
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
ENV PATH=/root/.cargo/bin:$PATH

WORKDIR /src
COPY Cargo.toml Cargo.lock build.rs ./
COPY src ./src
COPY kernels ./kernels
COPY tests ./tests
COPY fixtures ./fixtures
# sm_100 (B200/B300) and sm_120 (RTX 5090, RTX PRO 6000); set DECISION_CUDA_ARCHS=120 for a faster 5090-only build.
ARG DECISION_CUDA_ARCHS=100,120
ENV DECISION_CUDA_ARCHS=${DECISION_CUDA_ARCHS}
RUN cargo build --release --features cuda && cp target/release/decisions-rust-nvfp4 /decision-server

FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

# Only what start.sh needs to fetch the checkpoint: Python with huggingface_hub (no torch, no transformers).
RUN apt-get update && apt-get install -y --no-install-recommends python3 python3-pip ca-certificates && \
    pip3 install --no-cache-dir "huggingface_hub>=0.30" && rm -rf /var/lib/apt/lists/*

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HF_HOME=/workspace/huggingface \
    MODEL_HOME=/workspace/models \
    DECISION_MODEL_ID=machinez/pplx-decider-v1.2-NVFP4 \
    DECISION_MODEL_REVISION=fdeb7a9694eb633c29103f7a39181f0bda2fa0f9 \
    RUST_LOG=info \
    PORT=8000

COPY --from=build /decision-server /usr/local/bin/decision-server
COPY scripts/fetch_model.py scripts/start.sh /
RUN chmod +x /start.sh

EXPOSE 8000
CMD ["/start.sh"]
