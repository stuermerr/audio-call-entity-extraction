# syntax=docker/dockerfile:1
#
# Single Dockerfile for CPU/API, GPU, and GPU-benchmark paths.
#
# CPU image (default):
#   docker build -t phonebot:cpu .
#
# GPU image (whisperx + onnxruntime-gpu + nemo):
#   docker build -t phonebot:gpu --build-arg INSTALL_GROUP=gpu .
#
# GPU-benchmark image (onnxruntime-gpu + FastEnhancer only — no whisperx/nemo):
#   docker build -t phonebot:gpu-benchmark --build-arg INSTALL_GROUP=gpu-benchmark .
#
# See README.md for runtime usage.

ARG INSTALL_GROUP=cpu

FROM python:3.13-slim

# ---------------------------------------------------------------------------
# uv — install from the official distroless image (no curl/pip required)
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Improve layer caching and container-fs compatibility
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# GPU dependencies can pull source distributions that need a compiler when
# Python-version-specific wheels are unavailable.
ARG INSTALL_GROUP
RUN if [ "$INSTALL_GROUP" = "gpu" ] || [ "$INSTALL_GROUP" = "gpu-benchmark" ]; then \
        apt-get update && \
        apt-get install -y --no-install-recommends g++ && \
        rm -rf /var/lib/apt/lists/*; \
    fi

# ---------------------------------------------------------------------------
# Dependencies — copy manifests first so this layer is cached independently
# from source changes.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# Install all dependencies but not the project package itself yet.
# gpu:           whisperx, onnxruntime-gpu, librosa, scipy, nemo_toolkit[asr]
#                (+ torch from pytorch-cu128 index declared in pyproject.toml)
# gpu-benchmark: onnxruntime-gpu, librosa, scipy only — no whisperx/nemo.
#                Use with transcriber: openai_llm or deepgram + denoising_enabled: true.
RUN if [ "$INSTALL_GROUP" = "gpu" ]; then \
        uv sync --frozen --no-dev --group gpu --no-install-project; \
    elif [ "$INSTALL_GROUP" = "gpu-benchmark" ]; then \
        uv sync --frozen --no-dev --group gpu-benchmark --no-install-project; \
    else \
        uv sync --frozen --no-dev --no-install-project; \
    fi

# ---------------------------------------------------------------------------
# Application source — copied after deps to avoid invalidating the dep layer
# on every code change.
# ---------------------------------------------------------------------------
COPY src/ ./src/
COPY prompts/ ./prompts/

# data/ is committed to the repo and baked into the image so the pipeline
# runs without any host volume for recordings or ground truth.
# Mount a host data/ only to override with different recordings:
#   -v "$PWD/data:/app/data:ro"
COPY data/ ./data/

# Install the project package itself. Dependencies are already cached from the
# earlier sync layer, but keep the selected dependency group consistent.
RUN if [ "$INSTALL_GROUP" = "gpu" ]; then \
        uv sync --frozen --no-dev --group gpu; \
    elif [ "$INSTALL_GROUP" = "gpu-benchmark" ]; then \
        uv sync --frozen --no-dev --group gpu-benchmark; \
    else \
        uv sync --frozen --no-dev; \
    fi

# Outputs directory must exist so the pipeline can write even if the volume
# is not mounted (useful for quick one-off runs).
RUN mkdir -p /app/outputs

# ---------------------------------------------------------------------------
# Runtime
#
# The following are NOT baked into the image — mount them at runtime:
#   config.yaml  →  -v "$PWD/config_cpu.yaml:/app/config.yaml:ro"
#   outputs/     →  -v "$PWD/outputs:/app/outputs"   (to retrieve results)
#   secrets      →  --env-file .env
# ---------------------------------------------------------------------------
ENTRYPOINT ["uv", "run", "--no-sync", "phonebot"]
