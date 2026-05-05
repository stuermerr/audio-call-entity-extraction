# syntax=docker/dockerfile:1
#
# Single Dockerfile for both CPU/API and GPU paths.
#
# CPU image (default):
#   docker build -t phonebot:cpu .
#
# GPU image (whisperx + onnxruntime-gpu):
#   docker build -t phonebot:gpu --build-arg INSTALL_GROUP=gpu .
#
# See README.md for runtime usage.

ARG INSTALL_GROUP=cpu

FROM python:3.11-slim

# ---------------------------------------------------------------------------
# uv — install from the official distroless image (no curl/pip required)
# ---------------------------------------------------------------------------
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

# Improve layer caching and container-fs compatibility
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# ---------------------------------------------------------------------------
# Dependencies — copy manifests first so this layer is cached independently
# from source changes.
# ---------------------------------------------------------------------------
COPY pyproject.toml uv.lock ./

# Install all dependencies but not the project package itself yet.
# GPU group adds: whisperx, onnxruntime-gpu, librosa, scipy (+ torch from
# the pytorch-cu128 index declared in pyproject.toml).
ARG INSTALL_GROUP
RUN if [ "$INSTALL_GROUP" = "gpu" ]; then \
        uv sync --frozen --no-dev --group gpu --no-install-project; \
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

# Install the project package itself (deps already present, so this is fast).
RUN uv sync --frozen --no-dev --no-deps

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
ENTRYPOINT ["uv", "run", "phonebot"]
CMD ["run", "--help"]
