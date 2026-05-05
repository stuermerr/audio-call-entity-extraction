# Phonebot — Call Recording Extraction Pipeline

A post-processing pipeline that transcribes German phone-call recordings and extracts structured caller data (`first_name`, `last_name`, `email`, `phone_number`) using a configurable transcription backend and a schema-constrained LLM extractor.

## Pipeline overview

```
config.yaml + .env
        │
        ▼
  AudioInput list  (data/splits.json or data/recordings/*.wav)
        │
        ▼
  Preprocessor     (no-op  │  FastEnhancer GPU denoising)
        │
        ▼
  Transcriber      (openai_llm  │  deepgram  │  whisperx  │  parakeet)
        │
        ▼
  LLM Extractor    (schema-constrained, prompt driven via YAML/Jinja2)
        │
        ▼
  outputs/{run_id}/results.json  +  eval.json  +  case_report.json
```


| Backend      | Requires           | Notes                                             |
| ------------ | ------------------ | ------------------------------------------------- |
| `openai_llm` | `OPENAI_API_KEY`   | Default CPU/API path; `gpt-4o-transcribe`         |
| `deepgram`   | `DEEPGRAM_API_KEY` | Fast, low-cost alternative                        |
| `whisperx`   | GPU image + CUDA   | Best local accuracy; large-v3                     |
| `parakeet`   | GPU image + CUDA   | NVIDIA NeMo; downloads ~1.2 GB model on first run |


The transcription backend, models, and all tuneable parameters are controlled by `config.yaml` — no code changes required.

---

## Prerequisites

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```


| Variable            | Required for                                       |
| ------------------- | -------------------------------------------------- |
| `OPENAI_API_KEY`    | `openai_llm` transcriber, `llm` extractor (always) |
| `DEEPGRAM_API_KEY`  | `deepgram` transcriber                             |
| `HF_TOKEN`          | WhisperX diarization (`diarization_enabled: true`) |
| `LANGSMITH_API_KEY` | LangSmith tracing (`langsmith_tracing: true`)      |


---

## Docker

### Build

```bash
# CPU/API image (default — uses openai_llm transcription)
docker build -t phonebot:cpu .

# GPU image (adds whisperx, onnxruntime-gpu, librosa, scipy + torch/CUDA)
docker build -t phonebot:gpu --build-arg INSTALL_GROUP=gpu .
```

### Run — CPU/API path

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/config_cpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:cpu run --eval true
```

### Run — GPU path (best accuracy)

Requires the NVIDIA Container Toolkit and a CUDA-capable GPU.

```bash
docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/config_gpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:gpu run --eval true
```

### Volume mounts

`data/` (recordings + ground truth) is committed to the repo and baked into the image — no data mount is needed.


| Mount                                                       | Purpose                            |
| ----------------------------------------------------------- | ---------------------------------- |
| `config_cpu.yaml` or `config_gpu.yaml` → `/app/config.yaml` | Pipeline configuration (read-only) |
| `outputs/` → `/app/outputs`                                 | Run artifacts written here         |


> **Note on `data/splits.json`:** The dev/test split index is generated locally via `uv run python scripts/split.py` and is not committed. When absent, the pipeline falls back to enumerating all `*.wav` files in order. With `sample: all` in the shipped configs this is transparent — all 30 recordings are processed either way.

---

## Configuration

`config_cpu.yaml` selects the `openai_llm` transcriber with `gpu_enabled: false` and `denoising_enabled: false`.

`config_gpu.yaml` selects `whisperx` with `gpu_enabled: true` and `denoising_enabled: true` (FastEnhancer ONNX preprocessing).

Key fields:


| Field                   | Effect                                                       |
| ----------------------- | ------------------------------------------------------------ |
| `transcriber`           | `openai_llm` / `deepgram` / `whisperx` / `parakeet`          |
| `gpu_enabled`           | Must be `true` for `whisperx` / `parakeet` / FastEnhancer    |
| `denoising_enabled`     | Enables FastEnhancer GPU denoising (GPU image only)          |
| `llm_extractor_model`   | OpenAI chat model used for extraction                        |
| `extractor_prompt_file` | Path to a YAML/Jinja2 prompt file; swap without code changes |
| `langsmith_tracing`     | Enable LangSmith tracing                                     |
| `sample`                | `dev` / `test` / `failed` / `all`                            |


---

## CLI reference

```
phonebot run [OPTIONS]

Options:
  -s, --samples TEXT           Split to run: dev|test|failed|all
  -t, --transcriber TEXT       Override transcriber (e.g. openai_llm)
  -e, --extractor TEXT         Override extractor (e.g. llm)
      --eval true|false        Run accuracy evaluation after extraction [default: true]
      --extraction-only        Skip transcription; read from --transcriptions-path
      --transcriptions-path    Path to a saved transcriptions.json artifact
      --output-dir PATH        Output root [default: outputs]
```

---

## Extraction-only mode

After a full run, `outputs/{run_id}/transcriptions.json` contains saved transcripts. You can iterate on the extraction prompt without paying transcription cost again:

```bash
docker run --rm \
  --env-file .env \
  -v "$PWD/config_cpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:cpu run \
    --extraction-only \
    --transcriptions-path outputs/<run_id>/transcriptions.json \
    --eval true
```

---

## Output artifacts

Each run writes to `outputs/{run_id}/`:


| File                  | Contents                                                      |
| --------------------- | ------------------------------------------------------------- |
| `results.json`        | Predicted caller fields for every recording                   |
| `transcriptions.json` | Raw transcripts (reusable for extraction-only mode)           |
| `eval.json`           | Per-field and overall accuracy against ground truth           |
| `case_report.json`    | Transcript + prediction + expected + per-field match per call |
| `config.yaml`         | Non-secret config snapshot for reproducibility                |
| `run.log`             | Structured run log                                            |


---

## Local development

```bash
# Install CPU dependencies
uv sync --frozen --no-dev

# Install GPU dependencies
uv sync --frozen --no-dev --group gpu

# Run the pipeline
uv run phonebot run --eval true

# Run tests
uv run pytest

# Lint / type-check
uv run ruff check src tests
uv run mypy src
```

