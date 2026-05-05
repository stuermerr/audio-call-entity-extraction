# 📞 Phonebot — Call Recording Extraction Pipeline

A post-processing pipeline that transcribes German phone-call recordings and extracts structured caller data (`first_name`, `last_name`, `email`, `phone_number`) using a configurable transcription backend and a schema-constrained LLM extractor.

---

## 🔍 What it does

Phonebot takes raw `.wav` recordings from a phonebot IVR system, transcribes them (cloud or local), and extracts structured caller fields:


| Field          | Description               |
| -------------- | ------------------------- |
| `first_name`   | Caller's first name       |
| `last_name`    | Caller's last name        |
| `email`        | Normalized e-mail address |
| `phone_number` | Normalized phone number   |


Results are written to a timestamped output directory alongside accuracy metrics (when ground truth is available) and a per-call case report for debugging.

### Pipeline

```
config.yaml + .env
        │
        ▼
  AudioInput list  (all *.wav recordings in data/recordings/)
        │
        ▼
  Preprocessor     (no-op  │  FastEnhancer GPU denoising)
        │
        ▼
  Transcriber      (openai_llm  │  deepgram  │  whisperx  │  parakeet)
        │
        ▼
  LLM Extractor    (schema-constrained, prompt-driven via YAML/Jinja2)
        │
        ▼
  outputs/{run_id}/results.json + results.md  +  eval.json  +  case_report.json + config.yaml (snapshot) [+ transcriptions.json]
```

### Transcription backends


| Backend      | Requires           | Notes                                                                        |
| ------------ | ------------------ | ---------------------------------------------------------------------------- |
| `whisperx`   | GPU image + CUDA   | `large-v3` (default); local; best accuracy, comparable with deepgram backend |
| `parakeet`   | GPU image + CUDA   | parakeet-tdt-0.6b-v3; local                                                  |
| `openai_llm` | `OPENAI_API_KEY`   | `gpt-4o-transcribe` (default); cloud (api)                                   |
| `deepgram`   | `DEEPGRAM_API_KEY` | `nova-2` (default); cloud (api); fast, high-quality alternative              |


The transcription backend, models, and all tuneable parameters are controlled by `config.yaml` — no code changes required.

---

## ✨ Key features

- **Multiple transcription backends** — swap between cloud APIs (`openai_llm`, `deepgram`) and local GPU models (`whisperx`, `parakeet`) via a single config field
- **Extraction-only mode** — reuse saved `transcriptions.json` from a previous run to iterate on prompts without incurring transcription costs
- **Versioned prompts** — extraction prompts are external YAML/Jinja2 files under `prompts/`, swappable at runtime
- **Built-in evaluation** — per-field and overall accuracy scored against `data/ground_truth.json`
- **GPU denoising** — optional FastEnhancer ONNX preprocessing step for noisy recordings (GPU image only)
- **LangSmith tracing** — optional observability via LangSmith for extraction LLM calls
- **Reproducible runs** — each run's output directory includes a non-secret config snapshot and structured log

---

## Benchmark

Best run on the provided 30-call dataset:


| Field          | Accuracy  |
| -------------- | --------- |
| `first_name`   | 100.0%    |
| `last_name`    | 86.7%     |
| `email`        | 90.0%     |
| `phone_number` | 100.0%    |
| **Overall**    | **94.2%** |


Default GPU-settings are the same as for the Benchmark run.

---

## 🚀 Usage

### Prerequisites

Copy `.env.example` to `.env` and fill in the keys you need:

```bash
cp .env.example .env
```


| Variable                                                      | Required for                                                           |
| ------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `OPENAI_API_KEY`                                              | `openai_llm` transcriber and `llm` extractor (ALWAYS REQUIRED)         |
| `DEEPGRAM_API_KEY`                                            | `deepgram` transcriber (OPTIONAL)                                      |
| `HF_TOKEN`                                                    | WhisperX diarization (OPTIONAL, default: `diarization_enabled: false`) |
| `LANGSMITH_API_KEY`,`LANGSMITH_PROJECT`, `LANGSMITH_ENDPOINT` | LangSmith tracing (OPTIONAL, default: `langsmith_tracing: false`)      |


---

### Local

Requires Python `>=3.11, <3.14` and [uv](https://github.com/astral-sh/uv).

**GPU** (adds whisperx, onnxruntime-gpu, librosa, scipy, torch/CUDA)

```bash
cp config_gpu.yaml config.yaml
uv sync --frozen --no-dev --group gpu
uv run phonebot
```

**CPU/API**

```bash
cp config_cpu.yaml config.yaml
uv sync --frozen --no-dev
uv run phonebot
```

---

### Docker

`data/` (recordings + ground truth) is baked into the image — no data mount needed.

**GPU** — requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and a CUDA-capable GPU

```bash
docker build -t phonebot:gpu --build-arg INSTALL_GROUP=gpu .

docker run --rm --gpus all \
  --env-file .env \
  -v "$PWD/config_gpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:gpu
```

**CPU/API**

```bash
docker build -t phonebot:cpu .

docker run --rm \
  --env-file .env \
  -v "$PWD/config_cpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:cpu
```

---

### Extraction-only mode

After a full run, `outputs/{run_id}/transcriptions.json` contains saved transcripts. Iterate on the extraction prompt without re-transcribing:

```bash
# local
uv run phonebot \
  --extraction-only \
  --transcriptions-path outputs/<run_id>/transcriptions.json

# Docker (cpu)
docker run --rm \
  --env-file .env \
  -v "$PWD/config_cpu.yaml:/app/config.yaml:ro" \
  -v "$PWD/outputs:/app/outputs" \
  phonebot:cpu \
    --extraction-only \
    --transcriptions-path outputs/<run_id>/transcriptions.json
```

---

### CLI reference

```
uv run phonebot [OPTIONS]

Options:
  -t, --transcriber TEXT        Override transcriber: openai_llm|deepgram|whisperx|parakeet
  -e, --extractor TEXT          Override extractor (currently only: llm)
      --eval true|false         Run accuracy evaluation after extraction  [default: true]
      --extraction-only         Skip transcription; read from --transcriptions-path
      --transcriptions-path     Path to a saved transcriptions.json artifact
      --extractor-prompt-file   Path to a custom YAML/Jinja2 extractor prompt file
      --output-dir PATH         Output root  [default: outputs]
```

---

## ⚙️ Configuration

`config_gpu.yaml` — `whisperx` transcriber, `gpu_enabled: true`, `denoising_enabled: true`  

`config_cpu.yaml` — `openai_llm` transcriber, `gpu_enabled: false`, `denoising_enabled: false`

Key fields:


| Field                   | Effect                                                     |
| ----------------------- | ---------------------------------------------------------- |
| `transcriber`           | `openai_llm` / `deepgram` / `whisperx` / `parakeet`        |
| `gpu_enabled`           | Must be `true` for `whisperx` / `parakeet` / FastEnhancer  |
| `denoising_enabled`     | Enables FastEnhancer GPU denoising (GPU image only)        |
| `llm_extractor_model`   | OpenAI chat model used for extraction                      |
| `extractor_prompt_file` | Path to YAML/Jinja2 prompt file; swap without code changes |
| `langsmith_tracing`     | Enable LangSmith tracing                                   |


---

## 📁 Output artifacts

Each run writes to `outputs/{run_id}/`:


| File                  | Contents                                                                                    |
| --------------------- | ------------------------------------------------------------------------------------------- |
| `results.json`        | Extracted caller fields for every recording                                                 |
| `results.md`          | Human-readable run summary with predictions and eval results                                |
| `transcriptions.json` | Raw transcripts (if not run with --extraction-only flag; reusable for extraction-only mode) |
| `eval.json`           | Per-field and overall accuracy against ground truth                                         |
| `case_report.json`    | Transcript + prediction + expected + per-field match per call                               |
| `config.yaml`         | Non-secret config snapshot for reproducibility                                              |
| `run.log`             | Structured run log                                                                          |


---

## 🛠️ Development

```bash
# Install runtime + dev dependencies
uv sync --frozen

# Run tests
uv run pytest

# Lint
uv run ruff check src tests

# Type-check
uv run mypy src
```

### Debug script

`scripts/debug_single_call.py` — single-call debug harness for development. Runs the full pipeline on one recording and writes all artifacts to `outputs/debug/<run_id>/`.

```bash
uv run python scripts/debug_single_call.py [AUDIO_FILE] [OPTIONS]
```

```
Positional:
  audio_file                    WAV file to process  [default: data/recordings/call_01.wav]

Options:
  --file PATH                   Audio file (alternative to positional arg)
  --record-id TEXT              Override record id (defaults to filename stem)
  --output-dir PATH             Output directory  [default: outputs/debug]
  --ground-truth PATH           Ground truth JSON for evaluation  [default: data/ground_truth.json]
                                Omit or point to a missing file to skip evaluation (prints a notice).
  --transcriber TEXT            Transcriber backend registry key
  --extractor TEXT              Extractor backend registry key
  --extractor-prompt-file PATH  Path to a custom YAML/Jinja2 extractor prompt file
  --extraction-only             Skip transcription; requires --transcriptions-path
  --transcriptions-path PATH    Path to a saved transcriptions.json artifact
  --diarization                 Enable speaker diarization (requires HF_TOKEN)
  --gpu                         Enable GPU acceleration
  --denoising                   Enable FastEnhancer denoising (requires --gpu)
  --langsmith-tracing           Enable LangSmith tracing for this run
```

### Project structure

```
src/phonebot/
├── cli.py              # Typer CLI entry point
├── pipeline.py         # Async batch orchestrator
├── config.py           # PipelineConfig (Pydantic)
├── schemas.py          # CallerInfo output schema
├── evaluation.py       # Accuracy scoring
├── normalization.py    # Phone / email normalizers
├── observability.py    # LangSmith tracing helpers
├── extraction/         # LLM extractor
├── transcription/      # Transcriber implementations
└── preprocessing/      # FastEnhancer denoiser
prompts/extraction/     # Versioned YAML/Jinja2 prompt files
data/
├── recordings/         # 30 German WAV call recordings
└── ground_truth.json   # Expected CallerInfo per recording
```

### Adding a new transcriber

1. Create `src/phonebot/transcription/my_backend.py` implementing `TranscriberBase`
2. Register it in the transcriber registry (see `transcription/base.py`)
3. Add any new config fields to `PipelineConfig` in `config.py`
4. Add a corresponding entry to the backend table in this README

