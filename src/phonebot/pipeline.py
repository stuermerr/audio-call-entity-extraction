from __future__ import annotations

import importlib.util
import json
import logging
import os
import shutil
from pathlib import Path

from phonebot.config import PipelineConfig
from phonebot.extraction.base import (
    _DEFAULT_PROMPT,
    ExtractorBase,
    PromptTemplate,
)
from phonebot.extraction.base import (
    REGISTRY as EXTRACTION_REGISTRY,
)
from phonebot.observability import (
    configure_langsmith_tracing,
    get_logger,
    make_run_id,
    save_config_snapshot,
)
from phonebot.preprocessing.base import PreprocessorBase
from phonebot.schemas import (
    AudioInput,
    CallerInfo,
    ExtractedFields,
    PipelineCaseResult,
    PipelineOutput,
    RecordingResult,
    ResultsFile,
    TranscriptionArtifact,
    TranscriptionResult,
)
from phonebot.transcription import REGISTRY as TRANSCRIPTION_REGISTRY
from phonebot.transcription import TranscriberBase

# ---------------------------------------------------------------------------
# Required environment variable per backend key.
# Only keys present here are validated; registry validation handles unknown
# backend names before any batch work starts.
# ---------------------------------------------------------------------------
_REQUIRED_ENV_VARS: dict[str, str] = {
    "openai_llm": "OPENAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}

# Backends that require CUDA / GPU libraries at runtime.
_GPU_BACKENDS: frozenset[str] = frozenset({"parakeet", "whisperx"})


def _assert_gpu_available(transcriber: str) -> None:
    """Raise RuntimeError before any I/O if the configured backend needs CUDA but it is absent.

    Checks are only performed when the transcriber is a known GPU-only backend.
    Uses a lightweight importlib probe so torch is not imported unconditionally.
    """
    if transcriber not in _GPU_BACKENDS:
        return

    torch_spec = importlib.util.find_spec("torch")
    if torch_spec is None:
        raise RuntimeError(
            f"Transcriber '{transcriber}' requires PyTorch with CUDA, but 'torch' is not "
            "installed. Install the gpu dependency group (`uv sync --group gpu`) or switch to a "
            "CPU-friendly config (e.g. copy config_cpu.yaml → config.yaml and set "
            "transcriber=openai_llm)."
        )

    import torch  # noqa: PLC0415

    if not torch.cuda.is_available():
        raise RuntimeError(
            f"Transcriber '{transcriber}' requires a CUDA-capable GPU, but "
            "torch.cuda.is_available() returned False. "
            "Switch to a CPU-friendly config (e.g. copy config_cpu.yaml → config.yaml "
            "and set transcriber=openai_llm), or run on a machine with a supported GPU."
        )


async def run_single(
    audio: AudioInput,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    transcriber: TranscriberBase,
    extractor: ExtractorBase,
    prompt: PromptTemplate,
    preprocessor: PreprocessorBase,
    transcription_results: list[TranscriptionResult] | None = None,
    idx: int = 0,
    total: int = 1,
) -> PipelineCaseResult:
    """Transcribe and extract caller info from one audio file.

    Returns a ``PipelineCaseResult`` in all cases:
    - ``transcript`` is ``None`` on file-not-found or transcription failure.
    - ``transcript`` is populated (``result.raw_text``) even if extraction fails,
      so the raw transcript is always preserved for debugging.
    - ``caller_info`` falls back to a null CallerInfo (all entity fields ``None``)
      on any unrecoverable failure so the batch loop can continue.
    """
    prefix = f"[{idx + 1}/{total}] {audio.id}"

    # 5a. File guard: missing file → skip + null CallerInfo, no transcript
    if not audio.file.exists():
        logger.error("%s — audio file not found: %s", prefix, audio.file)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=None,
        )

    # 5b. Preprocessing (passthrough unless denoising is enabled, never raises)
    if config.denoising_enabled:
        logger.info("%s — enhancing audio", prefix)
    original_audio = audio
    audio = await preprocessor.preprocess(audio)
    _enhanced_file = audio.file if audio.file != original_audio.file else None

    # 5c. Transcription – tenacity reraises after retries, outer except fires once
    logger.info("%s — transcribing", prefix)
    try:
        result = await transcriber.transcribe(audio)
    except Exception:
        logger.error("%s — transcription failed", prefix, exc_info=True)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=None,
        )
    finally:
        if _enhanced_file is not None and _enhanced_file.exists():
            _enhanced_file.unlink()

    if transcription_results is not None:
        transcription_results.append(result)

    # 5e. Extraction — transcript is preserved even on failure
    logger.info("%s — extracting", prefix)
    try:
        caller_info = await extractor.extract(audio.id, str(audio.file), result.raw_text, prompt)
    except Exception:
        logger.error("%s — extraction failed", prefix, exc_info=True)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=result.raw_text,
        )

    logger.info("%s — done", prefix)
    # 5f. Return
    return PipelineCaseResult(caller_info=caller_info, transcript=result.raw_text)


def _load_transcriptions(path: Path) -> dict[str, TranscriptionResult]:
    """Load a transcriptions.json artifact and return a call-id lookup."""
    if not path.exists():
        raise ValueError(f"Transcriptions file not found: {path}")

    try:
        artifact = TranscriptionArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"Invalid transcriptions file: {path}") from exc

    by_id: dict[str, TranscriptionResult] = {}
    duplicates: set[str] = set()
    for transcription in artifact.transcriptions:
        if transcription.id in by_id:
            duplicates.add(transcription.id)
        by_id[transcription.id] = transcription

    if duplicates:
        duplicate_list = ", ".join(sorted(duplicates))
        raise ValueError(f"Duplicate transcript id(s) in {path}: {duplicate_list}")

    return by_id


def _save_transcriptions(
    transcriptions: list[TranscriptionResult],
    run_id: str,
    output_dir: Path,
) -> None:
    """Write the canonical per-run transcript artifact."""
    transcriptions_path = Path(output_dir) / run_id / "transcriptions.json"
    transcriptions_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = TranscriptionArtifact(transcriptions=transcriptions)
    transcriptions_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")


async def _run_single_extraction(
    audio: AudioInput,
    transcript: TranscriptionResult,
    logger: logging.Logger,
    *,
    extractor: ExtractorBase,
    prompt: PromptTemplate,
    idx: int = 0,
    total: int = 1,
) -> PipelineCaseResult:
    """Extract caller info from a preloaded transcript without touching audio."""
    prefix = f"[{idx + 1}/{total}] {audio.id}"

    logger.info("%s — extracting", prefix)
    try:
        caller_info = await extractor.extract(
            audio.id, str(audio.file), transcript.raw_text, prompt
        )
    except Exception:
        logger.error("%s — extraction failed", prefix, exc_info=True)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=transcript.raw_text,
        )

    logger.info("%s — done", prefix)
    return PipelineCaseResult(caller_info=caller_info, transcript=transcript.raw_text)


async def run_batch(
    inputs: list[AudioInput],
    config: PipelineConfig,
    *,
    output_dir: Path = Path("outputs"),
) -> PipelineOutput:
    """Run the full pipeline over a list of audio inputs.

    Validates backend keys and API env vars before any I/O, then processes
    inputs sequentially, writing a results.json snapshot to
    ``output_dir/{run_id}/results.json``.
    """
    # 6a. Registry key validation – fail before any I/O
    if not config.extraction_only and config.transcriber not in TRANSCRIPTION_REGISTRY:
        raise ValueError(
            f"Unknown transcriber '{config.transcriber}'. Valid: {sorted(TRANSCRIPTION_REGISTRY)}"
        )
    if config.extractor not in EXTRACTION_REGISTRY:
        raise ValueError(
            f"Unknown extractor '{config.extractor}'. Valid: {sorted(EXTRACTION_REGISTRY)}"
        )

    # 6b. API key validation
    backend_keys = [config.extractor]
    if not config.extraction_only:
        backend_keys.append(config.transcriber)
    for key in backend_keys:
        required_var = _REQUIRED_ENV_VARS.get(key)
        if required_var and not os.environ.get(required_var):
            raise RuntimeError(f"Missing required env var: {required_var} (needed by {key})")

    # 6b2. GPU availability guard — hard-fail before any I/O if CUDA is required but absent
    if not config.extraction_only:
        _assert_gpu_available(config.transcriber)

    # 6c. Run ID + logger
    configure_langsmith_tracing(config)
    run_id = make_run_id(config)
    logger = get_logger(run_id, output_dir)

    # 6d. Create backends once per batch (avoids repeated client/prompt init overhead)
    transcriber: TranscriberBase | None = None
    if not config.extraction_only:
        try:
            transcriber = TRANSCRIPTION_REGISTRY[config.transcriber](config)  # type: ignore[call-arg]
        except ImportError as exc:
            raise RuntimeError(
                f"Failed to initialise transcriber '{config.transcriber}': {exc}. "
                "If using a GPU backend (parakeet, whisperx), ensure the gpu dependency "
                "group is installed (`uv sync --group gpu`) and CUDA is available, "
                "or switch to a CPU-friendly config (e.g. copy config_cpu.yaml → config.yaml)."
            ) from exc
    try:
        extractor = EXTRACTION_REGISTRY[config.extractor](config)  # type: ignore[call-arg]
    except ImportError as exc:
        raise RuntimeError(f"Failed to initialise extractor '{config.extractor}': {exc}.") from exc

    # 6e. Load prompt once per batch
    prompt_path = (
        Path(config.extractor_prompt_file) if config.extractor_prompt_file else _DEFAULT_PROMPT
    )
    try:
        prompt = ExtractorBase.load_prompt(prompt_path)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Prompt file not found: {prompt_path}. "
            "Check the 'extractor_prompt_file' setting in config.yaml."
        ) from exc
    logger.info("Using extraction prompt: %s", prompt_path)

    # 6f. Preprocessor
    _preprocessed_dir: Path | None = None
    if config.denoising_enabled and not config.extraction_only:
        from phonebot.preprocessing.fastenhancer import FastEnhancerPreprocessor

        _preprocessed_dir = Path(output_dir) / run_id / "preprocessed"
        preprocessor: PreprocessorBase = FastEnhancerPreprocessor(
            _preprocessed_dir,
            model_path=Path(config.fastenhancer_model_path)
            if config.fastenhancer_model_path
            else None,
            model_url=config.fastenhancer_model_url,
            hop_size=config.fastenhancer_hop_size,
        )
        logger.info("Denoising enabled: writing enhanced audio to %s", _preprocessed_dir)
    else:
        preprocessor = PreprocessorBase()
    # 6g. Persist config snapshot — include resolved prompt path so it is never null
    save_config_snapshot(
        config,
        run_id,
        output_dir,
        extra={"extractor_prompt_file": str(prompt_path)},
    )

    transcriptions_by_id: dict[str, TranscriptionResult] = {}
    if config.extraction_only:
        if config.transcriptions_path is None:
            raise ValueError("transcriptions_path is required when extraction_only=True")
        transcriptions_by_id = _load_transcriptions(Path(config.transcriptions_path))
        missing_ids = sorted(audio.id for audio in inputs if audio.id not in transcriptions_by_id)
        if missing_ids:
            raise ValueError("Missing transcript(s) for call id(s): " + ", ".join(missing_ids))

    # 6h. Sequential processing loop
    total = len(inputs)
    logger.info("Starting batch: %d file(s) [sample=%s]", total, config.sample)
    cases: list[PipelineCaseResult] = []
    transcriptions: list[TranscriptionResult] = []
    for idx, audio in enumerate(inputs):
        if config.extraction_only:
            case = await _run_single_extraction(
                audio,
                transcriptions_by_id[audio.id],
                logger,
                extractor=extractor,
                prompt=prompt,
                idx=idx,
                total=total,
            )
        else:
            if transcriber is None:
                raise RuntimeError("Transcriber was not initialised")
            case = await run_single(
                audio,
                config,
                logger,
                transcriber=transcriber,
                extractor=extractor,
                prompt=prompt,
                preprocessor=preprocessor,
                transcription_results=transcriptions,
                idx=idx,
                total=total,
            )
        cases.append(case)

    logger.info("Batch complete: %d/%d processed", len(cases), total)

    # Clean up preprocessed dir (individual files already removed after each transcription)
    if _preprocessed_dir is not None and _preprocessed_dir.exists():
        shutil.rmtree(_preprocessed_dir, ignore_errors=True)
        logger.debug("Removed preprocessed directory: %s", _preprocessed_dir)

    if not config.extraction_only:
        _save_transcriptions(transcriptions, run_id, Path(output_dir))

    # 6i. Build output model — include resolved prompt path in snapshot
    output = PipelineOutput(
        results=[c.caller_info for c in cases],
        run_id=run_id,
        config_snapshot={**config.model_dump(), "extractor_prompt_file": str(prompt_path)},
        cases=cases,
    )

    # 6j. Persist results
    results_path = Path(output_dir) / run_id / "results.json"
    results_file = ResultsFile(
        recordings=[
            RecordingResult(
                id=c.id,
                file=c.file,
                extracted=ExtractedFields(
                    first_name=c.first_name,
                    last_name=c.last_name,
                    email=c.email,
                    phone_number=c.phone_number,
                ),
            )
            for c in output.results
        ]
    )
    results_path.write_text(results_file.model_dump_json(indent=2), encoding="utf-8")

    # 6k. Return
    return output
