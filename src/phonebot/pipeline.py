from __future__ import annotations

import json  # noqa: F401 – reserved for future JSON utilities
import logging
import os
from pathlib import Path

from phonebot.config import PipelineConfig
from phonebot.diarization.pyannote import PyAnnoteDiarizer
from phonebot.extraction.base import (
    _DEFAULT_PROMPT,
    ExtractorBase,
    PromptTemplate,
)
from phonebot.extraction.base import (
    REGISTRY as EXTRACTION_REGISTRY,
)
from phonebot.observability import get_logger, make_run_id, save_config_snapshot
from phonebot.preprocessing.base import PreprocessorBase
from phonebot.schemas import AudioInput, CallerInfo, PipelineCaseResult, PipelineOutput
from phonebot.transcription import REGISTRY as TRANSCRIPTION_REGISTRY
from phonebot.transcription import TranscriberBase

# ---------------------------------------------------------------------------
# Required environment variable per backend key.
# Only keys present here are validated; unknown keys skip the check silently
# (stubs raise NotImplementedError on use anyway).
# ---------------------------------------------------------------------------
_REQUIRED_ENV_VARS: dict[str, str] = {
    "openai_llm": "OPENAI_API_KEY",
    "deepgram": "DEEPGRAM_API_KEY",
}


async def run_single(
    audio: AudioInput,
    config: PipelineConfig,
    logger: logging.Logger,
    *,
    transcriber: TranscriberBase,
    extractor: ExtractorBase,
    prompt: PromptTemplate,
    preprocessor: PreprocessorBase,
    diarizer: PyAnnoteDiarizer,
) -> PipelineCaseResult:
    """Transcribe, optionally diarize, and extract caller info from one audio file.

    Returns a ``PipelineCaseResult`` in all cases:
    - ``transcript`` is ``None`` on file-not-found or transcription failure.
    - ``transcript`` is populated (``result.raw_text``) even if extraction fails,
      so the raw transcript is always preserved for debugging.
    - ``caller_info`` falls back to a null CallerInfo (all entity fields ``None``)
      on any unrecoverable failure so the batch loop can continue.
    """
    # 5a. File guard: missing file → skip + null CallerInfo, no transcript
    if not audio.file.exists():
        logger.error("Audio file not found: %s", audio.file)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=None,
        )

    # 5b. Preprocessing (passthrough in MVP, never raises)
    audio = await preprocessor.preprocess(audio)

    # 5c. Transcription – tenacity reraises after retries, outer except fires once
    try:
        result = await transcriber.transcribe(audio)
    except Exception:
        logger.error("transcription failed", exc_info=True)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=None,
        )

    # 5d. Diarization: only when enabled AND transcriber lacks native speaker segments
    if config.diarization_enabled and not transcriber.supports_diarization:
        try:
            result = await diarizer.diarize(result)
        except Exception:
            logger.warning("diarization failed, using raw_text", exc_info=True)
            # result unchanged → passthrough

    # 5e. Extraction — transcript is preserved even on failure
    try:
        caller_info = await extractor.extract(audio.id, str(audio.file), result.raw_text, prompt)
    except Exception:
        logger.error("extraction failed", exc_info=True)
        return PipelineCaseResult(
            caller_info=CallerInfo(id=audio.id, file=str(audio.file)),
            transcript=result.raw_text,
        )

    # 5f. Return
    return PipelineCaseResult(caller_info=caller_info, transcript=result.raw_text)


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
    if config.transcriber not in TRANSCRIPTION_REGISTRY:
        raise ValueError(
            f"Unknown transcriber '{config.transcriber}'. Valid: {sorted(TRANSCRIPTION_REGISTRY)}"
        )
    if config.extractor not in EXTRACTION_REGISTRY:
        raise ValueError(
            f"Unknown extractor '{config.extractor}'. Valid: {sorted(EXTRACTION_REGISTRY)}"
        )

    # 6b. API key validation
    for key in (config.transcriber, config.extractor):
        required_var = _REQUIRED_ENV_VARS.get(key)
        if required_var and not os.environ.get(required_var):
            raise RuntimeError(f"Missing required env var: {required_var} (needed by {key})")

    # 6c. Run ID + logger
    run_id = make_run_id(config)
    logger = get_logger(run_id, output_dir)

    # 6d. Create backends once per batch (avoids repeated client/prompt init overhead)
    transcriber = TRANSCRIPTION_REGISTRY[config.transcriber](config)  # type: ignore[call-arg]
    extractor = EXTRACTION_REGISTRY[config.extractor](config)  # type: ignore[call-arg]

    # 6e. Load prompt once per batch
    prompt_path = (
        Path(config.extractor_prompt_file) if config.extractor_prompt_file else _DEFAULT_PROMPT
    )
    prompt = ExtractorBase.load_prompt(prompt_path)

    # 6f. Preprocessor + diarizer
    preprocessor = PreprocessorBase()
    diarizer = PyAnnoteDiarizer()

    # 6g. Persist config snapshot
    save_config_snapshot(config, run_id, output_dir)

    # 6h. Sequential processing loop
    cases: list[PipelineCaseResult] = []
    for audio in inputs:
        case = await run_single(
            audio,
            config,
            logger,
            transcriber=transcriber,
            extractor=extractor,
            prompt=prompt,
            preprocessor=preprocessor,
            diarizer=diarizer,
        )
        cases.append(case)

    # 6i. Build output model
    output = PipelineOutput(
        results=[c.caller_info for c in cases],
        run_id=run_id,
        config_snapshot=config.model_dump(),
        cases=cases,
    )

    # 6j. Persist results — exclude 'cases' to keep results.json backward-compatible
    results_path = Path(output_dir) / run_id / "results.json"
    results_path.write_text(output.model_dump_json(indent=2, exclude={"cases"}), encoding="utf-8")

    # 6k. Return
    return output
