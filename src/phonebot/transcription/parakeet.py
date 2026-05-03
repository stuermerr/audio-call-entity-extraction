from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable
from typing import Any, ClassVar

from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.observability import maybe_traceable
from phonebot.schemas import AudioInput, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class ParakeetTranscriber(TranscriberBase):
    """NVIDIA Parakeet transcription backend (GPU required).

    Uses the NeMo FastConformer-TDT model (parakeet-tdt-0.6b-v3) for
    transcription-only inference. Produces word/segment timestamps but no
    speaker diarization. Requires nemo_toolkit[asr] (gpu dependency group)
    and a CUDA-capable GPU (gpu_enabled=True in config).

    Outer @maybe_traceable traces the entire call as one LangSmith span.
    Inner @retry handles transient failures with exponential back-off (3 attempts).
    All blocking GPU calls are dispatched via asyncio.to_thread() to avoid
    freezing the event loop.
    """

    supports_diarization: ClassVar[bool] = False

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("Parakeet requires gpu_enabled=True in config")

        # Deferred import: nemo requires CUDA and lives in the optional 'gpu'
        # dependency group.  Stored as an instance attribute to allow test
        # injection via obj._nemo_asr = MagicMock() without installing the
        # real package.
        import nemo.collections.asr as _nemo_asr  # type: ignore[import-not-found]

        self._nemo_asr = _nemo_asr
        self._model_name = config.parakeet_model
        self._language = _parakeet_language_arg(config.parakeet_language)

        # Eager model load: surfaces GPU OOM or missing-model errors at batch
        # start, not mid-run — same reasoning as WhisperX's eager load.
        self._model = _nemo_asr.models.ASRModel.from_pretrained(model_name=self._model_name)

    @maybe_traceable("parakeet.transcribe")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Offload blocking GPU work to a thread so the event loop stays free."""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: AudioInput) -> TranscriptionResult:
        """Synchronous transcription; runs in a thread-pool thread via asyncio.to_thread.

        NeMo returns a list[Hypothesis]; .text is the transcript string.
        """
        hypotheses = self._model.transcribe(
            [str(audio.file)],
            **_parakeet_language_kwargs(self._model.transcribe, self._language),
        )
        raw_text = hypotheses[0].text.strip()
        return TranscriptionResult(
            id=audio.id,
            raw_text=raw_text,
            segments=None,
            supports_diarization=False,
        )


def _parakeet_language_arg(config_language: str) -> str:
    """Translate the public config sentinel to Parakeet's auto-detect language code."""
    if config_language == "auto":
        return "multi"
    return config_language


def _parakeet_language_kwargs(
    transcribe: Callable[..., Any],
    language: str,
) -> dict[str, str]:
    """Return the supported language kwarg for the loaded Parakeet implementation.

    NVIDIA NIM exposes this knob as ``language`` / ``language-code``. The NeMo
    prompt-enabled Parakeet class accepts the same values as ``target_lang``.
    Older non-prompt NeMo ASR classes have no language override, so no kwarg is
    sent to avoid breaking local inference.
    """
    try:
        parameters = inspect.signature(transcribe).parameters
    except (TypeError, ValueError):
        return {"target_lang": language}

    if "language" in parameters:
        return {"language": language}
    if "target_lang" in parameters:
        return {"target_lang": language}
    if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return {"target_lang": language}
    return {}
