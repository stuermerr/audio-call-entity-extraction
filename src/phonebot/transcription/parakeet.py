from __future__ import annotations

import asyncio
from typing import ClassVar

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
        hypotheses = self._model.transcribe([str(audio.file)])
        raw_text = hypotheses[0].text.strip()
        return TranscriptionResult(
            id=audio.id,
            raw_text=raw_text,
            segments=None,
            supports_diarization=False,
        )
