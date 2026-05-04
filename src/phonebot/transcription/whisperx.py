from __future__ import annotations

import asyncio
import gc
from typing import ClassVar

from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.observability import maybe_traceable
from phonebot.schemas import AudioInput, SpeakerSegment, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class WhisperXTranscriber(TranscriberBase):
    """WhisperX transcription backend (GPU required).

    Supports word-level timestamps and optional speaker diarization via pyannote.
    Requires the whisperx package (uv sync --group gpu) and a CUDA-capable GPU.

    Config knobs (all via config.yaml):
      - gpu_enabled: must be true to use this backend
      - whisperx_model: model size (e.g. large-v2)
      - whisperx_compute_type: float16 | int8 | float32 (VRAM/accuracy tradeoff)
      - whisperx_language: language hint, e.g. "de"; "auto" = auto-detect
      - diarization_enabled: whether to run speaker diarization (requires hf_token)

    Outer @maybe_traceable traces the entire call as one LangSmith span.
    Inner @retry handles transient failures with exponential back-off (3 attempts).
    All blocking GPU calls are dispatched via asyncio.to_thread() to avoid
    freezing the event loop.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("WhisperX requires gpu_enabled=True in config")

        # Deferred imports: whisperx requires CUDA and lives in the optional 'gpu'
        # dependency group.  Stored as instance attributes to allow test injection
        # via obj._wx = MagicMock() / obj._wxd = MagicMock() without installing
        # the real package.  DiarizationPipeline is in whisperx.diarize, NOT on
        # the whisperx top-level namespace, hence a separate import.
        import whisperx as _wx  # type: ignore[import-not-found]
        import whisperx.diarize as _wxd  # type: ignore[import-not-found]

        self._wx = _wx
        self._wxd = _wxd

        self._device = "cuda"
        self._model_name = config.whisperx_model
        self._compute_type = config.whisperx_compute_type
        self._language: str | None = _whisperx_language_arg(config.whisperx_language)
        self._batch_size = config.whisperx_batch_size
        self._diarization_enabled = config.diarization_enabled
        self._hf_token = config.hf_token

        # Eager model load: surfaces GPU OOM or missing-model errors at batch
        # start, not mid-run.
        self._model = _wx.load_model(
            self._model_name,
            self._device,
            compute_type=self._compute_type,
        )

        # Patch the pyannote VAD segmentation batch_size.  whisperx's load_model
        # accepts batch_size only for the Whisper ASR path; the VAD model (pyannote
        # Inference) keeps its own hardcoded default of 32 which causes CUDA OOM on
        # small GPUs.  We override it here after load.
        try:
            self._model.vad_model.vad_pipeline._segmentation.batch_size = (
                config.whisperx_vad_batch_size
            )
        except AttributeError:
            pass  # non-pyannote VAD backend — no-op

    @maybe_traceable("whisperx.transcribe")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Offload blocking GPU work to a thread so the event loop stays free."""
        return await asyncio.to_thread(self._transcribe_sync, audio)

    def _transcribe_sync(self, audio: AudioInput) -> TranscriptionResult:
        """Synchronous transcription; runs in a thread-pool thread via asyncio.to_thread."""
        audio_array = self._wx.load_audio(str(audio.file))

        result = self._model.transcribe(
            audio_array,
            batch_size=self._batch_size,
            language=self._language,
        )

        if not self._diarization_enabled:
            raw_text = " ".join(seg["text"].strip() for seg in result["segments"])
            return TranscriptionResult(
                id=audio.id,
                raw_text=raw_text,
                segments=None,
                supports_diarization=False,
            )

        # --- Diarized path ---------------------------------------------------
        # 1. Align: produces word-level timestamps required for speaker assignment.
        model_a, metadata = self._wx.load_align_model(
            language_code=result["language"],
            device=self._device,
        )
        result = self._wx.align(
            result["segments"],
            model_a,
            metadata,
            audio_array,
            self._device,
            return_char_alignments=False,
            print_progress=False,
        )

        # 2. Free alignment model VRAM before loading diarization model.
        del model_a
        gc.collect()
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()

        # 3. Diarize and assign speaker labels.
        diarize_model = self._wxd.DiarizationPipeline(
            token=self._hf_token,
            device=self._device,
        )
        diarize_segments = diarize_model(audio_array)
        result = self._wx.assign_word_speakers(diarize_segments, result)

        segments = [
            SpeakerSegment(
                speaker=seg.get("speaker", "UNKNOWN"),
                start=seg["start"],
                end=seg["end"],
                text=seg["text"].strip(),
            )
            for seg in result["segments"]
        ]
        raw_text = " ".join(seg.text for seg in segments)
        return TranscriptionResult(
            id=audio.id,
            raw_text=raw_text,
            segments=segments,
            supports_diarization=True,
        )


def _whisperx_language_arg(config_language: str) -> str | None:
    """Translate the public config sentinel to WhisperX's auto-detect API value."""
    if config_language == "auto":
        return None
    return config_language
