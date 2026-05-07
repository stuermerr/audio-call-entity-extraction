from __future__ import annotations

import asyncio
import gc
from typing import ClassVar

from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, SpeakerSegment, TranscriptionResult
from phonebot.transcription.base import TranscriberBase, load_transcription_prompt


class WhisperXTranscriber(TranscriberBase):
    """WhisperX transcription backend (GPU required).

    Supports word-level timestamps and optional speaker diarization through
    WhisperX's integrated diarization path.
    Requires the whisperx package (uv sync --group gpu) and a CUDA-capable GPU.

    Config knobs (all via config.yaml):
      - gpu_enabled: must be true to use this backend
      - whisperx_model: model size (e.g. large-v2)
      - whisperx_compute_type: float16 | int8 | float32 (VRAM/accuracy tradeoff)
      - whisperx_language: language hint, e.g. "de"; "auto" = auto-detect
      - diarization_enabled: whether to run integrated speaker diarization

    @retry handles transient failures with exponential back-off (3 attempts).
    All blocking GPU calls are dispatched via asyncio.to_thread() to avoid freezing
    the event loop.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("WhisperX requires gpu_enabled=True in config")

        # Deferred imports: whisperx requires CUDA and lives in the optional 'gpu'
        # dependency group.  Stored as instance attributes to allow test injection
        # via obj._wx = MagicMock() / obj._wxd = MagicMock() without installing
        # the real package. DiarizationPipeline is in whisperx.diarize, not on
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

        # Load transcription injection prompt if configured.
        # Must be injected via asr_options at load_model() time — passing
        # initial_prompt directly to model.transcribe() raises an unexpected
        # keyword argument error in some WhisperX versions.
        _transcription_prompt = load_transcription_prompt(config.transcription_prompt_file)
        _asr_options: dict = {}
        if _transcription_prompt is not None:
            _asr_options["initial_prompt"] = _transcription_prompt

        # Eager model load: surfaces GPU OOM or missing-model errors at batch
        # start, not mid-run.
        self._model = _wx.load_model(
            self._model_name,
            self._device,
            compute_type=self._compute_type,
            asr_options=_asr_options if _asr_options else None,
        )

        # Patch VAD segmentation batch size after load for smaller GPUs.
        try:
            self._model.vad_model.vad_pipeline._segmentation.batch_size = (
                config.whisperx_vad_batch_size
            )
        except AttributeError:
            pass

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

        del model_a
        gc.collect()
        import torch  # type: ignore[import-not-found]

        torch.cuda.empty_cache()

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
