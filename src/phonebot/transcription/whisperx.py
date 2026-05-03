from __future__ import annotations

from typing import ClassVar

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class WhisperXTranscriber(TranscriberBase):
    """WhisperX transcription backend (GPU required).

    # TODO: Implement WhisperX transcription.
    #   - Requires whisperx package and a CUDA-capable GPU.
    #   - Supports word-level timestamps and speaker diarization via pyannote.
    #   - Set gpu_enabled=True in config to use this backend.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("WhisperX requires gpu_enabled=True in config")

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise NotImplementedError("WhisperX not implemented")
