from __future__ import annotations

from typing import ClassVar

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class DeepgramTranscriber(TranscriberBase):
    """Deepgram transcription backend stub.

    # TODO: Implement Deepgram transcription.
    #   - Requires DEEPGRAM_API_KEY environment variable.
    #   - Supports speaker diarization via the Deepgram Nova model.
    #   - Install the deepgram-sdk package to enable this backend.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:  # noqa: ARG002
        pass

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise NotImplementedError("Deepgram not implemented")
