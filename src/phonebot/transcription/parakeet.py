from __future__ import annotations

from typing import ClassVar

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class ParakeetTranscriber(TranscriberBase):
    """NVIDIA Parakeet transcription backend stub (GPU required).

    # TODO: Implement Parakeet transcription.
    #   - Requires NVIDIA NeMo / nemo_toolkit and a CUDA-capable GPU.
    #   - supports_diarization is set to True provisionally (ARCHITECTURE.md: "to be
    #     verified"); update once the backend is wired up.
    #   - Set gpu_enabled=True in config to use this backend.
    """

    # TODO: Verify diarization support for Parakeet once implemented.
    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("Parakeet requires gpu_enabled=True in config")

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise NotImplementedError("Parakeet not implemented")
