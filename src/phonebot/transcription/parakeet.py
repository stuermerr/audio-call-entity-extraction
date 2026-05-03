from __future__ import annotations

from typing import ClassVar

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class ParakeetTranscriber(TranscriberBase):
    """NVIDIA Parakeet transcription backend (GPU required).

    Uses the NeMo FastConformer-TDT model (parakeet-tdt-0.6b-v3) for
    transcription-only inference. Produces word/segment timestamps but no
    speaker diarization. Requires nemo_toolkit[asr] (gpu dependency group)
    and a CUDA-capable GPU (gpu_enabled=True in config).
    """

    supports_diarization: ClassVar[bool] = False

    def __init__(self, config: PipelineConfig) -> None:
        if not config.gpu_enabled:
            raise RuntimeError("Parakeet requires gpu_enabled=True in config")

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise NotImplementedError("Parakeet not implemented")
