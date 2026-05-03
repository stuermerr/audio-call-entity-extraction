from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, ClassVar

from phonebot.schemas import AudioInput, TranscriptionResult

if TYPE_CHECKING:
    pass

REGISTRY: dict[str, type[TranscriberBase]] = {}
"""Backend registry. Populated by __init__.py to avoid circular imports."""


class TranscriberBase(ABC):
    """Abstract base class for all transcription backends."""

    supports_diarization: ClassVar[bool] = False

    @abstractmethod
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Transcribe the given audio input and return a TranscriptionResult."""
        ...
