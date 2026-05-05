"""Abstract base class and backend registry for transcription backends.

All transcription backends implement ``TranscriberBase`` and register themselves
in the module-level ``REGISTRY`` dict (populated by ``transcription/__init__.py``
to avoid circular imports).  The active backend is selected at runtime via the
``transcriber`` field in ``PipelineConfig``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from phonebot.schemas import AudioInput, TranscriptionResult

REGISTRY: dict[str, type[TranscriberBase]] = {}
"""Backend registry. Populated by __init__.py to avoid circular imports."""


class TranscriberBase(ABC):
    """Abstract base class for all transcription backends."""

    supports_diarization: ClassVar[bool] = False

    @abstractmethod
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Transcribe the given audio input and return a TranscriptionResult."""
        ...
