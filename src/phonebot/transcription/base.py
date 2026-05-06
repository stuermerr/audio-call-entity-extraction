"""Abstract base class and backend registry for transcription backends.

All transcription backends implement ``TranscriberBase`` and register themselves
in the module-level ``REGISTRY`` dict (populated by ``transcription/__init__.py``
to avoid circular imports).  The active backend is selected at runtime via the
``transcriber`` field in ``PipelineConfig``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import ClassVar

import yaml

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


def load_transcription_prompt(path_str: str | None) -> str | None:
    """Load the transcription injection prompt from a YAML file.

    The file must contain a top-level ``prompt`` key whose value is a plain
    string.  Returns ``None`` when ``path_str`` is ``None`` (feature disabled).

    Raises ``FileNotFoundError`` if the file is missing, or ``ValueError`` if
    the YAML does not contain a ``prompt`` key.
    """
    if path_str is None:
        return None
    path = Path(path_str)
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict) or "prompt" not in data:
        raise ValueError(
            f"Transcription prompt file {path} must contain a top-level 'prompt' key; "
            f"got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
        )
    return str(data["prompt"])
