from __future__ import annotations

from phonebot.transcription.base import REGISTRY, TranscriberBase
from phonebot.transcription.deepgram import DeepgramTranscriber
from phonebot.transcription.openai_llm import OpenAILLMTranscriber
from phonebot.transcription.parakeet import ParakeetTranscriber
from phonebot.transcription.whisperx import WhisperXTranscriber

REGISTRY.update(
    {
        "openai_llm": OpenAILLMTranscriber,
        "whisperx": WhisperXTranscriber,
        "deepgram": DeepgramTranscriber,
        "parakeet": ParakeetTranscriber,
    }
)

__all__ = ["TranscriberBase", "REGISTRY"]
