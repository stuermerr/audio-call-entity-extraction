from __future__ import annotations

from phonebot.extraction.base import REGISTRY, ExtractorBase
from phonebot.extraction.llm import LLMExtractor

REGISTRY.update(
    {
        "llm": LLMExtractor,
    }
)

__all__ = ["ExtractorBase", "REGISTRY"]
