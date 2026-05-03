from __future__ import annotations

from phonebot.extraction.base import REGISTRY, ExtractorBase
from phonebot.extraction.gliner import GLiNERExtractor
from phonebot.extraction.llm import LLMExtractor
from phonebot.extraction.presidio import PresidioExtractor
from phonebot.extraction.privacy_filter import PrivacyFilterExtractor

REGISTRY.update(
    {
        "llm": LLMExtractor,
        "privacy_filter": PrivacyFilterExtractor,
        "presidio": PresidioExtractor,
        "gliner": GLiNERExtractor,
    }
)

__all__ = ["ExtractorBase", "REGISTRY"]
