from __future__ import annotations

from phonebot.extraction.base import ExtractorBase, PromptTemplate
from phonebot.schemas import CallerInfo


class PresidioExtractor(ExtractorBase):
    """Presidio NER extraction backend (not yet implemented).

    # TODO: Implement using Microsoft Presidio for local NER-based PII detection.
    # Presidio runs fully offline and supports custom recognizers for German PII.
    """

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        raise NotImplementedError("Presidio extractor not yet implemented")
