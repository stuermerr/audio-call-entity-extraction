from __future__ import annotations

from phonebot.extraction.base import ExtractorBase, PromptTemplate
from phonebot.schemas import CallerInfo


class GLiNERExtractor(ExtractorBase):
    """GLiNER NER extraction backend (not yet implemented).

    # TODO: Implement using GLiNER for zero-shot local NER.
    # GLiNER supports arbitrary entity types without fine-tuning,
    # making it suitable for German PII extraction without GPU requirements.
    """

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        raise NotImplementedError("GLiNER extractor not yet implemented")
