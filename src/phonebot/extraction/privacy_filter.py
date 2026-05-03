from __future__ import annotations

from phonebot.extraction.base import ExtractorBase, PromptTemplate
from phonebot.schemas import CallerInfo


class PrivacyFilterExtractor(ExtractorBase):
    """Privacy Filter extraction backend (not yet implemented).

    # TODO: Implement using the OpenAI Privacy Filter API.
    # See DECISIONS.md "Privacy filter extractor approach" for the planned design:
    # detect PII spans (person name, email, phone), then map to schema fields.
    # Requires a second step to split 'person name' into first_name / last_name.
    """

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        raise NotImplementedError("PrivacyFilter extractor not yet implemented")
