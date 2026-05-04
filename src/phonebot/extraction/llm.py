from __future__ import annotations

import logging
import re
from typing import Annotated

import openai
from pydantic import BaseModel, Field, field_validator
from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.extraction.base import _DEFAULT_PROMPT, ExtractorBase, PromptTemplate
from phonebot.normalization import clean_phone
from phonebot.observability import maybe_traceable
from phonebot.schemas import CallerInfo

_logger = logging.getLogger(__name__)


class _ExtractedFields(BaseModel):
    """Structured output schema for OpenAI parse(); kept internal to avoid polluting schemas.py.

    Validators run in ``mode="before"`` so they fire before type coercion and before
    the ``Field(pattern=...)`` constraint check.  This means the pattern always sees
    a pre-cleaned value, making the constraint a hard post-normalization assertion.
    """

    first_name: str | None = None
    last_name: str | None = None
    email: Annotated[str, Field(pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$")] | None = None
    phone_number: Annotated[str, Field(pattern=r"^\+\d+$")] | None = None

    @field_validator("phone_number", mode="before")
    @classmethod
    def _normalise_phone(cls, v: object) -> object:
        """Strip spaces, hyphens, and parentheses before the pattern constraint fires."""
        if v is None:
            return None
        return clean_phone(str(v))

    @field_validator("email", mode="before")
    @classmethod
    def _normalise_email(cls, v: object) -> object:
        """Lowercase and strip email addresses — Python owns this transform."""
        if v is None:
            return None
        return str(v).lower().strip()

    @field_validator("first_name", "last_name", mode="before")
    @classmethod
    def _normalise_name(cls, v: object) -> object:
        """Remove digit and ``@`` parse artifacts from name fields."""
        if v is None:
            return None
        result = re.sub(r"\d+", "", str(v)).replace("@", "").strip()
        return None if result == "" else result


class LLMExtractor(ExtractorBase):
    """Extraction backend using a configurable OpenAI chat model with structured output.

    Outer @maybe_traceable traces the entire call (including all retry attempts)
    as one LangSmith span. Inner @retry handles transient API failures with
    exponential back-off (up to 3 attempts); reraise=True re-raises the last
    exception on exhaustion.

    On refusal or parse failure a single manual re-prompt is attempted before
    falling back to a null CallerInfo. This separates semantic/output failures
    (manual re-prompt) from transient API failures (tenacity) — they are
    orthogonal concerns.
    """

    def __init__(self, config: PipelineConfig) -> None:
        prompt_path = (
            _DEFAULT_PROMPT
            if config.extractor_prompt_file is None
            else __import__("pathlib").Path(config.extractor_prompt_file)
        )
        self._prompt: PromptTemplate = self.load_prompt(prompt_path)
        self._model = config.llm_extractor_model
        # Defer client creation so instantiation does not require OPENAI_API_KEY.
        # The key is validated by the SDK only when the first API call is made.
        self._client: openai.AsyncOpenAI | None = None

    @maybe_traceable("llm.extract")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        if self._client is None:
            self._client = openai.AsyncOpenAI()
        user_msg = self.render_user(prompt, transcript)
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt.system},
            {"role": "user", "content": user_msg},
        ]

        response = await self._client.beta.chat.completions.parse(
            model=self._model,
            messages=messages,  # type: ignore[arg-type]
            response_format=_ExtractedFields,
        )
        choice = response.choices[0]

        # If the model refused, do one manual re-prompt before giving up.
        if choice.message.refusal:
            _logger.warning(
                "LLMExtractor: model refused first attempt for record %s; re-prompting.",
                record_id,
            )
            stricter_messages: list[dict[str, str]] = messages + [
                {
                    "role": "system",
                    "content": (
                        "Return ONLY valid JSON matching the schema. "
                        "Do not include any explanation."
                    ),
                }
            ]
            retry_response = await self._client.beta.chat.completions.parse(
                model=self._model,
                messages=stricter_messages,  # type: ignore[arg-type]
                response_format=_ExtractedFields,
            )
            retry_choice = retry_response.choices[0]
            if retry_choice.message.refusal or retry_choice.message.parsed is None:
                _logger.warning(
                    "LLMExtractor: model refused re-prompt for record %s; "
                    "returning null CallerInfo. Reason: %s",
                    record_id,
                    retry_choice.message.refusal,
                )
                return CallerInfo(id=record_id, file=record_file)
            parsed = retry_choice.message.parsed
        else:
            if choice.message.parsed is None:
                _logger.warning(
                    "LLMExtractor: parsed result is None for record %s; returning null CallerInfo.",
                    record_id,
                )
                return CallerInfo(id=record_id, file=record_file)
            parsed = choice.message.parsed

        return CallerInfo(id=record_id, file=record_file, **parsed.model_dump())
