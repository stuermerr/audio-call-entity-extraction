from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import jinja2
import yaml

from phonebot.schemas import CallerInfo

_DEFAULT_PROMPT = Path("prompts/extraction/llm.yaml")

REGISTRY: dict[str, type[ExtractorBase]] = {}
"""Backend registry. Populated by __init__.py to avoid circular imports."""


@dataclass
class PromptTemplate:
    """Extraction-internal prompt container loaded from a YAML file."""

    system: str
    user: str


class ExtractorBase(ABC):
    """Abstract base class for all extraction backends."""

    @abstractmethod
    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        """Extract caller information from a transcript."""
        ...

    @staticmethod
    def load_prompt(path: Path) -> PromptTemplate:
        """Load and validate a YAML prompt file, returning a PromptTemplate.

        Raises ValueError if `system` or `user` keys are missing.
        """
        with path.open(encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict) or "system" not in data or "user" not in data:
            raise ValueError(
                f"Prompt file {path} must contain top-level 'system' and 'user' keys; "
                f"got keys: {list(data.keys()) if isinstance(data, dict) else type(data)}"
            )
        return PromptTemplate(system=str(data["system"]), user=str(data["user"]))

    @staticmethod
    def render_user(template: PromptTemplate, transcript: str) -> str:
        """Render the user prompt template with the given transcript.

        Uses StrictUndefined so a missing {{ transcript }} placeholder raises
        immediately rather than silently rendering an empty string.
        """
        env = jinja2.Environment(undefined=jinja2.StrictUndefined)
        return env.from_string(template.user).render(transcript=transcript)
