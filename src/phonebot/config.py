from __future__ import annotations

from typing import Literal

from dotenv import load_dotenv
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.main import YamlConfigSettingsSource

# Load .env into os.environ early so all consumers (SDKs, observability) see the vars.
load_dotenv()

# ---------------------------------------------------------------------------
# Key-to-backend matrix (update when adding new backends)
# ---------------------------------------------------------------------------
# OPENAI_API_KEY  : transcriber="openai_llm"  OR  extractor in ("llm", "privacy_filter")
# DEEPGRAM_API_KEY: transcriber="deepgram"
# HF_TOKEN        : diarization_enabled=True with WhisperX or pyannote fallback
# LANGSMITH_API_KEY: langsmith_tracing=True
# ---------------------------------------------------------------------------

_OPENAI_TRANSCRIBERS = {"openai_llm"}
_OPENAI_EXTRACTORS = {"llm", "privacy_filter"}
_HF_DIARIZATION_TRANSCRIBERS = {"whisperx", "parakeet"}


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_file=".env",
        extra="ignore",
    )

    # --- pipeline settings ---
    transcriber: str = "openai_llm"
    extractor: str = "llm"
    sample: Literal["dev", "test", "all"] = "dev"
    diarization_enabled: bool = False
    gpu_enabled: bool = False
    langsmith_tracing: bool = False
    extractor_prompt_file: str | None = None
    openai_llm_transcriber_model: str = "gpt-4o-mini-transcribe"
    openai_llm_diarization_model: str = "gpt-4o-transcribe-diarize"
    llm_extractor_model: str = "gpt-4.1-mini"
    whisperx_model: str = "large-v2"
    whisperx_compute_type: str = "float16"
    whisperx_language: str = "auto"
    deepgram_model: str = "nova-3"
    deepgram_language: str = "default"
    deepgram_smart_format: bool = True
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"

    # --- API keys ---
    # Excluded from model_dump() / config-snapshot serialisation to avoid
    # writing secrets to disk.  Validated conditionally: a missing key is only
    # an error when the configured backend actually needs it.
    openai_api_key: str = Field(default="", exclude=True)
    deepgram_api_key: str = Field(default="", exclude=True)
    hf_token: str = Field(default="", exclude=True)
    langsmith_api_key: str = Field(default="", exclude=True)

    @model_validator(mode="after")
    def _validate_required_keys(self) -> "PipelineConfig":
        """Fail fast only when the configured backend actually needs a key."""
        if (
            self.transcriber in _OPENAI_TRANSCRIBERS or self.extractor in _OPENAI_EXTRACTORS
        ) and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for "
                f"transcriber={self.transcriber!r} / extractor={self.extractor!r}"
            )

        if self.transcriber == "deepgram" and not self.deepgram_api_key:
            raise ValueError("DEEPGRAM_API_KEY is required when transcriber='deepgram'")

        if (
            self.diarization_enabled
            and self.transcriber in _HF_DIARIZATION_TRANSCRIBERS
            and not self.hf_token
        ):
            raise ValueError(
                "HF_TOKEN is required when diarization_enabled=True "
                f"with transcriber={self.transcriber!r}"
            )

        if self.langsmith_tracing and not self.langsmith_api_key:
            raise ValueError("LANGSMITH_API_KEY is required when langsmith_tracing=True")

        return self

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # First source wins:
        # explicit init kwargs (CLI overrides) > env/.env > config.yaml > defaults.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
