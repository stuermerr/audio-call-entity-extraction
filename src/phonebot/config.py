from __future__ import annotations

from typing import Literal

from pydantic_settings import BaseSettings, PydanticBaseSettingsSource, SettingsConfigDict
from pydantic_settings.main import YamlConfigSettingsSource


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_file=".env",
        extra="ignore",
    )

    transcriber: str = "openai_llm"
    extractor: str = "llm"
    sample: Literal["dev", "test", "all"] = "dev"
    diarization_enabled: bool = False
    gpu_enabled: bool = False
    langsmith_tracing: bool = False
    extractor_prompt_file: str | None = None
    whisperx_model: str = "large-v2"
    whisperx_vad: bool = True

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )
