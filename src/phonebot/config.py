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
# OPENAI_API_KEY  : transcriber="openai_llm" OR extractor="llm"
# DEEPGRAM_API_KEY: transcriber="deepgram"
# HF_TOKEN        : diarization_enabled=True with transcriber="whisperx"
# LANGSMITH_API_KEY: langsmith_tracing=True
# ---------------------------------------------------------------------------

_OPENAI_TRANSCRIBERS = {"openai_llm"}
_OPENAI_EXTRACTORS = {"llm"}
_HF_DIARIZATION_TRANSCRIBERS = {"whisperx"}


class PipelineConfig(BaseSettings):
    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_file=".env",
        extra="ignore",
    )

    # --- pipeline settings ---
    transcriber: str = "openai_llm"
    extractor: str = "llm"
    sample: Literal["dev", "test", "failed", "all"] = "dev"
    extraction_only: bool = False
    transcriptions_path: str | None = None
    diarization_enabled: bool = False
    gpu_enabled: bool = False
    denoising_enabled: bool = False
    langsmith_tracing: bool = False
    extractor_prompt_file: str | None = None
    openai_llm_transcriber_model: str = "gpt-4o-mini-transcribe"
    openai_llm_diarization_model: str = "gpt-4o-transcribe-diarize"
    llm_extractor_model: str = "gpt-4.1-mini"
    whisperx_model: str = "large-v2"
    whisperx_compute_type: str = "float16"
    whisperx_language: str = "auto"
    whisperx_batch_size: int = 16  # ASR inference batch; reduce to 4-8 if CUDA OOM occurs
    whisperx_vad_batch_size: int = 8  # VAD segmentation batch; reduce to 2-4 if CUDA OOM occurs
    deepgram_model: str = "nova-3"
    deepgram_language: str = "default"
    deepgram_smart_format: bool = True
    parakeet_model: str = "nvidia/parakeet-tdt-0.6b-v3"
    parakeet_language: str = "auto"
    fastenhancer_model_url: str = (
        "https://github.com/aask1357/fastenhancer/releases/download/"
        "onnx-dns-v1.0.0/fastenhancer_b.onnx"
    )
    fastenhancer_model_path: str | None = None  # local path override; skips download when set
    fastenhancer_hop_size: int = 256  # must match model variant: 256 T/B/S, 160 M, 100 L

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
        if self.extraction_only and not self.transcriptions_path:
            raise ValueError("transcriptions_path is required when extraction_only=True")

        if self.denoising_enabled and not self.gpu_enabled:
            raise ValueError("denoising_enabled=True requires gpu_enabled=True")

        if (
            (not self.extraction_only and self.transcriber in _OPENAI_TRANSCRIBERS)
            or self.extractor in _OPENAI_EXTRACTORS
        ) and not self.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required for "
                f"transcriber={self.transcriber!r} / extractor={self.extractor!r}"
            )

        if (
            not self.extraction_only
            and self.transcriber == "deepgram"
            and not self.deepgram_api_key
        ):
            raise ValueError("DEEPGRAM_API_KEY is required when transcriber='deepgram'")

        if (
            not self.extraction_only
            and self.diarization_enabled
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
