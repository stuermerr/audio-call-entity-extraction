"""Pipeline configuration loaded from config.yaml, environment variables, and .env.

``PipelineConfig`` is a ``pydantic-settings`` model; sources are resolved in priority
order: explicit init kwargs (CLI overrides) > env/.env > config.yaml > field defaults.
Cross-field validation (API key requirements, GPU constraints) is enforced in
``_validate_required_keys``.
"""

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
    """Unified runtime configuration for the phonebot pipeline.

    Loaded from (in priority order): explicit init kwargs, environment variables /
    ``.env`` file, ``config.yaml``, and field defaults.  API keys are excluded from
    serialisation so they are never written to on-disk config snapshots.
    """

    model_config = SettingsConfigDict(
        yaml_file="config.yaml",
        env_file=".env",
        extra="ignore",
    )

    # --- pipeline settings ---
    transcriber: str = Field(
        default="openai_llm",
        description="Transcription backend key. One of: openai_llm, deepgram, whisperx, parakeet.",
    )
    extractor: str = Field(
        default="llm",
        description="Extraction backend key. Currently only 'llm' is supported.",
    )
    sample: Literal["dev", "test", "failed", "all"] = Field(
        default="all",
        description="Recording subset to process: 'all' runs every WAV in data/recordings/.",
    )
    extraction_only: bool = Field(
        default=False,
        description="Skip transcription and load transcripts from transcriptions_path instead.",
    )
    transcriptions_path: str | None = Field(
        default=None,
        description=(
            "Path to a saved transcriptions.json artifact; required when extraction_only=True."
        ),
    )
    diarization_enabled: bool = Field(
        default=False,
        description=(
            "Enable speaker diarization. Requires HF_TOKEN and a diarization-capable transcriber."
        ),
    )
    gpu_enabled: bool = Field(
        default=True,
        description=(
            "Enable GPU acceleration. Required for whisperx, parakeet, and FastEnhancer denoising."
        ),
    )
    denoising_enabled: bool = Field(
        default=True,
        description=(
            "Enable FastEnhancer ONNX audio denoising. Requires gpu_enabled=True (GPU image only)."
        ),
    )
    langsmith_tracing: bool = Field(
        default=False,
        description=(
            "Enable LangSmith tracing for extraction LLM calls. Requires LANGSMITH_API_KEY."
        ),
    )
    extractor_prompt_file: str | None = Field(
        default="prompts/extraction/llm_v8_de.yaml",
        description=(
            "Path to a YAML/Jinja2 extraction prompt file. Swap without code changes "
            "for A/B testing."
        ),
    )
    openai_llm_transcriber_model: str = Field(
        default="gpt-4o-transcribe",
        description="OpenAI model used by the openai_llm transcription backend.",
    )
    openai_llm_diarization_model: str = Field(
        default="gpt-4o-transcribe-diarize",
        description=(
            "OpenAI model for diarized transcription "
            "(openai_llm backend, diarization_enabled=True)."
        ),
    )
    llm_extractor_model: str = Field(
        default="gpt-5.4-mini",
        description="OpenAI chat model used for schema-constrained entity extraction.",
    )
    whisperx_model: str = Field(
        default="large-v3",
        description="WhisperX model size. GPU-only. large-v3 gives the best accuracy.",
    )
    whisperx_compute_type: str = Field(
        default="float16",
        description=(
            "Compute precision for WhisperX inference (float16 requires a CUDA-capable GPU)."
        ),
    )
    whisperx_language: str = Field(
        default="de",
        description="BCP-47 language code passed to WhisperX for forced alignment.",
    )
    whisperx_batch_size: int = Field(
        default=32,
        description="ASR inference batch size for WhisperX. Reduce to 4–8 if CUDA OOM occurs.",
    )
    whisperx_vad_batch_size: int = Field(
        default=8,
        description="VAD segmentation batch size for WhisperX. Reduce to 2–4 if CUDA OOM occurs.",
    )
    deepgram_model: str = Field(
        default="nova-2",
        description="Deepgram model name used for transcription.",
    )
    deepgram_language: str = Field(
        default="de",
        description="BCP-47 language code sent to the Deepgram API.",
    )
    deepgram_smart_format: bool = Field(
        default=True,
        description="Enable Deepgram smart formatting (punctuation, capitalisation).",
    )
    parakeet_model: str = Field(
        default="nvidia/parakeet-tdt-0.6b-v3",
        description="HuggingFace model ID for the Parakeet transcription backend. GPU-only.",
    )
    parakeet_language: str = Field(
        default="de-DE",
        description="BCP-47 language code for Parakeet inference.",
    )
    fastenhancer_model_url: str = Field(
        default=(
            "https://github.com/aask1357/fastenhancer/releases/download/"
            "onnx-dns-v1.0.0/fastenhancer_l.onnx"
        ),
        description=(
            "URL to download the FastEnhancer ONNX model. "
            "Used when fastenhancer_model_path is not set."
        ),
    )
    fastenhancer_model_path: str | None = Field(
        default=None,
        description="Local path to a FastEnhancer ONNX model file. Skips download when set.",
    )
    fastenhancer_hop_size: int = Field(
        default=100,
        description=(
            "Hop size for FastEnhancer inference. "
            "Must match model variant: 256 T/B/S, 160 M, 100 L."
        ),
    )

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
