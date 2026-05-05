from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import yaml

from phonebot.config import PipelineConfig

CONFIG_ENV_VARS = [
    "TRANSCRIBER",
    "EXTRACTOR",
    "SAMPLE",
    "DIARIZATION_ENABLED",
    "GPU_ENABLED",
    "LANGSMITH_TRACING",
    "EXTRACTOR_PROMPT_FILE",
    "OPENAI_LLM_TRANSCRIBER_MODEL",
    "OPENAI_LLM_DIARIZATION_MODEL",
    "LLM_EXTRACTOR_MODEL",
    "WHISPERX_MODEL",
    "WHISPERX_COMPUTE_TYPE",
    "WHISPERX_LANGUAGE",
    "DEEPGRAM_MODEL",
    "DEEPGRAM_LANGUAGE",
    "DEEPGRAM_SMART_FORMAT",
    "PARAKEET_MODEL",
    "PARAKEET_LANGUAGE",
    "OPENAI_API_KEY",
    "DEEPGRAM_API_KEY",
    "HF_TOKEN",
    "LANGSMITH_API_KEY",
]

NON_SECRET_CONFIG_FIELDS = [
    "transcriber",
    "extractor",
    "sample",
    "diarization_enabled",
    "gpu_enabled",
    "langsmith_tracing",
    "extractor_prompt_file",
    "openai_llm_transcriber_model",
    "openai_llm_diarization_model",
    "llm_extractor_model",
    "whisperx_model",
    "whisperx_compute_type",
    "whisperx_language",
    "deepgram_model",
    "deepgram_language",
    "deepgram_smart_format",
    "parakeet_model",
    "parakeet_language",
]

YAML_VALUES: dict[str, Any] = {
    "transcriber": "whisperx",
    "extractor": "custom_yaml",
    "sample": "test",
    "diarization_enabled": True,
    "gpu_enabled": True,
    "langsmith_tracing": True,
    "extractor_prompt_file": "yaml_prompt.yaml",
    "openai_llm_transcriber_model": "yaml-transcribe",
    "openai_llm_diarization_model": "yaml-diarize",
    "llm_extractor_model": "yaml-llm",
    "whisperx_model": "yaml-whisper",
    "whisperx_compute_type": "int8",
    "whisperx_language": "de",
    "deepgram_model": "yaml-deepgram",
    "deepgram_language": "de",
    "deepgram_smart_format": False,
    "parakeet_model": "yaml-parakeet",
    "parakeet_language": "de-DE",
}

DOTENV_VALUES: dict[str, Any] = {
    "transcriber": "parakeet",
    "extractor": "custom_dotenv",
    "sample": "all",
    "diarization_enabled": False,
    "gpu_enabled": False,
    "langsmith_tracing": False,
    "extractor_prompt_file": "dotenv_prompt.yaml",
    "openai_llm_transcriber_model": "dotenv-transcribe",
    "openai_llm_diarization_model": "dotenv-diarize",
    "llm_extractor_model": "dotenv-llm",
    "whisperx_model": "dotenv-whisper",
    "whisperx_compute_type": "float32",
    "whisperx_language": "auto",
    "deepgram_model": "dotenv-deepgram",
    "deepgram_language": "default",
    "deepgram_smart_format": True,
    "parakeet_model": "dotenv-parakeet",
    "parakeet_language": "auto",
}

INIT_VALUES: dict[str, Any] = {
    "transcriber": "deepgram",
    "extractor": "custom_init",
    "sample": "dev",
    "diarization_enabled": True,
    "gpu_enabled": True,
    "langsmith_tracing": True,
    "extractor_prompt_file": "init_prompt.yaml",
    "openai_llm_transcriber_model": "init-transcribe",
    "openai_llm_diarization_model": "init-diarize",
    "llm_extractor_model": "init-llm",
    "whisperx_model": "init-whisper",
    "whisperx_compute_type": "float16",
    "whisperx_language": "fr",
    "deepgram_model": "init-deepgram",
    "deepgram_language": "fr",
    "deepgram_smart_format": False,
    "parakeet_model": "init-parakeet",
    "parakeet_language": "fr-FR",
}

SECRETS = {
    "OPENAI_API_KEY": "test-openai",
    "DEEPGRAM_API_KEY": "test-deepgram",
    "HF_TOKEN": "test-hf",
    "LANGSMITH_API_KEY": "test-langsmith",
}


@pytest.fixture(autouse=True)
def _isolated_settings_sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    for name in CONFIG_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    yield tmp_path


def _write_yaml_config(path: Path, values: dict[str, Any]) -> None:
    (path / "config.yaml").write_text(yaml.safe_dump(values), encoding="utf-8")


def _write_dotenv(path: Path, values: dict[str, Any]) -> None:
    lines = [f"{key}={value}" for key, value in SECRETS.items()]
    lines.extend(
        f"{field.upper()}={str(value).lower() if isinstance(value, bool) else value}"
        for field, value in values.items()
    )
    (path / ".env").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _assert_config_values(config: PipelineConfig, expected: dict[str, Any]) -> None:
    for field in NON_SECRET_CONFIG_FIELDS:
        assert getattr(config, field) == expected[field], field


def test_config_yaml_overrides_model_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key, value in SECRETS.items():
        monkeypatch.setenv(key, value)
    _write_yaml_config(tmp_path, YAML_VALUES)

    config = PipelineConfig()

    _assert_config_values(config, YAML_VALUES)
    for field in NON_SECRET_CONFIG_FIELDS:
        assert getattr(config, field) != PipelineConfig.model_fields[field].default, field


def test_dotenv_overrides_config_yaml(tmp_path: Path) -> None:
    _write_yaml_config(tmp_path, YAML_VALUES)
    _write_dotenv(tmp_path, DOTENV_VALUES)

    config = PipelineConfig()

    _assert_config_values(config, DOTENV_VALUES)


def test_init_kwargs_override_dotenv_and_config_yaml(tmp_path: Path) -> None:
    _write_yaml_config(tmp_path, YAML_VALUES)
    _write_dotenv(tmp_path, DOTENV_VALUES)

    config = PipelineConfig(**INIT_VALUES)

    _assert_config_values(config, INIT_VALUES)


def test_deepgram_diarization_does_not_require_hf_token() -> None:
    config = PipelineConfig(
        transcriber="deepgram",
        extractor="custom",
        diarization_enabled=True,
        deepgram_api_key="test-deepgram",
        hf_token="",
    )

    assert config.transcriber == "deepgram"
    assert config.diarization_enabled is True
    assert config.hf_token == ""


def test_whisperx_diarization_requires_hf_token() -> None:
    with pytest.raises(ValueError, match="HF_TOKEN is required"):
        PipelineConfig(
            transcriber="whisperx",
            extractor="custom",
            diarization_enabled=True,
            gpu_enabled=True,
            hf_token="",
        )


def test_non_diarized_run_does_not_require_hf_token() -> None:
    config = PipelineConfig(
        transcriber="whisperx",
        extractor="custom",
        diarization_enabled=False,
        gpu_enabled=True,
        hf_token="",
    )

    assert config.transcriber == "whisperx"
    assert config.diarization_enabled is False
    assert config.hf_token == ""


def test_denoising_requires_gpu_enabled() -> None:
    with pytest.raises(ValueError, match="denoising_enabled=True requires gpu_enabled=True"):
        PipelineConfig(
            transcriber="deepgram",
            extractor="custom",
            denoising_enabled=True,
            gpu_enabled=False,
            deepgram_api_key="test-deepgram",
        )


def test_failed_sample_is_valid() -> None:
    config = PipelineConfig(transcriber="whisperx", extractor="custom", sample="failed")

    assert config.sample == "failed"


def test_extraction_only_does_not_require_transcriber_key(tmp_path: Path) -> None:
    config = PipelineConfig(
        transcriber="deepgram",
        extractor="custom",
        extraction_only=True,
        transcriptions_path=str(tmp_path / "transcriptions.json"),
        deepgram_api_key="",
    )

    assert config.extraction_only is True
    assert config.deepgram_api_key == ""


def test_extraction_only_requires_transcriptions_path() -> None:
    with pytest.raises(ValueError, match="transcriptions_path is required"):
        PipelineConfig(
            transcriber="deepgram",
            extractor="custom",
            extraction_only=True,
            transcriptions_path=None,
        )
