from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from phonebot.config import PipelineConfig
from phonebot.extraction.base import PromptTemplate
from phonebot.extraction.llm import LLMExtractor
from phonebot.schemas import AudioInput
from phonebot.transcription.openai_llm import OpenAILLMTranscriber


def _make_config(**overrides: object) -> PipelineConfig:
    values: dict[str, object] = {
        "transcriber": "openai_llm",
        "extractor": "llm",
        "sample": "dev",
        "diarization_enabled": False,
        "gpu_enabled": False,
        "langsmith_tracing": False,
        "extractor_prompt_file": None,
        "openai_llm_transcriber_model": "custom-transcribe-model",
        "openai_llm_diarization_model": "custom-diarize-model",
        "llm_extractor_model": "custom-extractor-model",
        "whisperx_model": "large-v2",
        "whisperx_vad": True,
    }
    values.update(overrides)
    return PipelineConfig.model_construct(**values)  # type: ignore[arg-type]


class _FakeTranscriptions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        if kwargs["response_format"] == "diarized_json":
            segment = SimpleNamespace(speaker="speaker_0", start=0.0, end=1.0, text="Hallo")
            return SimpleNamespace(text="Hallo", segments=[segment])
        return SimpleNamespace(text="Hallo")


class _FakeOpenAITranscriptionClient:
    def __init__(self) -> None:
        self.transcriptions = _FakeTranscriptions()
        self.audio = SimpleNamespace(transcriptions=self.transcriptions)


async def test_openai_llm_transcriber_uses_configured_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = _FakeOpenAITranscriptionClient()
    monkeypatch.setattr("phonebot.transcription.openai_llm.openai.AsyncOpenAI", lambda: fake_client)
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"RIFF")

    transcriber = OpenAILLMTranscriber(_make_config())
    await transcriber.transcribe(AudioInput(id="call", file=wav))

    assert fake_client.transcriptions.calls[0]["model"] == "custom-transcribe-model"


async def test_openai_llm_transcriber_uses_configured_diarization_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = _FakeOpenAITranscriptionClient()
    monkeypatch.setattr("phonebot.transcription.openai_llm.openai.AsyncOpenAI", lambda: fake_client)
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"RIFF")

    transcriber = OpenAILLMTranscriber(_make_config(diarization_enabled=True))
    await transcriber.transcribe(AudioInput(id="call", file=wav))

    assert fake_client.transcriptions.calls[0]["model"] == "custom-diarize-model"


class _ParsedFields:
    def model_dump(self) -> dict[str, str]:
        return {"first_name": "Max"}


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def parse(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        message = SimpleNamespace(refusal=None, parsed=_ParsedFields())
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])


class _FakeOpenAIExtractionClient:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()
        self.beta = SimpleNamespace(
            chat=SimpleNamespace(completions=self.completions),
        )


async def test_llm_extractor_uses_configured_model() -> None:
    fake_client = _FakeOpenAIExtractionClient()
    extractor = LLMExtractor(_make_config())
    extractor._client = fake_client  # type: ignore[assignment]

    caller = await extractor.extract(
        "call",
        "call.wav",
        "Hallo ich bin Max.",
        PromptTemplate(system="Extract.", user="{{ transcript }}"),
    )

    assert fake_client.completions.calls[0]["model"] == "custom-extractor-model"
    assert caller.first_name == "Max"
