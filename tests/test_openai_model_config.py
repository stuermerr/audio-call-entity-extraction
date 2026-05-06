from __future__ import annotations

import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from openai.lib._parsing._completions import parse_chat_completion
from openai.types.chat.chat_completion import ChatCompletion, Choice
from openai.types.chat.chat_completion_message import ChatCompletionMessage

import phonebot.observability as observability
from phonebot.config import PipelineConfig
from phonebot.extraction.base import PromptTemplate
from phonebot.extraction.llm import LLMExtractor, _ExtractedFields
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
    monkeypatch.setattr("phonebot.transcription.openai_llm.openai.AsyncOpenAI", lambda **_kw: fake_client)
    wav = tmp_path / "call.wav"
    wav.write_bytes(b"RIFF")

    transcriber = OpenAILLMTranscriber(_make_config())
    await transcriber.transcribe(AudioInput(id="call", file=wav))

    assert fake_client.transcriptions.calls[0]["model"] == "custom-transcribe-model"


async def test_openai_llm_transcriber_uses_configured_diarization_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake_client = _FakeOpenAITranscriptionClient()
    monkeypatch.setattr("phonebot.transcription.openai_llm.openai.AsyncOpenAI", lambda **_kw: fake_client)
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


def _make_openai_parsed_completion():
    message = ChatCompletionMessage.model_construct(
        role="assistant",
        content='{"first_name":"Max"}',
        refusal=None,
    )
    choice = Choice.model_construct(index=0, finish_reason="stop", message=message)
    completion = ChatCompletion.model_construct(
        id="chatcmpl-test",
        object="chat.completion",
        created=1,
        model="test-model",
        choices=[choice],
    )
    return parse_chat_completion(
        response_format=_ExtractedFields,
        input_tools=[],
        chat_completion=completion,
    )


def _is_openai_parsed_response_serializer_warning(warning: warnings.WarningMessage) -> bool:
    message = str(warning.message)
    return (
        "Pydantic serializer warnings:" in message
        and "field_name='parsed'" in message
        and "input_type=_ExtractedFields" in message
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


def test_openai_parsed_response_model_dump_emits_known_serializer_warning() -> None:
    parsed_completion = _make_openai_parsed_completion()

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        parsed_completion.model_dump()

    assert any(_is_openai_parsed_response_serializer_warning(w) for w in caught)


def test_openai_parsed_response_serializer_warning_is_suppressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_completion = _make_openai_parsed_completion()
    monkeypatch.setattr(
        observability,
        "_OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED",
        False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observability.suppress_openai_parsed_response_serializer_warning()
        parsed_completion.model_dump()

    assert not any(_is_openai_parsed_response_serializer_warning(w) for w in caught)


def test_openai_parsed_response_warning_filter_keeps_unrelated_warnings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        observability,
        "_OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED",
        False,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        observability.suppress_openai_parsed_response_serializer_warning()
        warnings.warn("unrelated serializer warning", UserWarning, stacklevel=1)

    assert [str(w.message) for w in caught] == ["unrelated serializer warning"]


def test_llm_extractor_wraps_openai_client_for_langsmith(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_client = _FakeOpenAIExtractionClient()
    wrap_calls: list[dict[str, Any]] = []

    def fake_wrap_openai(client: object, **kwargs: Any) -> object:
        wrap_calls.append({"client": client, **kwargs})
        return client

    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.setattr("phonebot.extraction.llm.openai.AsyncOpenAI", lambda **_kw: fake_client)
    monkeypatch.setattr("phonebot.extraction.llm.langsmith_tracing_enabled", lambda: True)
    monkeypatch.setattr("langsmith.wrappers.wrap_openai", fake_wrap_openai)

    extractor = LLMExtractor(_make_config())

    client = extractor._build_client()

    assert client is fake_client
    assert wrap_calls[0]["client"] is fake_client
    assert wrap_calls[0]["chat_name"] == "llm.extract.openai_parse"
    assert wrap_calls[0]["tracing_extra"]["metadata"] == {
        "phonebot_stage": "extraction",
        "phonebot_backend": "llm",
        "extractor_prompt_file": "prompts/extraction/llm_v8_de.yaml",
    }
