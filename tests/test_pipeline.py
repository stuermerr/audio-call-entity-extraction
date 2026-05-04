from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest

from phonebot.config import PipelineConfig
from phonebot.diarization.pyannote import PyAnnoteDiarizer
from phonebot.extraction.base import (
    REGISTRY as EXTRACTION_REGISTRY,
)
from phonebot.extraction.base import (
    ExtractorBase,
    PromptTemplate,
)
from phonebot.pipeline import run_batch, run_single
from phonebot.preprocessing.base import PreprocessorBase
from phonebot.schemas import (
    AudioInput,
    CallerInfo,
    PipelineCaseResult,
    TranscriptionArtifact,
    TranscriptionResult,
)
from phonebot.transcription import REGISTRY as TRANSCRIPTION_REGISTRY
from phonebot.transcription import TranscriberBase


# ---------------------------------------------------------------------------
# Step 7 – MockTranscriber: happy-path stub returning a fixed transcript
# ---------------------------------------------------------------------------
class MockTranscriber(TranscriberBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        return TranscriptionResult(
            id=audio.id,
            raw_text="Max Muster max@test.com 015201234567",
        )


# ---------------------------------------------------------------------------
# Step 8 – FailingTranscriber: always raises, exercises transcription boundary
# ---------------------------------------------------------------------------
class FailingTranscriber(TranscriberBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise RuntimeError("api error")


# ---------------------------------------------------------------------------
# Step 9 – MockExtractor: returns a fixed CallerInfo with first/last name
# ---------------------------------------------------------------------------
class MockExtractor(ExtractorBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        return CallerInfo(
            id=record_id,
            file=record_file,
            first_name="Max",
            last_name="Muster",
        )


# ---------------------------------------------------------------------------
# Step 10 – FailingExtractor: always raises, exercises extraction boundary
# ---------------------------------------------------------------------------
class FailingExtractor(ExtractorBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        raise RuntimeError("extraction error")


class ExplodingTranscriber(TranscriberBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        raise AssertionError("transcriber should not be initialised")

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        raise AssertionError("transcriber should not be called")


class AlternateTranscriber(TranscriberBase):
    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        return TranscriptionResult(id=audio.id, raw_text="Alternate transcript")


class TranscriptEchoExtractor(ExtractorBase):
    seen_transcripts: list[str] = []

    def __init__(self, config=None) -> None:  # noqa: ANN001
        pass

    async def extract(
        self,
        record_id: str,
        record_file: str,
        transcript: str,
        prompt: PromptTemplate,
    ) -> CallerInfo:
        self.seen_transcripts.append(transcript)
        return CallerInfo(id=record_id, file=record_file, first_name=transcript.split()[0])


# ---------------------------------------------------------------------------
# Step 11 – _make_config: model_construct skips settings-source resolution so
# tests don't need config.yaml or .env on any particular cwd.
# ---------------------------------------------------------------------------
def _make_config() -> PipelineConfig:
    return PipelineConfig.model_construct(
        transcriber="openai_llm",
        extractor="llm",
        sample="dev",
        extraction_only=False,
        transcriptions_path=None,
        diarization_enabled=False,
        gpu_enabled=False,
        langsmith_tracing=False,
        extractor_prompt_file=None,
        openai_llm_transcriber_model="gpt-4o-mini-transcribe",
        openai_llm_diarization_model="gpt-4o-transcribe-diarize",
        llm_extractor_model="gpt-4.1-mini",
        whisperx_model="large-v2",
    )


# ---------------------------------------------------------------------------
# Step 12 – _make_prompt: in-memory PromptTemplate avoids disk access in
# run_single unit tests.
# ---------------------------------------------------------------------------
def _make_prompt() -> PromptTemplate:
    return PromptTemplate(system="Extract.", user="{{ transcript }}")


# ---------------------------------------------------------------------------
# Step 13 – _make_logger: stdlib logger; avoids creating outputs/ dirs in
# run_single unit tests.
# ---------------------------------------------------------------------------
def _make_logger() -> logging.Logger:
    return logging.getLogger("test_pipeline")


# ---------------------------------------------------------------------------
# Internal helpers: shared backend stubs for run_single tests
# ---------------------------------------------------------------------------
def _make_preprocessor() -> PreprocessorBase:
    return PreprocessorBase()


def _make_diarizer() -> PyAnnoteDiarizer:
    return PyAnnoteDiarizer()


def _write_prompt(tmp_path: Path) -> Path:
    prompt_yaml = tmp_path / "prompt.yaml"
    prompt_yaml.write_text("system: Extract.\nuser: '{{ transcript }}'\n", encoding="utf-8")
    return prompt_yaml


# ---------------------------------------------------------------------------
# Step 14 – Happy path: file exists, transcription + extraction succeed
# ---------------------------------------------------------------------------
async def test_run_single_happy_path(tmp_path: Path) -> None:
    wav = tmp_path / "c1.wav"
    wav.write_bytes(b"RIFF....")
    audio = AudioInput(id="c1", file=wav)

    case = await run_single(
        audio,
        _make_config(),
        _make_logger(),
        transcriber=MockTranscriber(),
        extractor=MockExtractor(),
        prompt=_make_prompt(),
        preprocessor=_make_preprocessor(),
        diarizer=_make_diarizer(),
    )

    assert isinstance(case, PipelineCaseResult)
    assert case.caller_info.first_name == "Max"
    assert case.caller_info.last_name == "Muster"
    assert case.transcript == "Max Muster max@test.com 015201234567"


# ---------------------------------------------------------------------------
# Step 15 – File guard: missing file → skip + null CallerInfo, transcript None
# ---------------------------------------------------------------------------
async def test_run_single_missing_file(tmp_path: Path) -> None:
    audio = AudioInput(id="c1", file=tmp_path / "missing.wav")

    case = await run_single(
        audio,
        _make_config(),
        _make_logger(),
        transcriber=MockTranscriber(),
        extractor=MockExtractor(),
        prompt=_make_prompt(),
        preprocessor=_make_preprocessor(),
        diarizer=_make_diarizer(),
    )

    assert case.caller_info.first_name is None
    assert case.caller_info.last_name is None
    assert case.transcript is None


# ---------------------------------------------------------------------------
# Step 16 – Transcription failure → null CallerInfo, transcript None
# ---------------------------------------------------------------------------
async def test_run_single_transcription_failure(tmp_path: Path) -> None:
    wav = tmp_path / "c1.wav"
    wav.write_bytes(b"RIFF....")
    audio = AudioInput(id="c1", file=wav)

    case = await run_single(
        audio,
        _make_config(),
        _make_logger(),
        transcriber=FailingTranscriber(),
        extractor=MockExtractor(),
        prompt=_make_prompt(),
        preprocessor=_make_preprocessor(),
        diarizer=_make_diarizer(),
    )

    assert case.caller_info.first_name is None
    assert case.caller_info.last_name is None
    assert case.caller_info.email is None
    assert case.caller_info.phone_number is None
    assert case.transcript is None


# ---------------------------------------------------------------------------
# Step 17 – Extraction failure → null CallerInfo, but transcript IS preserved
# ---------------------------------------------------------------------------
async def test_run_single_extraction_failure(tmp_path: Path) -> None:
    wav = tmp_path / "c1.wav"
    wav.write_bytes(b"RIFF....")
    audio = AudioInput(id="c1", file=wav)

    case = await run_single(
        audio,
        _make_config(),
        _make_logger(),
        transcriber=MockTranscriber(),
        extractor=FailingExtractor(),
        prompt=_make_prompt(),
        preprocessor=_make_preprocessor(),
        diarizer=_make_diarizer(),
    )

    assert case.caller_info.first_name is None
    assert case.caller_info.last_name is None
    assert case.caller_info.email is None
    assert case.caller_info.phone_number is None
    # Transcript is preserved even when extraction fails
    assert case.transcript == "Max Muster max@test.com 015201234567"


# ---------------------------------------------------------------------------
# Step 18 – Integration smoke test: run_batch writes correct output shape
# ---------------------------------------------------------------------------
async def test_run_batch_output_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Satisfy API-key env-var check for "openai_llm"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    # Inject mock backends into the live registry dicts (same objects pipeline.py holds)
    monkeypatch.setitem(TRANSCRIPTION_REGISTRY, "openai_llm", MockTranscriber)
    monkeypatch.setitem(EXTRACTION_REGISTRY, "llm", MockExtractor)

    # Self-contained prompt file so the test is independent of cwd
    prompt_yaml = _write_prompt(tmp_path)

    # Two temp wav files
    wav1 = tmp_path / "c1.wav"
    wav2 = tmp_path / "c2.wav"
    wav1.write_bytes(b"RIFF....")
    wav2.write_bytes(b"RIFF....")
    inputs = [
        AudioInput(id="c1", file=wav1),
        AudioInput(id="c2", file=wav2),
    ]

    config = PipelineConfig.model_construct(
        transcriber="openai_llm",
        extractor="llm",
        sample="dev",
        extraction_only=False,
        transcriptions_path=None,
        diarization_enabled=False,
        gpu_enabled=False,
        langsmith_tracing=False,
        extractor_prompt_file=str(prompt_yaml),
        openai_llm_transcriber_model="gpt-4o-mini-transcribe",
        openai_llm_diarization_model="gpt-4o-transcribe-diarize",
        llm_extractor_model="gpt-4.1-mini",
        whisperx_model="large-v2",
    )

    output = await run_batch(inputs, config, output_dir=tmp_path)

    assert len(output.results) == 2
    assert isinstance(output.run_id, str) and output.run_id != ""
    assert isinstance(output.config_snapshot, dict)

    # cases is populated in-memory with 2 PipelineCaseResult objects
    assert len(output.cases) == 2
    assert all(isinstance(c, PipelineCaseResult) for c in output.cases)

    results_path = tmp_path / output.run_id / "results.json"
    assert results_path.exists()

    data = json.loads(results_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert len(data["results"]) == 2
    # cases must NOT appear in the serialised results.json
    assert "cases" not in data

    transcriptions_path = tmp_path / output.run_id / "transcriptions.json"
    artifact = TranscriptionArtifact.model_validate_json(
        transcriptions_path.read_text(encoding="utf-8")
    )
    assert [t.id for t in artifact.transcriptions] == ["c1", "c2"]
    assert all(
        t.raw_text == "Max Muster max@test.com 015201234567" for t in artifact.transcriptions
    )


async def test_run_batch_overwrites_current_transcriptions_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("phonebot.pipeline.make_run_id", lambda config: "fixed_run")
    monkeypatch.setitem(EXTRACTION_REGISTRY, "llm", MockExtractor)

    prompt_yaml = _write_prompt(tmp_path)
    wav = tmp_path / "c1.wav"
    wav.write_bytes(b"RIFF....")
    inputs = [AudioInput(id="c1", file=wav)]
    config = _make_config().model_copy(update={"extractor_prompt_file": str(prompt_yaml)})

    monkeypatch.setitem(TRANSCRIPTION_REGISTRY, "openai_llm", MockTranscriber)
    await run_batch(inputs, config, output_dir=tmp_path)

    monkeypatch.setitem(TRANSCRIPTION_REGISTRY, "openai_llm", AlternateTranscriber)
    await run_batch(inputs, config, output_dir=tmp_path)

    artifact = TranscriptionArtifact.model_validate_json(
        (tmp_path / "fixed_run" / "transcriptions.json").read_text(encoding="utf-8")
    )
    assert len(artifact.transcriptions) == 1
    assert artifact.transcriptions[0].raw_text == "Alternate transcript"


async def test_extraction_only_uses_transcripts_and_skips_transcriber(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    TranscriptEchoExtractor.seen_transcripts = []
    monkeypatch.setitem(TRANSCRIPTION_REGISTRY, "deepgram", ExplodingTranscriber)
    monkeypatch.setitem(EXTRACTION_REGISTRY, "presidio", TranscriptEchoExtractor)

    transcriptions_path = tmp_path / "transcriptions.json"
    transcriptions_path.write_text(
        TranscriptionArtifact(
            transcriptions=[TranscriptionResult(id="c1", raw_text="Saved transcript text")]
        ).model_dump_json(),
        encoding="utf-8",
    )
    prompt_yaml = _write_prompt(tmp_path)
    config = PipelineConfig.model_construct(
        transcriber="deepgram",
        extractor="presidio",
        sample="dev",
        extraction_only=True,
        transcriptions_path=str(transcriptions_path),
        diarization_enabled=False,
        gpu_enabled=False,
        langsmith_tracing=False,
        extractor_prompt_file=str(prompt_yaml),
        openai_llm_transcriber_model="gpt-4o-mini-transcribe",
        openai_llm_diarization_model="gpt-4o-transcribe-diarize",
        llm_extractor_model="gpt-4.1-mini",
        whisperx_model="large-v2",
    )

    output = await run_batch(
        [AudioInput(id="c1", file=tmp_path / "missing.wav")],
        config,
        output_dir=tmp_path,
    )

    assert output.results[0].first_name == "Saved"
    assert output.cases[0].transcript == "Saved transcript text"
    assert TranscriptEchoExtractor.seen_transcripts == ["Saved transcript text"]


async def test_extraction_only_fails_fast_for_missing_transcript(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(EXTRACTION_REGISTRY, "presidio", TranscriptEchoExtractor)
    transcriptions_path = tmp_path / "transcriptions.json"
    transcriptions_path.write_text(
        TranscriptionArtifact(transcriptions=[]).model_dump_json(),
        encoding="utf-8",
    )
    prompt_yaml = _write_prompt(tmp_path)
    config = PipelineConfig.model_construct(
        transcriber="deepgram",
        extractor="presidio",
        sample="dev",
        extraction_only=True,
        transcriptions_path=str(transcriptions_path),
        diarization_enabled=False,
        gpu_enabled=False,
        langsmith_tracing=False,
        extractor_prompt_file=str(prompt_yaml),
    )

    with pytest.raises(ValueError, match="Missing transcript.*c1"):
        await run_batch(
            [AudioInput(id="c1", file=tmp_path / "missing.wav")],
            config,
            output_dir=tmp_path,
        )


async def test_extraction_only_preserves_transcript_on_extraction_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(EXTRACTION_REGISTRY, "presidio", FailingExtractor)
    transcriptions_path = tmp_path / "transcriptions.json"
    transcriptions_path.write_text(
        TranscriptionArtifact(
            transcriptions=[TranscriptionResult(id="c1", raw_text="Saved transcript text")]
        ).model_dump_json(),
        encoding="utf-8",
    )
    prompt_yaml = _write_prompt(tmp_path)
    config = PipelineConfig.model_construct(
        transcriber="deepgram",
        extractor="presidio",
        sample="dev",
        extraction_only=True,
        transcriptions_path=str(transcriptions_path),
        diarization_enabled=False,
        gpu_enabled=False,
        langsmith_tracing=False,
        extractor_prompt_file=str(prompt_yaml),
    )

    output = await run_batch(
        [AudioInput(id="c1", file=tmp_path / "missing.wav")],
        config,
        output_dir=tmp_path,
    )

    assert output.results[0].first_name is None
    assert output.cases[0].transcript == "Saved transcript text"
