from __future__ import annotations

from pathlib import Path

from phonebot.schemas import (
    AudioInput,
    CallerInfo,
    EvalReport,
    EvalResult,
    PipelineOutput,
    TranscriptionResult,
)


def test_audio_input_instantiation() -> None:
    """Path field accepts str coercion."""
    audio = AudioInput(id="call_01", file="data/recordings/call_01.wav")
    assert audio.id == "call_01"
    assert isinstance(audio.file, Path)
    assert audio.file == Path("data/recordings/call_01.wav")


def test_transcription_result_defaults() -> None:
    """segments=None and supports_diarization=False by default."""
    result = TranscriptionResult(id="call_01", raw_text="Hallo, ich bin Max.")
    assert result.segments is None
    assert result.supports_diarization is False


def test_caller_info_all_none() -> None:
    """All 4 entity fields default to None."""
    info = CallerInfo(id="call_01", file="call_01.wav")
    assert info.first_name is None
    assert info.last_name is None
    assert info.email is None
    assert info.phone_number is None


def test_pipeline_output_roundtrip() -> None:
    """.model_dump() → .model_validate() is lossless."""
    output = PipelineOutput(
        results=[CallerInfo(id="call_01", file="call_01.wav", first_name="Max")],
        run_id="20240101_120000_openai_llm_llm_dev",
        config_snapshot={"transcriber": "openai_llm", "extractor": "llm"},
    )
    dumped = output.model_dump()
    restored = PipelineOutput.model_validate(dumped)
    assert restored.run_id == output.run_id
    assert restored.results[0].first_name == "Max"
    assert restored.config_snapshot == output.config_snapshot


def test_eval_report_shape() -> None:
    """run_id str, per_entity_accuracy dict, overall_accuracy float, results list present."""
    report = EvalReport(
        run_id="20240101_120000_openai_llm_llm_dev",
        per_entity_accuracy={"first_name": 0.9, "last_name": 0.85},
        overall_accuracy=0.875,
        results=[EvalResult(id="call_01", per_field={"first_name": True, "last_name": False})],
    )
    assert isinstance(report.run_id, str)
    assert isinstance(report.per_entity_accuracy, dict)
    assert isinstance(report.overall_accuracy, float)
    assert isinstance(report.results, list)
    assert report.results[0].id == "call_01"
