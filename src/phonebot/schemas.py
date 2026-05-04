from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class AudioInput(BaseModel):
    id: str
    file: Path


class SpeakerSegment(BaseModel):
    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    id: str
    raw_text: str
    segments: list[SpeakerSegment] | None = None
    supports_diarization: bool = False


class TranscriptionArtifact(BaseModel):
    transcriptions: list[TranscriptionResult]


class CallerInfo(BaseModel):
    id: str
    file: str
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone_number: str | None = None


class PipelineCaseResult(BaseModel):
    """Per-case pipeline result carrying the transcript alongside the extracted CallerInfo.

    ``id`` and ``file`` are intentionally omitted here — access via ``.caller_info.id``
    and ``.caller_info.file`` to avoid duplicated, potentially diverging values.

    ``transcript`` is ``None`` when transcription fails (or the file is missing);
    it is populated even if downstream extraction fails, so the raw transcript is
    always preserved for debugging.
    """

    caller_info: CallerInfo
    transcript: str | None


class PipelineOutput(BaseModel):
    results: list[CallerInfo]
    run_id: str
    config_snapshot: dict[str, Any]
    # Not serialised to results.json — excluded at call site via model_dump_json(exclude={'cases'}).
    cases: list[PipelineCaseResult] = []


class EvalResult(BaseModel):
    id: str
    per_field: dict[str, bool]  # field_name → matched


class EvalReport(BaseModel):
    run_id: str
    per_entity_accuracy: dict[str, float]
    overall_accuracy: float
    results: list[EvalResult]


class CaseReportEntry(BaseModel):
    """One row in case_report.json — combines pipeline data with evaluation results."""

    id: str
    file: str
    transcript: str | None
    predicted: CallerInfo
    expected: dict[str, Any] | None  # raw ground-truth dict; None if id was absent
    per_field: dict[str, bool]


class CaseReport(BaseModel):
    """Top-level structure of outputs/<run_id>/case_report.json."""

    run_id: str
    cases: list[CaseReportEntry]
