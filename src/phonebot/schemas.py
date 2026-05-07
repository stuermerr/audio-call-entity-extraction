"""Pydantic models used across the phonebot pipeline.

Defines the data contracts for audio inputs, transcription outputs, extraction
results, evaluation reports, and all on-disk serialisation formats.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel


class AudioInput(BaseModel):
    """A single audio recording identified by a unique string id and its file path."""

    id: str
    file: Path


class SpeakerSegment(BaseModel):
    """One diarized speaker turn with its speaker label, time boundaries, and transcript text."""

    speaker: str
    start: float
    end: float
    text: str


class TranscriptionResult(BaseModel):
    """Raw transcription output for a single recording, optionally including diarized segments."""

    id: str
    raw_text: str
    segments: list[SpeakerSegment] | None = None
    supports_diarization: bool = False


class TranscriptionArtifact(BaseModel):
    """On-disk format of transcriptions.json — a list of TranscriptionResult objects."""

    transcriptions: list[TranscriptionResult]


class CallerInfo(BaseModel):
    """Extracted caller entity fields for a single recording.

    Entity fields (first_name, last_name, email, phone_number) are nullable;
    ``None`` indicates the field was not extracted.
    """

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


class ExtractedFields(BaseModel):
    """Extracted entity fields stored in results.json (without id/file metadata)."""

    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone_number: str | None = None


class RecordingResult(BaseModel):
    id: str
    file: str
    extracted: ExtractedFields


class ResultsFile(BaseModel):
    """On-disk format of output/<run_id>/results.json — mirrors ground_truth.json structure."""

    recordings: list[RecordingResult]


class PipelineOutput(BaseModel):
    """Aggregated output produced by ``pipeline.run_batch``, passed to downstream consumers."""

    results: list[CallerInfo]
    run_id: str
    config_snapshot: dict[str, Any]
    cases: list[PipelineCaseResult] = []


class EvalResult(BaseModel):
    """Per-recording evaluation outcome: a boolean match flag for each entity field."""

    id: str
    per_field: dict[str, bool]  # field_name → matched


class EvalReport(BaseModel):
    """Aggregated evaluation report written to ``outputs/<run_id>/eval.json``."""

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
