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


class CallerInfo(BaseModel):
    id: str
    file: str
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    phone_number: str | None = None


class PipelineOutput(BaseModel):
    results: list[CallerInfo]
    run_id: str
    config_snapshot: dict[str, Any]


class EvalResult(BaseModel):
    id: str
    per_field: dict[str, bool]  # field_name → matched


class EvalReport(BaseModel):
    run_id: str
    per_entity_accuracy: dict[str, float]
    overall_accuracy: float
    results: list[EvalResult]
