from __future__ import annotations

from typing import ClassVar

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.observability import maybe_traceable
from phonebot.schemas import AudioInput, SpeakerSegment, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class OpenAILLMTranscriber(TranscriberBase):
    """Transcription backend using OpenAI's gpt-4o-mini-transcribe (default) or
    gpt-4o-transcribe-diarize (when diarization_enabled=True).

    Outer @maybe_traceable traces the entire call (including all retry attempts)
    as one LangSmith span. Inner @retry handles transient API failures with
    exponential back-off (up to 3 attempts); reraise=True re-raises the last
    exception on exhaustion so the pipeline can catch it and produce null CallerInfo.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        self._diarization_enabled = config.diarization_enabled
        self._client = openai.AsyncOpenAI()

    @maybe_traceable("openai_llm.transcribe")
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        with open(audio.file, "rb") as fh:
            if self._diarization_enabled:
                response = await self._client.audio.transcriptions.create(  # type: ignore[call-overload]
                    model="gpt-4o-transcribe-diarize",
                    file=fh,
                    response_format="diarized_json",
                    chunking_strategy="auto",
                )
                raw_segments = getattr(response, "segments", []) or []
                segments = [
                    SpeakerSegment(
                        speaker=seg.speaker,
                        start=seg.start,
                        end=seg.end,
                        text=seg.text,
                    )
                    for seg in raw_segments
                ]
                # Use response.text if present, else join segment texts.
                raw_text: str = getattr(response, "text", None) or " ".join(
                    seg.text for seg in segments
                )
                return TranscriptionResult(
                    id=audio.id,
                    raw_text=raw_text,
                    segments=segments,
                    supports_diarization=True,
                )
            else:
                response = await self._client.audio.transcriptions.create(
                    model="gpt-4o-mini-transcribe",
                    file=fh,
                    response_format="json",
                )
                # response.text is a str for json format; guard against SDK fallback.
                text_val = response.text
                raw_text = text_val if isinstance(text_val, str) else str(text_val)
                return TranscriptionResult(
                    id=audio.id,
                    raw_text=raw_text,
                    segments=None,
                    supports_diarization=False,
                )
