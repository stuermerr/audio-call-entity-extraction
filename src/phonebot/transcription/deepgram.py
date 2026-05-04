from __future__ import annotations

from typing import ClassVar

from deepgram import AsyncDeepgramClient  # type: ignore[import-untyped]
from tenacity import retry, stop_after_attempt, wait_exponential

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput, SpeakerSegment, TranscriptionResult
from phonebot.transcription.base import TranscriberBase


class DeepgramTranscriber(TranscriberBase):
    """Deepgram transcription backend using the official deepgram-sdk.

    Supports both plain transcript (non-diarized) and speaker-segmented
    (diarized) paths via the Deepgram pre-recorded audio API.

    @retry handles transient API failures with exponential back-off (up to
    3 attempts); reraise=True re-raises the last exception on exhaustion so the
    pipeline can catch it and produce null CallerInfo.
    """

    supports_diarization: ClassVar[bool] = True

    def __init__(self, config: PipelineConfig) -> None:
        self._diarization_enabled = config.diarization_enabled
        self._model = config.deepgram_model
        self._language = _deepgram_language_arg(config.deepgram_language)
        self._smart_format = config.deepgram_smart_format
        self._client = AsyncDeepgramClient(api_key=config.deepgram_api_key)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    async def transcribe(self, audio: AudioInput) -> TranscriptionResult:
        """Transcribe the given audio and return a TranscriptionResult.

        Reads the audio file as bytes and submits it to the Deepgram pre-recorded
        audio API (listen/v1/media). When diarization is enabled, utterance-level
        speaker segments are extracted and returned alongside the full transcript.

        The ``language`` parameter is omitted entirely when ``deepgram_language``
        is ``"default"`` so the SDK default (``en``) is preserved.
        """
        audio_bytes = audio.file.read_bytes()

        call_kwargs: dict = {  # type: ignore[type-arg]
            "request": audio_bytes,
            "model": self._model,
            "smart_format": self._smart_format,
        }
        if self._language is not None:
            call_kwargs["language"] = self._language
        if self._diarization_enabled:
            call_kwargs["diarize"] = True
            call_kwargs["utterances"] = True

        response = await self._client.listen.v1.media.transcribe_file(**call_kwargs)

        channel = response.results.channels[0]
        raw_text: str = channel.alternatives[0].transcript or ""

        if self._diarization_enabled:
            segments = [
                SpeakerSegment(
                    speaker=f"SPEAKER_{u.speaker:02d}",
                    start=u.start or 0.0,
                    end=u.end or 0.0,
                    text=u.transcript or "",
                )
                for u in (response.results.utterances or [])
            ]
            return TranscriptionResult(
                id=audio.id,
                raw_text=raw_text,
                segments=segments,
                supports_diarization=True,
            )

        return TranscriptionResult(
            id=audio.id,
            raw_text=raw_text,
            segments=None,
            supports_diarization=False,
        )


def _deepgram_language_arg(config_language: str) -> str | None:
    """Translate the public config sentinel to Deepgram's omitted-language default."""
    if config_language == "default":
        return None
    return config_language
