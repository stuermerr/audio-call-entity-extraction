from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from phonebot.schemas import AudioInput
from phonebot.transcription.deepgram import DeepgramTranscriber, _deepgram_language_arg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcriber(
    *,
    diarization_enabled: bool = False,
    language: str | None = None,
) -> DeepgramTranscriber:
    """Build a DeepgramTranscriber bypassing __init__ and inject mocked deps."""
    t = DeepgramTranscriber.__new__(DeepgramTranscriber)
    t._diarization_enabled = diarization_enabled
    t._model = "nova-3"
    t._language = language
    t._smart_format = True
    t._client = MagicMock()
    return t


def _make_audio(tmp_path: Path) -> AudioInput:
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF....")
    return AudioInput(id="t1", file=wav)


def _make_mock_response(transcript: str, utterances: list | None = None) -> MagicMock:
    """Build a mock Deepgram v7 API response matching ListenV1Response structure."""
    alt = MagicMock()
    alt.transcript = transcript
    channel = MagicMock()
    channel.alternatives = [alt]
    results = MagicMock()
    results.channels = [channel]
    results.utterances = utterances
    response = MagicMock()
    response.results = results
    return response


def _mock_transcribe_file(t: DeepgramTranscriber, response: MagicMock) -> AsyncMock:
    """Wire an AsyncMock for client.listen.v1.media.transcribe_file."""
    mock = AsyncMock(return_value=response)
    t._client.listen.v1.media.transcribe_file = mock  # type: ignore[attr-defined]
    return mock


# ---------------------------------------------------------------------------
# Test 1: non-diarized path — segments=None, raw_text returned
# ---------------------------------------------------------------------------


def test_transcribe_non_diarized(tmp_path: Path) -> None:
    t = _make_transcriber(diarization_enabled=False)
    audio = _make_audio(tmp_path)

    mock = _mock_transcribe_file(t, _make_mock_response("Guten Tag"))

    result = asyncio.run(t.transcribe(audio))

    assert result.raw_text == "Guten Tag"
    assert result.segments is None
    assert result.supports_diarization is False
    assert result.id == "t1"
    mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 2: diarized path — segments populated, speaker labels formatted correctly
# ---------------------------------------------------------------------------


def test_transcribe_diarized(tmp_path: Path) -> None:
    t = _make_transcriber(diarization_enabled=True)
    audio = _make_audio(tmp_path)

    utterance = MagicMock()
    utterance.speaker = 0
    utterance.start = 0.0
    utterance.end = 1.2
    utterance.transcript = "Hallo"

    mock = _mock_transcribe_file(t, _make_mock_response("Hallo", utterances=[utterance]))

    result = asyncio.run(t.transcribe(audio))

    assert result.supports_diarization is True
    assert len(result.segments) == 1  # type: ignore[arg-type]
    assert result.segments[0].speaker == "SPEAKER_00"  # type: ignore[index]
    assert result.segments[0].text == "Hallo"  # type: ignore[index]
    assert result.raw_text == "Hallo"
    mock.assert_awaited_once()


# ---------------------------------------------------------------------------
# Test 3: default language sentinel preserves SDK default
# ---------------------------------------------------------------------------


def test_deepgram_language_default_maps_to_sdk_default() -> None:
    assert _deepgram_language_arg("default") is None
    assert _deepgram_language_arg("de") == "de"


def test_language_omitted_for_default(tmp_path: Path) -> None:
    """When _language is None, language must not appear in the transcribe_file kwargs."""
    t = _make_transcriber(diarization_enabled=False, language=_deepgram_language_arg("default"))
    audio = _make_audio(tmp_path)

    mock = _mock_transcribe_file(t, _make_mock_response("Hello"))

    asyncio.run(t.transcribe(audio))

    assert mock.called
    assert "language" not in mock.call_args.kwargs
