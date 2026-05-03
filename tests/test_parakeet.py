from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput
from phonebot.transcription.parakeet import ParakeetTranscriber

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcriber() -> ParakeetTranscriber:
    """Build a ParakeetTranscriber bypassing __init__ and inject mocked deps."""
    t = ParakeetTranscriber.__new__(ParakeetTranscriber)
    t._model = MagicMock()
    t._model_name = "nvidia/parakeet-tdt-0.6b-v3"
    return t


def _make_audio(tmp_path: Path) -> AudioInput:
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF....")
    return AudioInput(id="t1", file=wav)


# ---------------------------------------------------------------------------
# Test 1: gpu_enabled=False raises RuntimeError before any NeMo import
# ---------------------------------------------------------------------------


def test_gpu_disabled_raises() -> None:
    config = PipelineConfig.model_construct(
        gpu_enabled=False,
        parakeet_model="nvidia/parakeet-tdt-0.6b-v3",
    )
    with pytest.raises(RuntimeError, match="gpu_enabled"):
        ParakeetTranscriber(config)


# ---------------------------------------------------------------------------
# Test 2: primary success path — raw_text, segments=None, supports_diarization=False
# ---------------------------------------------------------------------------


def test_transcribe_non_diarized(tmp_path: Path) -> None:
    t = _make_transcriber()
    audio = _make_audio(tmp_path)

    hypothesis = MagicMock()
    hypothesis.text = "Hallo Welt"
    t._model.transcribe.return_value = [hypothesis]

    result = t._transcribe_sync(audio)

    assert result.id == "t1"
    assert result.raw_text == "Hallo Welt"
    assert result.segments is None
    assert result.supports_diarization is False

    t._model.transcribe.assert_called_once_with([str(audio.file)])


# ---------------------------------------------------------------------------
# Test 3: .strip() guards against leading/trailing whitespace from NeMo output
# ---------------------------------------------------------------------------


def test_transcribe_strips_whitespace(tmp_path: Path) -> None:
    t = _make_transcriber()
    audio = _make_audio(tmp_path)

    hypothesis = MagicMock()
    hypothesis.text = "  Guten Morgen  "
    t._model.transcribe.return_value = [hypothesis]

    result = t._transcribe_sync(audio)

    assert result.raw_text == "Guten Morgen"


# ---------------------------------------------------------------------------
# Test 4: blocking GPU call is offloaded to a thread via asyncio.to_thread
# ---------------------------------------------------------------------------


def test_transcribe_dispatches_to_thread(tmp_path: Path) -> None:
    t = _make_transcriber()
    audio = _make_audio(tmp_path)

    with patch("asyncio.to_thread", new_callable=AsyncMock) as mock_to_thread:
        asyncio.run(t.transcribe(audio))

    mock_to_thread.assert_called_once_with(t._transcribe_sync, audio)
