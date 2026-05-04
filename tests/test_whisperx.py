from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from phonebot.config import PipelineConfig
from phonebot.schemas import AudioInput
from phonebot.transcription.whisperx import WhisperXTranscriber, _whisperx_language_arg

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_transcriber(*, diarization_enabled: bool = False) -> WhisperXTranscriber:
    """Build a WhisperXTranscriber bypassing __init__ and inject mocked deps."""
    t = WhisperXTranscriber.__new__(WhisperXTranscriber)
    t._device = "cuda"
    t._model_name = "large-v2"
    t._compute_type = "float16"
    t._language = None
    t._batch_size = 16
    t._diarization_enabled = diarization_enabled
    t._hf_token = "fake-hf-token"
    t._wx = MagicMock()
    t._wxd = MagicMock()
    t._model = MagicMock()
    return t


def _make_audio(tmp_path: Path) -> AudioInput:
    wav = tmp_path / "test.wav"
    wav.write_bytes(b"RIFF....")
    return AudioInput(id="t1", file=wav)


# ---------------------------------------------------------------------------
# Test 1: gpu_enabled=False raises RuntimeError before any whisperx import
# ---------------------------------------------------------------------------


def test_gpu_disabled_raises() -> None:
    config = PipelineConfig.model_construct(
        gpu_enabled=False,
        whisperx_model="large-v2",
        whisperx_compute_type="float16",
        whisperx_language="auto",
        diarization_enabled=False,
        hf_token="",
    )
    with pytest.raises(RuntimeError, match="gpu_enabled"):
        WhisperXTranscriber(config)


def test_whisperx_language_auto_maps_to_auto_detection() -> None:
    assert _whisperx_language_arg("auto") is None
    assert _whisperx_language_arg("de") == "de"


# ---------------------------------------------------------------------------
# Test 2: non-diarized path — segments=None, raw_text joined, no diarize calls
# ---------------------------------------------------------------------------


def test_transcribe_non_diarized(tmp_path: Path) -> None:
    t = _make_transcriber(diarization_enabled=False)
    audio = _make_audio(tmp_path)

    # load_audio returns a numpy array stand-in
    t._wx.load_audio.return_value = MagicMock()
    t._model.transcribe.return_value = {
        "segments": [{"text": " Hallo Welt", "start": 0.0, "end": 1.5}],
        "language": "de",
    }

    result = t._transcribe_sync(audio)

    assert result.id == "t1"
    assert result.raw_text == "Hallo Welt"
    assert result.segments is None
    assert result.supports_diarization is False

    # Diarization machinery must not have been touched
    t._wx.load_align_model.assert_not_called()
    t._wxd.DiarizationPipeline.assert_not_called()


# ---------------------------------------------------------------------------
# Test 3: diarized path — segments populated, speaker labels assigned
# ---------------------------------------------------------------------------


def test_transcribe_diarized(tmp_path: Path) -> None:
    t = _make_transcriber(diarization_enabled=True)
    audio = _make_audio(tmp_path)

    audio_array = MagicMock()
    t._wx.load_audio.return_value = audio_array
    t._model.transcribe.return_value = {
        "segments": [{"text": " Hallo", "start": 0.0, "end": 1.0}],
        "language": "de",
    }

    aligned_result = {"segments": [{"text": "Hallo", "start": 0.0, "end": 1.0}], "language": "de"}
    t._wx.load_align_model.return_value = (MagicMock(), MagicMock())
    t._wx.align.return_value = aligned_result

    diarize_segments = MagicMock()
    t._wxd.DiarizationPipeline.return_value = MagicMock(return_value=diarize_segments)

    t._wx.assign_word_speakers.return_value = {
        "segments": [{"text": "Hallo", "start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}],
        "language": "de",
    }

    # torch is imported lazily inside _transcribe_sync with a local `import torch`.
    # Inject a lightweight fake into sys.modules so the import doesn't fail when
    # the real torch package isn't installed in the test environment.
    import sys
    import types

    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = MagicMock()  # type: ignore[attr-defined]
    with patch.dict(sys.modules, {"torch": fake_torch}):
        result = t._transcribe_sync(audio)

    assert result.supports_diarization is True
    assert len(result.segments) == 1  # type: ignore[arg-type]
    assert result.segments[0].speaker == "SPEAKER_00"  # type: ignore[index]
    assert result.segments[0].text == "Hallo"  # type: ignore[index]
    assert result.raw_text == "Hallo"
