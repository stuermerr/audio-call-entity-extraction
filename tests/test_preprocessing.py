"""Unit tests for DeepFilterPreprocessor.

These tests run without the ``deepfilternet`` package installed.  The
``df.enhance`` module is mocked at the ``sys.modules`` level before the
class is imported so no real DeepFilterNet weights or torch are needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from phonebot.schemas import AudioInput

# ---------------------------------------------------------------------------
# Helpers: build a minimal fake df.enhance module so the top-level import
# in deepfilter.py resolves without the real package being installed.
# ---------------------------------------------------------------------------


def _make_df_module() -> types.ModuleType:
    """Return a mock ``df.enhance`` module whose callables are MagicMocks."""
    df_enhance = types.ModuleType("df.enhance")
    df_enhance.init_df = MagicMock()  # type: ignore[attr-defined]
    df_enhance.load_audio = MagicMock()  # type: ignore[attr-defined]
    df_enhance.enhance = MagicMock()  # type: ignore[attr-defined]
    df_enhance.save_audio = MagicMock()  # type: ignore[attr-defined]

    df_pkg = types.ModuleType("df")
    df_pkg.enhance = df_enhance  # type: ignore[attr-defined]
    return df_enhance


def _inject_df(monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    """Inject fake ``df`` / ``df.enhance`` into sys.modules and return the mock."""
    df_enhance = _make_df_module()
    df_pkg = types.ModuleType("df")
    df_pkg.enhance = df_enhance  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "df", df_pkg)
    monkeypatch.setitem(sys.modules, "df.enhance", df_enhance)
    return df_enhance


# ---------------------------------------------------------------------------
# Test 1: happy path — enhanced WAV is returned with same id and _enhanced suffix
# ---------------------------------------------------------------------------


async def test_deepfilter_preprocess_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df_mod = _inject_df(monkeypatch)

    # Ensure the module is not cached from a previous import
    monkeypatch.delitem(sys.modules, "phonebot.preprocessing.deepfilter", raising=False)

    from phonebot.preprocessing.deepfilter import DeepFilterPreprocessor

    fake_state = MagicMock()
    fake_state.sr.return_value = 48000
    df_mod.init_df.return_value = (MagicMock(), fake_state, MagicMock())
    df_mod.load_audio.return_value = (MagicMock(), 48000)
    df_mod.enhance.return_value = MagicMock()

    work_dir = tmp_path / "preprocessed"
    preprocessor = DeepFilterPreprocessor(work_dir)

    src = tmp_path / "call_01.wav"
    src.write_bytes(b"RIFF....")
    audio = AudioInput(id="call_01", file=src)

    result = await preprocessor.preprocess(audio)

    assert result.id == "call_01"
    assert result.file == work_dir / "call_01_enhanced.wav"
    df_mod.save_audio.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2: enhancement failure → original audio returned (graceful degradation)
# ---------------------------------------------------------------------------


async def test_deepfilter_preprocess_failure_returns_original(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df_mod = _inject_df(monkeypatch)
    monkeypatch.delitem(sys.modules, "phonebot.preprocessing.deepfilter", raising=False)

    from phonebot.preprocessing.deepfilter import DeepFilterPreprocessor

    fake_state = MagicMock()
    fake_state.sr.return_value = 48000
    df_mod.init_df.return_value = (MagicMock(), fake_state, MagicMock())
    df_mod.load_audio.return_value = (MagicMock(), 48000)
    df_mod.enhance.side_effect = RuntimeError("GPU OOM")

    work_dir = tmp_path / "preprocessed"
    preprocessor = DeepFilterPreprocessor(work_dir)

    src = tmp_path / "call_01.wav"
    src.write_bytes(b"RIFF....")
    audio = AudioInput(id="call_01", file=src)

    result = await preprocessor.preprocess(audio)

    # Must be the exact same object (unchanged)
    assert result is audio


# ---------------------------------------------------------------------------
# Test 3: init_df is called exactly once even when preprocess is called twice
# ---------------------------------------------------------------------------


async def test_deepfilter_model_loaded_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    df_mod = _inject_df(monkeypatch)
    monkeypatch.delitem(sys.modules, "phonebot.preprocessing.deepfilter", raising=False)

    from phonebot.preprocessing.deepfilter import DeepFilterPreprocessor

    fake_state = MagicMock()
    fake_state.sr.return_value = 48000
    df_mod.init_df.return_value = (MagicMock(), fake_state, MagicMock())
    df_mod.load_audio.return_value = (MagicMock(), 48000)
    df_mod.enhance.return_value = MagicMock()

    work_dir = tmp_path / "preprocessed"
    preprocessor = DeepFilterPreprocessor(work_dir)

    for i in range(2):
        src = tmp_path / f"call_0{i}.wav"
        src.write_bytes(b"RIFF....")
        audio = AudioInput(id=f"call_0{i}", file=src)
        await preprocessor.preprocess(audio)

    df_mod.init_df.assert_called_once()
