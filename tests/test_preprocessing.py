"""Unit tests for FastEnhancerPreprocessor.

The optional GPU/audio packages are mocked before importing the module so these
tests do not require ``onnxruntime-gpu`` or real CUDA hardware.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from phonebot.schemas import AudioInput


def _install_fake_fastenhancer_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    available_providers: list[str],
    session_providers: list[str] | None = None,
) -> MagicMock:
    """Install fake optional modules and return the InferenceSession mock."""
    session = MagicMock()
    session.get_providers.return_value = session_providers or available_providers
    session.get_inputs.return_value = [
        types.SimpleNamespace(name="wav_in", shape=(1, 256)),
        types.SimpleNamespace(name="cache_in_0", shape=(1, 2, 3)),
    ]

    ort = types.ModuleType("onnxruntime")
    ort.get_available_providers = MagicMock(return_value=available_providers)  # type: ignore[attr-defined]
    ort.InferenceSession = MagicMock(return_value=session)  # type: ignore[attr-defined]

    librosa = types.ModuleType("librosa")
    librosa.load = MagicMock()  # type: ignore[attr-defined]

    numpy = types.ModuleType("numpy")
    numpy.float32 = "float32"  # type: ignore[attr-defined]
    numpy.int16 = "int16"  # type: ignore[attr-defined]
    numpy.ndarray = object  # type: ignore[attr-defined]
    numpy.zeros = MagicMock()  # type: ignore[attr-defined]
    numpy.pad = MagicMock()  # type: ignore[attr-defined]
    numpy.clip = MagicMock()  # type: ignore[attr-defined]
    numpy.concatenate = MagicMock()  # type: ignore[attr-defined]

    scipy = types.ModuleType("scipy")
    scipy_io = types.ModuleType("scipy.io")
    scipy_wavfile = types.ModuleType("scipy.io.wavfile")
    scipy_wavfile.write = MagicMock()  # type: ignore[attr-defined]
    scipy_io.wavfile = scipy_wavfile  # type: ignore[attr-defined]
    scipy.io = scipy_io  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    monkeypatch.setitem(sys.modules, "numpy", numpy)
    monkeypatch.setitem(sys.modules, "scipy", scipy)
    monkeypatch.setitem(sys.modules, "scipy.io", scipy_io)
    monkeypatch.setitem(sys.modules, "scipy.io.wavfile", scipy_wavfile)
    monkeypatch.delitem(sys.modules, "phonebot.preprocessing.fastenhancer", raising=False)
    return ort.InferenceSession  # type: ignore[attr-defined]


def test_fastenhancer_rejects_cpu_only_onnxruntime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fastenhancer_deps(
        monkeypatch,
        available_providers=["AzureExecutionProvider", "CPUExecutionProvider"],
    )

    from phonebot.preprocessing.fastenhancer import FastEnhancerPreprocessor

    model_path = tmp_path / "fastenhancer.onnx"
    model_path.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="CUDAExecutionProvider"):
        FastEnhancerPreprocessor(tmp_path / "preprocessed", model_path=model_path)


def test_fastenhancer_initializes_with_cuda_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inference_session = _install_fake_fastenhancer_deps(
        monkeypatch,
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    from phonebot.preprocessing.fastenhancer import FastEnhancerPreprocessor

    model_path = tmp_path / "fastenhancer.onnx"
    model_path.write_bytes(b"fake")
    preprocessor = FastEnhancerPreprocessor(tmp_path / "preprocessed", model_path=model_path)

    inference_session.assert_called_once_with(
        str(model_path),
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    assert preprocessor._cache_meta == [("cache_in_0", (1, 2, 3))]


def test_fastenhancer_rejects_session_without_cuda_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fastenhancer_deps(
        monkeypatch,
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        session_providers=["CPUExecutionProvider"],
    )

    from phonebot.preprocessing.fastenhancer import FastEnhancerPreprocessor

    model_path = tmp_path / "fastenhancer.onnx"
    model_path.write_bytes(b"fake")

    with pytest.raises(RuntimeError, match="did not activate CUDAExecutionProvider"):
        FastEnhancerPreprocessor(tmp_path / "preprocessed", model_path=model_path)


async def test_fastenhancer_preprocess_failure_returns_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_fastenhancer_deps(
        monkeypatch,
        available_providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )

    from phonebot.preprocessing.fastenhancer import FastEnhancerPreprocessor

    model_path = tmp_path / "fastenhancer.onnx"
    model_path.write_bytes(b"fake")
    preprocessor = FastEnhancerPreprocessor(tmp_path / "preprocessed", model_path=model_path)
    preprocessor._enhance_sync = MagicMock(side_effect=RuntimeError("GPU OOM"))  # type: ignore[method-assign]

    src = tmp_path / "call_01.wav"
    src.write_bytes(b"RIFF....")
    audio = AudioInput(id="call_01", file=src)

    result = await preprocessor.preprocess(audio)

    assert result is audio
