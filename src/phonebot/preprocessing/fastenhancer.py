"""FastEnhancer ONNX-based audio preprocessor.

This module is only usable when the ``gpu`` dependency group is installed::

    uv sync --python 3.11 --group gpu

FastEnhancer uses ONNX Runtime GPU (no PyTorch dependency), so it is fully
compatible with whisperx's PyTorch stack when ``onnxruntime-gpu`` is the only
installed ONNX Runtime distribution.

The default model is the ``onnx-dns-v1.0.0`` Base variant (16 kHz, DNS
Challenge), which is well-suited for telephone recordings.  It is downloaded
once to ``~/.cache/phonebot/fastenhancer/`` on first use.

It is imported lazily inside ``pipeline.py`` when
``config.denoising_enabled=True``, so the ``onnxruntime`` / ``librosa``
packages are never imported on the default code path.
"""

from __future__ import annotations

import asyncio
import logging
import urllib.request
from pathlib import Path

import librosa
import numpy as np
import onnxruntime as ort
import scipy.io.wavfile

from phonebot.preprocessing.base import PreprocessorBase
from phonebot.schemas import AudioInput

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default model: DNS-v1.0.0 Base — 16 kHz, trained on DNS Challenge data,
# recommended for telephone audio.  T/B/S all share the same n_fft/hop_size.
# ---------------------------------------------------------------------------
_DEFAULT_MODEL_URL = (
    "https://github.com/aask1357/fastenhancer/releases/download/onnx-dns-v1.0.0/fastenhancer_b.onnx"
)
_DEFAULT_CACHE_DIR = Path.home() / ".cache" / "phonebot" / "fastenhancer"

_N_FFT = 512
_HOP_SIZE = 256
_SAMPLE_RATE = 16_000


class FastEnhancerPreprocessor(PreprocessorBase):
    """Audio preprocessor that denoises recordings with FastEnhancer (ONNX).

    Uses FastEnhancer's wav2wav ONNX model through ONNX Runtime GPU. CPU-only
    ONNX Runtime is rejected at startup because denoising is part of the local
    GPU path in this project.

    The ONNX session is initialised once in ``__init__`` (not per recording)
    to avoid repeated cold-start overhead.  Enhanced WAV files are written to
    ``work_dir`` so they can be inspected for debugging.

    On any exception during enhancement the original ``AudioInput`` is
    returned unchanged — a denoising failure must never kill a batch.
    """

    def __init__(
        self,
        work_dir: Path,
        *,
        model_path: Path | None = None,
        model_url: str = _DEFAULT_MODEL_URL,
        cache_dir: Path = _DEFAULT_CACHE_DIR,
        hop_size: int = _HOP_SIZE,
    ) -> None:
        # Preload CUDA/cuDNN shared libraries from PyPI nvidia-* wheels before
        # any ORT API call that triggers provider library loading.
        # Must come first: get_available_providers() and InferenceSession() both
        # trigger lazy provider library loading, so preload_dlls must precede both.
        if hasattr(ort, "preload_dlls"):
            ort.preload_dlls(cuda=True, cudnn=True, msvc=False, directory="")  # type: ignore[attr-defined]

        work_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = work_dir
        self._hop_size = hop_size

        available_providers = ort.get_available_providers()
        if "CUDAExecutionProvider" not in available_providers:
            raise RuntimeError(
                "FastEnhancer denoising requires onnxruntime-gpu with "
                "CUDAExecutionProvider available; got providers: "
                f"{available_providers}"
            )

        if model_path is None:
            model_path = self._ensure_model(model_url, cache_dir)

        logger.info("Loading FastEnhancer ONNX model from %s", model_path)
        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        session_providers = self._session.get_providers()
        if "CUDAExecutionProvider" not in session_providers:
            raise RuntimeError(
                "FastEnhancer ONNX session did not activate CUDAExecutionProvider; "
                f"session providers: {session_providers}"
            )

        # Cache input names/shapes — all session inputs that begin with "cache_in_"
        self._cache_meta: list[tuple[str, tuple[int, ...]]] = [
            (inp.name, tuple(inp.shape))
            for inp in self._session.get_inputs()
            if inp.name.startswith("cache_in_")
        ]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ensure_model(url: str, cache_dir: Path) -> Path:
        """Download the ONNX model to *cache_dir* if not present."""
        cache_dir.mkdir(parents=True, exist_ok=True)
        filename = url.split("/")[-1]
        local_path = cache_dir / filename
        if not local_path.exists():
            logger.info("Downloading FastEnhancer model: %s → %s", url, local_path)
            urllib.request.urlretrieve(url, local_path)
            logger.info("Model download complete")
        return local_path

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def preprocess(self, audio: AudioInput) -> AudioInput:
        """Denoise *audio* and return a new ``AudioInput`` pointing at the enhanced file.

        Falls back to the original *audio* on any error so the pipeline can
        continue without denoising.
        """
        out_path = self._work_dir / f"{audio.id}_enhanced.wav"
        try:
            result = await asyncio.to_thread(self._enhance_sync, audio.id, audio.file, out_path)
        except Exception:
            logger.warning(
                "Denoising failed for %s — using original audio", audio.id, exc_info=True
            )
            return audio
        return result

    # ------------------------------------------------------------------
    # Synchronous inference (runs in a thread via asyncio.to_thread)
    # ------------------------------------------------------------------

    def _enhance_sync(self, audio_id: str, src: Path, dst: Path) -> AudioInput:
        """Blocking enhancement; called via ``asyncio.to_thread``."""
        # Load and resample to 16 kHz mono
        wav, _ = librosa.load(str(src), sr=_SAMPLE_RATE, mono=True)
        wav = wav.reshape(1, -1).astype(np.float32)  # (1, T)
        wav = np.clip(wav, -1.0, 1.0)
        length = wav.shape[-1]

        # Pad so the sliding window always has a full frame at the end
        wav_padded = np.pad(wav, ((0, 0), (0, _N_FFT)), mode="constant")

        # Initialise stateful cache tensors (one per call — not shared across files)
        cache: dict[str, np.ndarray] = {
            name: np.zeros(shape, dtype=np.float32) for name, shape in self._cache_meta
        }

        wav_out_chunks: list[np.ndarray] = []
        for start in range(0, length + _N_FFT - self._hop_size, self._hop_size):
            chunk = wav_padded[:, start : start + self._hop_size]
            if chunk.shape[-1] < self._hop_size:
                # Pad the last partial chunk
                chunk = np.pad(
                    chunk, ((0, 0), (0, self._hop_size - chunk.shape[-1])), mode="constant"
                )
            feed: dict[str, np.ndarray] = {"wav_in": chunk, **cache}
            outputs = self._session.run(None, feed)
            wav_out_chunks.append(outputs[0][0])  # first output: (hop_size,)
            # Update rolling cache from remaining outputs
            for j, (name, _) in enumerate(self._cache_meta):
                cache[name] = outputs[j + 1]

        wav_out = np.concatenate(wav_out_chunks, axis=0)
        # Trim: model introduces a _N_FFT - hop_size lookahead
        start_idx = _N_FFT - self._hop_size
        wav_out = wav_out[start_idx : start_idx + length]
        wav_out = np.clip(wav_out, -1.0, 1.0)

        # Write 16-bit PCM WAV at 16 kHz
        wav_int16 = (wav_out * 32767).astype(np.int16)
        scipy.io.wavfile.write(str(dst), _SAMPLE_RATE, wav_int16)
        return AudioInput(id=audio_id, file=dst)
