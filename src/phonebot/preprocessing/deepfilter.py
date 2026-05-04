"""DeepFilterNet-based audio preprocessor.

This module is only usable when the ``denoise`` dependency group is installed::

    uv sync --group denoise

It is imported lazily inside ``pipeline.py`` when ``config.denoising_enabled=True``,
so the ``deepfilternet`` package (and its transitive torch dependency) are never
imported on the default CPU-safe code path.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from df.enhance import enhance, init_df, load_audio, save_audio  # type: ignore[import-untyped]

from phonebot.preprocessing.base import PreprocessorBase
from phonebot.schemas import AudioInput

logger = logging.getLogger(__name__)


class DeepFilterPreprocessor(PreprocessorBase):
    """Audio preprocessor that denoises recordings with DeepFilterNet.

    The model is loaded once in ``__init__`` (not per recording) to avoid
    repeated cold-start cost (~1–3 s per load).  Enhanced WAV files are
    written to ``work_dir`` so they can be inspected for debugging.

    On any exception during enhancement the original ``AudioInput`` is
    returned unchanged — a denoising failure must never kill a batch.
    """

    def __init__(self, work_dir: Path) -> None:
        work_dir.mkdir(parents=True, exist_ok=True)
        self._work_dir = work_dir
        self._model, self._df_state, _ = init_df()

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

    def _enhance_sync(self, audio_id: str, src: Path, dst: Path) -> AudioInput:
        """Blocking enhancement call; run via ``asyncio.to_thread``."""
        audio_array, _ = load_audio(str(src), sr=self._df_state.sr())
        enhanced = enhance(self._model, self._df_state, audio_array)
        save_audio(str(dst), enhanced, self._df_state.sr())
        return AudioInput(id=audio_id, file=dst)
