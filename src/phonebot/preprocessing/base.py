from __future__ import annotations

from phonebot.schemas import AudioInput


class PreprocessorBase:
    """Passthrough audio preprocessor.

    In MVP this class is a no-op: it returns the AudioInput unchanged.
    Use ``DeepFilterPreprocessor`` (``phonebot.preprocessing.deepfilter``) when
    ``config.denoising_enabled=True`` for DeepFilterNet-based noise reduction.
    """

    async def preprocess(self, audio: AudioInput) -> AudioInput:
        """Return *audio* unchanged (passthrough)."""
        return audio
