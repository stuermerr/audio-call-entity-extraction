from __future__ import annotations

from phonebot.schemas import AudioInput


class PreprocessorBase:
    """Passthrough audio preprocessor.

    In MVP this class is a no-op: it returns the AudioInput unchanged.
    Use ``FastEnhancerPreprocessor`` (``phonebot.preprocessing.fastenhancer``) when
    ``config.denoising_enabled=True`` for GPU-backed ONNX noise reduction with
    no PyTorch dependency.
    """

    async def preprocess(self, audio: AudioInput) -> AudioInput:
        """Return *audio* unchanged (passthrough)."""
        return audio
