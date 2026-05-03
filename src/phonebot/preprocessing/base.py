from __future__ import annotations

from phonebot.schemas import AudioInput


class PreprocessorBase:
    """Passthrough audio preprocessor.

    In MVP this class is a no-op: it returns the AudioInput unchanged.

    # TODO: future normalization hooks to implement here:
    #   - gain normalization (e.g. ffmpeg loudnorm to -23 LUFS)
    #   - resampling to 16 kHz mono (required by WhisperX / Parakeet)
    #   - noise reduction (e.g. noisereduce or RNNoise)
    # Each hook should be opt-in via PipelineConfig so the passthrough
    # behaviour is preserved when none are enabled.
    """

    async def preprocess(self, audio: AudioInput) -> AudioInput:
        """Return *audio* unchanged (passthrough)."""
        return audio
