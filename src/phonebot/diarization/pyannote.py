from __future__ import annotations

import logging

from phonebot.schemas import TranscriptionResult

logger = logging.getLogger(__name__)

# TODO: full pyannote.audio integration deferred.
#   Dependencies required when implemented:
#     - pyannote.audio >= 3.x  (pip: pyannote.audio)
#     - A HuggingFace account with access to the gated model
#       pyannote/speaker-diarization-3.x; set HF_TOKEN env var.
#   What needs doing:
#     1. Load Pipeline("pyannote/speaker-diarization-3.x", use_auth_token=HF_TOKEN)
#     2. Run pipeline on the audio file to produce a pyannote Annotation.
#     3. Map annotation segments onto TranscriptionResult.segments, aligning
#        speaker labels to SpeakerSegment.speaker strings.
#     4. Resolve speaker-label → caller identity (e.g. SPEAKER_00 = "caller")
#        based on call-centre conventions or a voice-print reference.
#   Until then, this class is a passthrough stub and emits a runtime warning
#   so the limitation is visible in run.log without raising errors.


class PyAnnoteDiarizer:
    """Stub diarizer backed by pyannote.audio (not yet implemented).

    Returns *result* unchanged.  Speaker-label → caller identity mapping
    is unimplemented; ``raw_text`` remains the uniform contract to the
    extraction stage.
    """

    async def diarize(self, result: TranscriptionResult) -> TranscriptionResult:
        """Return *result* unchanged (passthrough stub).

        Emits a WARNING so the limitation is visible in logs.
        """
        logger.warning(
            "PyAnnoteDiarizer is a stub: speaker-label → caller identity mapping "
            "is unimplemented.  raw_text is used as the fallback contract.  "
            "Install pyannote.audio and set HF_TOKEN to enable full diarization."
        )
        return result
