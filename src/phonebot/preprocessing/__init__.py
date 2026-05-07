"""Optional audio preprocessing layer applied before transcription.

Provides ``PreprocessorBase`` as the passthrough default and ``FastEnhancerPreprocessor``
(GPU image only) for ONNX-based noise suppression.  The active
preprocessor is selected via ``denoising_enabled`` in ``PipelineConfig``.
"""
