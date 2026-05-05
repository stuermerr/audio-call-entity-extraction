"""Optional audio preprocessing layer applied before transcription.

Provides ``PreprocessorBase`` as the passthrough default and ``DeepFilterPreprocessor``
(GPU image only) for ONNX-based noise suppression via FastEnhancer.  The active
preprocessor is selected via ``denoising_enabled`` in ``PipelineConfig``.
"""
