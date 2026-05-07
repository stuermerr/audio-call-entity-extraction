from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


async def _run(args: argparse.Namespace) -> None:
    load_dotenv()
    if args.langsmith_tracing:
        os.environ["LANGSMITH_TRACING"] = "true"

    # Import after applying debug env flags. Some tracing decorators are chosen
    # at module import time from os.environ.
    from phonebot.config import PipelineConfig
    from phonebot.evaluation import Evaluator
    from phonebot.pipeline import run_batch
    from phonebot.schemas import AudioInput, TranscriptionArtifact

    audio_path = _resolve_audio_path(args.file or args.audio_file)
    config_overrides = {
        key: value
        for key, value in {
            "transcriber": args.transcriber,
            "extractor": args.extractor,
            "diarization_enabled": args.diarization,
            "langsmith_tracing": args.langsmith_tracing,
            "extraction_only": args.extraction_only or None,
            "transcriptions_path": args.transcriptions_path,
            "extractor_prompt_file": args.extractor_prompt_file,
            "transcription_prompt_file": args.transcription_prompt_file,
            "openai_transcription_prompt_file": args.openai_transcription_prompt_file,
            "gpu_enabled": args.gpu,
            "denoising_enabled": args.denoising,
        }.items()
        if value is not None
    }
    config = PipelineConfig(**config_overrides)

    output = await run_batch(
        [AudioInput(id=args.record_id or audio_path.stem, file=audio_path)],
        config,
        output_dir=Path(args.output_dir),
    )

    run_dir = Path(args.output_dir) / output.run_id

    print(output.model_dump_json(indent=2, exclude={"cases"}))
    print(f"\nWrote: {run_dir / 'results.json'}")

    print("\nTranscription:")
    for case in output.cases:
        print(f"\n--- {case.caller_info.id} ---")
        print(case.transcript or "<no transcript>")

    # Print diarization speaker segments when present
    transcriptions_artifact_path = run_dir / "transcriptions.json"
    if transcriptions_artifact_path.exists():
        artifact = TranscriptionArtifact.model_validate_json(
            transcriptions_artifact_path.read_text(encoding="utf-8")
        )
        has_segments = any(t.segments for t in artifact.transcriptions if t.segments)
        if has_segments:
            print("\nSpeaker Segments (diarization):")
            for t in artifact.transcriptions:
                if t.segments:
                    print(f"\n--- {t.id} ---")
                    for seg in t.segments:
                        print(f"  [{seg.speaker}] {seg.start:.2f}s–{seg.end:.2f}s: {seg.text}")

    # Evaluation — optional; skip gracefully when ground truth is absent
    ground_truth = _load_ground_truth(args.ground_truth)
    if ground_truth is None:
        return

    evaluator = Evaluator(run_id=output.run_id, output_dir=Path(args.output_dir))
    eval_report = evaluator.compare(
        output.results,
        ground_truth,
        cases=output.cases,
    )
    print("\nEvaluation:")
    print(eval_report.model_dump_json(indent=2))
    print(f"\nWrote: {run_dir / 'eval.json'}")
    print(f"Wrote: {run_dir / 'case_report.json'}")

    md_path = evaluator.write_results_md(
        output.cases, ground_truth, output.config_snapshot, eval_report
    )
    print(f"Wrote: {md_path}")


def _resolve_audio_path(file_arg: str) -> Path:
    audio_path = Path(file_arg)
    if audio_path.parent == Path("."):
        recordings_path = Path("data/recordings") / audio_path
        if recordings_path.exists():
            return recordings_path
    return audio_path


def _load_ground_truth(path_arg: str) -> dict[str, dict[str, Any]] | None:
    """Load ground truth from *path_arg*.

    Returns ``None`` (and prints a notice) when the path is not provided or the
    file does not exist, so callers can skip evaluation gracefully.
    """
    path = Path(path_arg)
    if not path.exists():
        print(f"\n[notice] Ground truth file not found: {path} — skipping evaluation.")
        return None

    data = json.loads(path.read_text(encoding="utf-8"))
    recordings = data.get("recordings", [])
    if not isinstance(recordings, list):
        raise ValueError(f"Ground truth file {path} must contain a 'recordings' list.")

    ground_truth: dict[str, dict[str, Any]] = {}
    for recording in recordings:
        if not isinstance(recording, dict):
            continue
        record_id = recording.get("id")
        expected = recording.get("expected")
        if isinstance(record_id, str) and isinstance(expected, dict):
            ground_truth[record_id] = expected
    return ground_truth


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the current Phonebot pipeline on a single audio recording."
    )
    parser.add_argument(
        "audio_file",
        nargs="?",
        default="data/recordings/call_01.wav",
        help="Path to one WAV file to process. Kept for quick positional runs.",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Audio file to process. Accepts call_01.wav or a full path.",
    )
    parser.add_argument(
        "--record-id",
        default=None,
        help="Optional record id. Defaults to the audio filename stem.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/debug",
        help="Directory where debug run artifacts are written.",
    )
    parser.add_argument(
        "--ground-truth",
        default="data/ground_truth.json",
        help="Ground truth JSON used for evaluation. Omit or point to a missing file to skip eval.",
    )
    parser.add_argument(
        "--transcriber",
        default=None,
        help="Transcriber backend registry key.",
    )
    parser.add_argument(
        "--extractor",
        default=None,
        help="Extractor backend registry key.",
    )
    parser.add_argument(
        "--diarization",
        action="store_const",
        const=True,
        default=None,
        help="Enable diarization for this debug run. Current config validation requires HF_TOKEN.",
    )
    parser.add_argument(
        "--langsmith-tracing",
        action="store_const",
        const=True,
        default=None,
        help="Enable LangSmith tracing for this debug run.",
    )
    parser.add_argument(
        "--extraction-only",
        action="store_true",
        default=False,
        help="Skip transcription and run extraction only. Requires --transcriptions-path.",
    )
    parser.add_argument(
        "--transcriptions-path",
        default=None,
        help="Path to a transcriptions.json artifact. Required when --extraction-only is set.",
    )
    parser.add_argument(
        "--extractor-prompt-file",
        default=None,
        help="Path to a custom extractor prompt file.",
    )
    parser.add_argument(
        "--transcription-prompt-file",
        default=None,
        help="Path to a YAML transcription injection prompt file (top-level 'prompt' key). "
        "Injected into WhisperX at model-load time.",
    )
    parser.add_argument(
        "--openai-transcription-prompt-file",
        default=None,
        help="Path to a YAML transcription prompt file for openai_llm (top-level 'prompt' key).",
    )
    parser.add_argument(
        "--gpu",
        action="store_const",
        const=True,
        default=None,
        help="Enable GPU acceleration (sets gpu_enabled=True in config).",
    )
    parser.add_argument(
        "--denoising",
        action="store_const",
        const=True,
        default=None,
        help="Enable audio denoising. Requires --gpu.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(_run(_parse_args()))
