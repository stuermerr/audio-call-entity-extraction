from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Optional

import typer

from phonebot.config import PipelineConfig
from phonebot.evaluation import Evaluator
from phonebot.pipeline import run_batch
from phonebot.schemas import AudioInput

app = typer.Typer(help="Phone-call extraction pipeline CLI.")


def _parse_eval_option(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise typer.BadParameter("--eval must be true or false")


def _resolve_inputs(sample: str, data_dir: Path) -> list[AudioInput]:
    """Resolve the list of AudioInput objects for the given sample split.

    - If ``data/splits.json`` exists: use the split index to select inputs.
    - If missing and ``sample == "all"``: enumerate ``data/recordings/*.wav`` sorted.
    - If missing and ``sample != "all"``: exit with an actionable error message.
    """
    splits_path = data_dir / "splits.json"
    recordings_dir = data_dir / "recordings"

    if splits_path.exists():
        splits: dict[str, list[str]] = json.loads(splits_path.read_text(encoding="utf-8"))
        if sample == "all":
            call_ids = sorted(set(splits.get("dev", [])) | set(splits.get("test", [])))
        else:
            call_ids = sorted(splits.get(sample, []))
        if not call_ids:
            typer.echo(
                f"Warning: split '{sample}' resolved to 0 recordings. "
                "Check data/splits.json or run `uv run scripts/split.py`.",
                err=True,
            )
        return [AudioInput(id=cid, file=recordings_dir / f"{cid}.wav") for cid in call_ids]

    if sample != "all":
        typer.echo(
            "Error: data/splits.json not found. "
            "Run `uv run scripts/split.py` to generate it, "
            "or use --samples all to process every recording without a split.",
            err=True,
        )
        raise typer.Exit(1)

    # Fallback: enumerate wav files when splits.json is absent and sample == "all"
    if not recordings_dir.exists():
        typer.echo(
            f"Error: recordings directory not found: {recordings_dir}. "
            "Ensure data/recordings/ exists and contains the WAV files.",
            err=True,
        )
        raise typer.Exit(1)
    wav_files = sorted(recordings_dir.glob("*.wav"))
    if not wav_files:
        typer.echo(
            f"Warning: no .wav files found in {recordings_dir}. "
            "The pipeline will process 0 recordings.",
            err=True,
        )
    return [AudioInput(id=f.stem, file=f) for f in wav_files]


def _build_ground_truth(data_dir: Path) -> dict[str, dict]:  # type: ignore[type-arg]
    """Load ground_truth.json and return a ``{id: expected_fields}`` mapping.

    Returns an empty dict (with a warning) when the file is missing so that
    callers can proceed without crashing; missing entries count as all-miss in
    the evaluator.
    """
    gt_path = data_dir / "ground_truth.json"
    if not gt_path.exists():
        logging.getLogger("phonebot.cli").warning(
            "ground_truth.json not found at %s; eval will count all as miss.", gt_path
        )
        return {}
    data = json.loads(gt_path.read_text(encoding="utf-8"))
    return {rec["id"]: rec["expected"] for rec in data["recordings"]}


@app.command()
def run(
    samples: Optional[str] = typer.Option(
        None,
        "--samples",
        "-s",
        help="Split to process: dev|test|failed|all. If omitted, config/env/defaults apply.",
    ),
    transcriber: Optional[str] = typer.Option(
        None, "--transcriber", "-t", help="Transcriber registry key, e.g. openai_llm"
    ),
    extractor: Optional[str] = typer.Option(
        None, "--extractor", "-e", help="Extractor registry key, e.g. llm"
    ),
    evaluate: str = typer.Option(
        "true",
        "--eval",
        metavar="true|false",
        help="Run eval after extraction: true|false",
    ),
    extraction_only: bool = typer.Option(
        False,
        "--extraction-only",
        help="Skip transcription and extract from --transcriptions-path.",
    ),
    transcriptions_path: Optional[Path] = typer.Option(
        None,
        "--transcriptions-path",
        help="Path to a transcriptions.json artifact for --extraction-only.",
    ),
    extractor_prompt_file: Optional[str] = typer.Option(
        None,
        "--extractor-prompt-file",
        help="Path to a custom YAML/Jinja2 extractor prompt file.",
    ),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", help="Output root directory"),
) -> None:
    """Run the phonebot extraction pipeline over a set of recordings."""
    # Build PipelineConfig; CLI values override yaml/env for explicitly-provided args only.
    overrides: dict[str, object] = {}
    if samples is not None:
        overrides["sample"] = samples
    if transcriber is not None:
        overrides["transcriber"] = transcriber
    if extractor is not None:
        overrides["extractor"] = extractor
    if extraction_only:
        overrides["extraction_only"] = extraction_only
    if transcriptions_path is not None:
        overrides["transcriptions_path"] = str(transcriptions_path)
    if extractor_prompt_file is not None:
        overrides["extractor_prompt_file"] = extractor_prompt_file
    evaluate_enabled = _parse_eval_option(evaluate)

    try:
        config = PipelineConfig(**overrides)  # type: ignore[arg-type]
    except ValueError as exc:
        typer.echo(f"Configuration error: {exc}", err=True)
        raise typer.Exit(1)

    inputs = _resolve_inputs(config.sample, Path("data"))
    typer.echo(f"Processing {len(inputs)} recordings [{config.sample}]…")

    try:
        output = asyncio.run(run_batch(inputs, config, output_dir=output_dir))
    except (RuntimeError, ValueError, FileNotFoundError, ImportError) as exc:
        typer.echo(f"Pipeline error: {exc}", err=True)
        raise typer.Exit(1)

    if evaluate_enabled:
        ground_truth = _build_ground_truth(Path("data"))
        evaluator = Evaluator(run_id=output.run_id, output_dir=output_dir)
        report = evaluator.compare(output.results, ground_truth, cases=output.cases)
        evaluator.write_results_md(output.cases, ground_truth, output.config_snapshot, report)
        typer.echo(f"\n--- Eval summary: {output.run_id} ---")
        for field, pct in report.per_entity_accuracy.items():
            typer.echo(f"{field:<14}{pct:.1%}")
        typer.echo(f"{'Overall':<14}{report.overall_accuracy:.1%}")
