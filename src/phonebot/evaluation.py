from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from phonebot.schemas import (
    CallerInfo,
    CaseReport,
    CaseReportEntry,
    EvalReport,
    EvalResult,
    PipelineCaseResult,
)

ENTITY_FIELDS: tuple[str, ...] = ("first_name", "last_name", "email", "phone_number")

_ENTITY_HEADERS: dict[str, str] = {
    "first_name": "First Name",
    "last_name": "Last Name",
    "email": "Email",
    "phone_number": "Phone",
}

_MAX_TRANSCRIPT_CHARS = 5000


def normalize_phone(s: str) -> str:
    """Normalise a German phone number to E.164 format where possible.

    Handles three common prefix forms:
    - ``+49…``   → strip internal whitespace/dashes/parens, keep as-is
    - ``0049…``  → replace leading ``0049`` with ``+49``
    - ``0…``     → replace leading ``0`` with ``+49`` (national trunk)
    - Otherwise  → return stripped string unchanged (avoids silent corruption)
    """
    stripped = re.sub(r"[\s\-()]", "", s)
    if stripped.startswith("+49"):
        return stripped
    if stripped.startswith("0049"):
        return "+49" + stripped[4:]
    if stripped.startswith("0"):
        return "+49" + stripped[1:]
    return stripped


def match_field(
    predicted: str | None,
    expected: str | list[str],
    field: str,
) -> bool:
    """Return True when *predicted* matches *expected* for the given *field*.

    Rules:
    - ``predicted is None`` → always False
    - ``expected`` is a list → True if any element matches (any acceptable spelling)
    - ``field == "phone_number"`` → compare E.164-normalised strings
    - all other fields → case-insensitive stripped string equality
    """
    if predicted is None:
        return False

    if isinstance(expected, list):
        return any(match_field(predicted, e, field) for e in expected)

    if field == "phone_number":
        return normalize_phone(predicted) == normalize_phone(expected)

    return predicted.lower().strip() == expected.lower().strip()


class Evaluator:
    """Compares pipeline extraction results against a ground-truth dict."""

    def __init__(
        self,
        run_id: str,
        output_dir: Path = Path("outputs"),
        logger: logging.Logger | None = None,
    ) -> None:
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.logger = logger or logging.getLogger("phonebot.evaluator")

    def compare(
        self,
        results: list[CallerInfo],
        ground_truth: dict[str, dict],  # type: ignore[type-arg]
        *,
        cases: list[PipelineCaseResult] | None = None,
    ) -> EvalReport:
        """Evaluate *results* against *ground_truth* and return an EvalReport.

        *ground_truth* maps caller id → dict of expected field values.

        Missing ids count as all-miss; fields absent from ground truth are not penalised.

        When *cases* is provided, ``_write_case_report`` is called to write
        ``outputs/<run_id>/case_report.json`` combining pipeline data with evaluation
        results.  When *cases* is ``None`` the file is not written.
        """
        eval_results: list[EvalResult] = []

        for caller in results:
            gt = ground_truth.get(caller.id)

            if gt is None:
                self.logger.warning(
                    "Caller id %r not found in ground_truth; counting as all-miss.",
                    caller.id,
                )
                per_field: dict[str, bool] = {f: False for f in ENTITY_FIELDS}
            else:
                per_field = {}
                for f in ENTITY_FIELDS:
                    if f not in gt:
                        # Field not annotated in ground truth → not penalised
                        per_field[f] = True
                    else:
                        per_field[f] = match_field(getattr(caller, f), gt[f], f)

            eval_results.append(EvalResult(id=caller.id, per_field=per_field))

        n = len(results)
        if n == 0:
            per_entity_accuracy: dict[str, float] = {f: 0.0 for f in ENTITY_FIELDS}
            overall_accuracy = 0.0
        else:
            per_entity_accuracy = {
                f: sum(r.per_field[f] for r in eval_results) / n for f in ENTITY_FIELDS
            }
            total_pairs = n * len(ENTITY_FIELDS)
            overall_accuracy = (
                sum(v for r in eval_results for v in r.per_field.values()) / total_pairs
            )

        report = EvalReport(
            run_id=self.run_id,
            per_entity_accuracy=per_entity_accuracy,
            overall_accuracy=overall_accuracy,
            results=eval_results,
        )

        out_path = self.output_dir / self.run_id / "eval.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report.model_dump_json(indent=2))

        if cases is not None:
            self._write_case_report(cases, eval_results, ground_truth)

        return report

    def _write_case_report(
        self,
        cases: list[PipelineCaseResult],
        eval_results: list[EvalResult],
        ground_truth: dict[str, Any],
    ) -> None:
        """Build and write ``outputs/<run_id>/case_report.json``.

        Entries are written in input order (matching ``run_batch`` processing order).
        Each entry aligns the pipeline case with its matching ``EvalResult`` by id.
        ``expected`` is the raw ground-truth dict for that id, or ``None`` if absent.
        """
        # Build a lookup from caller id → EvalResult for O(1) alignment
        eval_by_id: dict[str, EvalResult] = {er.id: er for er in eval_results}

        entries: list[CaseReportEntry] = []
        for case in cases:
            cid = case.caller_info.id
            er = eval_by_id.get(cid)
            per_field: dict[str, bool] = (
                er.per_field if er is not None else {f: False for f in ENTITY_FIELDS}
            )
            entries.append(
                CaseReportEntry(
                    id=cid,
                    file=case.caller_info.file,
                    transcript=case.transcript,
                    predicted=case.caller_info,
                    expected=ground_truth.get(cid),
                    per_field=per_field,
                )
            )

        case_report = CaseReport(run_id=self.run_id, cases=entries)
        out_path = self.output_dir / self.run_id / "case_report.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(case_report.model_dump_json(indent=2))

    def write_results_md(
        self,
        cases: list[PipelineCaseResult],
        ground_truth: dict[str, Any],
        config_snapshot: dict[str, Any],
        eval_report: EvalReport,
    ) -> Path:
        """Write a human-readable ``results.md`` to the run output directory.

        The file contains:
        - Run ID and config parameters table at the top.
        - A per-case results table with transcript, extracted entities, expected
          entities, and per-field match indicators (✅/⛔).

        Returns the path to the written file.
        """
        eval_by_id: dict[str, EvalResult] = {er.id: er for er in eval_report.results}

        lines: list[str] = []

        # --- Header ---
        lines.append(f"# Run Results: {self.run_id}")
        lines.append("")

        # --- Config table ---
        lines.append("## Config")
        lines.append("")
        lines.append("| Parameter | Value |")
        lines.append("|-----------|-------|")
        for key, value in sorted(config_snapshot.items()):
            lines.append(f"| `{key}` | `{value}` |")
        lines.append("")

        # --- Accuracy summary ---
        lines.append("## Accuracy Summary")
        lines.append("")
        lines.append("| Field | Accuracy |")
        lines.append("|-------|----------|")
        for field in ENTITY_FIELDS:
            pct = eval_report.per_entity_accuracy.get(field, 0.0)
            lines.append(f"| {_ENTITY_HEADERS[field]} | {pct:.1%} |")
        lines.append(f"| **Overall** | **{eval_report.overall_accuracy:.1%}** |")
        lines.append("")

        # --- Results table ---
        lines.append(f"## Results ({len(cases)} cases)")
        lines.append("")

        # Build header row
        entity_cols = " | ".join(_ENTITY_HEADERS[f] for f in ENTITY_FIELDS)
        lines.append(f"| File | Transcript | {entity_cols} |")
        sep_cols = " | ".join("---" for _ in ENTITY_FIELDS)
        lines.append(f"|------|------------|{sep_cols}|")

        for case in cases:
            cid = case.caller_info.id
            file_name = Path(case.caller_info.file).name
            gt = ground_truth.get(cid)
            er = eval_by_id.get(cid)

            # Transcript cell — cap at _MAX_TRANSCRIPT_CHARS, escape pipe chars
            raw_transcript = case.transcript or "*(transcription failed)*"
            if len(raw_transcript) > _MAX_TRANSCRIPT_CHARS:
                raw_transcript = raw_transcript[:_MAX_TRANSCRIPT_CHARS] + " *(truncated)*"
            transcript_cell = raw_transcript.replace("|", "\\|").replace("\n", " ")

            # Entity cells: "extracted ✅" on match, "extracted ⛔ → expected" on mismatch
            entity_cells: list[str] = []
            for field in ENTITY_FIELDS:
                extracted = getattr(case.caller_info, field)
                matched = er.per_field.get(field, False) if er else False
                cell_extracted = f"`{extracted}`" if extracted is not None else "*(null)*"

                if matched:
                    cell = f"{cell_extracted} ✅"
                else:
                    expected_raw = gt.get(field) if gt else None
                    if isinstance(expected_raw, list):
                        expected_str = " / ".join(expected_raw)
                    else:
                        expected_str = (
                            str(expected_raw) if expected_raw is not None else "*(missing)*"
                        )
                    cell = f"{cell_extracted} ⛔ → `{expected_str}`"

                entity_cells.append(cell)

            entity_row = " | ".join(entity_cells)
            lines.append(f"| {file_name} | {transcript_cell} | {entity_row} |")

        lines.append("")

        out_path = self.output_dir / self.run_id / "results.md"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(lines), encoding="utf-8")
        self.logger.info("results.md written to %s", out_path)
        return out_path
