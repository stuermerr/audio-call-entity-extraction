from __future__ import annotations

import logging
import re
from pathlib import Path

from phonebot.schemas import CallerInfo, EvalReport, EvalResult

ENTITY_FIELDS: tuple[str, ...] = ("first_name", "last_name", "email", "phone_number")


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
    ) -> EvalReport:
        """Evaluate *results* against *ground_truth* and return an EvalReport.

        *ground_truth* maps caller id → dict of expected field values.

        Missing ids count as all-miss; fields absent from ground truth are not penalised.
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

        return report
