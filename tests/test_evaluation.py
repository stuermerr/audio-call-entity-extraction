from __future__ import annotations

import json
from pathlib import Path

import pytest

from phonebot.evaluation import Evaluator, match_field, normalize_phone
from phonebot.schemas import CallerInfo, PipelineCaseResult

# ---------------------------------------------------------------------------
# normalize_phone
# ---------------------------------------------------------------------------


def test_normalize_phone_e164_strips_spaces() -> None:
    assert normalize_phone("+49 152 11223456") == "+4915211223456"


def test_normalize_phone_0049_prefix() -> None:
    assert normalize_phone("0049152 11223456") == "+4915211223456"


def test_normalize_phone_0_national_trunk() -> None:
    assert normalize_phone("0152 11223456") == "+4915211223456"


def test_normalize_phone_no_spaces_passthrough() -> None:
    assert normalize_phone("+4915211223456") == "+4915211223456"


# ---------------------------------------------------------------------------
# match_field
# ---------------------------------------------------------------------------


def test_match_field_name_case_insensitive() -> None:
    assert match_field("max", "Max", "first_name") is True


def test_match_field_name_none_predicted() -> None:
    assert match_field(None, "Max", "first_name") is False


def test_match_field_list_expected_hit() -> None:
    assert match_field("Max", ["max", "Maximilian"], "first_name") is True


def test_match_field_list_expected_miss() -> None:
    assert match_field("Klaus", ["max", "Maximilian"], "first_name") is False


def test_match_field_phone_normalised_match() -> None:
    assert match_field("0049152 11223456", "+49 152 11223456", "phone_number") is True


# ---------------------------------------------------------------------------
# Evaluator.compare
# ---------------------------------------------------------------------------


def test_compare_missing_gt_key_counts_as_miss(tmp_path: Path) -> None:
    evaluator = Evaluator(run_id="test_run", output_dir=tmp_path)
    caller = CallerInfo(id="call_01", file="call_01.wav", first_name="Max")

    report = evaluator.compare([caller], {})

    assert len(report.results) == 1
    result = report.results[0]
    assert all(v is False for v in result.per_field.values())
    assert report.overall_accuracy == 0.0


def test_compare_overall_accuracy(tmp_path: Path) -> None:
    evaluator = Evaluator(run_id="test_run", output_dir=tmp_path)

    caller_match = CallerInfo(
        id="call_01",
        file="call_01.wav",
        first_name="Max",
        last_name="Mustermann",
        email="max@example.com",
        phone_number="+4915211223456",
    )
    caller_miss = CallerInfo(
        id="call_02",
        file="call_02.wav",
        first_name="Wrong",
        last_name="Wrong",
        email="wrong@example.com",
        phone_number="+49000000000",
    )

    ground_truth = {
        "call_01": {
            "first_name": "Max",
            "last_name": "Mustermann",
            "email": "max@example.com",
            "phone_number": "+4915211223456",
        },
        "call_02": {
            "first_name": "Hans",
            "last_name": "Schmidt",
            "email": "hans@example.com",
            "phone_number": "+4917611223456",
        },
    }

    report = evaluator.compare([caller_match, caller_miss], ground_truth)

    # call_01: all 4 fields correct → 4 hits; call_02: all 4 wrong → 0 hits
    # overall = 4 / (2 * 4) = 0.5
    assert report.overall_accuracy == pytest.approx(0.5)


def test_compare_saves_eval_json(tmp_path: Path) -> None:
    run_id = "test_run_save"
    evaluator = Evaluator(run_id=run_id, output_dir=tmp_path)
    caller = CallerInfo(id="call_01", file="call_01.wav", first_name="Max")
    ground_truth = {
        "call_01": {
            "first_name": "Max",
            "last_name": "Mustermann",
            "email": "max@example.com",
            "phone_number": "+4915211223456",
        }
    }

    evaluator.compare([caller], ground_truth)

    assert (tmp_path / run_id / "eval.json").exists()


# ---------------------------------------------------------------------------
# case_report.json: written when cases provided, not written when cases=None
# ---------------------------------------------------------------------------


def _make_cases(
    callers: list[CallerInfo], transcripts: list[str | None]
) -> list[PipelineCaseResult]:
    return [PipelineCaseResult(caller_info=c, transcript=t) for c, t in zip(callers, transcripts)]


def test_compare_writes_case_report(tmp_path: Path) -> None:
    """case_report.json is written when cases are provided; content is correct."""
    run_id = "test_case_report"
    evaluator = Evaluator(run_id=run_id, output_dir=tmp_path)

    caller = CallerInfo(
        id="call_01",
        file="call_01.wav",
        first_name="Max",
        last_name="Mustermann",
        email="max@example.com",
        phone_number="+4915211223456",
    )
    ground_truth = {
        "call_01": {
            "first_name": "Max",
            "last_name": "Mustermann",
            "email": "max@example.com",
            "phone_number": "+4915211223456",
        }
    }
    cases = _make_cases([caller], ["Hallo ich bin Max Mustermann."])

    evaluator.compare([caller], ground_truth, cases=cases)

    report_path = tmp_path / run_id / "case_report.json"
    assert report_path.exists()

    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["run_id"] == run_id
    assert len(data["cases"]) == 1
    entry = data["cases"][0]
    assert entry["id"] == "call_01"
    assert entry["file"] == "call_01.wav"
    assert entry["transcript"] == "Hallo ich bin Max Mustermann."
    assert entry["predicted"]["first_name"] == "Max"
    assert entry["expected"]["first_name"] == "Max"
    assert entry["per_field"]["first_name"] is True


def test_compare_no_cases_no_case_report(tmp_path: Path) -> None:
    """case_report.json is NOT written when cases=None."""
    run_id = "test_no_case_report"
    evaluator = Evaluator(run_id=run_id, output_dir=tmp_path)
    caller = CallerInfo(id="call_01", file="call_01.wav", first_name="Max")

    evaluator.compare([caller], {"call_01": {"first_name": "Max"}})

    assert not (tmp_path / run_id / "case_report.json").exists()


def test_case_report_missing_gt_entry(tmp_path: Path) -> None:
    """Case whose id is absent from ground truth shows expected=None and all-False per_field."""
    run_id = "test_missing_gt"
    evaluator = Evaluator(run_id=run_id, output_dir=tmp_path)

    caller = CallerInfo(id="call_99", file="call_99.wav", first_name="Anna")
    cases = _make_cases([caller], [None])

    evaluator.compare([caller], {}, cases=cases)

    data = json.loads((tmp_path / run_id / "case_report.json").read_text(encoding="utf-8"))
    entry = data["cases"][0]
    assert entry["id"] == "call_99"
    assert entry["expected"] is None
    assert entry["transcript"] is None
    assert all(v is False for v in entry["per_field"].values())


def test_results_md_uses_emoji_match_indicators(tmp_path: Path) -> None:
    run_id = "test_results_md"
    evaluator = Evaluator(run_id=run_id, output_dir=tmp_path)
    caller = CallerInfo(
        id="call_01",
        file="call_01.wav",
        first_name="Max",
        last_name="Wrong",
        email="max@example.com",
        phone_number="+4915211223456",
    )
    ground_truth = {
        "call_01": {
            "first_name": "Max",
            "last_name": "Mustermann",
            "email": "max@example.com",
            "phone_number": "+4915211223456",
        }
    }
    cases = _make_cases([caller], ["Hallo ich bin Max."])
    eval_report = evaluator.compare([caller], ground_truth)

    results_path = evaluator.write_results_md(cases, ground_truth, {}, eval_report)

    results_md = results_path.read_text(encoding="utf-8")
    assert "`Max` ✅" in results_md
    assert "`Wrong` ❌ → `Mustermann`" in results_md
    assert "✓" not in results_md
    assert "✗" not in results_md
