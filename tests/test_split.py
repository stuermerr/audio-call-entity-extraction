from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType


def _load_split_script() -> ModuleType:
    script_path = Path(__file__).parents[1] / "scripts" / "split.py"
    spec = importlib.util.spec_from_file_location("phonebot_split_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_ground_truth(path: Path, call_ids: list[str]) -> None:
    path.write_text(
        json.dumps({"recordings": [{"id": call_id} for call_id in call_ids]}),
        encoding="utf-8",
    )


def test_write_split_includes_failed_diagnostic_set(tmp_path: Path) -> None:
    split_script = _load_split_script()
    ground_truth_path = tmp_path / "ground_truth.json"
    output_path = tmp_path / "splits.json"
    _write_ground_truth(
        ground_truth_path,
        sorted(set(split_script.DEV_CALLS) | set(split_script.TEST_CALLS)),
    )

    split_script.write_split(ground_truth_path, output_path)

    split = json.loads(output_path.read_text(encoding="utf-8"))
    assert set(split) == {"dev", "test", "failed"}
    assert split["dev"] == sorted(split_script.DEV_CALLS)
    assert split["test"] == sorted(split_script.TEST_CALLS)
    assert split["failed"] == sorted(split_script.FAILED_CALLS)
