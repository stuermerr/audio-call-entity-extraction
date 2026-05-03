from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from phonebot.cli import app
from phonebot.config import PipelineConfig
from phonebot.schemas import PipelineOutput

runner = CliRunner()


async def _fake_run_batch(
    inputs: list[object],
    config: PipelineConfig,
    *,
    output_dir: Path = Path("outputs"),
) -> PipelineOutput:
    return PipelineOutput(
        results=[],
        run_id="test_run",
        config_snapshot=config.model_dump(),
        cases=[],
    )


def test_eval_option_accepts_explicit_false(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("phonebot.cli._resolve_inputs", lambda sample, data_dir: [])
    monkeypatch.setattr("phonebot.cli.run_batch", _fake_run_batch)

    result = runner.invoke(app, ["--eval", "false"])

    assert result.exit_code == 0
    assert "Processing 0 recordings [dev]" in result.output
    assert "Eval summary" not in result.output


def test_help_shows_eval_value_option_only() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "--eval" in result.output
    assert "true|false" in result.output
