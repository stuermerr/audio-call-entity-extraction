from __future__ import annotations

from pathlib import Path

import yaml
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


def _write_cli_config(tmp_path: Path) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "transcriber": "whisperx",
                "extractor": "presidio",
                "sample": "test",
            }
        ),
        encoding="utf-8",
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


def test_omitted_config_flags_respect_config_yaml(tmp_path: Path, monkeypatch) -> None:
    _write_cli_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    captured: list[PipelineConfig] = []
    monkeypatch.setattr("phonebot.cli._resolve_inputs", lambda sample, data_dir: [])

    async def fake_run_batch(
        inputs: list[object],
        config: PipelineConfig,
        *,
        output_dir: Path = Path("outputs"),
    ) -> PipelineOutput:
        captured.append(config)
        return await _fake_run_batch(inputs, config, output_dir=output_dir)

    monkeypatch.setattr("phonebot.cli.run_batch", fake_run_batch)

    result = runner.invoke(app, ["--eval", "false"])

    assert result.exit_code == 0
    assert "Processing 0 recordings [test]" in result.output
    assert captured[0].sample == "test"
    assert captured[0].transcriber == "whisperx"
    assert captured[0].extractor == "presidio"


def test_explicit_config_flags_override_config_yaml(tmp_path: Path, monkeypatch) -> None:
    _write_cli_config(tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai")
    monkeypatch.setenv("DEEPGRAM_API_KEY", "test-deepgram")
    captured: list[PipelineConfig] = []
    monkeypatch.setattr("phonebot.cli._resolve_inputs", lambda sample, data_dir: [])

    async def fake_run_batch(
        inputs: list[object],
        config: PipelineConfig,
        *,
        output_dir: Path = Path("outputs"),
    ) -> PipelineOutput:
        captured.append(config)
        return await _fake_run_batch(inputs, config, output_dir=output_dir)

    monkeypatch.setattr("phonebot.cli.run_batch", fake_run_batch)

    result = runner.invoke(
        app,
        [
            "--samples",
            "all",
            "--transcriber",
            "deepgram",
            "--extractor",
            "privacy_filter",
            "--eval",
            "false",
        ],
    )

    assert result.exit_code == 0
    assert "Processing 0 recordings [all]" in result.output
    assert captured[0].sample == "all"
    assert captured[0].transcriber == "deepgram"
    assert captured[0].extractor == "privacy_filter"
