from __future__ import annotations

import json
from pathlib import Path

import yaml
from typer.testing import CliRunner

from phonebot.cli import _resolve_inputs, _build_ground_truth, app
from phonebot.config import PipelineConfig
from phonebot.schemas import CallerInfo, PipelineOutput

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
                "extractor": "custom_extractor",
                "sample": "test",
            }
        ),
        encoding="utf-8",
    )


def test_eval_option_accepts_explicit_false(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("SAMPLE", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("phonebot.cli._resolve_inputs", lambda sample, data_dir: [])
    monkeypatch.setattr("phonebot.cli.run_batch", _fake_run_batch)

    result = runner.invoke(app, ["--eval", "false"])

    assert result.exit_code == 0
    assert "Processing 0 recordings [all]" in result.output
    assert "Eval summary" not in result.output


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
    assert captured[0].extractor == "custom_extractor"


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
            "llm",
            "--eval",
            "false",
        ],
    )

    assert result.exit_code == 0
    assert "Processing 0 recordings [all]" in result.output
    assert captured[0].sample == "all"
    assert captured[0].transcriber == "deepgram"
    assert captured[0].extractor == "llm"


def test_failed_sample_flag_is_accepted(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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

    result = runner.invoke(app, ["--samples", "failed", "--eval", "false"])

    assert result.exit_code == 0
    assert "Processing 0 recordings [failed]" in result.output
    assert captured[0].sample == "failed"


def test_cli_prints_run_results(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("phonebot.cli._resolve_inputs", lambda sample, data_dir: [])

    async def fake_run_batch(
        inputs: list[object],
        config: PipelineConfig,
        *,
        output_dir: Path = Path("outputs"),
    ) -> PipelineOutput:
        return PipelineOutput(
            results=[
                CallerInfo(
                    id="call_01",
                    file="data/recordings/call_01.wav",
                    first_name="Max",
                    last_name="Mustermann",
                    email="max@example.com",
                    phone_number="+491701234567",
                )
            ],
            run_id="test_run",
            config_snapshot=config.model_dump(),
            cases=[],
        )

    monkeypatch.setattr("phonebot.cli.run_batch", fake_run_batch)

    result = runner.invoke(app, ["--eval", "false"])

    assert result.exit_code == 0
    assert "--- Results: test_run ---" in result.output
    assert "Artifacts: outputs/test_run" in result.output
    assert "call_01" in result.output
    assert "Max" in result.output
    assert "max@example.com" in result.output


def test_extraction_only_flags_reach_pipeline(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
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
    transcriptions_path = tmp_path / "transcriptions.json"

    result = runner.invoke(
        app,
        [
            "--extraction-only",
            "--transcriptions-path",
            str(transcriptions_path),
            "--eval",
            "false",
        ],
    )

    assert result.exit_code == 0
    assert captured[0].extraction_only is True
    assert captured[0].transcriptions_path == str(transcriptions_path)


def test_extraction_only_can_be_enabled_from_config_yaml(
    tmp_path: Path,
    monkeypatch,
) -> None:
    transcriptions_path = tmp_path / "saved_transcriptions.json"
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "transcriber": "deepgram",
                "extractor": "custom_extractor",
                "sample": "failed",
                "extraction_only": True,
                "transcriptions_path": str(transcriptions_path),
            }
        ),
        encoding="utf-8",
    )
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
    assert "Processing 0 recordings [failed]" in result.output
    assert captured[0].extraction_only is True
    assert captured[0].transcriptions_path == str(transcriptions_path)
    assert captured[0].transcriber == "deepgram"
    assert captured[0].extractor == "custom_extractor"


def test_extraction_only_requires_transcriptions_path(
    tmp_path: Path,
    monkeypatch,
) -> None:
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"extraction_only": False, "transcriptions_path": None}),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    result = runner.invoke(app, ["--extraction-only", "--eval", "false"])

    assert result.exit_code == 1
    assert "transcriptions_path is required when extraction_only=True" in result.output


def test_resolve_inputs_failed_uses_failed_split_only(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "splits.json").write_text(
        json.dumps(
            {
                "dev": ["call_01", "call_02"],
                "test": ["call_03"],
                "failed": ["call_02", "call_03"],
            }
        ),
        encoding="utf-8",
    )

    inputs = _resolve_inputs("failed", data_dir)

    assert [audio.id for audio in inputs] == ["call_02", "call_03"]
    assert [audio.file for audio in inputs] == [
        data_dir / "recordings" / "call_02.wav",
        data_dir / "recordings" / "call_03.wav",
    ]


def test_build_ground_truth_missing_file_returns_empty_dict_and_warns(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """_build_ground_truth returns {} and logs a warning when ground_truth.json is absent."""
    import logging

    with caplog.at_level(logging.WARNING, logger="phonebot.cli"):
        result = _build_ground_truth(tmp_path)

    assert result == {}
    assert any("ground_truth.json" in msg for msg in caplog.messages)
