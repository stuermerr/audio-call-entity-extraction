"""Tests for phonebot.observability."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from phonebot.observability import (
    _JsonFormatter,
    configure_langsmith_tracing,
    get_logger,
    langsmith_tracing_enabled,
    make_run_id,
    maybe_traceable,
    save_config_snapshot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    *,
    extraction_only: bool = False,
    transcriber: str = "openai_llm",
    extractor: str = "llm",
    sample: str = "all",
    langsmith_tracing: bool = False,
    langsmith_api_key: str = "",
    openai_api_key: str = "sk-test",
):
    """Build a minimal PipelineConfig without touching .env or config.yaml."""
    from phonebot.config import PipelineConfig

    return PipelineConfig(
        extraction_only=extraction_only,
        transcriptions_path="outputs/dummy/transcriptions.json" if extraction_only else None,
        transcriber=transcriber,
        extractor=extractor,
        sample=sample,  # type: ignore[arg-type]
        langsmith_tracing=langsmith_tracing,
        langsmith_api_key=langsmith_api_key,
        openai_api_key=openai_api_key,
    )


# ---------------------------------------------------------------------------
# make_run_id
# ---------------------------------------------------------------------------


def test_make_run_id_full_pipeline_contains_transcriber_and_extractor():
    config = _make_config(transcriber="openai_llm", extractor="llm", sample="dev")
    run_id = make_run_id(config)
    assert "openai_llm" in run_id
    assert "llm" in run_id
    assert "dev" in run_id
    # Should NOT contain "extraction_only" when running full pipeline
    assert "extraction_only" not in run_id


def test_make_run_id_extraction_only_label():
    config = _make_config(extraction_only=True, extractor="llm", sample="test")
    run_id = make_run_id(config)
    assert "extraction_only" in run_id
    assert "llm" in run_id
    assert "test" in run_id


def test_make_run_id_starts_with_timestamp():
    import re

    config = _make_config()
    run_id = make_run_id(config)
    # Timestamp prefix: YYYYMMDD_HHMMSS
    assert re.match(r"^\d{8}_\d{6}_", run_id), f"Unexpected run_id format: {run_id}"


# ---------------------------------------------------------------------------
# _JsonFormatter
# ---------------------------------------------------------------------------


def _make_log_record(msg: str = "hello", level: int = logging.INFO, **extra) -> logging.LogRecord:
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname="",
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_formatter_produces_valid_json():
    formatter = _JsonFormatter()
    record = _make_log_record("test message")
    output = formatter.format(record)
    # Must not raise
    parsed = json.loads(output)
    assert isinstance(parsed, dict)


def test_json_formatter_contains_required_keys():
    formatter = _JsonFormatter()
    record = _make_log_record("hello world", level=logging.WARNING)
    parsed = json.loads(formatter.format(record))
    assert "timestamp" in parsed
    assert "level" in parsed
    assert "name" in parsed
    assert "message" in parsed
    assert parsed["level"] == "WARNING"
    assert parsed["message"] == "hello world"
    assert parsed["name"] == "test.logger"


def test_json_formatter_merges_extra_fields():
    formatter = _JsonFormatter()
    record = _make_log_record("msg", run_id="abc123", stage="extraction")
    parsed = json.loads(formatter.format(record))
    assert parsed["run_id"] == "abc123"
    assert parsed["stage"] == "extraction"


def test_json_formatter_includes_exc_info_as_string():
    formatter = _JsonFormatter()
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        exc_info = sys.exc_info()
    record = logging.LogRecord(
        name="t", level=logging.ERROR, pathname="", lineno=0, msg="err", args=(), exc_info=exc_info
    )
    parsed = json.loads(formatter.format(record))
    assert "exc_info" in parsed
    assert "ValueError" in parsed["exc_info"]


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


def test_get_logger_creates_file_and_stream_handlers(tmp_path):
    run_id = "test_run_abc"
    logger = get_logger(run_id, output_dir=tmp_path)
    handler_types = [type(h).__name__ for h in logger.handlers]
    assert "FileHandler" in handler_types
    assert "StreamHandler" in handler_types


def test_get_logger_creates_log_file(tmp_path):
    run_id = "test_run_logfile"
    get_logger(run_id, output_dir=tmp_path)
    assert (tmp_path / run_id / "run.log").exists()


def test_get_logger_no_duplicate_handlers(tmp_path):
    run_id = "test_run_dedup"
    logger1 = get_logger(run_id, output_dir=tmp_path)
    handler_count = len(logger1.handlers)
    logger2 = get_logger(run_id, output_dir=tmp_path)
    assert logger1 is logger2
    assert len(logger2.handlers) == handler_count


def test_get_logger_file_handler_emits_json(tmp_path):
    run_id = "test_run_json_emit"
    logger = get_logger(run_id, output_dir=tmp_path)
    logger.debug("structured message", extra={"call_id": "call_01"})
    # Flush handlers
    for h in logger.handlers:
        h.flush()
    log_path = tmp_path / run_id / "run.log"
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert lines, "Log file is empty"
    parsed = json.loads(lines[0])
    assert parsed["message"] == "structured message"
    assert parsed["call_id"] == "call_01"


# ---------------------------------------------------------------------------
# configure_langsmith_tracing / langsmith_tracing_enabled
# ---------------------------------------------------------------------------


def test_configure_langsmith_tracing_sets_env_true(monkeypatch):
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    config = _make_config(langsmith_tracing=True, langsmith_api_key="lskey-abc")
    configure_langsmith_tracing(config)
    assert os.environ["LANGSMITH_TRACING"] == "true"
    assert os.environ["LANGSMITH_API_KEY"] == "lskey-abc"


def test_configure_langsmith_tracing_sets_env_false(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    config = _make_config(langsmith_tracing=False)
    configure_langsmith_tracing(config)
    assert os.environ["LANGSMITH_TRACING"] == "false"


def test_langsmith_tracing_enabled_true(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lskey-xyz")
    assert langsmith_tracing_enabled() is True


def test_langsmith_tracing_enabled_false_missing_key(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    assert langsmith_tracing_enabled() is False


def test_langsmith_tracing_enabled_false_tracing_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.setenv("LANGSMITH_API_KEY", "lskey-xyz")
    assert langsmith_tracing_enabled() is False


# ---------------------------------------------------------------------------
# maybe_traceable
# ---------------------------------------------------------------------------


def test_maybe_traceable_passthrough_sync_when_tracing_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    calls: list[tuple] = []

    @maybe_traceable("test.sync")
    def my_fn(x: int, y: int) -> int:
        calls.append((x, y))
        return x + y

    result = my_fn(2, 3)
    assert result == 5
    assert calls == [(2, 3)]


@pytest.mark.asyncio
async def test_maybe_traceable_passthrough_async_when_tracing_off(monkeypatch):
    monkeypatch.setenv("LANGSMITH_TRACING", "false")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)

    calls: list[tuple] = []

    @maybe_traceable("test.async")
    async def my_async_fn(x: int) -> str:
        calls.append((x,))
        return f"result-{x}"

    result = await my_async_fn(7)
    assert result == "result-7"
    assert calls == [(7,)]


# ---------------------------------------------------------------------------
# save_config_snapshot
# ---------------------------------------------------------------------------


def test_save_config_snapshot_writes_yaml(tmp_path):
    config = _make_config(transcriber="deepgram", extractor="llm", sample="dev")
    run_id = "snapshot_run"
    save_config_snapshot(config, run_id, output_dir=tmp_path)
    snapshot_path = tmp_path / run_id / "config.yaml"
    assert snapshot_path.exists()
    data = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert data["transcriber"] == "deepgram"
    assert data["extractor"] == "llm"
    assert data["sample"] == "dev"


def test_save_config_snapshot_excludes_api_keys(tmp_path):
    config = _make_config(openai_api_key="sk-secret")
    run_id = "snapshot_no_secrets"
    save_config_snapshot(config, run_id, output_dir=tmp_path)
    snapshot_path = tmp_path / run_id / "config.yaml"
    data = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    # API keys are excluded from model_dump() via Field(exclude=True)
    assert "openai_api_key" not in data
    assert "deepgram_api_key" not in data
    assert "langsmith_api_key" not in data


def test_save_config_snapshot_merges_extra(tmp_path):
    config = _make_config()
    run_id = "snapshot_extra"
    save_config_snapshot(config, run_id, output_dir=tmp_path, extra={"resolved_prompt": "v8_de"})
    data = yaml.safe_load((tmp_path / run_id / "config.yaml").read_text(encoding="utf-8"))
    assert data["resolved_prompt"] == "v8_de"
