from __future__ import annotations

import json
import logging
import os
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import TypeVar

import yaml

from phonebot.config import PipelineConfig

__all__ = ["make_run_id", "get_logger", "save_config_snapshot", "maybe_traceable"]

F = TypeVar("F", bound=Callable)  # type: ignore[type-arg]


def make_run_id(config: PipelineConfig) -> str:
    """Return a run identifier encoding timestamp and config dimensions."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{ts}_{config.transcriber}_{config.extractor}_{config.sample}"


class _JsonFormatter(logging.Formatter):
    """Emit one JSON object per log record."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        # Merge any extra fields attached to the record
        skip = {
            "args",
            "created",
            "exc_info",
            "exc_text",
            "filename",
            "funcName",
            "levelname",
            "levelno",
            "lineno",
            "message",
            "module",
            "msecs",
            "msg",
            "name",
            "pathname",
            "process",
            "processName",
            "relativeCreated",
            "stack_info",
            "taskName",
            "thread",
            "threadName",
        }
        for key, value in record.__dict__.items():
            if key not in skip:
                payload[key] = value
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def get_logger(run_id: str, output_dir: Path = Path("outputs")) -> logging.Logger:
    """Return a per-run logger writing JSON to outputs/{run_id}/run.log."""
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"phonebot.{run_id}")
    logger.setLevel(logging.DEBUG)

    # Avoid duplicate handlers if called more than once for the same run_id
    if logger.handlers:
        return logger

    file_handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(_JsonFormatter())

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.WARNING)
    stream_handler.setFormatter(_JsonFormatter())

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config_snapshot(
    config: PipelineConfig,
    run_id: str,
    output_dir: Path = Path("outputs"),
) -> None:
    """Serialize config to YAML at outputs/{run_id}/config.yaml."""
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config.model_dump(), fh)


def maybe_traceable(name: str) -> Callable[[F], F]:
    """Return @langsmith.traceable(name=name) when tracing is enabled, else identity."""
    tracing_on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    if tracing_on and has_key:
        import langsmith  # noqa: PLC0415

        return langsmith.traceable(name=name)  # type: ignore[return-value]
    return lambda fn: fn  # type: ignore[return-value]
