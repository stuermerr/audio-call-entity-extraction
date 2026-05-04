from __future__ import annotations

import json
import logging
import os
import warnings
from collections.abc import Callable
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import TypeVar

import yaml

from phonebot.config import PipelineConfig

__all__ = [
    "make_run_id",
    "get_logger",
    "save_config_snapshot",
    "configure_langsmith_tracing",
    "langsmith_tracing_enabled",
    "maybe_traceable",
    "suppress_openai_parsed_response_serializer_warning",
]

F = TypeVar("F", bound=Callable)  # type: ignore[type-arg]
_OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED = False


def make_run_id(config: PipelineConfig) -> str:
    """Return a run identifier encoding timestamp and config dimensions."""
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    if config.extraction_only:
        return f"{ts}_extraction_only_{config.extractor}_{config.sample}"
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


_CONSOLE_FORMATTER = logging.Formatter(
    fmt="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)


def get_logger(run_id: str, output_dir: Path = Path("outputs")) -> logging.Logger:
    """Return a per-run logger writing JSON to outputs/{run_id}/run.log.

    The file handler captures all DEBUG+ records as JSON lines.
    The console (stderr) handler captures INFO+ records as plain human-readable text.
    """
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
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(_CONSOLE_FORMATTER)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger


def save_config_snapshot(
    config: PipelineConfig,
    run_id: str,
    output_dir: Path = Path("outputs"),
    *,
    extra: dict | None = None,
) -> None:
    """Serialize config to YAML at outputs/{run_id}/config.yaml.

    *extra* is merged into the snapshot after ``model_dump()`` so callers can
    inject resolved values (e.g. the resolved ``extractor_prompt_file`` path).
    """
    run_dir = Path(output_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    config_path = run_dir / "config.yaml"
    data = config.model_dump()
    if extra:
        data.update(extra)
    with config_path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh)


def configure_langsmith_tracing(config: PipelineConfig) -> None:
    """Mirror PipelineConfig tracing fields into env vars used by LangSmith."""
    os.environ["LANGSMITH_TRACING"] = "true" if config.langsmith_tracing else "false"
    if config.langsmith_api_key:
        os.environ["LANGSMITH_API_KEY"] = config.langsmith_api_key


def langsmith_tracing_enabled() -> bool:
    """Return whether LangSmith tracing can emit runs in this process."""
    tracing_on = os.environ.get("LANGSMITH_TRACING", "").lower() == "true"
    has_key = bool(os.environ.get("LANGSMITH_API_KEY"))
    return tracing_on and has_key


def suppress_openai_parsed_response_serializer_warning() -> None:
    """Hide only the known OpenAI parsed-response serializer warning.

    LangSmith's OpenAI wrapper serializes ``ParsedChatCompletion`` objects for
    traces. With OpenAI SDK parsed responses, Pydantic may warn that the
    ``parsed`` field expected ``None`` even though it serializes correctly.
    Keep the filter narrow so unrelated serializer warnings remain visible.
    """
    global _OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED
    if _OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED:
        return

    warnings.filterwarnings(
        "ignore",
        message=(
            r"(?s)Pydantic serializer warnings:.*"
            r"field_name='parsed'.*"
            r"input_type=_ExtractedFields"
        ),
        category=UserWarning,
        module=r"pydantic\.main",
    )
    _OPENAI_PARSED_RESPONSE_WARNING_FILTER_INSTALLED = True


def maybe_traceable(name: str) -> Callable[[F], F]:
    """Trace calls when LangSmith is enabled at runtime, else call through."""

    def decorator(fn: F) -> F:
        @wraps(fn)
        async def async_wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            if langsmith_tracing_enabled():
                import langsmith  # noqa: PLC0415

                traced = langsmith.traceable(name=name)(fn)
                return await traced(*args, **kwargs)
            return await fn(*args, **kwargs)

        @wraps(fn)
        def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
            if langsmith_tracing_enabled():
                import langsmith  # noqa: PLC0415

                traced = langsmith.traceable(name=name)(fn)
                return traced(*args, **kwargs)
            return fn(*args, **kwargs)

        import inspect  # noqa: PLC0415

        if inspect.iscoroutinefunction(fn):
            return async_wrapper  # type: ignore[return-value]
        return wrapper  # type: ignore[return-value]

    return decorator
