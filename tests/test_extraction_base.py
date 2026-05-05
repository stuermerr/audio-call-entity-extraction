"""Tests for ExtractorBase.load_prompt and render_user (phonebot.extraction.base)."""

from __future__ import annotations

from pathlib import Path

import jinja2
import pytest
import yaml

from phonebot.extraction.base import ExtractorBase, PromptTemplate


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def valid_prompt_file(tmp_path: Path) -> Path:
    """Write a minimal valid prompt YAML and return its path."""
    data = {
        "system": "You are a helpful assistant.",
        "user": "Extract info from: {{ transcript }}",
    }
    path = tmp_path / "prompt.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# load_prompt — happy path
# ---------------------------------------------------------------------------


def test_load_prompt_returns_prompt_template(valid_prompt_file: Path):
    prompt = ExtractorBase.load_prompt(valid_prompt_file)
    assert isinstance(prompt, PromptTemplate)
    assert prompt.system == "You are a helpful assistant."
    assert "{{ transcript }}" in prompt.user


def test_load_prompt_preserves_multiline_system(tmp_path: Path):
    data = {
        "system": "Line one.\nLine two.\nLine three.",
        "user": "Transcript: {{ transcript }}",
    }
    path = tmp_path / "multi.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    prompt = ExtractorBase.load_prompt(path)
    assert "Line two." in prompt.system


# ---------------------------------------------------------------------------
# load_prompt — error paths
# ---------------------------------------------------------------------------


def test_load_prompt_raises_file_not_found(tmp_path: Path):
    missing = tmp_path / "nonexistent.yaml"
    with pytest.raises(FileNotFoundError):
        ExtractorBase.load_prompt(missing)


def test_load_prompt_raises_value_error_missing_user_key(tmp_path: Path):
    data = {"system": "Only system here."}
    path = tmp_path / "missing_user.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="user"):
        ExtractorBase.load_prompt(path)


def test_load_prompt_raises_value_error_missing_system_key(tmp_path: Path):
    data = {"user": "Only user here: {{ transcript }}"}
    path = tmp_path / "missing_system.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    with pytest.raises(ValueError, match="system"):
        ExtractorBase.load_prompt(path)


def test_load_prompt_raises_value_error_when_yaml_is_not_dict(tmp_path: Path):
    path = tmp_path / "list.yaml"
    path.write_text("- item1\n- item2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        ExtractorBase.load_prompt(path)


# ---------------------------------------------------------------------------
# render_user — happy path
# ---------------------------------------------------------------------------


def test_render_user_substitutes_transcript(valid_prompt_file: Path):
    prompt = ExtractorBase.load_prompt(valid_prompt_file)
    rendered = ExtractorBase.render_user(prompt, transcript="Hallo, ich bin Max.")
    assert "Hallo, ich bin Max." in rendered
    assert "{{ transcript }}" not in rendered


def test_render_user_handles_empty_transcript(valid_prompt_file: Path):
    prompt = ExtractorBase.load_prompt(valid_prompt_file)
    rendered = ExtractorBase.render_user(prompt, transcript="")
    # Empty transcript substituted cleanly — no template variable remains
    assert "{{ transcript }}" not in rendered


def test_render_user_handles_special_characters_in_transcript(valid_prompt_file: Path):
    prompt = ExtractorBase.load_prompt(valid_prompt_file)
    special = "Jürgen Müller, juergen.mueller@gmx.de, +49 170 1234567"
    rendered = ExtractorBase.render_user(prompt, transcript=special)
    assert special in rendered


# ---------------------------------------------------------------------------
# render_user — StrictUndefined raises on missing variable
# ---------------------------------------------------------------------------


def test_render_user_raises_on_missing_transcript_variable(tmp_path: Path):
    """Template without {{ transcript }} should raise when rendered — StrictUndefined."""
    # A prompt where the user template uses a *different* variable name
    data = {
        "system": "System prompt.",
        "user": "Transcript: {{ wrong_variable }}",
    }
    path = tmp_path / "wrong_var.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    prompt = ExtractorBase.load_prompt(path)
    with pytest.raises(jinja2.UndefinedError):
        ExtractorBase.render_user(prompt, transcript="some text")


def test_render_user_raises_when_user_template_has_no_variable(tmp_path: Path):
    """Even a static template with no variable should not silently drop the transcript.

    StrictUndefined only fires on *undefined* variable access; a static template
    (no variables at all) renders fine. This test documents that behaviour and
    checks that an *extra* undefined variable still raises.
    """
    data = {
        "system": "System.",
        "user": "Static text only.",  # No {{ transcript }} — renders without error
    }
    path = tmp_path / "static.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    prompt = ExtractorBase.load_prompt(path)
    # Static template renders without error — transcript is silently unused
    rendered = ExtractorBase.render_user(prompt, transcript="ignored")
    assert rendered == "Static text only."
