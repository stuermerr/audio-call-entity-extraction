"""Unit tests for src/phonebot/normalization.py and _ExtractedFields validators."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from phonebot.extraction.llm import _ExtractedFields
from phonebot.normalization import clean_phone

# ---------------------------------------------------------------------------
# clean_phone
# ---------------------------------------------------------------------------


def test_clean_phone_strips_spaces() -> None:
    assert clean_phone("+49 152 11223456") == "+4915211223456"


def test_clean_phone_strips_hyphens() -> None:
    assert clean_phone("+49-152-11223456") == "+4915211223456"


def test_clean_phone_strips_parens() -> None:
    assert clean_phone("+49(152)11223456") == "+4915211223456"


def test_clean_phone_preserves_plus() -> None:
    assert clean_phone("+4915211223456") == "+4915211223456"


# ---------------------------------------------------------------------------
# _ExtractedFields — phone_number validator + pattern
# ---------------------------------------------------------------------------


def test_extracted_fields_phone_validator() -> None:
    ef = _ExtractedFields(phone_number="+49 123 456789")
    assert ef.phone_number == "+49123456789"


def test_extracted_fields_phone_pattern_failure() -> None:
    with pytest.raises(ValidationError):
        _ExtractedFields(phone_number="not-a-phone")


# ---------------------------------------------------------------------------
# _ExtractedFields — email validator + pattern
# ---------------------------------------------------------------------------


def test_extracted_fields_email_lowercased() -> None:
    ef = _ExtractedFields(email="Max@Gmail.COM")
    assert ef.email == "max@gmail.com"


def test_extracted_fields_email_pattern_failure() -> None:
    with pytest.raises(ValidationError):
        _ExtractedFields(email="noemail")


# ---------------------------------------------------------------------------
# _ExtractedFields — name validator
# ---------------------------------------------------------------------------


def test_extracted_fields_name_strips_digits() -> None:
    ef = _ExtractedFields(first_name="Hans3")
    assert ef.first_name == "Hans"


def test_extracted_fields_name_strips_at() -> None:
    ef = _ExtractedFields(first_name="Hans@")
    assert ef.first_name == "Hans"


# ---------------------------------------------------------------------------
# _ExtractedFields — None passthrough
# ---------------------------------------------------------------------------


def test_extracted_fields_none_fields_passthrough() -> None:
    ef = _ExtractedFields(first_name=None, last_name=None, email=None, phone_number=None)
    assert ef.first_name is None
    assert ef.last_name is None
    assert ef.email is None
    assert ef.phone_number is None
