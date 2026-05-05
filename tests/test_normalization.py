"""Unit tests for src/phonebot/normalization.py and _ExtractedFields validators."""

from __future__ import annotations

import pytest

from phonebot.extraction.llm import _ExtractedFields
from phonebot.normalization import (
    clean_phone,
    normalize_phone,
    validate_and_normalize_email,
    validate_and_normalize_phone,
)

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


def test_clean_phone_mixed_symbols() -> None:
    assert clean_phone("+49 (0)89-123 456") == "+49089123456"


# ---------------------------------------------------------------------------
# normalize_phone
# ---------------------------------------------------------------------------


def test_normalize_phone_uses_phonenumbers_for_e164() -> None:
    assert normalize_phone("+49 152 11223456") == "+4915211223456"


def test_normalize_phone_uses_phonenumbers_for_0049() -> None:
    assert normalize_phone("0049152 11223456") == "+4915211223456"


def test_normalize_phone_uses_phonenumbers_for_german_national_trunk() -> None:
    assert normalize_phone("0152 11223456") == "+4915211223456"


# ---------------------------------------------------------------------------
# validate_and_normalize_phone
# ---------------------------------------------------------------------------


def test_validate_and_normalize_phone_normalizes_to_e164() -> None:
    """Covers +49, 0049, and national-trunk (0xxx) formats — all produce E.164."""
    assert validate_and_normalize_phone("+49 152 11223456") == "+4915211223456"
    assert validate_and_normalize_phone("0049152 11223456") == "+4915211223456"
    assert validate_and_normalize_phone("0152 11223456") == "+4915211223456"


def test_validate_and_normalize_phone_rejects_impossible_number() -> None:
    with pytest.raises(ValueError):
        validate_and_normalize_phone("+49 1")


# ---------------------------------------------------------------------------
# validate_and_normalize_email
# ---------------------------------------------------------------------------


def test_validate_and_normalize_email_lowercases_normalized_form() -> None:
    assert validate_and_normalize_email("Max@Gmail.COM") == "max@gmail.com"


def test_validate_and_normalize_email_rejects_invalid_syntax() -> None:
    with pytest.raises(ValueError):
        validate_and_normalize_email("noemail")


def test_validate_and_normalize_email_does_not_require_dns_deliverability() -> None:
    assert (
        validate_and_normalize_email("person@does-not-exist-hopefully-zzzzzz.com")
        == "person@does-not-exist-hopefully-zzzzzz.com"
    )


# ---------------------------------------------------------------------------
# _ExtractedFields — phone_number validator
# ---------------------------------------------------------------------------


def test_extracted_fields_phone_validator() -> None:
    ef = _ExtractedFields(phone_number="+49 123 456789")
    assert ef.phone_number == "+49123456789"


def test_extracted_fields_phone_validation_failure() -> None:
    ef = _ExtractedFields(phone_number="not-a-phone")
    assert ef.phone_number is None


# ---------------------------------------------------------------------------
# _ExtractedFields — email validator
# ---------------------------------------------------------------------------


def test_extracted_fields_email_lowercased() -> None:
    ef = _ExtractedFields(email="Max@Gmail.COM")
    assert ef.email == "max@gmail.com"


def test_extracted_fields_email_validation_failure() -> None:
    ef = _ExtractedFields(email="noemail")
    assert ef.email is None


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
