"""Shared phone and text normalization utilities.

``clean_phone`` strips formatting characters (spaces, hyphens, parentheses)
while preserving any leading ``+`` prefix.  This is a format-only transform.

``normalize_phone`` is intentionally non-throwing so the Evaluator can compare
messy predictions against ground truth without failing the whole report.

``validate_and_normalize_*`` helpers are strict enough for extraction-time
validation and raise ``ValueError`` when a model output is malformed.
"""

from __future__ import annotations

import re

import phonenumbers
from email_validator import EmailNotValidError, validate_email

_DEFAULT_REGION = "DE"


def clean_phone(s: str) -> str:
    """Strip spaces, hyphens, and parentheses from *s*, preserving any leading ``+``.

    This is a formatting-only transform; country-code prefix rewriting is not
    performed here.  Callers that need full E.164 normalisation should call
    :func:`normalize_phone`.

    Examples::

        >>> clean_phone("+49 152 11223456")
        '+4915211223456'
        >>> clean_phone("+49-152-11223456")
        '+4915211223456'
        >>> clean_phone("+49(152)11223456")
        '+4915211223456'
    """
    return re.sub(r"[\s\-()]", "", s)


def normalize_phone(s: str, default_region: str = _DEFAULT_REGION) -> str:
    """Normalize a phone number to E.164 where possible, otherwise return a cleaned value.

    This function is used for evaluation comparisons and must not throw.  When
    ``phonenumbers`` cannot parse *s* or the parsed number is not possible, the
    raw cleaned string (spaces/hyphens/parens stripped) is returned as-is.
    """
    try:
        parsed = phonenumbers.parse(clean_phone(s), default_region)
    except phonenumbers.NumberParseException:
        return clean_phone(s)

    if phonenumbers.is_possible_number(parsed):
        return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)
    return clean_phone(s)


def validate_and_normalize_phone(s: str, default_region: str = _DEFAULT_REGION) -> str:
    """Validate *s* as a possible phone number and return E.164 format.

    ``is_possible_number`` is intentionally used instead of ``is_valid_number``
    so synthetic-but-plausible challenge numbers are not rejected.
    """
    cleaned = clean_phone(s)
    try:
        parsed = phonenumbers.parse(cleaned, default_region)
    except phonenumbers.NumberParseException as exc:
        raise ValueError(f"Invalid phone number: {s!r}") from exc

    if not phonenumbers.is_possible_number(parsed):
        raise ValueError(f"Invalid phone number: {s!r}")

    return phonenumbers.format_number(parsed, phonenumbers.PhoneNumberFormat.E164)


def validate_and_normalize_email(s: str) -> str:
    """Validate email syntax and return the normalized lowercase address.

    DNS deliverability checks are disabled because extracted transcript data and
    challenge fixtures should be judged by syntax, not external DNS state.
    """
    try:
        result = validate_email(s.strip(), check_deliverability=False)
    except EmailNotValidError as exc:
        raise ValueError(f"Invalid email address: {s!r}") from exc
    return result.normalized.lower()
