"""Shared phone and text normalization utilities.

``clean_phone`` strips formatting characters (spaces, hyphens, parentheses)
while preserving any leading ``+`` prefix.  This is a format-only transform
used by ``_ExtractedFields`` validators so that the ``Field(pattern=...)``
check always sees a clean ``+digits`` string regardless of how the LLM
formatted the raw output.

``normalize_phone`` extends ``clean_phone`` with E.164 prefix rewriting for
German numbers.  It is the same logic previously inlined in
``evaluation.py`` and is used by the Evaluator for ground-truth comparison.
"""

from __future__ import annotations

import re


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


def normalize_phone(s: str) -> str:
    """Normalise a German phone number to E.164 format where possible.

    Handles three common prefix forms:

    - ``+49…``   → strip internal whitespace/dashes/parens, keep as-is
    - ``0049…``  → replace leading ``0049`` with ``+49``
    - ``0…``     → replace leading ``0`` with ``+49`` (national trunk)
    - Otherwise  → return stripped string unchanged (avoids silent corruption)
    """
    stripped = clean_phone(s)
    if stripped.startswith("+49"):
        return stripped
    if stripped.startswith("0049"):
        return "+49" + stripped[4:]
    if stripped.startswith("0"):
        return "+49" + stripped[1:]
    return stripped
