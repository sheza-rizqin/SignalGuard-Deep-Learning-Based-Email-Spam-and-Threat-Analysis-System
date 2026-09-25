from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .preprocessing import MAX_EMAIL_CHARS, clean_text
from .unicode_forensics import (
    classify_character,
    fold_confusables,
    strip_invisible,
)

WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class NormalizationView:
    """One representation of the same email handed to the model."""

    key: str
    label: str
    description: str
    subject: str
    body: str

    @property
    def combined_text(self) -> str:
        return f"subject: {self.subject}\nbody: {self.body}".strip()

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "label": self.label,
            "description": self.description,
            "subject": self.subject,
            "body": self.body,
            "combined_text": self.combined_text,
        }


def _clip(value: Any, limit: int = MAX_EMAIL_CHARS) -> str:
    return "" if value is None else str(value)[:limit]


def harden_text(text: Any) -> str:
    """Remove invisible characters and fold mixed-script homoglyphs.

    This is applied *after* ``clean_text`` so the only difference between the
    as-shipped and hardened representations is the hardening step itself.
    """
    cleaned = clean_text(text)
    stripped = strip_invisible(cleaned)
    folded = " ".join(fold_confusables(token) for token in stripped.split())
    return WHITESPACE_RE.sub(" ", folded).strip()


def build_views(subject: Any, body: Any) -> list[NormalizationView]:
    """Return the raw, as-shipped and hardened representations of one email."""
    raw_subject = _clip(subject, 2_000)
    raw_body = _clip(body)
    shipped = clean_text(raw_body)
    return [
        NormalizationView(
            key="raw",
            label="Raw input",
            description="Exactly what was submitted, with no normalisation applied.",
            subject=raw_subject,
            body=raw_body,
        ),
        NormalizationView(
            key="shipped",
            label="As-shipped pipeline",
            description=(
                "html.unescape -> NFKC -> HTML strip -> URL/email/phone masking -> "
                "whitespace collapse. This is the representation the deployed model "
                "was trained and evaluated on."
            ),
            subject=clean_text(raw_subject, 2_000),
            body=shipped,
        ),
        NormalizationView(
            key="hardened",
            label="Security-hardened",
            description=(
                "As-shipped output with Unicode Cf/Cc/bidi characters removed and "
                "mixed-script homoglyphs folded to Latin. Non-Latin scripts are "
                "never rewritten."
            ),
            subject=harden_text(raw_subject),
            body=harden_text(raw_body),
        ),
    ]


def removed_character_counts(raw: str, processed: str) -> dict[str, int]:
    """Count which suspicious characters the given pipeline stage eliminated."""
    counts: dict[str, int] = {}
    for character in raw:
        kind = classify_character(character)
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) + 1
    for character in processed:
        kind = classify_character(character)
        if kind is None:
            continue
        counts[kind] = counts.get(kind, 0) - 1
    return {kind: max(value, 0) for kind, value in counts.items()}


def token_differences(shipped: str, hardened: str, limit: int = 40) -> list[dict[str, str]]:
    """Report positions where hardening changed a whitespace-delimited token."""
    left = shipped.split()
    right = hardened.split()
    changes: list[dict[str, str]] = []
    for index in range(max(len(left), len(right))):
        before = left[index] if index < len(left) else ""
        after = right[index] if index < len(right) else ""
        if before != after:
            changes.append({"before": before[:80], "after": after[:80]})
        if len(changes) >= limit:
            break
    return changes
