"""Unicode security forensics: detect invisible, confusable and mixed-script content.

The shipped inference pipeline applies ``html.unescape``, NFKC, HTML stripping and
whitespace collapsing but deliberately does not strip Unicode *format* characters
(``Cf``).  Python's ``\\s`` does not match ``Cf`` characters, so zero-width and
bidi characters survive into the tokenised model input and silently break tokens.
This module measures that surface instead of silently deleting evidence.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Iterable

ZERO_WIDTH = frozenset({0x200B, 0x200C, 0x200D, 0x2060, 0x180E, 0xFEFF})
BIDI_CONTROLS = frozenset(
    {0x200E, 0x200F, 0x061C, *range(0x202A, 0x202F), *range(0x2066, 0x206A)}
)
ALLOWED_CONTROLS = frozenset({0x09, 0x0A, 0x0D})

CONFUSABLES: dict[str, str] = {
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "х": "x", "у": "y",
    "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C", "Х": "X", "У": "Y",
    "і": "i", "ѕ": "s", "ј": "j", "ԁ": "d", "ɡ": "g", "ӏ": "l", "ν": "v",
    "α": "a", "ο": "o", "ρ": "p", "τ": "t", "ε": "e", "ι": "i", "κ": "k",
    "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H", "Ι": "I", "Μ": "M",
    "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T", "Υ": "Y", "Χ": "X",
}

SCRIPT_RANGES: tuple[tuple[str, int, int], ...] = (
    ("LATIN", 0x0041, 0x024F), ("LATIN_EXT", 0x1E00, 0x1EFF),
    ("GREEK", 0x0370, 0x03FF), ("CYRILLIC", 0x0400, 0x04FF),
    ("ARMENIAN", 0x0530, 0x058F), ("HEBREW", 0x0590, 0x05FF),
    ("ARABIC", 0x0600, 0x06FF), ("DEVANAGARI", 0x0900, 0x097F),
    ("HIRAGANA", 0x3040, 0x309F), ("KATAKANA", 0x30A0, 0x30FF),
    ("HANGUL", 0xAC00, 0xD7AF), ("CJK", 0x4E00, 0x9FFF),
    ("FULLWIDTH", 0xFF00, 0xFFEF),
)

LATIN_SCRIPTS = frozenset({"LATIN", "LATIN_EXT", "FULLWIDTH"})


@dataclass(frozen=True)
class UnicodeFinding:
    """One distinct suspicious character observed in the raw email."""

    kind: str
    character: str
    codepoint: str
    name: str
    category: str
    count: int
    contexts: tuple[str, ...] = ()
    removed_by_hardening: bool = False

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "character": self.character,
            "codepoint": self.codepoint,
            "name": self.name,
            "category": self.category,
            "count": self.count,
            "contexts": list(self.contexts),
            "removed_by_hardening": self.removed_by_hardening,
        }


@dataclass
class UnicodeReport:
    findings: list[UnicodeFinding] = field(default_factory=list)
    mixed_script_tokens: list[dict] = field(default_factory=list)
    invisible_count: int = 0

    def as_dict(self) -> dict:
        return {
            "findings": [finding.as_dict() for finding in self.findings],
            "mixed_script_tokens": self.mixed_script_tokens,
            "invisible_count": self.invisible_count,
            "total_findings": len(self.findings),
        }


def script_of(character: str) -> str | None:
    point = ord(character)
    for name, start, end in SCRIPT_RANGES:
        if start <= point <= end:
            return name
    return None


def classify_character(character: str) -> str | None:
    """Return a threat-relevant kind for a character, or None if unremarkable."""
    point = ord(character)
    category = unicodedata.category(character)
    if point in ZERO_WIDTH:
        return "zero_width"
    if point in BIDI_CONTROLS:
        return "bidi_control"
    if category == "Cf":
        return "format_char"
    if category == "Cc" and point not in ALLOWED_CONTROLS:
        return "control_char"
    if category in {"Cs", "Co", "Cn"}:
        return "unassigned_or_private"
    if category in {"Zs", "Zl", "Zp"} and point != 0x20:
        return "unusual_space"
    return None


def _token_context(text: str, index: int, width: int = 48) -> str:
    start = index
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    end = index + 1
    while end < len(text) and not text[end].isspace():
        end += 1
    token = text[start:end]
    if len(token) > width:
        return token[:width] + "..."
    return token


def scan_unicode(text: str, hardened_text: str | None = None) -> UnicodeReport:
    """Scan raw email text for invisible, confusable and mixed-script content."""
    report = UnicodeReport()
    grouped: dict[tuple[str, str], dict] = {}
    for index, character in enumerate(text):
        kind = classify_character(character)
        if kind is None:
            continue
        key = (kind, character)
        entry = grouped.setdefault(
            key,
            {"count": 0, "contexts": [], "point": ord(character), "category": unicodedata.category(character)},
        )
        entry["count"] += 1
        if len(entry["contexts"]) < 4:
            entry["contexts"].append(_token_context(text, index))

    for (kind, character), entry in sorted(
        grouped.items(), key=lambda item: (item[0][0], item[0][1])
    ):
        if kind in {"zero_width", "bidi_control", "format_char", "control_char"}:
            report.invisible_count += entry["count"]
        report.findings.append(
            UnicodeFinding(
                kind=kind,
                character=character,
                codepoint=f"U+{entry['point']:04X}",
                name=unicodedata.name(character, "UNKNOWN"),
                category=entry["category"],
                count=entry["count"],
                contexts=tuple(entry["contexts"]),
                removed_by_hardening=bool(hardened_text is not None and character not in hardened_text),
            )
        )

    report.mixed_script_tokens = _find_mixed_script_tokens(text)
    return report


def _find_mixed_script_tokens(text: str, limit: int = 25) -> list[dict]:
    """Flag tokens that mix Latin with another script (the homoglyph attack shape)."""
    seen: dict[str, dict] = {}
    for token in text.split():
        scripts = {script_of(character) for character in token}
        if None in scripts:
            scripts.discard(None)
        if len(scripts) < 2:
            continue
        has_latin = bool(scripts & LATIN_SCRIPTS)
        names = sorted(scripts)
        if not has_latin:
            continue
        folded = fold_confusables(token)
        entry = seen.get(token)
        if entry is None:
            seen[token] = {"token": token[:80], "scripts": names, "folded": folded[:80], "count": 1}
        else:
            entry["count"] += 1
        if len(seen) >= limit:
            break
    return list(seen.values())


def fold_confusables(token: str) -> str:
    """Fold non-Latin homoglyphs to their Latin skeleton *within* a mixed-script token.

    Folding is intentionally restricted to tokens that already mix scripts so that
    genuinely non-Latin emails are never rewritten.
    """
    scripts = {script_of(character) for character in token}
    scripts.discard(None)
    if not (scripts & LATIN_SCRIPTS):
        return token
    return "".join(CONFUSABLES.get(character, character) for character in token)


def strip_invisible(text: str) -> str:
    """Remove format/control/bidi characters and normalise exotic spaces."""
    output: list[str] = []
    for character in text:
        point = ord(character)
        category = unicodedata.category(character)
        if category == "Cf" or point in ZERO_WIDTH or point in BIDI_CONTROLS:
            continue
        if category == "Cc":
            output.append("\n" if character == "\n" else " ")
            continue
        if category in {"Cs", "Co"}:
            continue
        output.append(character)
    return "".join(output)


def iter_removed_characters(text: str) -> Iterable[str]:
    for character in text:
        if unicodedata.category(character) == "Cf" or ord(character) in ZERO_WIDTH:
            yield character
