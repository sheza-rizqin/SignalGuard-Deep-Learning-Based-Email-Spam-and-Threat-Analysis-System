from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup

from .unicode_forensics import strip_invisible

MAX_EMAIL_CHARS = 80_000


@dataclass(frozen=True)
class EmailRecord:
    subject: str
    body: str

    @property
    def combined_text(self) -> str:
        return f"subject: {self.subject}\nbody: {self.body}".strip()


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if value != value:
            return ""
    except Exception:
        return ""
    return str(value)


def clean_text(text: Any, max_chars: int = MAX_EMAIL_CHARS) -> str:
    """Normalize email text without removing useful spam indicators."""
    value = html.unescape(_as_text(text))
    value = unicodedata.normalize("NFKC", value)
    value = strip_invisible(value)
    value = BeautifulSoup(value, "html.parser").get_text(" ")
    value = re.sub(r"https?://\S+|www\.\S+", " URLTOKEN ", value, flags=re.I)
    value = re.sub(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}", " EMAILTOKEN ", value)
    value = re.sub(r"\b\+?\d[\d\s().-]{6,}\d\b", " PHONETOKEN ", value)
    value = re.sub(r"(.)\1{3,}", r"\1\1\1", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value[:max_chars]


def normalize_email(subject: Any, body: Any) -> EmailRecord:
    """Create the exact representation consumed by the neural model."""
    return EmailRecord(clean_text(subject, 2_000), clean_text(body))


def parse_eml(content: bytes) -> EmailRecord:
    """Parse an RFC email as untrusted text; never execute HTML or attachments."""
    message = BytesParser(policy=policy.default).parsebytes(content)
    subject = message.get("subject", "")
    parts: list[str] = []
    if message.is_multipart():
        for part in message.walk():
            if part.get_content_maintype() == "multipart":
                continue
            if part.get_content_disposition() == "attachment":
                continue
            if part.get_content_type() not in {"text/plain", "text/html"}:
                continue
            try:
                parts.append(part.get_content())
            except (LookupError, UnicodeError):
                continue
    else:
        try:
            parts.append(message.get_content())
        except (LookupError, UnicodeError):
            parts.append(content.decode("utf-8", errors="replace"))
    return normalize_email(subject, "\n".join(parts))


def parse_uploaded_file(filename: str, content: bytes) -> EmailRecord:
    """Parse only supported text email formats and reject unsafe extensions."""
    suffix = Path(filename).suffix.lower()
    if suffix == ".eml":
        return parse_eml(content[:MAX_EMAIL_CHARS * 2])
    if suffix == ".txt":
        return normalize_email("", content[:MAX_EMAIL_CHARS].decode("utf-8", errors="replace"))
    raise ValueError("Only .eml and .txt files are supported.")


def normalize_label(value: Any) -> int:
    """Map common spam/ham labels to 1/0 without case assumptions."""
    label = _as_text(value).strip().lower()
    if label in {"spam", "1", "true", "yes", "junk", "phishing"}:
        return 1
    if label in {"ham", "not spam", "not_spam", "0", "false", "no", "legitimate"}:
        return 0
    raise ValueError(f"Unsupported label: {value!r}")
