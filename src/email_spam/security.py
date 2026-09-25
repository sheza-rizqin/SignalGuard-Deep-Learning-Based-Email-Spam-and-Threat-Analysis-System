from __future__ import annotations

from pathlib import Path

MAX_UPLOAD_BYTES = 512 * 1024
MAX_RAW_PREVIEW_CHARS = 20_000
ALLOWED_EXTENSIONS = frozenset({".eml", ".txt"})
BINARY_SNIFF_BYTES = 2048


class UploadRejected(ValueError):
    pass


def sanitize_filename(filename: str | None) -> str:
    if not filename:
        return ""
    base = Path(str(filename).replace("\\", "/")).name.strip()
    return "".join(character for character in base if character.isprintable())


def validate_upload(filename: str | None, content: bytes) -> str:
    if not content:
        raise UploadRejected("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        raise UploadRejected(
            f"The uploaded file is {len(content) / 1024:.0f} KB, which exceeds the "
            f"{MAX_UPLOAD_BYTES // 1024} KB limit."
        )
    safe_name = sanitize_filename(filename)
    if not safe_name:
        raise UploadRejected("The uploaded file has no usable name.")
    if ".." in str(filename):
        raise UploadRejected("The uploaded file name contains a path traversal sequence.")
    extension = Path(safe_name).suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise UploadRejected(
            "Only .eml and .txt files are supported. Got: " + (extension or "no extension")
        )
    if b"\x00" in content[:BINARY_SNIFF_BYTES]:
        raise UploadRejected("The uploaded file appears to be binary, not a text email.")
    return safe_name


def decode_upload(content: bytes) -> str:
    if content.startswith(b"\xef\xbb\xbf"):
        return content.decode("utf-8-sig", errors="replace")
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        return content.decode("utf-16", errors="replace")
    if content.startswith(b"=?") and content.count(b"?") > 4:
        return content.decode("utf-8", errors="replace")
    try:
        return content.decode("utf-8")
    except UnicodeDecodeError:
        return content.decode("latin-1", errors="replace")


def inert_preview(raw: str, limit: int = MAX_RAW_PREVIEW_CHARS) -> str:
    if len(raw) <= limit:
        return raw
    return raw[:limit] + f"\n\n... [truncated for display at {limit} characters]"
def extract_raw_email(filename: str, content: bytes) -> tuple[str, str]:
    suffix = Path(sanitize_filename(filename)).suffix.lower()
    if suffix == ".txt":
        return "", decode_upload(content)[:MAX_UPLOAD_BYTES]

    import email
    from email import policy
    from email.parser import BytesParser

    try:
        message = BytesParser(policy=policy.default).parsebytes(content)
    except Exception:
        return "", decode_upload(content)
    subject = str(message.get("subject", "") or "")
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
            parts.append(decode_upload(content))
    body = "\n".join(part for part in parts if isinstance(part, str))
    return subject, body
