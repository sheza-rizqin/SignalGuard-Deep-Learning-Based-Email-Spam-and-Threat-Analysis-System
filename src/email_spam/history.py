from __future__ import annotations

import hashlib
import threading
from collections import deque
from datetime import datetime, timezone

MAX_ENTRIES = 200

PRIVACY_NOTICE = (
    "Privacy: this application does not store email content. History keeps only a truncated "
    "fingerprint of the analysed text plus decision metadata (time, prediction, probability, "
    "calibrated confidence, model version and indicator counts). History is in-memory and is "
    "cleared when the server restarts. Nothing is written to disk."
)


def fingerprint(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:16]


class HistoryStore:

    def __init__(self, max_entries: int = MAX_ENTRIES) -> None:
        self._entries: deque[dict] = deque(maxlen=max_entries)
        self._lock = threading.Lock()

    def record(
        self,
        *,
        prediction: str,
        probability: float,
        calibrated: float | None,
        model_version: str,
        text: str,
        unicode_findings: int = 0,
        url_findings: int = 0,
        persuasion_signals: int = 0,
        prediction_stability: float | None = None,
        token_count: int = 0,
        oov_rate: float = 0.0,
        inference_ms: float = 0.0,
        total_ms: float = 0.0,
    ) -> dict:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "fingerprint": fingerprint(text),
            "prediction": prediction,
            "probability": round(float(probability), 6),
            "calibrated": None if calibrated is None else round(float(calibrated), 6),
            "model_version": model_version,
            "unicode_findings": int(unicode_findings),
            "url_findings": int(url_findings),
            "persuasion_signals": int(persuasion_signals),
            "prediction_stability": prediction_stability,
            "token_count": int(token_count),
            "oov_rate": round(float(oov_rate), 6),
            "inference_ms": round(float(inference_ms), 3),
            "total_ms": round(float(total_ms), 3),
        }
        with self._lock:
            self._entries.append(entry)
        return entry

    def entries(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return list(self._entries)[-limit:][::-1]

    def latest(self) -> dict | None:
        with self._lock:
            return dict(self._entries[-1]) if self._entries else None

    def summary(self) -> dict:
        with self._lock:
            entries = list(self._entries)
        spam = sum(1 for entry in entries if entry["prediction"] == "SPAM")
        return {
            "analyzed": len(entries),
            "spam_count": spam,
            "legitimate_count": len(entries) - spam,
            "with_hidden_characters": sum(1 for entry in entries if entry["unicode_findings"] > 0),
            "recent": entries[-6:][::-1],
            "stores_content": False,
            "privacy_notice": PRIVACY_NOTICE,
        }

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()
