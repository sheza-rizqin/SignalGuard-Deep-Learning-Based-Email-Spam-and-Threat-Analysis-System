from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from .preprocessing import normalize_email, normalize_label

DATE_SOURCES: dict[str, dict[str, str]] = {
    "SpamAssasin.csv": {"date": "date", "subject": "subject", "body": "body", "label": "label"},
    "Nigerian_Fraud.csv": {"date": "date", "subject": "subject", "body": "body", "label": "label"},
    "CEAS_08.csv": {"date": "date", "subject": "subject", "body": "body", "label": "label"},
    "Nazario.csv": {"date": "date", "subject": "subject", "body": "body", "label": "label"},
}

MIN_YEAR = 1998
MAX_YEAR = 2025


def available_sources(directory: str | Path = "data/raw") -> list[str]:
    """Timestamped files that are actually present on disk."""
    base = Path(directory)
    return [name for name in DATE_SOURCES if (base / name).exists()]


def load_source(path: str | Path, limit: int | None = None) -> pd.DataFrame:
    """Load one timestamped CSV into a normalised frame with parsed dates."""
    path = Path(path)
    spec = DATE_SOURCES[path.name]
    frame = pd.read_csv(path, usecols=[spec["date"], spec["body"], spec["label"]] +
                        ([spec["subject"]] if spec["subject"] else []))
    if limit is not None and len(frame) > limit:
        frame = frame.sample(n=limit, random_state=42)
    parsed = pd.to_datetime(frame[spec["date"]], format="mixed", utc=True, errors="coerce")
    records = []
    for position, (_, row) in enumerate(frame.iterrows()):
        timestamp = parsed.iloc[position]
        if pd.isna(timestamp) or not (MIN_YEAR <= timestamp.year <= MAX_YEAR):
            continue
        try:
            label = normalize_label(row[spec["label"]])
        except ValueError:
            continue
        email = normalize_email(row.get(spec["subject"], ""), row[spec["body"]])
        if not email.combined_text:
            continue
        records.append({"text": email.combined_text, "label": label, "date": timestamp, "source": path.name})
    return pd.DataFrame(records)


def load_pooled(
    directory: str | Path = "data/raw",
    sources: Sequence[str] | None = None,
    limit_per_source: int | None = None,
) -> pd.DataFrame:
    """Concatenate every timestamped source into one dated frame."""
    base = Path(directory)
    names = list(sources) if sources else available_sources(directory)
    frames = [load_source(base / name, limit_per_source) for name in names]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        return pd.DataFrame(columns=["text", "label", "date", "source"])
    pooled = pd.concat(frames, ignore_index=True)
    return pooled.sort_values("date").reset_index(drop=True)


def temporal_split(frame: pd.DataFrame, train: float = 0.7, validation: float = 0.15) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological split: earliest rows train, latest rows test."""
    ordered = frame.sort_values("date").reset_index(drop=True)
    train_end = int(len(ordered) * train)
    validation_end = int(len(ordered) * (train + validation))
    return (
        ordered.iloc[:train_end].reset_index(drop=True),
        ordered.iloc[train_end:validation_end].reset_index(drop=True),
        ordered.iloc[validation_end:].reset_index(drop=True),
    )


def random_split(frame: pd.DataFrame, train: float = 0.7, validation: float = 0.15, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified random split over the same pooled rows, for a like-for-like comparison."""
    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train)
    validation_end = int(len(shuffled) * (train + validation))
    return (
        shuffled.iloc[:train_end].reset_index(drop=True),
        shuffled.iloc[train_end:validation_end].reset_index(drop=True),
        shuffled.iloc[validation_end:].reset_index(drop=True),
    )


def describe_split(frame: pd.DataFrame) -> dict:
    """Describe a split honestly, including its class balance and era composition."""
    if frame.empty:
        return {"rows": 0, "spam_ratio": None, "start": None, "end": None, "sources": {}}
    return {
        "rows": int(len(frame)),
        "spam_ratio": float(frame["label"].mean()),
        "start": frame["date"].min().isoformat(),
        "end": frame["date"].max().isoformat(),
        "sources": {str(key): int(value) for key, value in frame["source"].value_counts().items()},
    }


def era_label(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "unknown"
    start, end = frame["date"].min().year, frame["date"].max().year
    return str(start) if start == end else f"{start}-{end}"


def source_summary(frame: pd.DataFrame) -> list[dict]:
    """Per-source class balance and date range, so confounds are visible."""
    rows: list[dict] = []
    for name, group in frame.groupby("source"):
        rows.append(
            {
                "source": str(name),
                "rows": int(len(group)),
                "spam": int(group["label"].sum()),
                "ham": int((group["label"] == 0).sum()),
                "start": group["date"].min().isoformat(),
                "end": group["date"].max().isoformat(),
            }
        )
    return sorted(rows, key=lambda row: row["start"])


def text_column(frame: pd.DataFrame) -> Iterable[str]:
    return frame["text"].tolist()
