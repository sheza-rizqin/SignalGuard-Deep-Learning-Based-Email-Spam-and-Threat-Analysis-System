from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

from .preprocessing import normalize_email, normalize_label


def load_email_csv(path: str | Path, require_both_labels: bool = True) -> pd.DataFrame:
    """Load common subject/body/label CSV schemas into a normalized frame."""
    frame = pd.read_csv(path)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    label_column = next((columns[name] for name in ("label", "class", "category", "target") if name in columns), None)
    if label_column is None:
        raise ValueError("Dataset must contain a label, class, category, or target column.")
    body_column = next((columns[name] for name in ("body", "text", "message", "content") if name in columns), None)
    if body_column is None:
        raise ValueError("Dataset must contain a body, text, message, or content column.")
    subject_column = next((columns[name] for name in ("subject", "title") if name in columns), None)
    records = []
    for _, row in frame.iterrows():
        email = normalize_email(row[subject_column] if subject_column else "", row[body_column])
        try:
            label = normalize_label(row[label_column])
        except ValueError:
            continue
        if email.combined_text:
            records.append({"text": email.combined_text, "label": label})
    result = pd.DataFrame(records).drop_duplicates(subset=["text", "label"]).reset_index(drop=True)
    if result.empty:
        raise ValueError("Dataset contains no usable email examples.")
    if require_both_labels and result["label"].nunique() < 2:
        raise ValueError("Dataset must contain both spam and legitimate examples.")
    return result


def split_dataset(frame: pd.DataFrame, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create stratified train/validation/test partitions before fitting a tokenizer."""
    train, holdout = train_test_split(frame, test_size=0.2, random_state=seed, stratify=frame["label"])
    validation, test = train_test_split(holdout, test_size=0.5, random_state=seed, stratify=holdout["label"])
    return train.reset_index(drop=True), validation.reset_index(drop=True), test.reset_index(drop=True)
