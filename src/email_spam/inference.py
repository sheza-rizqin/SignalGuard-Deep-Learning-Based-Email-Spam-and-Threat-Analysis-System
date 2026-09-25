from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .calibration import apply_temperature
from .model_io import load_spam_model
from .preprocessing import EmailRecord, normalize_email


class EmailClassifier:
    def __init__(self, artifact_dir: str | Path = "models"):
        artifact_dir = Path(artifact_dir)
        model_path = artifact_dir / "email_spam_lstm.keras"
        tokenizer_path = artifact_dir / "tokenizer.json"
        metadata_path = artifact_dir / "metadata.json"
        missing = [str(path.name) for path in (model_path, tokenizer_path, metadata_path) if not path.exists()]
        if missing:
            raise FileNotFoundError("Model artifacts are missing: " + ", ".join(missing) + ". Train an email dataset first.")
        self.model = load_spam_model(model_path)
        self.tokenizer = tokenizer_from_json(tokenizer_path.read_text(encoding="utf-8"))
        self.metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        calibration_path = artifact_dir / "calibration.json"
        self.calibration = json.loads(calibration_path.read_text(encoding="utf-8")) if calibration_path.exists() else None

    def predict(self, subject: str, body: str) -> dict:
        email = normalize_email(subject, body)
        if not email.subject and not email.body:
            raise ValueError("Enter an email subject or body before analyzing.")
        sequence = self.tokenizer.texts_to_sequences([email.combined_text])
        padded = pad_sequences(sequence, maxlen=self.metadata["sequence_length"], padding="post", truncating="post")
        raw_probability = float(self.model.predict(padded, verbose=0)[0][0])
        temperature = float((self.calibration or {}).get("temperature", 1.0))
        probability = float(apply_temperature(np.asarray([raw_probability]), temperature)[0])
        threshold = float(self.metadata.get("threshold", 0.5))
        is_spam = raw_probability >= threshold
        return {
            "prediction": "SPAM" if is_spam else "NOT SPAM",
            "probability_spam": probability,
            "raw_probability_spam": raw_probability,
            "confidence": probability if is_spam else 1 - probability,
            "risk": "HIGH" if probability >= 0.8 else "MEDIUM" if probability >= 0.5 else "LOW",
            "explanations": explain(email, probability),
        }


def explain(email: EmailRecord, probability: float) -> list[dict[str, str]]:
    """Report observable, deterministic signals; never claim they are model attributions."""
    text = email.combined_text.lower()
    signals: list[dict[str, str]] = []
    patterns = [
        (r"\b(urgent|act now|limited time|verify your account|final notice)\b", "Urgency or account-pressure language detected."),
        (r"\b(free|winner|prize|cash|bonus|claim)\b", "Promotional or reward-oriented language detected."),
        (r"urltoken", "One or more URLs were detected and normalized."),
        (r"emailtoken", "An email address was detected and normalized."),
    ]
    for pattern, message in patterns:
        if re.search(pattern, text):
            signals.append({"type": "supporting_signal", "message": message})
    if text.count("!") >= 3:
        signals.append({"type": "supporting_signal", "message": "Unusually frequent exclamation marks detected."})
    if not signals:
        signals.append({"type": "model_context", "message": "No common surface-level spam signals were found; the result comes from the trained LSTM."})
    signals.append({"type": "model_context", "message": f"The trained LSTM assigned a spam probability of {probability:.1%}."})
    return signals
