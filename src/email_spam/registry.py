
from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

from .model_io import load_spam_model
ARTIFACT_FILES = {
    "model": "email_spam_lstm.keras",
    "tokenizer": "tokenizer.json",
    "metadata": "metadata.json",
}
OPTIONAL_ARTIFACTS = {
    "evaluation": "evaluation.json",
    "calibration": "calibration.json",
}

MAX_DISPLAY_EVIDENCE_WEIGHT = 0.75


@dataclass(frozen=True)
class Prediction:
    """A single measured model outcome."""

    text: str
    probability: float
    label: str
    calibrated: float | None
    evidence_adjusted: float | None
    evidence: dict | None
    latency_ms: float

    def as_dict(self) -> dict:
        return {
            "probability": self.probability,
            "label": self.label,
            "calibrated": self.calibrated,
            "evidence_adjusted": self.evidence_adjusted,
            "evidence": self.evidence,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True)
class BatchOutcome:
    predictions: list[Prediction]
    total_latency_ms: float
    per_item_latency_ms: float
    batch_size: int


def _read_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _sha256(path: Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()[:16]


def _logit(probability: float) -> float:
    clipped = min(max(probability, 1e-9), 1 - 1e-9)
    return float(np.log(clipped / (1 - clipped)))


def _sigmoid(value: float) -> float:
    return float(1.0 / (1.0 + np.exp(-value)))


class ModelRegistry:
    """Owns the only in-process copy of the trained network and its tokenizer."""

    _lock = threading.Lock()
    _instances: dict[str, "ModelRegistry"] = {}

    def __init__(self, artifact_dir: str | Path = "models") -> None:
        self.artifact_dir = Path(artifact_dir)
        paths = {key: self.artifact_dir / name for key, name in ARTIFACT_FILES.items()}
        missing = [path.name for path in paths.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Model artifacts are missing: " + ", ".join(missing) + ". Train an email dataset first."
            )
        self.model_path = paths["model"]
        self.tokenizer = tokenizer_from_json(paths["tokenizer"].read_text(encoding="utf-8"))
        self.metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))
        self.sequence_length = int(self.metadata.get("sequence_length", 300))
        self.threshold = float(self.metadata.get("threshold", 0.5))
        self.auxiliary = {
            key: _read_json(self.artifact_dir / name) for key, name in OPTIONAL_ARTIFACTS.items()
        }
        self.model = load_spam_model(self.model_path)
        self._forward = tf.function(
            lambda values: self.model(values, training=False),
            input_signature=[tf.TensorSpec(shape=(None, self.sequence_length), dtype=tf.int32)],
            reduce_retracing=True,
        )
        self._inference_lock = threading.Lock()
        self.parameter_count = int(self.model.count_params())
        self.model_bytes = int(self.model_path.stat().st_size)
        self.model_hash = _sha256(self.model_path)
        self.loaded_at = time.time()
        self.warmup_latency_ms = self._warm_up()

    @classmethod
    def instance(cls, artifact_dir: str | Path = "models") -> "ModelRegistry":
        """Return the process-wide singleton for the given artifact directory."""
        key = str(Path(artifact_dir).resolve())
        with cls._lock:
            registry = cls._instances.get(key)
            if registry is None:
                registry = cls(artifact_dir)
                cls._instances[key] = registry
            return registry

    def _warm_up(self) -> float:
        """Pay the first-call graph-tracing cost at load time, not on a request."""
        probe = np.zeros((1, self.sequence_length), dtype="int32")
        started = time.perf_counter()
        self._forward(probe)
        return (time.perf_counter() - started) * 1000.0

    def version(self) -> str:
        return self.model_hash

    def calibration(self) -> dict | None:
        return self.auxiliary.get("calibration")

    def temperature(self) -> float:
        payload = self.calibration() or {}
        return float(payload.get("temperature", 1.0)) or 1.0

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        """Tokenise and pad text into the exact tensor shape the model expects."""
        sequences = self.tokenizer.texts_to_sequences(list(texts))
        return pad_sequences(
            sequences,
            maxlen=self.sequence_length,
            padding="post",
            truncating="post",
        )

    def probabilities(self, encoded: np.ndarray) -> np.ndarray:
        """Forward pass over an already-encoded tensor."""
        with self._inference_lock:
            return np.asarray(self._forward(encoded)).reshape(-1)

    def predict_batch(self, texts: Sequence[str]) -> BatchOutcome:
        """Run one batched forward pass and time the whole batch."""
        values = list(texts)
        if not values:
            return BatchOutcome([], 0.0, 0.0, 0)
        encoded = self.encode(values)
        start = time.perf_counter()
        probabilities = self.probabilities(encoded)
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        temperature = self.temperature()
        predictions = [
            self._to_prediction(text, float(probability), temperature, self._evidence(encoded[index]))
            for index, (text, probability) in enumerate(zip(values, probabilities))
        ]
        return BatchOutcome(
            predictions=predictions,
            total_latency_ms=elapsed_ms,
            per_item_latency_ms=elapsed_ms / len(values),
            batch_size=len(values),
        )

    def predict_text(self, text: str) -> Prediction:
        return self.predict_batch([text]).predictions[0]

    def _evidence(self, encoded: np.ndarray) -> dict:
        """Quantify input support without running another model pass.

        A classifier cannot infer uncertainty from unfamiliar tokens by itself.
        This uses token count and OOV rate to expose when a confident score is
        based on too little recognised text. Twelve recognised tokens count as
        full support; a 50% OOV rate counts as no lexical support.
        """
        active = np.asarray(encoded)[np.asarray(encoded) != 0]
        count = int(len(active))
        oov_id = int(self.tokenizer.word_index.get(self.tokenizer.oov_token, 1))
        oov_rate = float(np.mean(active == oov_id)) if count else 1.0
        length_support = min(count / 12.0, 1.0)
        vocabulary_support = max(0.0, 1.0 - (oov_rate / 0.5))
        support = float(length_support * vocabulary_support)
        return {
            "token_count": count,
            "oov_rate": oov_rate,
            "support": support,
            "level": "HIGH" if support >= 0.8 else "MEDIUM" if support >= 0.5 else "LOW",
        }

    def _to_prediction(
        self, text: str, probability: float, temperature: float, evidence: dict | None = None
    ) -> Prediction:
        label = "SPAM" if probability >= self.threshold else "NOT SPAM"
        calibrated = None
        if self.calibration():
            calibrated = _sigmoid(_logit(probability) / temperature)
        base_probability = probability if calibrated is None else calibrated
        evidence_adjusted = None
        if evidence is not None:
            support = float(evidence["support"])
            display_weight = min(support, MAX_DISPLAY_EVIDENCE_WEIGHT)
            evidence_adjusted = float(0.5 + display_weight * (base_probability - 0.5))
            evidence = {**evidence, "display_weight": display_weight}
        return Prediction(
            text=text,
            probability=probability,
            label=label,
            calibrated=calibrated,
            evidence_adjusted=evidence_adjusted,
            evidence=evidence,
            latency_ms=0.0,
        )

    def vocabulary_size(self) -> int:
        return int(self.metadata.get("vocabulary_size", 0))

    def artifact_inventory(self) -> dict:
        """Report what is on disk, so the UI can show 'not available' honestly."""
        inventory = {}
        for key, name in {**ARTIFACT_FILES, **OPTIONAL_ARTIFACTS}.items():
            path = self.artifact_dir / name
            inventory[key] = {
                "file": name,
                "present": path.exists(),
                "bytes": int(path.stat().st_size) if path.exists() else 0,
            }
        return inventory


def token_ids(tokenizer, text: str, limit: int = 40) -> list[dict]:
    """Expose raw tokenisation so broken tokens become visible evidence."""
    values = tokenizer.texts_to_sequences([text])[0]
    index = {value: token for token, value in tokenizer.word_index.items()}
    return [{"id": int(value), "token": index.get(value, "<UNK>")} for value in values[:limit]]
