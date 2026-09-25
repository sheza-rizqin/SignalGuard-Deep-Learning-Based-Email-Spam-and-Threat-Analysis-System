from __future__ import annotations

import json
from pathlib import Path

import numpy as np

_EPSILON = 1e-9
_LOG_T_LOW = -3.0
_LOG_T_HIGH = 3.0


def logits(probabilities: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probabilities, dtype=float), _EPSILON, 1 - _EPSILON)
    return np.log(clipped / (1 - clipped))


def apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    scaled = logits(probabilities) / max(float(temperature), _EPSILON)
    return 1.0 / (1.0 + np.exp(-scaled))


def negative_log_likelihood(probabilities: np.ndarray, labels: np.ndarray) -> float:
    clipped = np.clip(np.asarray(probabilities, dtype=float), _EPSILON, 1 - _EPSILON)
    return float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log(1 - clipped)))


def fit_temperature(probabilities: np.ndarray, labels: np.ndarray) -> float:
    """Golden-section search for the temperature minimising held-out NLL."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)

    def objective(log_temperature: float) -> float:
        return negative_log_likelihood(apply_temperature(probabilities, np.exp(log_temperature)), labels)

    low, high = _LOG_T_LOW, _LOG_T_HIGH
    ratio = (5 ** 0.5 - 1) / 2
    left = high - ratio * (high - low)
    right = low + ratio * (high - low)
    left_value, right_value = objective(left), objective(right)
    for _ in range(200):
        if high - low < 1e-6:
            break
        if left_value < right_value:
            high, right, right_value = right, left, left_value
            left = high - ratio * (high - low)
            left_value = objective(left)
        else:
            low, left, left_value = left, right, right_value
            right = low + ratio * (high - low)
            right_value = objective(right)
    return float(np.exp((low + high) / 2))


def expected_calibration_error(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> float:
    return calibration_bins(probabilities, labels, bins)["expected_calibration_error"]


def brier_score(probabilities: np.ndarray, labels: np.ndarray) -> float:
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    return float(np.mean((probabilities - labels) ** 2))


def evaluate_temperature(
    probabilities: np.ndarray, labels: np.ndarray, temperature: float, bins: int = 15
) -> dict:
    """Score a fixed temperature on data that was not used to fit it.

    This is deliberately separate from :func:`calibration_report`: callers can
    fit the scalar on validation data, then use the untouched test split only
    to report generalisation of the calibration.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    calibrated = apply_temperature(probabilities, temperature)
    return {
        "samples": int(len(probabilities)),
        "before": {
            "brier": brier_score(probabilities, labels),
            "ece": expected_calibration_error(probabilities, labels, bins),
            "negative_log_likelihood": negative_log_likelihood(probabilities, labels),
        },
        "after": {
            "brier": brier_score(calibrated, labels),
            "ece": expected_calibration_error(calibrated, labels, bins),
            "negative_log_likelihood": negative_log_likelihood(calibrated, labels),
        },
        "reliability": calibration_bins(calibrated, labels, bins),
    }


def calibration_bins(probabilities: np.ndarray, labels: np.ndarray, bins: int = 15) -> dict:
    """Reliability diagram data plus the aggregate ECE."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(probabilities)
    rows: list[dict] = []
    ece = 0.0
    if total == 0:
        return {"expected_calibration_error": 0.0, "bins": []}
    for index in range(bins):
        low, high = edges[index], edges[index + 1]
        mask = (probabilities > low) & (probabilities <= high) if index else (probabilities >= low) & (probabilities <= high)
        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {"lower": float(low), "upper": float(high), "count": 0, "mean_confidence": 0.0,
                 "mean_accuracy": 0.0, "gap": 0.0}
            )
            continue
        mean_confidence = float(np.mean(probabilities[mask]))
        mean_accuracy = float(np.mean(labels[mask]))
        gap = abs(mean_confidence - mean_accuracy)
        ece += (count / total) * gap
        rows.append(
            {
                "lower": float(low),
                "upper": float(high),
                "count": count,
                "mean_confidence": mean_confidence,
                "mean_accuracy": mean_accuracy,
                "gap": float(gap),
            }
        )
    return {"expected_calibration_error": float(ece), "bins": rows}


def calibration_report(probabilities: np.ndarray, labels: np.ndarray, threshold: float = 0.5, bins: int = 15) -> dict:
    """Full before/after calibration comparison on one held-out set."""
    probabilities = np.asarray(probabilities, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if len(probabilities) == 0:
        return {"available": False, "reason": "No held-out predictions were supplied."}
    temperature = fit_temperature(probabilities, labels)
    accuracy = float(np.mean((probabilities >= threshold).astype(int) == labels.astype(int)))
    return {
        "available": True,
        "method": "temperature_scaling",
        "temperature": temperature,
        "samples": int(len(probabilities)),
        "accuracy_at_threshold": accuracy,
        **evaluate_temperature(probabilities, labels, temperature, bins),
        "note": (
            "Temperature scaling rescales the logit, so it changes confidence but not the ranking of "
            "examples; ROC-AUC and PR-AUC are therefore identical before and after. A temperature above 1 "
            "means the raw sigmoid was over-confident; below 1 means it was under-confident."
        ),
    }


def save_calibration(report: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(report, indent=2), encoding="utf-8")
