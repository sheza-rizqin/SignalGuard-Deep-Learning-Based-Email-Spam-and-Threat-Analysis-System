
from __future__ import annotations

import time

from .attribution import attribute_tokens
from .hardening import build_views, removed_character_counts, token_differences
from .history import PRIVACY_NOTICE
from .robustness import evaluate_email
from .security import inert_preview
from .signals import observe
from .unicode_forensics import scan_unicode

RISK_HIGH = 0.8
RISK_MEDIUM = 0.5
DEFAULT_ATTRIBUTION_BUDGET = 48


def _risk(probability: float) -> str:
    if probability >= RISK_HIGH:
        return "HIGH"
    if probability >= RISK_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _view_predictions(registry, views, include_comparisons: bool) -> dict:
    """Predict the production view, with diagnostics available on request."""
    selected_views = views if include_comparisons else [views[1]]
    outcome = registry.predict_batch([view.combined_text for view in selected_views])
    predictions = {
        view.key: prediction for view, prediction in zip(selected_views, outcome.predictions)
    }
    return {"predictions": predictions, "batch_latency_ms": outcome.total_latency_ms}


def analyse_email(
    registry,
    subject: str,
    body: str,
    run_robustness: bool = False,
    run_comparisons: bool = False,
    run_attribution: bool = False,
    attribution_budget: int = DEFAULT_ATTRIBUTION_BUDGET,
) -> dict:
    """Produce the complete multi-view security analysis for one email."""
    started = time.perf_counter()
    raw_subject = "" if subject is None else str(subject)
    raw_body = "" if body is None else str(body)
    if not raw_subject.strip() and not raw_body.strip():
        raise ValueError("Enter an email subject or body before analyzing.")

    views = build_views(raw_subject, raw_body)
    shipped_view = views[1]
    hardened_view = views[2]
    raw_combined = views[0].combined_text
    shipped_text = shipped_view.combined_text
    hardened_text = hardened_view.combined_text

    normalization_started = time.perf_counter()
    unicode_report = scan_unicode(raw_combined, hardened_text)
    normalization_ms = (time.perf_counter() - normalization_started) * 1000.0

    inference_started = time.perf_counter()
    prediction_result = _view_predictions(registry, views, run_comparisons)
    inference_ms = (time.perf_counter() - inference_started) * 1000.0

    shipped_prediction = prediction_result["predictions"]["shipped"]

    signal_started = time.perf_counter()
    signal_report = observe(raw_subject, raw_body, shipped_text)
    signals_ms = (time.perf_counter() - signal_started) * 1000.0

    robustness_payload: dict = {
        "available": False,
        "reason": "Robustness testing was not requested for this analysis.",
    }
    robustness_ms = 0.0
    if run_robustness:
        robustness_started = time.perf_counter()
        robustness_payload = evaluate_email(registry, raw_subject, raw_body).as_dict()
        robustness_ms = (time.perf_counter() - robustness_started) * 1000.0

    attribution_payload: dict = {
        "available": False,
        "reason": "Token attribution was not requested for this analysis.",
    }
    attribution_ms = 0.0
    if run_attribution:
        attribution_started = time.perf_counter()
        attribution_payload = attribute_tokens(registry, shipped_text, budget=attribution_budget)
        attribution_ms = (time.perf_counter() - attribution_started) * 1000.0

    total_ms = (time.perf_counter() - started) * 1000.0

    normalization = {
        "available": run_comparisons,
        "reason": None if run_comparisons else "Representation comparison was not requested for this quick analysis.",
    }
    if run_comparisons:
        predictions = prediction_result["predictions"]
        raw_prediction = predictions["raw"]
        hardened_prediction = predictions["hardened"]
        normalization.update({
            "views": [view.as_dict() for view in views],
            "differences": {
                "shipped_equals_hardened": shipped_text == hardened_text,
                "characters_removed_by_hardening": len(shipped_text) - len(hardened_text),
                "removed_character_kinds": removed_character_counts(raw_combined, hardened_text),
                "token_changes": token_differences(shipped_text, hardened_text),
            },
            "comparison": {
                "raw": raw_prediction.as_dict(), "shipped": shipped_prediction.as_dict(),
                "hardened": hardened_prediction.as_dict(),
                "raw_vs_shipped_delta": raw_prediction.probability - shipped_prediction.probability,
                "hardened_vs_shipped_delta": hardened_prediction.probability - shipped_prediction.probability,
                "hardening_changes_decision": hardened_prediction.label != shipped_prediction.label,
                "raw_changes_decision": raw_prediction.label != shipped_prediction.label,
                "note": "The as-shipped view is the production decision; the others are diagnostic comparison arms.",
            },
        })

    return {
        "_history_text": shipped_text,
        "model": {
            "prediction": shipped_prediction.label,
            "probability_spam": shipped_prediction.probability,
            "calibrated_confidence": shipped_prediction.calibrated,
            "evidence_adjusted_probability": shipped_prediction.evidence_adjusted,
            "evidence": shipped_prediction.evidence,
            "risk": _risk(shipped_prediction.probability),
            "threshold": registry.threshold,
            "representation": "as-shipped pipeline",
            "note": (
                "probability_spam is the raw sigmoid and calibrated_confidence is temperature-scaled. "
                "evidence_adjusted_probability moves low-support text toward 50% using token count and "
                "OOV rate. Its display weight is capped at 75% because familiar wording alone cannot "
                "prove an email comes from the training distribution; it does not alter the threshold decision."
            ),
        },
        "content": {
            "subject": inert_preview(raw_subject, 4_000),
            "body": inert_preview(raw_body),
            "characters": len(raw_body),
            "subject_characters": len(raw_subject),
        },
        "normalization": normalization,
        "unicode": unicode_report.as_dict(),
        "signals": signal_report.as_dict(),
        "robustness": robustness_payload,
        "attribution": attribution_payload,
        "model_info": {
            "version": registry.model_hash,
            "parameter_count": registry.parameter_count,
            "artifact_bytes": registry.model_bytes,
            "sequence_length": registry.sequence_length,
            "vocabulary_size": registry.vocabulary_size(),
            "architecture": registry.metadata.get("architecture"),
            "dataset": registry.metadata.get("dataset"),
        },
        "timings": {
            "normalization_ms": normalization_ms,
            "inference_ms": inference_ms,
            "signals_ms": signals_ms,
            "robustness_ms": robustness_ms,
            "attribution_ms": attribution_ms,
            "total_ms": total_ms,
        },
        "privacy": {
            "stores_content": False,
            "notice": PRIVACY_NOTICE,
        },
    }
