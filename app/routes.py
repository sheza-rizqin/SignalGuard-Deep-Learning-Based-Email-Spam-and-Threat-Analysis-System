"""HTTP routes for the email security intelligence console.

Every numeric value returned by these endpoints originates from one of:

* inference on the loaded LSTM,
* a precomputed experiment artifact produced by a script in ``tools/``,
* a measurement taken during the request.

No endpoint invents a metric. When an artifact has not been generated yet the
response says so explicitly instead of substituting a placeholder number.
"""

from __future__ import annotations

import time
import uuid

from flask import Blueprint, current_app, jsonify, render_template, request

from src.email_spam.adversarial import TRANSFORMATIONS
from src.email_spam.analysis import analyse_email
from src.email_spam.registry import OPTIONAL_ARTIFACTS
from src.email_spam.security import (
    MAX_UPLOAD_BYTES,
    UploadRejected,
    extract_raw_email,
    validate_upload,
)

from .state import (
    artifact_inventory,
    get_registry,
    history,
    model_available,
)

api = Blueprint("api", __name__)


@api.after_request
def prevent_api_caching(response):
    """Email analysis payloads must not remain in browser or proxy caches."""
    if request.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-store, no-cache, max-age=0, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
    return response


def _artifact_response(key: str) -> tuple:
    """Return a precomputed artifact, or an explicit not-available payload."""
    if not model_available():
        return jsonify({"available": False, "reason": "Model artifacts are not present."}), 200
    payload = get_registry().auxiliary.get(key)
    if not payload:
        return (
            jsonify(
                {
                    "available": False,
                    "reason": (
                        f"{OPTIONAL_ARTIFACTS[key]} has not been generated yet. Run the corresponding "
                        f"script in tools/ to produce it."
                    ),
                }
            ),
            200,
        )
    return jsonify(payload), 200


@api.get("/")
def index():
    return render_template("index.html")


@api.get("/api/status")
def status():
    """Model readiness, delivered metrics, and session counters."""
    ready = model_available()
    inventory = artifact_inventory()
    metrics = None
    metadata = None
    model_info = None
    if ready:
        registry = get_registry()
        metrics = registry.auxiliary.get("evaluation")
        metadata = registry.metadata
        model_info = {
            "version": registry.model_hash,
            "parameter_count": registry.parameter_count,
            "artifact_bytes": registry.model_bytes,
            "sequence_length": registry.sequence_length,
            "vocabulary_size": registry.vocabulary_size(),
        }
    return jsonify(
        {
            "ready": ready,
            "metrics": metrics,
            "metadata": metadata,
            "model_info": model_info,
            "artifacts": inventory,
            "history": history.summary(),
            "capabilities": {
                key: inventory[key]["present"]
                for key in ("calibration",)
            },
        }
    )


@api.post("/api/analyze")
def analyze():
    """Run the full multi-view analysis over a pasted or uploaded email."""
    subject = request.form.get("subject", "")
    body = request.form.get("body", "")
    run_robustness = request.form.get("run_robustness", "false").lower() == "true"
    run_comparisons = request.form.get("run_comparisons", "false").lower() == "true"
    run_attribution = request.form.get("run_attribution", "false").lower() == "true"
    attribution_budget = request.form.get("attribution_budget", type=int) or 48
    attribution_budget = max(8, min(attribution_budget, 128))

    upload = request.files.get("email_file")
    if upload and upload.filename:
        try:
            content = upload.read(MAX_UPLOAD_BYTES + 1)
            safe_name = validate_upload(upload.filename, content)
            subject, body = extract_raw_email(safe_name, content)
        except UploadRejected as error:
            return jsonify({"error": str(error)}), 400

    try:
        registry = get_registry()
        result = analyse_email(
            registry,
            subject,
            body,
            run_robustness=run_robustness,
            run_comparisons=run_comparisons,
            run_attribution=run_attribution,
            attribution_budget=attribution_budget,
        )
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        current_app.logger.exception("Email analysis failed")
        return jsonify({"error": "The email could not be analyzed. Check the input and model artifacts."}), 500

    shipped_text = result.pop("_history_text")
    robustness = result.get("robustness") or {}
    evidence = result["model"].get("evidence") or {}
    history.record(
        prediction=result["model"]["prediction"],
        probability=result["model"]["probability_spam"],
        calibrated=result["model"]["calibrated_confidence"],
        model_version=result["model_info"]["version"],
        text=shipped_text,
        unicode_findings=result["unicode"]["total_findings"],
        url_findings=len(result["signals"]["url_findings"]),
        persuasion_signals=len(result["signals"]["persuasion"]),
        prediction_stability=robustness.get("prediction_stability") if robustness.get("available") else None,
        token_count=evidence.get("token_count", 0),
        oov_rate=evidence.get("oov_rate", 0.0),
        inference_ms=result["timings"]["inference_ms"],
        total_ms=result["timings"]["total_ms"],
    )
    result["request_id"] = uuid.uuid4().hex
    return jsonify(result)


@api.get("/api/probes")
def probes():
    """The complete adversarial probe catalogue, with its stated hypotheses."""
    return jsonify(
        {
            "count": len(TRANSFORMATIONS),
            "families": sorted({transformation.family for transformation in TRANSFORMATIONS}),
            "transformations": [transformation.as_dict() for transformation in TRANSFORMATIONS],
            "definition": (
                "Prediction stability = probes that preserved the baseline decision / probes evaluated. "
                "Each probe inherits the baseline decision of the email it was applied to, so this measures "
                "decision persistence rather than classification accuracy."
            ),
        }
    )


@api.get("/api/health")
def health():
    """Live measurements: load cost, single-request latency, batch efficiency."""
    if not model_available():
        return jsonify({"available": False, "reason": "Model artifacts are not present."}), 200
    registry = get_registry()
    probe_text = "subject: account verification\nbody: please confirm your account details immediately"
    sequential_ms = 0.0
    repeats = 8
    for _ in range(repeats):
        start = time.perf_counter()
        registry.predict_text(probe_text)
        sequential_ms += (time.perf_counter() - start) * 1000.0
    batch_ms = registry.predict_batch([probe_text] * repeats).total_latency_ms
    return jsonify(
        {
            "available": True,
            "model": {
                "version": registry.model_hash,
                "parameters": registry.parameter_count,
                "artifact_bytes": registry.model_bytes,
                "sequence_length": registry.sequence_length,
                "vocabulary_size": registry.vocabulary_size(),
                "architecture": registry.metadata.get("architecture"),
                "dataset": registry.metadata.get("dataset"),
                "warmup_latency_ms": registry.warmup_latency_ms,
            },
            "live_measurement": {
                "repeats": repeats,
                "sequential_total_ms": sequential_ms,
                "sequential_per_item_ms": sequential_ms / repeats,
                "batched_total_ms": batch_ms,
                "batched_per_item_ms": batch_ms / repeats,
                "speedup_factor": (sequential_ms / batch_ms) if batch_ms else None,
                "note": (
                    "Both paths include tokenisation and padding. The sequential path performs one encode "
                    "and one forward pass per item; the batched path performs a single encode and a single "
                    "forward pass for all items."
                ),
            },
            "latest_analysis": history.latest(),
            "delivered_metrics": registry.auxiliary.get("evaluation"),
            "artifacts": registry.artifact_inventory(),
        }
    )


@api.get("/api/calibration")
def calibration():
    return _artifact_response("calibration")


@api.get("/api/evaluation")
def evaluation():
    return _artifact_response("evaluation")


@api.get("/api/history")
def history_entries():
    return jsonify({**history.summary(), "entries": history.entries(limit=25)})


@api.post("/api/history/clear")
def history_clear():
    history.clear()
    return jsonify(history.summary())
