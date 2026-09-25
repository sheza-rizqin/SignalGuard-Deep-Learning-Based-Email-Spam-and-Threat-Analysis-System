
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from .adversarial import TRANSFORMATIONS, Transformation, apply_transformation
from .hardening import build_views


@dataclass
class RobustnessReport:

    baseline_probability: float = 0.0
    baseline_label: str = "NOT SPAM"
    evaluated: int = 0
    preserved: int = 0
    changed: int = 0
    normalization_neutralised: int = 0
    tokenizer_neutralised: int = 0
    reached_model: int = 0
    variants: list[dict] = field(default_factory=list)
    by_family: list[dict] = field(default_factory=list)

    @property
    def prediction_stability(self) -> float:
        return self.preserved / self.evaluated if self.evaluated else 0.0

    @property
    def mean_absolute_shift(self) -> float:
        if not self.variants:
            return 0.0
        return sum(abs(item["probability_delta"]) for item in self.variants) / len(self.variants)

    @property
    def mean_signed_shift(self) -> float:
        if not self.variants:
            return 0.0
        return sum(item["probability_delta"] for item in self.variants) / len(self.variants)

    def most_damaging(self) -> dict | None:
        damaging = [item for item in self.variants if item["prediction_changed"]]
        pool = damaging or self.variants
        if not pool:
            return None
        return max(pool, key=lambda item: abs(item["probability_delta"]))

    def as_dict(self) -> dict:
        damaging = self.most_damaging()
        return {
            "baseline_probability": self.baseline_probability,
            "baseline_label": self.baseline_label,
            "evaluated": self.evaluated,
            "preserved": self.preserved,
            "changed": self.changed,
            "prediction_stability": self.prediction_stability,
            "mean_absolute_shift": self.mean_absolute_shift,
            "mean_signed_shift": self.mean_signed_shift,
            "normalization_neutralised": self.normalization_neutralised,
            "tokenizer_neutralised": self.tokenizer_neutralised,
            "reached_model": self.reached_model,
            "most_damaging": damaging,
            "by_family": self.by_family,
            "variants": self.variants,
            "definition": (
                "Prediction stability = probes that preserved the baseline decision / probes evaluated. "
                "The baseline decision is inherited by every probe, so this measures decision persistence, "
                "not classification accuracy."
            ),
        }


def _tokens(registry, text: str) -> list[int]:
    return list(registry.tokenizer.texts_to_sequences([text])[0])


def _build_variant_texts(raw_subject: str, raw_body: str, transformation: Transformation) -> tuple[str, str]:
    return (
        apply_transformation(raw_subject, transformation.key),
        apply_transformation(raw_body, transformation.key),
    )


def evaluate_email(
    registry,
    raw_subject: str,
    raw_body: str,
    transformations: Sequence[Transformation] | None = None,
) -> RobustnessReport:
    probes = list(transformations or TRANSFORMATIONS)
    baseline_views = build_views(raw_subject, raw_body)
    baseline_shipped = baseline_views[1].combined_text
    baseline_tokens = _tokens(registry, baseline_shipped)

    prepared: list[tuple[Transformation, dict]] = []
    model_inputs: list[str] = []
    for transformation in probes:
        variant_subject, variant_body = _build_variant_texts(raw_subject, raw_body, transformation)
        variant_views = build_views(variant_subject, variant_body)
        shipped = variant_views[1].combined_text
        hardened = variant_views[2].combined_text
        tokens = _tokens(registry, shipped)
        normalization_neutralised = shipped == baseline_shipped
        tokenizer_neutralised = (not normalization_neutralised) and tokens == baseline_tokens
        pipeline_stage = (
            "normalization_neutralised" if normalization_neutralised
            else "tokenizer_neutralised" if tokenizer_neutralised
            else "model_input_changed"
        )
        entry = {
            "transformation": transformation.as_dict(),
            "shipped_text": shipped,
            "hardened_text": hardened,
            "normalization_neutralised": normalization_neutralised,
            "tokenizer_neutralised": tokenizer_neutralised,
            "token_ids_changed": tokens != baseline_tokens,
            "pipeline_stage": pipeline_stage,
            "token_count": len(tokens),
            "baseline_token_count": len(baseline_tokens),
        }
        prepared.append((transformation, entry))
        model_inputs.append(shipped)

    batch = registry.predict_batch(model_inputs) if model_inputs else None
    baseline_prediction = registry.predict_text(baseline_shipped)

    report = RobustnessReport(
        baseline_probability=(
            baseline_prediction.evidence_adjusted
            if baseline_prediction.evidence_adjusted is not None
            else (baseline_prediction.calibrated if baseline_prediction.calibrated is not None else baseline_prediction.probability)
        ),
        baseline_label=baseline_prediction.label,
    )

    family_totals: dict[str, dict] = {}
    if batch is not None:
        for (transformation, entry), prediction in zip(prepared, batch.predictions):
            displayed_probability = (
                prediction.evidence_adjusted if prediction.evidence_adjusted is not None
                else (prediction.calibrated if prediction.calibrated is not None else prediction.probability)
            )
            delta = displayed_probability - report.baseline_probability
            prediction_changed = prediction.label != baseline_prediction.label
            entry.update(
                {
                    "probability": displayed_probability,
                    "raw_probability": prediction.probability,
                    "label": prediction.label,
                    "calibrated": prediction.calibrated,
                    "probability_delta": delta,
                    "absolute_delta": abs(delta),
                    "prediction_changed": prediction_changed,
                }
            )
            entry["transformation"]["hypothesis"] = transformation.hypothesis
            report.variants.append(entry)
            report.evaluated += 1
            report.preserved += 0 if prediction_changed else 1
            report.changed += 1 if prediction_changed else 0
            report.normalization_neutralised += 1 if entry["normalization_neutralised"] else 0
            report.tokenizer_neutralised += 1 if entry["tokenizer_neutralised"] else 0
            report.reached_model += 1 if entry["token_ids_changed"] else 0
            bucket = family_totals.setdefault(
                transformation.family, {"family": transformation.family, "evaluated": 0, "preserved": 0, "shift": 0.0}
            )
            bucket["evaluated"] += 1
            bucket["preserved"] += 0 if prediction_changed else 1
            bucket["shift"] += abs(delta)

    for bucket in family_totals.values():
        bucket["stability"] = bucket["preserved"] / bucket["evaluated"] if bucket["evaluated"] else 0.0
        bucket["measured_shift"] = bucket["shift"] / bucket["evaluated"] if bucket["evaluated"] else 0.0
    report.by_family = sorted(family_totals.values(), key=lambda item: item["stability"])
    report.variants.sort(key=lambda item: item["absolute_delta"], reverse=True)
    return report


def evaluate_corpus(registry, texts: Sequence[str], limit: int = 300) -> dict:
    sample = list(texts)[:limit]
    if not sample:
        return {"available": False, "reason": "No corpus texts supplied."}

    baseline = registry.predict_batch(sample)
    baseline_labels = [prediction.label for prediction in baseline.predictions]
    baseline_probabilities = [prediction.probability for prediction in baseline.predictions]

    per_probe: list[dict] = []
    for transformation in TRANSFORMATIONS:
        transformed = [
            build_views("", apply_transformation(text, transformation.key))[1].combined_text
            for text in sample
        ]
        outcome = registry.predict_batch(transformed)
        flips = 0
        shift = 0.0
        for index, prediction in enumerate(outcome.predictions):
            flips += 1 if prediction.label != baseline_labels[index] else 0
            shift += abs(prediction.probability - baseline_probabilities[index])
        per_probe.append(
            {
                "key": transformation.key,
                "name": transformation.name,
                "family": transformation.family,
                "evaluated": len(sample),
                "flips": flips,
                "stability": 1.0 - flips / len(sample),
                "mean_absolute_shift": shift / len(sample),
            }
        )

    corpus_stability = (
        sum(item["stability"] for item in per_probe) / len(per_probe) if per_probe else 0.0
    )
    families: dict[str, list[float]] = {}
    for item in per_probe:
        families.setdefault(item["family"], []).append(item["stability"])
    by_family = [
        {"family": family, "stability": sum(values) / len(values), "probes": len(values)}
        for family, values in sorted(families.items())
    ]
    per_probe.sort(key=lambda item: item["stability"])
    return {
        "available": True,
        "corpus_size": len(sample),
        "corpus_stability": corpus_stability,
        "baseline_spam_rate": sum(1 for label in baseline_labels if label == "SPAM") / len(sample),
        "per_probe": per_probe,
        "by_family": by_family,
        "definition": (
            "Corpus stability = mean over probes of (emails whose decision was preserved / emails evaluated). "
            "Probes inherit the baseline decision of each email; this is not accuracy."
        ),
    }
