from __future__ import annotations

from dataclasses import dataclass

import numpy as np

NEGLIGIBLE_DELTA = 0.01
DEFAULT_TOKEN_BUDGET = 64


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    token_id: int
    occurrences: int
    occluded_probability: float
    delta: float
    direction: str

    def as_dict(self) -> dict:
        return {
            "token": self.token,
            "token_id": self.token_id,
            "occurrences": self.occurrences,
            "occluded_probability": self.occluded_probability,
            "delta": self.delta,
            "direction": self.direction,
        }


def _oov_id(registry) -> int:
    """The reserved out-of-vocabulary token id used by the trained tokenizer."""
    return int(registry.tokenizer.word_index.get("<OOV>", 1))


def attribute_tokens(registry, text: str, budget: int = DEFAULT_TOKEN_BUDGET) -> dict:
    """Attribute the model's decision to individual tokens via occlusion."""
    encoded = registry.encode([text])[0]
    active = [int(value) for value in encoded if int(value) > 0]
    baseline_probability = float(registry.probabilities(encoded.reshape(1, -1))[0])

    if not active:
        return {
            "method": "position_preserving_occlusion",
            "available": False,
            "reason": "No in-vocabulary tokens were found in the model input.",
            "tokens": [],
        }

    order: list[int] = []
    counts: dict[int, int] = {}
    for token_id in active:
        counts[token_id] = counts.get(token_id, 0) + 1
        if token_id not in order:
            order.append(token_id)

    truncated = len(order) > budget
    selected = order[:budget]
    oov = _oov_id(registry)

    variants = np.tile(encoded.reshape(1, -1), (len(selected), 1))
    for row, token_id in enumerate(selected):
        variants[row][variants[row] == token_id] = oov

    probabilities = registry.probabilities(variants)
    index_word: dict[int, str] = getattr(registry.tokenizer, "index_word", {}) or {}
    attributions: list[TokenAttribution] = []
    for token_id, probability in zip(selected, probabilities):
        delta = baseline_probability - float(probability)
        if abs(delta) < NEGLIGIBLE_DELTA:
            direction = "negligible"
        elif delta > 0:
            direction = "supports_spam"
        else:
            direction = "suppresses_spam"
        attributions.append(
            TokenAttribution(
                token=str(index_word.get(token_id, f"<id:{token_id}>")),
                token_id=token_id,
                occurrences=counts[token_id],
                occluded_probability=float(probability),
                delta=float(delta),
                direction=direction,
            )
        )

    attributions.sort(key=lambda item: abs(item.delta), reverse=True)
    return {
        "method": "position_preserving_occlusion",
        "available": True,
        "description": (
            "Each token is replaced by the reserved OOV token so sequence length and every other "
            "position stay fixed. The reported change is the real model output difference; a positive "
            "delta means the token contributed spam probability."
        ),
        "baseline_probability": baseline_probability,
        "evaluated_tokens": len(attributions),
        "unique_tokens": len(order),
        "truncated": truncated,
        "negligible_threshold": NEGLIGIBLE_DELTA,
        "tokens": [attribution.as_dict() for attribution in attributions],
    }
