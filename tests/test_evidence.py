import numpy as np

from src.email_spam.registry import ModelRegistry


def test_sparse_or_oov_tokens_have_low_evidence_support():
    registry = object.__new__(ModelRegistry)
    registry.tokenizer = type("Tokenizer", (), {"oov_token": "<OOV>", "word_index": {"<OOV>": 1}})()
    evidence = registry._evidence(np.asarray([1, 1, 0, 0]))
    assert evidence["level"] == "LOW"
    assert evidence["support"] == 0.0


def test_full_known_text_has_high_evidence_support():
    registry = object.__new__(ModelRegistry)
    registry.tokenizer = type("Tokenizer", (), {"oov_token": "<OOV>", "word_index": {"<OOV>": 1}})()
    evidence = registry._evidence(np.asarray([2] * 12))
    assert evidence["level"] == "HIGH"
    assert evidence["support"] == 1.0


def test_display_probability_retains_uncertainty_for_known_text():
    registry = object.__new__(ModelRegistry)
    registry.threshold = 0.5
    registry.calibration = lambda: None

    prediction = registry._to_prediction("known text", 0.99, 1.0, {"support": 1.0})

    assert abs(prediction.evidence_adjusted - 0.8675) < 1e-9
    assert prediction.label == "SPAM"
