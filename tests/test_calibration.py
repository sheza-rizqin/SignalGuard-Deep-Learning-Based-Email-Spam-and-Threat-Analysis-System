import numpy as np

from src.email_spam.calibration import apply_temperature, evaluate_temperature


def test_temperature_one_preserves_probabilities():
    values = np.asarray([0.01, 0.25, 0.5, 0.9])
    assert np.allclose(apply_temperature(values, 1.0), values)


def test_temperature_scaling_preserves_the_half_threshold_decision():
    values = np.asarray([0.01, 0.49, 0.5, 0.99])
    calibrated = apply_temperature(values, 0.67)
    assert np.array_equal(values >= 0.5, calibrated >= 0.5)


def test_temperature_evaluation_reports_bounded_metrics():
    report = evaluate_temperature(np.asarray([0.1, 0.2, 0.8, 0.9]), np.asarray([0, 0, 1, 1]), 1.0)
    assert report["samples"] == 4
    assert 0 <= report["before"]["ece"] <= 1
    assert 0 <= report["after"]["brier"] <= 1
