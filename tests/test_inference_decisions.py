from src.email_spam.registry import ModelRegistry


def test_probability_threshold_maps_to_expected_prediction():
    registry = object.__new__(ModelRegistry)
    registry.threshold = 0.5
    registry.calibration = lambda: None

    low = registry._to_prediction("normal email", 0.0053, 1.0)
    high = registry._to_prediction("spam email", 0.95, 1.0)

    assert low.probability < registry.threshold
    assert low.label == "NOT SPAM"
    assert high.probability >= registry.threshold
    assert high.label == "SPAM"


def test_risk_bands_follow_probability():
    from src.email_spam.analysis import _risk

    assert _risk(0.0053) == "LOW"
    assert _risk(0.65) == "MEDIUM"
    assert _risk(0.95) == "HIGH"
