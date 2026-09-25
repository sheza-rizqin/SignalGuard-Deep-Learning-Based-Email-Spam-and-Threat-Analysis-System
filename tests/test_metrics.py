from src.email_spam.metrics import calculate_metrics


def test_metrics_are_real_and_bounded():
    metrics = calculate_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9])
    assert metrics["accuracy"] == 1.0
    assert 0 <= metrics["false_positive_rate"] <= 1
    assert 0 <= metrics["false_negative_rate"] <= 1
