import numpy as np
import pytest

from src import stats


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(0)
    y_true = rng.choice(["negative", "neutral", "positive"], size=300)
    y_pred = np.where(rng.random(300) < 0.8, y_true, "neutral")
    ci = stats.bootstrap_ci(y_true, y_pred, stats.accuracy, n_boot=500)
    assert ci["ci_low"] <= ci["value"] <= ci["ci_high"]
    assert 0.0 < ci["ci_high"] - ci["ci_low"] < 0.15


def test_bootstrap_ci_is_degenerate_for_perfect_predictions():
    y = ["negative", "neutral", "positive"] * 10
    ci = stats.bootstrap_ci(y, y, stats.accuracy, n_boot=200)
    assert ci == {"value": 1.0, "ci_low": 1.0, "ci_high": 1.0}


def test_bootstrap_ci_rejects_empty_input():
    with pytest.raises(ValueError):
        stats.bootstrap_ci([], [], stats.accuracy)


def test_paired_bootstrap_detects_a_clearly_better_model():
    rng = np.random.default_rng(1)
    y_true = rng.choice(["negative", "neutral", "positive"], size=400)
    good = np.where(rng.random(400) < 0.9, y_true, "neutral")
    bad = np.where(rng.random(400) < 0.5, y_true, "neutral")
    res = stats.paired_bootstrap_diff(y_true, good, bad, stats.accuracy, n_boot=500)
    assert res["diff"] > 0.15  # ~0.93 vs ~0.67 accuracy (wrong answers fall back to neutral)
    assert res["ci_low"] > 0
    assert res["p_value"] < 0.01


def test_paired_bootstrap_identical_models_has_zero_difference():
    y = ["negative", "neutral", "positive", "neutral"] * 20
    pred = ["neutral"] * len(y)
    res = stats.paired_bootstrap_diff(y, pred, pred, n_boot=200)
    assert res["diff"] == 0.0
    assert res["p_value"] == 1.0


def test_paired_bootstrap_requires_equal_lengths():
    with pytest.raises(ValueError):
        stats.paired_bootstrap_diff(["neutral"], ["neutral"], ["neutral", "neutral"])


def test_mcnemar_counts_disagreements():
    y = ["positive"] * 10
    a = ["positive"] * 10
    b = ["positive"] * 2 + ["negative"] * 8
    res = stats.mcnemar_test(y, a, b)
    assert res["only_a_correct"] == 8
    assert res["only_b_correct"] == 0
    assert res["p_value"] == pytest.approx(2 * 0.5 ** 8, abs=1e-6)
    assert stats.mcnemar_test(y, a, a)["p_value"] == 1.0


def test_ece_is_zero_when_confidence_matches_accuracy():
    # 10 examples at confidence 0.8, 8 of them correct -> perfectly calibrated
    probs = np.tile([0.8, 0.1, 0.1], (10, 1))
    y = np.array([0] * 8 + [1] * 2)
    res = stats.expected_calibration_error(probs, y)
    assert res["ece"] == pytest.approx(0.0)
    assert res["bins"][0]["count"] == 10


def test_ece_detects_overconfidence():
    probs = np.tile([0.99, 0.005, 0.005], (10, 1))
    y = np.array([0] * 5 + [1] * 5)  # only 50% right at 99% confidence
    assert stats.expected_calibration_error(probs, y)["ece"] == pytest.approx(0.49, abs=1e-6)


def test_apply_temperature_one_is_identity_and_preserves_argmax():
    probs = np.array([[0.7, 0.2, 0.1], [0.1, 0.3, 0.6]])
    np.testing.assert_allclose(stats.apply_temperature(probs, 1.0), probs, rtol=1e-6)
    softened = stats.apply_temperature(probs, 3.0)
    np.testing.assert_allclose(softened.sum(axis=1), 1.0)
    assert (softened.argmax(axis=1) == probs.argmax(axis=1)).all()
    assert softened.max() < probs.max()


def test_fit_temperature_softens_an_overconfident_model():
    rng = np.random.default_rng(2)
    n = 600
    y = rng.integers(0, 3, size=n)
    pred = np.where(rng.random(n) < 0.7, y, (y + 1) % 3)  # 70% accurate
    probs = np.full((n, 3), 0.005)
    probs[np.arange(n), pred] = 0.99  # but 99% confident
    t = stats.fit_temperature(probs, y)
    assert t > 1.5
    calibrated = stats.apply_temperature(probs, t)
    assert stats.negative_log_likelihood(calibrated, y) < stats.negative_log_likelihood(probs, y)
    assert stats.expected_calibration_error(calibrated, y)["ece"] < stats.expected_calibration_error(probs, y)["ece"]


def test_labels_to_ids():
    assert stats.labels_to_ids(["negative", "neutral", "positive"]).tolist() == [0, 1, 2]
