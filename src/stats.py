"""Uncertainty, significance and calibration helpers for classifier evaluation.

All functions work on plain label lists / probability arrays so they can be applied
to any saved prediction file without reloading a model.
"""

from typing import Callable, Dict, List, Sequence

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, f1_score

from src.config import LABEL_TO_ID, LABELS


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(f1_score(y_true, y_pred, labels=LABELS, average="macro", zero_division=0))


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(accuracy_score(y_true, y_pred))


METRICS: Dict[str, Callable[[Sequence[str], Sequence[str]], float]] = {
    "accuracy": accuracy,
    "macro_f1": macro_f1,
}


def _bootstrap_indices(n: int, n_boot: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, n, size=(n_boot, n))


def bootstrap_ci(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    metric: Callable = macro_f1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """Point estimate and percentile bootstrap CI (resampling examples with replacement)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if len(y_true) == 0:
        raise ValueError("bootstrap_ci needs at least one example")
    idx = _bootstrap_indices(len(y_true), n_boot, seed)
    samples = np.array([metric(y_true[i], y_pred[i]) for i in idx])
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return {"value": round(metric(y_true, y_pred), 4), "ci_low": round(float(lo), 4), "ci_high": round(float(hi), 4)}


def paired_bootstrap_diff(
    y_true: Sequence[str],
    pred_a: Sequence[str],
    pred_b: Sequence[str],
    metric: Callable = macro_f1,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, float]:
    """metric(A) - metric(B) on the same resampled examples.

    ``p_value`` is a two-sided bootstrap p-value for "no difference".
    """
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    if not (len(y_true) == len(pred_a) == len(pred_b)):
        raise ValueError("y_true, pred_a and pred_b must be the same length")
    idx = _bootstrap_indices(len(y_true), n_boot, seed)
    diffs = np.array([metric(y_true[i], pred_a[i]) - metric(y_true[i], pred_b[i]) for i in idx])
    lo, hi = np.quantile(diffs, [alpha / 2, 1 - alpha / 2])
    p_value = min(1.0, 2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))
    return {
        "diff": round(metric(y_true, pred_a) - metric(y_true, pred_b), 4),
        "ci_low": round(float(lo), 4),
        "ci_high": round(float(hi), 4),
        "p_value": round(float(p_value), 4),
    }


def mcnemar_test(y_true: Sequence[str], pred_a: Sequence[str], pred_b: Sequence[str]) -> Dict[str, float]:
    """Exact McNemar test on the examples where exactly one model is correct."""
    y_true, pred_a, pred_b = map(np.asarray, (y_true, pred_a, pred_b))
    a_right, b_right = pred_a == y_true, pred_b == y_true
    only_a = int(np.sum(a_right & ~b_right))
    only_b = int(np.sum(~a_right & b_right))
    n = only_a + only_b
    p = 1.0 if n == 0 else float(binomtest(only_a, n, 0.5).pvalue)
    return {"only_a_correct": only_a, "only_b_correct": only_b, "p_value": round(p, 6)}


def labels_to_ids(labels: Sequence[str]) -> np.ndarray:
    return np.array([LABEL_TO_ID[label] for label in labels])


def expected_calibration_error(probs: np.ndarray, y_true_ids: np.ndarray, n_bins: int = 10) -> Dict[str, object]:
    """Top-label ECE with equal-width confidence bins, plus the per-bin table."""
    probs = np.asarray(probs, dtype=float)
    y_true_ids = np.asarray(y_true_ids)
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true_ids).astype(float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: List[Dict[str, float]] = []
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if not mask.any():
            continue
        acc, conf, share = correct[mask].mean(), confidences[mask].mean(), mask.mean()
        ece += share * abs(acc - conf)
        bins.append({"bin_low": round(float(lo), 2), "bin_high": round(float(hi), 2), "count": int(mask.sum()),
                     "accuracy": round(float(acc), 4), "confidence": round(float(conf), 4)})
    return {"ece": round(float(ece), 4), "bins": bins}


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    """Rescale probabilities as softmax(log p / T).

    log-probabilities differ from logits only by a per-row constant, so this is exactly
    temperature scaling of the original logits.
    """
    logp = np.log(np.clip(np.asarray(probs, dtype=float), 1e-12, 1.0))
    return _softmax(logp / temperature)


def negative_log_likelihood(probs: np.ndarray, y_true_ids: np.ndarray) -> float:
    p = np.clip(np.asarray(probs)[np.arange(len(y_true_ids)), y_true_ids], 1e-12, 1.0)
    return float(-np.log(p).mean())


def fit_temperature(probs: np.ndarray, y_true_ids: np.ndarray, bounds=(0.05, 10.0)) -> float:
    """Temperature that minimises NLL on a held-out (validation) set."""
    result = minimize_scalar(
        lambda t: negative_log_likelihood(apply_temperature(probs, t), y_true_ids),
        bounds=bounds,
        method="bounded",
    )
    return float(result.x)
