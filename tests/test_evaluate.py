import json

import numpy as np
import pandas as pd
import pytest

from src.config import METRICS_DIR
from src.evaluate import (
    calculate_metrics,
    compute_confusion_matrix,
    compute_trainer_metrics,
    export_error_analysis,
    save_metrics_json,
)


def test_calculate_metrics_known_values():
    y_true = ["negative", "neutral", "positive", "positive"]
    y_pred = ["negative", "neutral", "neutral", "positive"]
    m = calculate_metrics(y_true, y_pred)

    assert m["accuracy"] == 0.75
    assert m["per_class_f1"] == {"negative": 1.0, "neutral": pytest.approx(0.6667), "positive": pytest.approx(0.6667)}
    assert m["macro_f1"] == pytest.approx(0.7778, abs=1e-4)


def test_calculate_metrics_handles_a_never_predicted_class():
    m = calculate_metrics(["negative", "positive"], ["positive", "positive"])
    assert m["per_class_f1"]["negative"] == 0.0
    assert m["per_class_f1"]["neutral"] == 0.0


def test_compute_trainer_metrics_from_logits():
    logits = np.array([[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0], [3.0, 0.0, 0.0]])
    labels = np.array([0, 1, 2, 2])
    m = compute_trainer_metrics((logits, labels))
    assert m["accuracy"] == 0.75


def test_confusion_matrix_uses_label_order():
    cm = compute_confusion_matrix(["negative", "positive"], ["positive", "positive"])
    assert cm.tolist() == [[0, 0, 1], [0, 0, 0], [0, 0, 1]]


def test_export_error_analysis_sorts_by_confidence(tmp_path):
    df = pd.DataFrame({
        "sentence": ["a", "b", "c"],
        "label": ["neutral", "neutral", "positive"],
        "predicted_label": ["positive", "neutral", "negative"],
        "confidence": [0.6, 0.9, 0.95],
    })
    out_path = tmp_path / "errors.csv"
    errors = export_error_analysis(df, output_path=out_path)

    assert errors["sentence"].tolist() == ["c", "a"]
    assert out_path.exists()


def test_save_metrics_json_roundtrip(tmp_path):
    path = tmp_path / "nested" / "m.json"
    save_metrics_json({"accuracy": 0.5}, path)
    assert json.loads(path.read_text(encoding="utf-8")) == {"accuracy": 0.5}


@pytest.mark.skipif(not (METRICS_DIR / "finetuned_test_metrics.json").exists(), reason="no committed metrics")
def test_committed_metric_files_are_internally_consistent():
    """The README tables are copied from these files; guard against hand edits drifting."""
    for name in ["finetuned_test_metrics.json", "zero_shot_test_metrics.json"]:
        m = json.loads((METRICS_DIR / name).read_text(encoding="utf-8"))
        assert m["num_examples"] == 726
        mean_f1 = sum(m["per_class_f1"].values()) / 3
        assert m["macro_f1"] == pytest.approx(mean_f1, abs=1e-3), name

    val = json.loads((METRICS_DIR / "finetuned_validation_metrics.json").read_text(encoding="utf-8"))
    from src.config import NUM_TRAIN_EPOCHS

    assert val["epochs"] == NUM_TRAIN_EPOCHS
