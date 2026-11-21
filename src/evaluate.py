import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
)

from src.config import LABELS


def calculate_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABELS,
    round_digits: int = 4,
) -> Dict[str, Any]:
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)
    per_class = f1_score(y_true, y_pred, average=None, labels=labels, zero_division=0)

    per_class_dict = {
        label: round(float(score), round_digits)
        for label, score in zip(labels, per_class)
    }

    return {
        "accuracy": round(float(acc), round_digits),
        "macro_f1": round(float(macro_f1), round_digits),
        "per_class_f1": per_class_dict,
    }


def compute_trainer_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    if isinstance(logits, tuple):
        logits = logits[0]
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    macro = f1_score(
        labels, preds, average="macro", labels=list(range(len(LABELS))), zero_division=0
    )
    return {
        "accuracy": round(float(acc), 4),
        "macro_f1": round(float(macro), 4),
    }


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABELS,
) -> np.ndarray:
    return confusion_matrix(y_true, y_pred, labels=labels)


def export_error_analysis(
    df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "label",
    pred_col: str = "predicted_label",
    output_path: Optional[Path] = None,
) -> pd.DataFrame:
    errors = df[df[label_col] != df[pred_col]].copy()

    if "confidence" in errors.columns:
        errors = errors.sort_values("confidence", ascending=False)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        errors.to_csv(output_path, index=False)

    return errors.reset_index(drop=True)


def save_metrics_json(metrics: Dict[str, Any], filepath: Path) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

