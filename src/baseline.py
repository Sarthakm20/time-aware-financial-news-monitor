from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

from src.config import (
    BASELINE_MODEL_NAME,
    LABELS,
    METRICS_DIR,
    PREDICTIONS_DIR,
)
from src.evaluate import calculate_metrics, export_error_analysis


def run_zero_shot_baseline(
    test_df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "label",
    model_name: str = BASELINE_MODEL_NAME,
    candidate_labels: List[str] = LABELS,
    hypothesis_template: str = "This text expresses {} financial sentiment.",
    batch_size: int = 16,
    device: Optional[int] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    sentences = test_df[text_col].tolist()
    all_predictions: List[str] = []
    all_confidences: List[float] = []
    neg_probs: List[float] = []
    neu_probs: List[float] = []
    pos_probs: List[float] = []

    for i in tqdm(range(0, len(sentences), batch_size), desc="Zero-shot Evaluation"):
        batch = sentences[i : i + batch_size]
        outputs = classifier(
            batch,
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
        )

        for out in outputs:
            top_label = out["labels"][0]
            top_score = out["scores"][0]
            all_predictions.append(top_label)
            all_confidences.append(top_score)

            score_map = dict(zip(out["labels"], out["scores"]))
            neg_probs.append(score_map.get("negative", 0.0))
            neu_probs.append(score_map.get("neutral", 0.0))
            pos_probs.append(score_map.get("positive", 0.0))

    results_df = test_df.copy()
    results_df["predicted_label"] = all_predictions
    results_df["confidence"] = all_confidences
    results_df["negative_probability"] = neg_probs
    results_df["neutral_probability"] = neu_probs
    results_df["positive_probability"] = pos_probs

    y_true = results_df[label_col].tolist()
    y_pred = results_df["predicted_label"].tolist()

    metrics = calculate_metrics(y_true, y_pred, labels=candidate_labels)
    metrics["model"] = model_name
    metrics["evaluation_split"] = "held-out test set"
    metrics["num_examples"] = len(results_df)

    return metrics, results_df

