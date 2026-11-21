from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import pipeline

from src.config import BASELINE_MODEL_NAME, LABELS, ZERO_SHOT_HYPOTHESIS_TEMPLATE
from src.evaluate import calculate_metrics


def run_zero_shot_baseline(
    test_df: pd.DataFrame,
    text_col: str = "sentence",
    label_col: str = "label",
    model_name: str = BASELINE_MODEL_NAME,
    candidate_labels: List[str] = LABELS,
    hypothesis_template: str = ZERO_SHOT_HYPOTHESIS_TEMPLATE,
    batch_size: int = 16,
    device: Optional[int] = None,
) -> Tuple[Dict[str, Any], pd.DataFrame]:
    """Score ``test_df`` with an NLI zero-shot classifier (no domain training)."""
    if device is None:
        device = 0 if torch.cuda.is_available() else -1

    classifier = pipeline(
        "zero-shot-classification",
        model=model_name,
        device=device,
    )

    sentences = test_df[text_col].tolist()
    outputs: List[Dict[str, Any]] = []
    for i in tqdm(range(0, len(sentences), batch_size), desc="Zero-shot evaluation"):
        batch_outputs = classifier(
            sentences[i : i + batch_size],
            candidate_labels=candidate_labels,
            hypothesis_template=hypothesis_template,
            multi_label=False,
            batch_size=batch_size,
        )
        if isinstance(batch_outputs, dict):  # a single sentence returns a dict
            batch_outputs = [batch_outputs]
        outputs.extend(batch_outputs)

    results_df = test_df.copy()
    results_df["predicted_label"] = [out["labels"][0] for out in outputs]
    results_df["confidence"] = [out["scores"][0] for out in outputs]
    for label in candidate_labels:
        results_df[f"{label}_probability"] = [
            dict(zip(out["labels"], out["scores"])).get(label, 0.0) for out in outputs
        ]

    metrics = calculate_metrics(
        results_df[label_col].tolist(),
        results_df["predicted_label"].tolist(),
        labels=candidate_labels,
    )
    metrics.update({
        "model": model_name,
        "hypothesis_template": hypothesis_template,
        "evaluation_split": "held-out test set",
        "num_examples": len(results_df),
    })

    return metrics, results_df
