import numpy as np
import pandas as pd
import torch
from sklearn.utils.class_weight import compute_class_weight
from torch.nn import CrossEntropyLoss
from transformers import Trainer

from src.config import LABELS


def get_class_weights(df: pd.DataFrame, label_col: str = "label") -> torch.Tensor:
    """Balanced class weights (n_samples / (n_classes * class_count)) in LABELS order."""
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array(LABELS),
        y=df[label_col].values,
    )
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):
    """Hugging Face Trainer whose loss is class-weighted cross-entropy."""

    def __init__(self, *args, class_weight_tensor: torch.Tensor = None, **kwargs):
        if class_weight_tensor is None:
            raise ValueError("WeightedLossTrainer needs class_weight_tensor")
        super().__init__(*args, **kwargs)
        self.class_weight_tensor = class_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weight_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
