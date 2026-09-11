import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer
from sklearn.utils.class_weight import compute_class_weight
import numpy as np


def get_class_weights(df, label_col="label"):
    labels = df[label_col].values
    classes = np.array(["negative", "neutral", "positive"])
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float)


class WeightedLossTrainer(Trainer):

    def __init__(self, *args, class_weight_tensor=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weight_tensor = class_weight_tensor

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = CrossEntropyLoss(weight=self.class_weight_tensor.to(logits.device))
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
