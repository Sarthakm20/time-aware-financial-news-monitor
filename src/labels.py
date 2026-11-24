"""Label-order mapping and probability summaries shared by the PyTorch and ONNX backends (no torch import)."""

from typing import List, Tuple

import numpy as np

from src.config import ID_TO_LABEL, LABELS


def label_column_order(model) -> List[int]:
    """Indices that reorder a model's output columns into LABELS order.

    Models trained here already use LABELS order. Third-party checkpoints (e.g.
    ProsusAI/finbert: positive, negative, neutral) are mapped by their ``id2label``.
    Models without a usable config are assumed to be in LABELS order.
    """
    id2label = getattr(getattr(model, "config", None), "id2label", None) or {}
    names = {str(name).strip().lower(): int(idx) for idx, name in id2label.items()}
    if set(names) == set(LABELS):
        return [names[label] for label in LABELS]
    return list(range(len(LABELS)))


def summarize_probabilities(probs: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Labels, the probabilities, confidences and P(pos) - P(neg) from a (n, 3) matrix in LABELS order."""
    if len(probs) == 0:
        empty = np.empty((0,), dtype=np.float32)
        return [], np.empty((0, len(LABELS)), dtype=np.float32), empty, empty
    labels = [ID_TO_LABEL[int(i)] for i in np.argmax(probs, axis=1)]
    return labels, probs, np.max(probs, axis=1), probs[:, 2] - probs[:, 0]
