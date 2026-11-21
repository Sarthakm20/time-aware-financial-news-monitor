from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score, precision_score, recall_score

from src.config import LABELS


def _per_class(metric, true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    scores = metric(true_labels, pred_labels, labels=LABELS, average=None, zero_division=0)
    return {label: float(score) for label, score in zip(LABELS, scores)}


def compute_per_class_f1(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    return _per_class(f1_score, true_labels, pred_labels)


def compute_per_class_precision(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    return _per_class(precision_score, true_labels, pred_labels)


def compute_per_class_recall(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    return _per_class(recall_score, true_labels, pred_labels)


def plot_per_class_f1(
    f1_a: Dict[str, float],
    f1_b: Dict[str, float],
    label_a: str = "Model A",
    label_b: str = "Model B",
    title: str = "Per-class F1 comparison",
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    df = pd.DataFrame({
        "Class": LABELS,
        label_a: [f1_a[c] for c in LABELS],
        label_b: [f1_b[c] for c in LABELS],
    })
    melted = pd.melt(df, id_vars=["Class"], var_name="Model", value_name="F1")

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.barplot(x="Class", y="F1", hue="Model", data=melted, ax=ax)
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 Score")
    fig.tight_layout()

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150)
        print(f"Figure saved to {save_path}")
    if show:
        plt.show()
    return fig
