import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, List
from sklearn.metrics import f1_score, precision_score, recall_score


def compute_per_class_f1(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    labels_order = ['negative', 'neutral', 'positive']
    scores = f1_score(true_labels, pred_labels, labels=labels_order, average=None)
    return {lbl: float(s) for lbl, s in zip(labels_order, scores)}


def compute_per_class_precision(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    labels_order = ['negative', 'neutral', 'positive']
    scores = precision_score(true_labels, pred_labels, labels=labels_order, average=None)
    return {lbl: float(s) for lbl, s in zip(labels_order, scores)}


def compute_per_class_recall(true_labels: List[str], pred_labels: List[str]) -> Dict[str, float]:
    labels_order = ['negative', 'neutral', 'positive']
    scores = recall_score(true_labels, pred_labels, labels=labels_order, average=None)
    return {lbl: float(s) for lbl, s in zip(labels_order, scores)}


def plot_per_class_f1(
    f1_a: Dict[str, float],
    f1_b: Dict[str, float],
    label_a: str = 'Model A',
    label_b: str = 'Model B',
    title: str = 'Per-class F1 comparison',
    save_path: str = None,
) -> None:
    classes = ['negative', 'neutral', 'positive']
    df = pd.DataFrame({
        'Class': classes,
        label_a: [f1_a[c] for c in classes],
        label_b: [f1_b[c] for c in classes],
    })
    melted = pd.melt(df, id_vars=['Class'], var_name='Model', value_name='F1')
    plt.figure(figsize=(7, 5))
    sns.barplot(x='Class', y='F1', hue='Model', data=melted)
    plt.title(title)
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f'Figure saved to {save_path}')
    plt.show()
