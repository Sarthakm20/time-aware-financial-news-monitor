from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import LABELS


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str] = LABELS,
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    cmap: str = "Blues",
) -> plt.Figure:
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=True,
        ax=ax,
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=10)
    ax.set_xlabel("Predicted Label", fontsize=10)
    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_baseline_vs_finetuned(
    baseline_metrics: Dict[str, Any],
    finetuned_metrics: Dict[str, Any],
    save_path: Optional[Path] = None,
) -> plt.Figure:
    categories = ["Accuracy", "Macro F1", "Negative F1", "Neutral F1", "Positive F1"]
    base_scores = [
        baseline_metrics["accuracy"],
        baseline_metrics["macro_f1"],
        baseline_metrics["per_class_f1"]["negative"],
        baseline_metrics["per_class_f1"]["neutral"],
        baseline_metrics["per_class_f1"]["positive"],
    ]
    ft_scores = [
        finetuned_metrics["accuracy"],
        finetuned_metrics["macro_f1"],
        finetuned_metrics["per_class_f1"]["negative"],
        finetuned_metrics["per_class_f1"]["neutral"],
        finetuned_metrics["per_class_f1"]["positive"],
    ]

    x = np.arange(len(categories))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    rects1 = ax.bar(x - width / 2, base_scores, width, label="Zero-Shot (BART-MNLI)", color="#7f7f7f")
    rects2 = ax.bar(x + width / 2, ft_scores, width, label="Fine-Tuned DistilBERT", color="#1f77b4")

    ax.set_ylabel("Score (0.0 - 1.0)", fontsize=11)
    ax.set_title("Classifier Performance: Baseline vs. Fine-Tuned DistilBERT", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0.0, 1.05)
    ax.legend(frameon=True)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for rect in list(rects1) + list(rects2):
        height = rect.get_height()
        ax.annotate(
            f"{height:.2f}",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig


def plot_rolling_sentiment_timeline(
    rolling_df: pd.DataFrame,
    ticker: str = "TSLA",
    window_days: int = 30,
    save_path: Optional[Path] = None,
) -> plt.Figure:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})

    ax1.plot(
        rolling_df["Date"],
        rolling_df["rolling_sentiment"],
        color="#1f77b4",
        linewidth=2,
        label=f"{window_days}-Day Rolling Sentiment",
    )
    ax1.axhline(0, color="gray", linestyle="--", linewidth=1, alpha=0.7)
    ax1.set_ylabel("Sentiment Score [-1, +1]", fontsize=11)
    ax1.set_title(
        f"{ticker} Historical Financial News Monitor ({window_days}-Day Rolling)",
        fontsize=13,
        fontweight="bold",
    )
    ax1.legend(loc="upper left")
    ax1.grid(True, linestyle="--", alpha=0.4)

    ax2.bar(
        rolling_df["Date"],
        rolling_df["rolling_article_count"],
        color="#aec7e8",
        width=2,
        label=f"{window_days}-Day Cumulative Articles",
    )
    ax2.set_ylabel("Article Count", fontsize=11)
    ax2.set_xlabel("Date", fontsize=11)
    ax2.legend(loc="upper left")
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches="tight")

    return fig



def plot_reliability_diagram(
    curves: Dict[str, List[Dict[str, float]]],
    title: str = "Reliability diagram",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """Accuracy vs confidence per bin for one or more models (bins from stats.expected_calibration_error)."""
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", linewidth=1, label="Perfect calibration")
    for (name, bins), color in zip(curves.items(), ["#1f77b4", "#d62728", "#2ca02c", "#9467bd"]):
        conf = [b["confidence"] for b in bins]
        acc = [b["accuracy"] for b in bins]
        sizes = [max(15, b["count"] / 2) for b in bins]
        ax.plot(conf, acc, marker="o", color=color, linewidth=1.5, label=name)
        ax.scatter(conf, acc, s=sizes, color=color, alpha=0.35)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
