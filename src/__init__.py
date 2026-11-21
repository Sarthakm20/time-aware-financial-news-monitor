"""Time-aware financial news sentiment monitor.

Public functions are re-exported lazily so that ``import src.aggregation`` (or the
test suite) does not pull in torch/transformers unless a model function is used.
Notebooks can keep using ``from src import calculate_metrics`` etc.
"""

from importlib import import_module
from typing import Any

from src.config import (  # noqa: F401  (lightweight, re-exported eagerly)
    DATA_DIR,
    DEFAULT_FINETUNED_MODEL_DIR,
    FIGURES_DIR,
    HALF_LIFE_DAYS,
    ID_TO_LABEL,
    LABEL_TO_ID,
    LABELS,
    METRICS_DIR,
    MODELS_DIR,
    MONITORING_WINDOWS,
    OUTPUTS_DIR,
    PREDICTIONS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
)

__version__ = "0.2.0"

_LAZY_EXPORTS = {
    "clean_phrasebank_dataframe": "src.preprocessing",
    "prepare_model_dataframe": "src.preprocessing",
    "add_label_ids": "src.preprocessing",
    "split_data_stratified": "src.preprocessing",
    "dataframe_to_hf_dataset": "src.preprocessing",
    "tokenize_hf_dataset": "src.preprocessing",
    "load_phrasebank_from_hub": "src.data_loading",
    "load_split_csvs": "src.data_loading",
    "stream_fnspid_slice": "src.data_loading",
    "calculate_metrics": "src.evaluate",
    "compute_trainer_metrics": "src.evaluate",
    "compute_confusion_matrix": "src.evaluate",
    "export_error_analysis": "src.evaluate",
    "save_metrics_json": "src.evaluate",
    "load_fine_tuned_model": "src.inference",
    "predict_sentiment_batch": "src.inference",
    "score_news_dataframe": "src.inference",
    "compute_recency_weights": "src.aggregation",
    "summarize_sentiment_windows": "src.aggregation",
    "compute_rolling_sentiment": "src.aggregation",
    "plot_confusion_matrix": "src.visualization",
    "plot_baseline_vs_finetuned": "src.visualization",
    "plot_rolling_sentiment_timeline": "src.visualization",
    "run_zero_shot_baseline": "src.baseline",
    "train_distilbert_classifier": "src.train",
}

__all__ = sorted(_LAZY_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module 'src' has no attribute {name!r}")
    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_LAZY_EXPORTS))
