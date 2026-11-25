from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_FINETUNED_MODEL_DIR = MODELS_DIR / "distilbert_financial_sentiment"
WEIGHTED_MODEL_DIR = MODELS_DIR / "distilbert_weighted"
ADAPTED_MODEL_DIR = MODELS_DIR / "distilbert_headline_adapted"
# ONNX Runtime exports of the headline-adapted model (``python -m src.pipeline onnx-export``)
ADAPTED_ONNX_DIR = MODELS_DIR / "distilbert_headline_adapted_onnx"
ADAPTED_ONNX_INT8_DIR = MODELS_DIR / "distilbert_headline_adapted_onnx_int8"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"
WEB_DIR = PROJECT_ROOT / "web"


def ensure_project_dirs() -> None:
    """Create the data/model/output folders. Called explicitly instead of on import."""
    for folder in [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        MODELS_DIR,
        METRICS_DIR,
        PREDICTIONS_DIR,
        FIGURES_DIR,
    ]:
        folder.mkdir(parents=True, exist_ok=True)


RANDOM_SEED: int = 42

BASE_MODEL_NAME: str = "distilbert-base-uncased"
BASELINE_MODEL_NAME: str = "facebook/bart-large-mnli"

# The single hypothesis template used for the reported zero-shot baseline
# (outputs/metrics/zero_shot_test_metrics.json).
ZERO_SHOT_HYPOTHESIS_TEMPLATE: str = "The financial sentiment of this statement is {}."

LABELS: List[str] = ["negative", "neutral", "positive"]

LABEL_TO_ID: Dict[str, int] = {label: idx for idx, label in enumerate(LABELS)}
ID_TO_LABEL: Dict[int, str] = {idx: label for idx, label in enumerate(LABELS)}

MAX_SEQUENCE_LENGTH: int = 256
TRAIN_BATCH_SIZE: int = 16
EVAL_BATCH_SIZE: int = 32
LEARNING_RATE: float = 2e-5
# The reported model (models/distilbert_financial_sentiment) was trained for 2 epochs
# (checkpoint-424 = 2 x 212 steps). Keep this in sync with the README.
NUM_TRAIN_EPOCHS: int = 2
WEIGHT_DECAY: float = 0.01

TRAIN_SPLIT_RATIO: float = 0.70
VAL_SPLIT_RATIO: float = 0.15
TEST_SPLIT_RATIO: float = 0.15

DEFAULT_TICKER: str = "TSLA"
DEFAULT_START_DATE: str = "2019-01-01"
DEFAULT_END_DATE: str = "2024-01-01"
HALF_LIFE_DAYS: int = 30
ROLLING_WINDOW_DAYS: int = 30

MONITORING_WINDOWS: Dict[str, Optional[int]] = {
    "Last 30 days": 30,
    "Last 180 days": 180,
    "Last 365 days": 365,
    "All available history": None,
}

# Headlines describing the whole market rather than the company (used for a sensitivity check:
# annotators label these neutral for the company, while models score their general tone).
MARKET_WIDE_HEADLINE_PATTERN: str = (
    r"^(?:US STOCKS|Wall St|Dow\b|Stock Market (?:News|Today)|Pre-Market Most Active|Stock Investors|"
    r"Fed Surprise|Position Scrimmage)"
)

FNSPID_HF_DATASET: str = "benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"
PHRASEBANK_HF_REPO: str = "takala/financial_phrasebank"
PHRASEBANK_ZIP_FILE: str = "data/FinancialPhraseBank-v1.0.zip"
PHRASEBANK_INNER_FILE: str = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"
