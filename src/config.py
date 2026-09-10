from pathlib import Path
from typing import Dict, List, Optional

SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
DEFAULT_FINETUNED_MODEL_DIR = MODELS_DIR / "distilbert_financial_sentiment"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"
METRICS_DIR = OUTPUTS_DIR / "metrics"
PREDICTIONS_DIR = OUTPUTS_DIR / "predictions"
FIGURES_DIR = OUTPUTS_DIR / "figures"

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

LABELS: List[str] = ["negative", "neutral", "positive"]

LABEL_TO_ID: Dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID_TO_LABEL: Dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}

MAX_SEQUENCE_LENGTH: int = 256
TRAIN_BATCH_SIZE: int = 16
EVAL_BATCH_SIZE: int = 32
LEARNING_RATE: float = 2e-5
NUM_TRAIN_EPOCHS: int = 3
WEIGHT_DECAY: float = 0.01

TRAIN_SPLIT_RATIO: float = 0.70
VAL_SPLIT_RATIO: float = 0.15
TEST_SPLIT_RATIO: float = 0.15

DEFAULT_TICKER: str = "TSLA"
HALF_LIFE_DAYS: int = 30

MONITORING_WINDOWS: Dict[str, Optional[int]] = {
    "Last 30 days": 30,
    "Last 180 days": 180,
    "Last 365 days": 365,
    "All available history": None,
}

FNSPID_HF_DATASET: str = "benstaf/FNSPID-nasdaq-100-post2019-1newsperrow"
PHRASEBANK_HF_REPO: str = "takala/financial_phrasebank"
PHRASEBANK_ZIP_FILE: str = "data/FinancialPhraseBank-v1.0.zip"
PHRASEBANK_INNER_FILE: str = "FinancialPhraseBank-v1.0/Sentences_50Agree.txt"

