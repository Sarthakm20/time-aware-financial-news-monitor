"""Shared prediction service for the API and the dashboard.

Model resolution order:
1. ``SENTIMENT_MODEL`` environment variable (local directory or Hugging Face Hub id),
2. the headline-adapted model in ``models/`` if it exists,
3. the PhraseBank fine-tuned model in ``models/``.

The model is loaded lazily on first use, so importing this module is cheap.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from src.config import ADAPTED_MODEL_DIR, DEFAULT_FINETUNED_MODEL_DIR, LABELS
from src.relevance import is_relevant

MAX_HEADLINES_PER_REQUEST = 256
MAX_HEADLINE_CHARS = 500


def resolve_model_source() -> str:
    env = os.environ.get("SENTIMENT_MODEL", "").strip()
    if env:
        return env
    for candidate in (ADAPTED_MODEL_DIR, DEFAULT_FINETUNED_MODEL_DIR):
        if (Path(candidate) / "config.json").exists():
            return str(candidate)
    raise FileNotFoundError(
        "No sentiment model found. Set SENTIMENT_MODEL to a Hugging Face Hub id or train one with "
        "`python -m src.pipeline train adapt`."
    )


def model_display_name(source: str) -> str:
    """Name safe to show publicly (never a local absolute path)."""
    path = Path(source)
    return path.name if path.exists() else source


@dataclass
class Prediction:
    headline: str
    label: str
    probabilities: Dict[str, float]
    sentiment_score: float
    confidence: float
    relevant: Optional[bool] = None

    def to_dict(self) -> dict:
        return {
            "headline": self.headline,
            "label": self.label,
            "probabilities": self.probabilities,
            "sentiment_score": self.sentiment_score,
            "confidence": self.confidence,
            "relevant": self.relevant,
        }


class SentimentService:
    """Thread-safe, lazily loaded wrapper around the fine-tuned classifier."""

    def __init__(self, model_source: Optional[str] = None):
        self._source = model_source
        self._lock = Lock()
        self._bundle = None  # (model, tokenizer, device)

    @property
    def source(self) -> str:
        if self._source is None:
            self._source = resolve_model_source()
        return self._source

    @property
    def model_name(self) -> str:
        return model_display_name(self.source)

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def _load(self):
        with self._lock:
            if self._bundle is None:
                from src.inference import load_fine_tuned_model

                self._bundle = load_fine_tuned_model(self.source)
        return self._bundle

    def predict(self, headlines: Sequence[str], ticker: Optional[str] = None) -> List[Prediction]:
        headlines = [str(h).strip()[:MAX_HEADLINE_CHARS] for h in headlines]
        if not headlines:
            return []
        if len(headlines) > MAX_HEADLINES_PER_REQUEST:
            raise ValueError(f"at most {MAX_HEADLINES_PER_REQUEST} headlines per request")
        if any(not h for h in headlines):
            raise ValueError("headlines must be non-empty")

        from src.inference import predict_sentiment_batch

        model, tokenizer, device = self._load()
        labels, probs, confs, scores = predict_sentiment_batch(headlines, model, tokenizer, device, max_length=64)
        return [
            Prediction(
                headline=h,
                label=label,
                probabilities={name: round(float(p[i]), 4) for i, name in enumerate(LABELS)},
                sentiment_score=round(float(score), 4),
                confidence=round(float(conf), 4),
                relevant=is_relevant(h, ticker) if ticker else None,
            )
            for h, label, p, conf, score in zip(headlines, labels, probs, confs, scores)
        ]
