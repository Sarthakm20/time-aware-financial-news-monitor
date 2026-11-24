"""Shared prediction service for the API and the dashboard.

Model resolution order:
1. ``SENTIMENT_MODEL`` environment variable (local directory or Hugging Face Hub id),
2. the headline-adapted model in ``models/`` if it exists,
3. the PhraseBank fine-tuned model in ``models/``,
4. the ONNX exports of the headline-adapted model (int8 first).

Backend: ``SENTIMENT_BACKEND`` = ``torch`` | ``onnx`` | ``auto`` (default). ``auto`` uses ONNX Runtime
when the chosen directory holds a ``model.onnx``; ``onnx`` also moves the ONNX exports to the front
of the search. The ONNX backend never imports torch (about 300 MB less RAM).

The model is loaded lazily on first use, so importing this module is cheap.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional, Sequence

from src.config import (
    ADAPTED_MODEL_DIR,
    ADAPTED_ONNX_DIR,
    ADAPTED_ONNX_INT8_DIR,
    DEFAULT_FINETUNED_MODEL_DIR,
    LABELS,
)
from src.relevance import is_relevant

MAX_HEADLINES_PER_REQUEST = 256
MAX_HEADLINE_CHARS = 500


BACKENDS = ("auto", "torch", "onnx")
ONNX_MODEL_DIRS = (ADAPTED_ONNX_INT8_DIR, ADAPTED_ONNX_DIR)
TORCH_MODEL_DIRS = (ADAPTED_MODEL_DIR, DEFAULT_FINETUNED_MODEL_DIR)


def requested_backend() -> str:
    backend = os.environ.get("SENTIMENT_BACKEND", "auto").strip().lower() or "auto"
    if backend not in BACKENDS:
        raise ValueError(f"SENTIMENT_BACKEND must be one of {', '.join(BACKENDS)}, not {backend!r}")
    return backend


def resolve_model_source(backend: Optional[str] = None) -> str:
    env = os.environ.get("SENTIMENT_MODEL", "").strip()
    if env:
        return env
    backend = backend or requested_backend()
    if backend == "onnx":
        candidates = ONNX_MODEL_DIRS
    elif backend == "torch":
        candidates = TORCH_MODEL_DIRS
    else:
        candidates = TORCH_MODEL_DIRS + ONNX_MODEL_DIRS
    for candidate in candidates:
        is_onnx = (Path(candidate) / "model.onnx").exists()
        if (Path(candidate) / "config.json").exists() and (is_onnx or candidate not in ONNX_MODEL_DIRS):
            return str(candidate)
    raise FileNotFoundError(
        "No sentiment model found. Set SENTIMENT_MODEL to a Hugging Face Hub id or a local directory, or build "
        "one with `python -m src.pipeline train adapt` (and `onnx-export` for the ONNX backend)."
    )


def resolve_backend(source: str, backend: Optional[str] = None) -> str:
    backend = backend or requested_backend()
    if backend != "auto":
        return backend
    return "onnx" if (Path(source) / "model.onnx").exists() else "torch"


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

    def __init__(self, model_source: Optional[str] = None, backend: Optional[str] = None):
        self._source = model_source
        self._backend = backend
        self._lock = Lock()
        self._bundle = None  # torch: (model, tokenizer, device); onnx: OnnxSentimentModel

    @property
    def source(self) -> str:
        if self._source is None:
            self._source = resolve_model_source(self._backend)
        return self._source

    @property
    def backend(self) -> str:
        return resolve_backend(self.source, self._backend)

    @property
    def model_name(self) -> str:
        return model_display_name(self.source)

    @property
    def loaded(self) -> bool:
        return self._bundle is not None

    def _load(self):
        with self._lock:
            if self._bundle is None:
                if self.backend == "onnx":
                    from src.onnx_backend import OnnxSentimentModel

                    model = OnnxSentimentModel(self.source)
                    model.self_check()  # refuse to serve an int8 model whose kernels misbehave on this CPU
                    self._bundle = model
                else:
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

        bundle = self._load()
        if self.backend == "onnx":
            from src.onnx_backend import predict_sentiment_batch_onnx

            labels, probs, confs, scores = predict_sentiment_batch_onnx(headlines, bundle, max_length=64)
        else:
            from src.inference import predict_sentiment_batch

            model, tokenizer, device = bundle
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
