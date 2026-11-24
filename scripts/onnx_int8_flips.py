"""How do int8 labels differ from fp32 on the 554 annotated real headlines? (polarity flips vs softening)

Needs both exports (``python -m src.pipeline onnx-export``). Writes outputs/metrics/onnx_int8_vs_fp32.json.
Ran on the Plaksha HPC (4 threads); loads only ONNX models, no torch.

    python scripts/onnx_int8_flips.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.onnx_backend import OnnxSentimentModel, predict_sentiment_batch_onnx  # noqa: E402

heads = pd.read_csv(ROOT / "outputs/predictions/multi_company_headline_predictions.csv")
heads = heads[heads["passes_filter"]]
x = heads["Article_title"].tolist()
f, _, _, fs = predict_sentiment_batch_onnx(x, OnnxSentimentModel(ROOT / "models/distilbert_headline_adapted_onnx"))
q, _, _, qs = predict_sentiment_batch_onnx(x, OnnxSentimentModel(ROOT / "models/distilbert_headline_adapted_onnx_int8"))
f, q = np.array(f), np.array(q)
polar = ((f == "positive") & (q == "negative")) | ((f == "negative") & (q == "positive"))
out = {"n": len(x), "label_disagreement": round(float(np.mean(f != q)), 4), "polarity_flips": int(polar.sum()),
       "polarity_flip_rate": round(float(polar.mean()), 4),
       "softened_to_neutral": int(((f != "neutral") & (q == "neutral")).sum()),
       "score_correlation": round(float(np.corrcoef(fs, qs)[0, 1]), 4),
       "mean_abs_score_diff": round(float(np.abs(fs - qs).mean()), 4),
       "confusion": {"rows": "fp32 label", "columns": ["negative", "neutral", "positive"],
                     **{a: [int(((f == a) & (q == b)).sum()) for b in ("negative", "neutral", "positive")]
                        for a in ("negative", "neutral", "positive")}}}
path = ROOT / "outputs/metrics/onnx_int8_vs_fp32.json"
path.write_text(json.dumps(out, indent=1), encoding="utf-8")
print(json.dumps(out, indent=1))
