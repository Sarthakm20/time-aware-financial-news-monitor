"""Which dynamic-quantization settings keep int8 closest to the fp32 ONNX model?

One-off experiment behind the per-channel default in ``src.onnx_backend.quantize_onnx``.
Needs ``models/distilbert_headline_adapted_onnx`` (``python -m src.pipeline onnx-export``).
Writes ``outputs/metrics/onnx_quantization_sweep.json``. Ran on 4 CPU threads (Plaksha HPC).

    python scripts/onnx_quant_sweep.py
"""

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from onnxruntime.quantization import QuantType, quantize_dynamic

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.onnx_backend import OnnxSentimentModel, _copy_side_files, predict_sentiment_batch_onnx  # noqa: E402

CONFIGS = {
    "default": {},
    "per_channel": {"per_channel": True},
    "matmul_only": {"op_types_to_quantize": ["MatMul"]},
    "per_channel_matmul_only": {"per_channel": True, "op_types_to_quantize": ["MatMul"]},
    "per_channel_matmul_reduce_range": {"per_channel": True, "op_types_to_quantize": ["MatMul"], "reduce_range": True},
    "uint8_weights_matmul": {"op_types_to_quantize": ["MatMul"], "weight_type": QuantType.QUInt8},
}


def main(threads: int = 4) -> None:
    fp32 = ROOT / "models" / "distilbert_headline_adapted_onnx"
    test = pd.read_csv(ROOT / "data/processed/test.csv")
    heads = pd.read_csv(ROOT / "outputs/predictions/multi_company_headline_predictions.csv")
    heads = heads[heads["passes_filter"]]
    sets = {"phrasebank": (test["sentence"].tolist(), test["label"].tolist(), 256),
            "headlines": (heads["Article_title"].tolist(), heads["label"].tolist(), 64)}
    ref_model = OnnxSentimentModel(fp32, num_threads=threads)
    ref = {k: predict_sentiment_batch_onnx(x, ref_model, max_length=m) for k, (x, _, m) in sets.items()}

    results = {}
    with tempfile.TemporaryDirectory() as tmp:
        for name, kw in CONFIGS.items():
            out = Path(tmp) / name
            out.mkdir()
            quantize_dynamic(str(fp32 / "model.onnx"), str(out / "model.onnx"), **{"weight_type": QuantType.QInt8,
                                                                                   **kw})
            _copy_side_files(fp32, out)
            model = OnnxSentimentModel(out, num_threads=threads)
            row = {"size_mb": round((out / "model.onnx").stat().st_size / 2**20, 1)}
            for k, (x, y, m) in sets.items():
                labels, probs, _, _ = predict_sentiment_batch_onnx(x, model, max_length=m)
                row[f"{k}_agreement"] = round(float(np.mean(np.array(labels) == np.array(ref[k][0]))), 4)
                row[f"{k}_mean_abs_prob_diff"] = round(float(np.abs(probs - ref[k][1]).mean()), 4)
                row[f"{k}_accuracy"] = round(float(np.mean(np.array(labels) == np.array(y))), 4)
            sample = sets["headlines"][0][:300]
            start = time.perf_counter()
            for h in sample:
                predict_sentiment_batch_onnx([h], model)
            row["ms_batch1"] = round((time.perf_counter() - start) * 1000 / len(sample), 3)
            results[name] = row
            print(name, row, flush=True)
    path = ROOT / "outputs" / "metrics" / "onnx_quantization_sweep.json"
    path.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
