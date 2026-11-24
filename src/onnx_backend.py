"""ONNX Runtime backend: export, int8 quantization and inference without PyTorch.

Inference here needs only ``onnxruntime``, ``tokenizers`` and ``numpy``. Not importing torch
saves about 300 MB of RAM and makes the backend usable on small machines.

    from src.onnx_backend import OnnxSentimentModel, predict_sentiment_batch_onnx
    model = OnnxSentimentModel("models/distilbert_headline_adapted_onnx_int8")
    labels, probs, confidences, scores = predict_sentiment_batch_onnx(["Intel cuts guidance"], model)

Export (needs torch + transformers + onnx): ``python -m src.pipeline onnx-export``.
Latency benchmark of one variant in a fresh process, which keeps memory figures clean:
``python -m src.onnx_backend bench --backend onnx --model DIR --texts headlines.txt``.
"""

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from src.config import EVAL_BATCH_SIZE, LABELS
from src.labels import label_column_order, summarize_probabilities

ONNX_FILE = "model.onnx"
SIDE_FILES = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt")
HEADLINE_MAX_LENGTH = 64
REFERENCE_FILE = "reference_outputs.json"
# Per-tensor int8 agreed with fp32 on 92.6% of real headlines; a broken int8 kernel lands near chance.
SELF_CHECK_MIN_AGREEMENT = 0.8

PathLike = Union[str, Path]


def is_onnx_dir(path: PathLike) -> bool:
    return (Path(path) / ONNX_FILE).exists()


def _copy_side_files(src: Path, dst: Path) -> None:
    for name in SIDE_FILES:
        if (src / name).exists():
            shutil.copy2(src / name, dst / name)


# --------------------------------------------------------------------------- export


def export_onnx(model_dir: PathLike, out_dir: PathLike, opset: int = 17) -> Path:
    """Export a fine-tuned sequence classifier to ``out_dir/model.onnx`` (fp32), with dynamic batch and length."""
    import torch
    from transformers import AutoModelForSequenceClassification

    model_dir, out_dir = Path(model_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForSequenceClassification.from_pretrained(str(model_dir)).eval()

    class LogitsOnly(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner = inner

        def forward(self, input_ids, attention_mask):
            return self.inner(input_ids=input_ids, attention_mask=attention_mask).logits

    wrapped = LogitsOnly(model).eval()
    ids = torch.ones((2, 16), dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[1, 9:] = 0  # trace a padded batch, so data-dependent "no padding" shortcuts are not baked in
    target = out_dir / ONNX_FILE
    names = {"input_names": ["input_ids", "attention_mask"], "output_names": ["logits"]}
    with torch.no_grad():
        try:  # TorchScript exporter: deprecated, but simple and exact for DistilBERT
            torch.onnx.export(wrapped, (ids, mask), str(target), opset_version=opset, dynamo=False,
                              dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                                            "attention_mask": {0: "batch", 1: "sequence"},
                                            "logits": {0: "batch"}}, **names)
        except Exception as exc:  # torch versions without the legacy exporter
            print(f"  legacy ONNX export unavailable ({type(exc).__name__}); using the dynamo exporter")
            batch, seq = torch.export.Dim("batch"), torch.export.Dim("sequence", max=512)
            torch.onnx.export(wrapped, (ids, mask), str(target), opset_version=max(opset, 18), dynamo=True,
                              dynamic_shapes={"input_ids": {0: batch, 1: seq}, "attention_mask": {0: batch, 1: seq}},
                              external_data=False, **names)
    _copy_side_files(model_dir, out_dir)
    return target


def quantize_onnx(onnx_dir: PathLike, out_dir: PathLike, per_channel: bool = False) -> Path:
    """Dynamic int8 quantization of the weights (activations are quantized on the fly at run time).

    Per-tensor scales (the default) are portable. Per-channel scales agreed better with fp32 on an AMD EPYC
    (97.3% vs 94.9% of PhraseBank labels; see scripts/onnx_quant_sweep.py), but on an Intel laptop CPU the
    same per-channel file returned near-uniform probabilities (every headline "neutral"). So per-channel is
    opt-in, and ``OnnxSentimentModel.self_check`` guards every machine against that kind of failure.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_dir, out_dir = Path(onnx_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / ONNX_FILE
    quantize_dynamic(str(onnx_dir / ONNX_FILE), str(target), weight_type=QuantType.QInt8, per_channel=per_channel)
    _copy_side_files(onnx_dir, out_dir)
    return target


# --------------------------------------------------------------------------- inference


def available_cpus() -> Optional[int]:
    """CPUs this process may use (respects Slurm/Docker CPU sets, unlike os.cpu_count())."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:  # Windows / macOS: let onnxruntime decide
        return None


class OnnxSentimentModel:
    """An exported classifier plus its tokenizer. Exposes ``config.id2label`` like a transformers model."""

    def __init__(self, path: PathLike, num_threads: Optional[int] = None):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        self.path = Path(path)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        num_threads = num_threads or available_cpus()
        if num_threads:
            opts.intra_op_num_threads = int(num_threads)
            opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(self.path / ONNX_FILE), opts, providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}

        cfg_path = self.path / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.exists() else {}
        self.config = SimpleNamespace(id2label={int(k): v for k, v in (cfg.get("id2label") or {}).items()})
        self._tokenizer_json = (self.path / "tokenizer.json").read_text(encoding="utf-8")
        self._pad_token = "[PAD]"
        tok_cfg = self.path / "tokenizer_config.json"
        if tok_cfg.exists():
            self._pad_token = json.loads(tok_cfg.read_text(encoding="utf-8")).get("pad_token", "[PAD]")
        self._tokenizers: Dict[int, "Tokenizer"] = {}
        self._lock = Lock()
        self._Tokenizer = Tokenizer

    def _tokenizer(self, max_length: int):
        # One configured tokenizer per max_length: encoding is thread-safe, re-configuring is not.
        with self._lock:
            tok = self._tokenizers.get(max_length)
            if tok is None:
                tok = self._Tokenizer.from_str(self._tokenizer_json)
                tok.enable_truncation(max_length=max_length)
                pad_id = tok.token_to_id(self._pad_token)
                tok.enable_padding(pad_id=0 if pad_id is None else pad_id, pad_token=self._pad_token)
                self._tokenizers[max_length] = tok
        return tok

    def logits(self, texts: Sequence[str], batch_size: int = EVAL_BATCH_SIZE,
               max_length: int = HEADLINE_MAX_LENGTH) -> np.ndarray:
        tok = self._tokenizer(max_length)
        out = []
        for start in range(0, len(texts), batch_size):
            enc = tok.encode_batch(list(texts[start:start + batch_size]))
            feeds = {"input_ids": np.asarray([e.ids for e in enc], dtype=np.int64),
                     "attention_mask": np.asarray([e.attention_mask for e in enc], dtype=np.int64)}
            out.append(self.session.run(["logits"], {k: v for k, v in feeds.items() if k in self.input_names})[0])
        return np.vstack(out) if out else np.empty((0, len(LABELS)), dtype=np.float32)

    def self_check(self, min_agreement: float = SELF_CHECK_MIN_AGREEMENT) -> Optional[float]:
        """Re-score the reference headlines saved at export and compare with the fp32 labels.

        int8 kernels differ between CPUs, so a model that is fine on one machine can be broken on another.
        Returns the agreement, or None if the model has no reference file; raises if it is too low.
        """
        ref_path = self.path / REFERENCE_FILE
        if not ref_path.exists():
            return None
        ref = json.loads(ref_path.read_text(encoding="utf-8"))
        labels, _, _, _ = predict_sentiment_batch_onnx(ref["headlines"], self, max_length=ref["max_length"])
        agreement = float(np.mean(np.array(labels) == np.array(ref["labels"])))
        if agreement < min_agreement:
            raise OnnxSelfCheckError(
                f"{self.path.name} agrees with the fp32 reference on only {agreement:.0%} of {len(labels)} headlines "
                f"on this CPU (need {min_agreement:.0%}); its int8 kernels are probably unsupported here. "
                "Re-export with per-tensor quantization or use SENTIMENT_BACKEND=torch.")
        return agreement


class OnnxSelfCheckError(RuntimeError):
    pass


def write_reference(fp32_dir: PathLike, headlines: Sequence[str], out_dirs: Sequence[PathLike],
                    max_length: int = HEADLINE_MAX_LENGTH) -> dict:
    """Save fp32 labels for ``headlines`` next to each exported model, for ``self_check``."""
    labels, _, _, _ = predict_sentiment_batch_onnx(list(headlines), OnnxSentimentModel(fp32_dir), max_length=max_length)
    ref = {"source_model": Path(fp32_dir).name, "max_length": max_length, "headlines": list(headlines),
           "labels": labels}
    for d in out_dirs:
        (Path(d) / REFERENCE_FILE).write_text(json.dumps(ref, indent=1), encoding="utf-8")
    return ref


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def predict_sentiment_batch_onnx(
    texts: List[str],
    model: OnnxSentimentModel,
    batch_size: int = EVAL_BATCH_SIZE,
    max_length: int = HEADLINE_MAX_LENGTH,
) -> Tuple[List[str], np.ndarray, np.ndarray, np.ndarray]:
    """Same outputs as ``src.inference.predict_sentiment_batch``: labels, probabilities, confidences, scores."""
    if len(texts) == 0:
        return summarize_probabilities(np.empty((0, len(LABELS)), dtype=np.float32))
    probs = softmax(model.logits(list(texts), batch_size=batch_size, max_length=max_length).astype(np.float64))
    return summarize_probabilities(probs[:, label_column_order(model)].astype(np.float32))


def model_size_mb(path: PathLike) -> float:
    path = Path(path)
    weights = [path / ONNX_FILE] if is_onnx_dir(path) else list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
    weights = [w for w in weights if w.name != "training_args.bin"]
    return round(sum(w.stat().st_size for w in weights) / 2**20, 1)


# --------------------------------------------------------------------------- latency benchmark (one process each)


def _peak_rss_mb() -> float:
    """Peak resident memory of this process. Linux: VmHWM (reset by exec, unlike ru_maxrss)."""
    status = Path("/proc/self/status")
    if status.exists():
        for line in status.read_text().splitlines():
            if line.startswith("VmHWM:"):
                return round(int(line.split()[1]) / 1024, 1)
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(peak / (2**20 if sys.platform == "darwin" else 2**10), 1)
    except ImportError:  # Windows
        import psutil

        return round(psutil.Process().memory_info().peak_wset / 2**20, 1)


def _rss_mb() -> float:
    import psutil

    return round(psutil.Process().memory_info().rss / 2**20, 1)


def benchmark_latency(backend: str, model_path: PathLike, texts: List[str], batch_sizes: Sequence[int] = (1, 32),
                      threads: int = 4, repeats: int = 3, max_length: int = HEADLINE_MAX_LENGTH) -> dict:
    """End-to-end (tokenize + forward + softmax) time per headline. Run in a fresh process for memory figures."""
    rss_start = _rss_mb()
    t0 = time.perf_counter()
    if backend == "torch":
        import torch

        from src.inference import load_fine_tuned_model, predict_sentiment_batch

        torch.set_num_threads(threads)
        model, tokenizer, device = load_fine_tuned_model(model_path, device=torch.device("cpu"))

        def run(batch, bs):
            return predict_sentiment_batch(batch, model, tokenizer, device, batch_size=bs, max_length=max_length)
    elif backend == "onnx":
        model = OnnxSentimentModel(model_path, num_threads=threads)

        def run(batch, bs):
            return predict_sentiment_batch_onnx(batch, model, batch_size=bs, max_length=max_length)
    else:
        raise ValueError("backend must be torch or onnx")
    load_seconds = time.perf_counter() - t0
    rss_loaded = _rss_mb()

    run(texts[:16], 16)  # warm-up
    results = {}
    for bs in batch_sizes:
        timings = []
        for _ in range(repeats):
            start = time.perf_counter()
            for i in range(0, len(texts), bs):  # one call per batch, so both backends pad per batch
                run(texts[i:i + bs], bs)
            timings.append((time.perf_counter() - start) * 1000 / len(texts))
        results[f"batch_{bs}"] = {"ms_per_headline_median": round(float(np.median(timings)), 3),
                                  "ms_per_headline_runs": [round(t, 3) for t in timings]}
    return {"backend": backend, "model_dir": Path(model_path).name, "model_size_mb": model_size_mb(model_path),
            "threads": threads, "n_headlines": len(texts), "max_length": max_length,
            "load_seconds": round(load_seconds, 2), "rss_before_load_mb": rss_start, "rss_after_load_mb": rss_loaded,
            "peak_rss_mb": _peak_rss_mb(), "latency": results}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bench", help="latency + memory of one backend, printed as JSON")
    b.add_argument("--backend", choices=["torch", "onnx"], required=True)
    b.add_argument("--model", type=Path, required=True)
    b.add_argument("--texts", type=Path, required=True, help="one headline per line")
    b.add_argument("--threads", type=int, default=4)
    b.add_argument("--repeats", type=int, default=3)
    b.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 32])
    args = parser.parse_args(argv)
    texts = [t for t in args.texts.read_text(encoding="utf-8").splitlines() if t.strip()]
    print(json.dumps(benchmark_latency(args.backend, args.model, texts, args.batch_sizes, args.threads,
                                       args.repeats)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
