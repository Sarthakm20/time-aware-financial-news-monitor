"""End-to-end command-line pipeline.

Examples (run from the project root)::

    python -m src.pipeline --list
    python -m src.pipeline prepare
    python -m src.pipeline train evaluate
    python -m src.pipeline score monitor --ticker TSLA
    python -m src.pipeline all            # every stage, in order (slow: baseline + training)

Each stage reads the previous stage's files from disk, so stages can be re-run
independently. ``--outputs-dir`` / ``--data-dir`` / ``--models-dir`` redirect files,
which is handy for checking a run against the committed results.
"""

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List

import pandas as pd

from src import config


@dataclass
class Paths:
    data: Path
    outputs: Path
    models: Path
    ticker: str

    @property
    def raw(self) -> Path:
        return self.data / "raw"

    @property
    def processed(self) -> Path:
        return self.data / "processed"

    @property
    def metrics(self) -> Path:
        return self.outputs / "metrics"

    @property
    def predictions(self) -> Path:
        return self.outputs / "predictions"

    @property
    def figures(self) -> Path:
        return self.outputs / "figures"

    @property
    def model(self) -> Path:
        return self.models / config.DEFAULT_FINETUNED_MODEL_DIR.name

    @property
    def weighted_model(self) -> Path:
        return self.models / config.WEIGHTED_MODEL_DIR.name

    @property
    def news_csv(self) -> Path:
        return self.processed / f"fnspid_{self.ticker.lower()}_2019_2023.csv"

    @property
    def scored_csv(self) -> Path:
        return self.predictions / f"{self.ticker.lower()}_scored_headlines_2019_2023.csv"

    def ensure(self) -> None:
        for folder in [self.raw, self.processed, self.metrics, self.predictions, self.figures, self.models]:
            folder.mkdir(parents=True, exist_ok=True)


def _require(path: Path, stage: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing - run the '{stage}' stage first.")
    return path


def _write_json(payload: dict, path: Path) -> None:
    from src.evaluate import save_metrics_json

    save_metrics_json(payload, path)
    print(f"  wrote {path}")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {path} ({len(df)} rows)")


# --------------------------------------------------------------------------- stages


def stage_prepare(p: Paths, args: argparse.Namespace) -> None:
    """Load PhraseBank (local raw CSV, else download), clean, de-conflict and split."""
    from src.data_loading import load_phrasebank_from_hub
    from src.preprocessing import clean_phrasebank_dataframe, prepare_model_dataframe, split_data_stratified

    raw_csv = p.raw / "financial_phrasebank_50agree_raw.csv"
    if raw_csv.exists():
        raw = pd.read_csv(raw_csv)
        print(f"  using local {raw_csv}")
    else:
        raw = load_phrasebank_from_hub()
        _write_csv(raw, raw_csv)

    clean = clean_phrasebank_dataframe(raw)
    _write_csv(clean, p.processed / "financial_phrasebank_50agree_clean.csv")

    model_df = prepare_model_dataframe(clean)
    train_df, val_df, test_df = split_data_stratified(model_df, seed=args.seed)
    for name, frame in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        _write_csv(frame, p.processed / f"{name}.csv")


def stage_baseline(p: Paths, args: argparse.Namespace) -> None:
    """Zero-shot BART-MNLI on the test split (slow on CPU: hours)."""
    from src.baseline import run_zero_shot_baseline
    from src.evaluate import compute_confusion_matrix
    from src.visualization import plot_confusion_matrix

    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    metrics, preds = run_zero_shot_baseline(test_df)
    _write_json(metrics, p.metrics / "zero_shot_test_metrics.json")
    _write_csv(preds, p.predictions / "zero_shot_test_predictions.csv")
    cm = compute_confusion_matrix(preds["label"], preds["predicted_label"])
    plot_confusion_matrix(cm, title="Zero-Shot Baseline Confusion Matrix",
                          save_path=p.figures / "zero_shot_confusion_matrix.png")


def stage_train(p: Paths, args: argparse.Namespace) -> None:
    """Fine-tune DistilBERT (and, with --weighted, a class-weighted variant)."""
    from src.train import train_distilbert_classifier

    train_df = pd.read_csv(_require(p.processed / "train.csv", "prepare"))
    val_df = pd.read_csv(_require(p.processed / "validation.csv", "prepare"))

    metrics = train_distilbert_classifier(train_df, val_df, output_dir=p.model,
                                          num_epochs=args.epochs, seed=args.seed)
    _write_json(metrics, p.metrics / "finetuned_validation_metrics.json")

    if args.weighted:
        from src.weighted_trainer import get_class_weights

        weights = get_class_weights(train_df)
        print(f"  class weights (negative, neutral, positive): {weights.tolist()}")
        w_metrics = train_distilbert_classifier(train_df, val_df, output_dir=p.weighted_model,
                                                num_epochs=args.epochs, seed=args.seed,
                                                class_weight_tensor=weights)
        _write_json(w_metrics, p.metrics / "weighted_validation_metrics.json")


def _evaluate_model(model_dir: Path, test_df: pd.DataFrame) -> pd.DataFrame:
    from src.inference import load_fine_tuned_model, predict_sentiment_batch

    model, tokenizer, device = load_fine_tuned_model(model_dir)
    labels, probs, confs, _ = predict_sentiment_batch(test_df["sentence"].tolist(), model, tokenizer, device)
    out = test_df.copy()
    out["predicted_label"] = labels
    out["confidence"] = confs
    for idx, label in config.ID_TO_LABEL.items():
        out[f"{label}_probability"] = probs[:, idx]
    return out


def stage_evaluate(p: Paths, args: argparse.Namespace) -> None:
    """Score the untouched test split with the fine-tuned model(s)."""
    from src.evaluate import calculate_metrics, compute_confusion_matrix, export_error_analysis
    from src.visualization import plot_baseline_vs_finetuned, plot_confusion_matrix

    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    _require(p.model, "train")

    preds = _evaluate_model(p.model, test_df)
    metrics = {
        "model": f"{config.BASE_MODEL_NAME} fine-tuned on Financial PhraseBank",
        "evaluation_split": "held-out test set",
        "num_examples": int(len(preds)),
        **calculate_metrics(preds["label"].tolist(), preds["predicted_label"].tolist()),
    }
    _write_json(metrics, p.metrics / "finetuned_test_metrics.json")
    _write_csv(preds, p.predictions / "finetuned_test_predictions.csv")
    export_error_analysis(preds, output_path=p.predictions / "finetuned_test_errors.csv")

    cm = compute_confusion_matrix(preds["label"], preds["predicted_label"])
    plot_confusion_matrix(cm, title="Fine-Tuned DistilBERT Test Confusion Matrix",
                          save_path=p.figures / "finetuned_test_confusion_matrix.png")

    zero_shot_json = p.metrics / "zero_shot_test_metrics.json"
    if zero_shot_json.exists():
        baseline = json.loads(zero_shot_json.read_text(encoding="utf-8"))
        plot_baseline_vs_finetuned(baseline, metrics,
                                   save_path=p.figures / "baseline_vs_finetuned_comparison.png")

    if p.weighted_model.exists():
        w_preds = _evaluate_model(p.weighted_model, test_df)
        w_metrics = {
            "model": f"{config.BASE_MODEL_NAME} fine-tuned with class-weighted loss",
            "evaluation_split": "held-out test set",
            "num_examples": int(len(w_preds)),
            **calculate_metrics(w_preds["label"].tolist(), w_preds["predicted_label"].tolist()),
        }
        _write_json(w_metrics, p.metrics / "weighted_test_metrics.json")

    print(f"  test accuracy={metrics['accuracy']}  macro_f1={metrics['macro_f1']}")


def stage_fetch_news(p: Paths, args: argparse.Namespace) -> None:
    """Stream one ticker's 2019-2023 headlines from FNSPID."""
    from src.data_loading import stream_fnspid_slice

    news = stream_fnspid_slice(ticker=args.ticker)
    _write_csv(news, p.news_csv)


def stage_score(p: Paths, args: argparse.Namespace) -> None:
    """Score each headline: label, class probabilities, P(pos) - P(neg)."""
    from src.inference import score_news_dataframe

    news = pd.read_csv(_require(p.news_csv, "fetch-news"))
    news["Date"] = pd.to_datetime(news["Date"], utc=True)
    news = news.sort_values("Date", kind="stable").reset_index(drop=True)
    scored = score_news_dataframe(news, model_path=_require(p.model, "train"))
    _write_csv(scored, p.scored_csv)


def stage_monitor(p: Paths, args: argparse.Namespace) -> None:
    """Window summaries (30/180/365 days/all) and the rolling sentiment timeline."""
    from src.aggregation import compute_rolling_sentiment, summarize_sentiment_windows
    from src.visualization import plot_rolling_sentiment_timeline

    scored = pd.read_csv(_require(p.scored_csv, "score"))
    scored["Date"] = pd.to_datetime(scored["Date"], utc=True)

    summary = summarize_sentiment_windows(scored, half_life_days=args.half_life)
    _write_csv(summary, p.metrics / f"{args.ticker.lower()}_sentiment_window_summary.csv")

    rolling = compute_rolling_sentiment(scored)
    plot_rolling_sentiment_timeline(rolling, ticker=args.ticker,
                                    save_path=p.figures / f"{args.ticker.lower()}_30day_rolling_sentiment.png")
    print(summary.to_string(index=False))


STAGES: Dict[str, Callable[[Paths, argparse.Namespace], None]] = {
    "prepare": stage_prepare,
    "baseline": stage_baseline,
    "train": stage_train,
    "evaluate": stage_evaluate,
    "fetch-news": stage_fetch_news,
    "score": stage_score,
    "monitor": stage_monitor,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.pipeline", description=__doc__.split("\n")[0])
    parser.add_argument("stages", nargs="*", help=f"stages to run, in order: {', '.join(STAGES)} or 'all'")
    parser.add_argument("--list", action="store_true", help="list stages and exit")
    parser.add_argument("--ticker", default=config.DEFAULT_TICKER)
    parser.add_argument("--epochs", type=int, default=config.NUM_TRAIN_EPOCHS)
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--half-life", type=float, default=config.HALF_LIFE_DAYS, help="recency half-life in days")
    parser.add_argument("--weighted", action="store_true", help="also train the class-weighted model")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    return parser


def resolve_stages(requested: List[str]) -> List[str]:
    if requested == ["all"]:
        return list(STAGES)
    unknown = [s for s in requested if s not in STAGES]
    if unknown:
        raise SystemExit(f"Unknown stage(s): {', '.join(unknown)}. Choose from: {', '.join(STAGES)}, all")
    return requested


def main(argv: List[str] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list or not args.stages:
        for name, fn in STAGES.items():
            print(f"{name:<11} {fn.__doc__.strip().splitlines()[0]}")
        return 0

    import matplotlib

    matplotlib.use("Agg")  # stages save figures to disk; never open windows

    paths = Paths(data=args.data_dir, outputs=args.outputs_dir, models=args.models_dir, ticker=args.ticker)
    paths.ensure()
    for name in resolve_stages(args.stages):
        print(f"[{name}]")
        STAGES[name](paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
