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
    def adapted_model(self) -> Path:
        return self.models / config.ADAPTED_MODEL_DIR.name

    @property
    def adapted_onnx(self) -> Path:
        return self.models / config.ADAPTED_ONNX_DIR.name

    @property
    def adapted_onnx_int8(self) -> Path:
        return self.models / config.ADAPTED_ONNX_INT8_DIR.name

    @property
    def live_db(self) -> Path:
        return self.data / "live" / "news.db"

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
        _write_csv(w_preds, p.predictions / "weighted_test_predictions.csv")

    print(f"  test accuracy={metrics['accuracy']}  macro_f1={metrics['macro_f1']}")


def stage_train_seeds(p: Paths, args: argparse.Namespace) -> None:
    """Train DistilBERT once per --seeds value and score each on the test split (resumable)."""
    from src.evaluate import calculate_metrics
    from src.train import train_distilbert_classifier

    train_df = pd.read_csv(_require(p.processed / "train.csv", "prepare"))
    val_df = pd.read_csv(_require(p.processed / "validation.csv", "prepare"))
    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    if args.max_train_samples:
        train_df = train_df.sample(n=min(args.max_train_samples, len(train_df)), random_state=0)

    runs_csv = p.metrics / "seed_runs.csv"
    rows = []
    for seed in args.seeds:
        seed_dir = p.models / "seeds" / f"seed_{seed}"
        if seed == config.RANDOM_SEED and (p.model / "model.safetensors").exists() and not args.max_train_samples:
            model_dir, source = p.model, "existing reported model"
        elif (seed_dir / "model.safetensors").exists():
            model_dir, source = seed_dir, "resumed"
        else:
            print(f"  training seed {seed} ...")
            train_distilbert_classifier(train_df, val_df, output_dir=seed_dir, num_epochs=args.epochs,
                                        seed=seed, keep_checkpoints=False)
            model_dir, source = seed_dir, "trained"

        preds = _evaluate_model(model_dir, test_df)
        _write_csv(preds, p.predictions / "seeds" / f"seed_{seed}_test_predictions.csv")
        m = calculate_metrics(preds["label"].tolist(), preds["predicted_label"].tolist())
        rows.append({"seed": seed, "source": source, "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     **{f"{k}_f1": v for k, v in m["per_class_f1"].items()}})
        _write_csv(pd.DataFrame(rows), runs_csv)  # written after every seed so progress survives interruptions
        print(f"  seed {seed} ({source}): accuracy={m['accuracy']} macro_f1={m['macro_f1']}")

    _write_json(summarize_seed_runs(pd.DataFrame(rows), epochs=args.epochs), p.metrics / "seed_summary.json")


def summarize_seed_runs(runs: pd.DataFrame, epochs: int = config.NUM_TRAIN_EPOCHS) -> dict:
    """Mean / sample std / min / max of every metric column in seed_runs.csv."""
    metric_cols = [c for c in runs.columns if c not in ("seed", "source")]
    return {
        "model": f"{config.BASE_MODEL_NAME} fine-tuned on Financial PhraseBank",
        "evaluation_split": "held-out test set",
        "epochs": epochs,
        "n_seeds": int(len(runs)),
        "seeds": [int(s) for s in runs["seed"]],
        **{col: {"mean": round(float(runs[col].mean()), 4),
                 "std": round(float(runs[col].std(ddof=1)), 4) if len(runs) > 1 else 0.0,
                 "min": round(float(runs[col].min()), 4),
                 "max": round(float(runs[col].max()), 4)} for col in metric_cols},
    }


def model_slug(model_name: str) -> str:
    """'ProsusAI/finbert' -> 'finbert' (used in output file names)."""
    return model_name.rstrip("/").split("/")[-1].lower().replace(".", "-")


def _headline_scored_csv(p: Paths, slug: str) -> Path:
    return p.predictions / f"{p.ticker.lower()}_scored_headlines_{slug}.csv"


def stage_benchmark(p: Paths, args: argparse.Namespace) -> None:
    """Score off-the-shelf HF classifiers (default FinBERT) on the test split and the headlines."""
    from src.evaluate import calculate_metrics
    from src.inference import score_news_dataframe

    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    news = pd.read_csv(p.news_csv) if p.news_csv.exists() else None
    if news is not None:
        news["Date"] = pd.to_datetime(news["Date"], utc=True)
        news = news.sort_values("Date", kind="stable").reset_index(drop=True)

    for name in args.external_models:
        slug = model_slug(name)
        test_preds_csv = p.predictions / f"{slug}_test_predictions.csv"
        if test_preds_csv.exists() and not args.rescore_test:
            print(f"  {name}: test-split predictions already exist (use --rescore-test to redo)")
        else:
            preds = _evaluate_model(name, test_df)
            metrics = {
                "model": name,
                "evaluation_split": "held-out test set",
                "num_examples": int(len(preds)),
                **calculate_metrics(preds["label"].tolist(), preds["predicted_label"].tolist()),
                "caveat": ("Off-the-shelf checkpoint. If it was fine-tuned on Financial PhraseBank (FinBERT was), "
                           "it has likely seen these test sentences, so this score is optimistic. "
                           "Compare models on the hand-labelled headlines instead."),
            }
            _write_json(metrics, p.metrics / f"{slug}_test_metrics.json")
            _write_csv(preds, test_preds_csv)
            print(f"  {name}: test accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']}")
        if news is not None:
            _write_csv(score_news_dataframe(news, model_path=name), _headline_scored_csv(p, slug))

    # our own model variants on this ticker's headlines
    for slug, model_dir in [("weighted", p.weighted_model), ("adapted", p.adapted_model)]:
        if news is not None and (model_dir / "config.json").exists():
            _write_csv(score_news_dataframe(news, model_path=model_dir), _headline_scored_csv(p, slug))


TEST_SET_MODELS = {
    "Zero-shot BART-MNLI": "zero_shot_test_predictions.csv",
    "DistilBERT fine-tuned": "finetuned_test_predictions.csv",
    "DistilBERT class-weighted": "weighted_test_predictions.csv",
    "DistilBERT headline-adapted": "adapted_test_predictions.csv",
}
REFERENCE_MODEL = "DistilBERT fine-tuned"


def _probs(df: pd.DataFrame):
    return df[[f"{label}_probability" for label in config.LABELS]].to_numpy(dtype=float)


def _compare_models(y_true, predictions: Dict[str, pd.DataFrame], reference: str, n_boot: int) -> pd.DataFrame:
    from src import stats

    rows = []
    for name, df in predictions.items():
        y_pred = df["predicted_label"].tolist()
        acc = stats.bootstrap_ci(y_true, y_pred, stats.accuracy, n_boot=n_boot)
        f1 = stats.bootstrap_ci(y_true, y_pred, stats.macro_f1, n_boot=n_boot)
        row = {"model": name, "n": len(y_true),
               "accuracy": acc["value"], "accuracy_ci_low": acc["ci_low"], "accuracy_ci_high": acc["ci_high"],
               "macro_f1": f1["value"], "macro_f1_ci_low": f1["ci_low"], "macro_f1_ci_high": f1["ci_high"]}
        if all(f"{label}_probability" in df for label in config.LABELS):
            row["ece"] = stats.expected_calibration_error(_probs(df), stats.labels_to_ids(y_true))["ece"]
        if name != reference and reference in predictions:
            ref_pred = predictions[reference]["predicted_label"].tolist()
            diff = stats.paired_bootstrap_diff(y_true, ref_pred, y_pred, stats.macro_f1, n_boot=n_boot)
            row.update({"reference_minus_model_macro_f1": diff["diff"],
                        "diff_ci_low": diff["ci_low"], "diff_ci_high": diff["ci_high"],
                        "bootstrap_p_value": diff["p_value"],
                        "mcnemar_p_value": stats.mcnemar_test(y_true, ref_pred, y_pred)["p_value"]})
        rows.append(row)
    return pd.DataFrame(rows)


def _majority_baseline(y_true) -> Dict[str, pd.DataFrame]:
    majority = max(sorted(set(y_true)), key=list(y_true).count)
    return {f"Majority class ({majority})": pd.DataFrame({"predicted_label": [majority] * len(y_true)})}


def _headline_agreement(p: Paths, args: argparse.Namespace) -> List[dict]:
    """Label agreement / Cohen's kappa / score correlation with the reference model on unlabelled headlines."""
    from sklearn.metrics import cohen_kappa_score

    if not p.scored_csv.exists():
        return []
    ref = pd.read_csv(p.scored_csv)
    others = {"DistilBERT class-weighted": _headline_scored_csv(p, "weighted"),
              "DistilBERT headline-adapted": _headline_scored_csv(p, "adapted")}
    others.update({f"{model_slug(m)} (off-the-shelf)": _headline_scored_csv(p, model_slug(m))
                   for m in args.external_models})
    rows = []
    for name, path in others.items():
        if not path.exists():
            continue
        other = pd.read_csv(path)
        if other["Article_title"].tolist() != ref["Article_title"].tolist():
            raise ValueError(f"{path.name} rows do not match {p.scored_csv.name}")
        rows.append({
            "model": name,
            "n_headlines": len(ref),
            "label_agreement": round(float((ref["predicted_label"] == other["predicted_label"]).mean()), 4),
            "cohen_kappa": round(float(cohen_kappa_score(ref["predicted_label"], other["predicted_label"])), 4),
            "score_spearman": round(float(ref["sentiment_score"].corr(other["sentiment_score"], method="spearman")), 4),
            "reference_label_share": ref["predicted_label"].value_counts(normalize=True).round(4).to_dict(),
            "model_label_share": other["predicted_label"].value_counts(normalize=True).round(4).to_dict(),
        })
    return rows


def stage_analyze(p: Paths, args: argparse.Namespace) -> None:
    """Bootstrap CIs, paired significance tests and calibration for every saved test-set model."""
    from src import stats
    from src.visualization import plot_reliability_diagram

    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    y_true = test_df["label"].tolist()

    files = dict(TEST_SET_MODELS)
    files.update({f"{model_slug(m)} (off-the-shelf)": f"{model_slug(m)}_test_predictions.csv"
                  for m in args.external_models})
    predictions = {}
    for name, fname in files.items():
        path = p.predictions / fname
        if not path.exists():
            print(f"  skipping {name}: {path.name} not found")
            continue
        df = pd.read_csv(path)
        if df["sentence"].tolist() != test_df["sentence"].tolist():
            raise ValueError(f"{path} rows do not match test.csv")
        predictions[name] = df
    predictions.update(_majority_baseline(y_true))

    table = _compare_models(y_true, predictions, REFERENCE_MODEL, args.n_boot)
    _write_csv(table, p.metrics / "test_set_model_comparison.csv")
    report = {"n_test": len(y_true), "bootstrap_resamples": args.n_boot, "models": table.to_dict(orient="records")}

    val_path = p.predictions / "finetuned_validation_predictions.csv"
    if REFERENCE_MODEL in predictions and val_path.exists():
        val = pd.read_csv(val_path)
        t = stats.fit_temperature(_probs(val), stats.labels_to_ids(val["label"]))
        test_probs = _probs(predictions[REFERENCE_MODEL])
        scaled = stats.apply_temperature(test_probs, t)
        ids = stats.labels_to_ids(y_true)
        before = stats.expected_calibration_error(test_probs, ids)
        after = stats.expected_calibration_error(scaled, ids)
        report["calibration"] = {
            "model": REFERENCE_MODEL,
            "temperature_fit_on": "validation split",
            "temperature": round(t, 4),
            "test_ece_before": before["ece"], "test_ece_after": after["ece"],
            "test_nll_before": round(stats.negative_log_likelihood(test_probs, ids), 4),
            "test_nll_after": round(stats.negative_log_likelihood(scaled, ids), 4),
            "bins_before": before["bins"], "bins_after": after["bins"],
        }
        plot_reliability_diagram(
            {"Before (T = 1)": before["bins"], f"After temperature scaling (T = {t:.2f})": after["bins"]},
            title="DistilBERT reliability on the test split",
            save_path=p.figures / "finetuned_reliability_diagram.png",
        )
        print(f"  temperature={t:.3f}  test ECE {before['ece']} -> {after['ece']}")

    agreement = _headline_agreement(p, args)
    if agreement:
        from sklearn.metrics import cohen_kappa_score

        ref_test = predictions.get(REFERENCE_MODEL)
        for row in agreement:
            other_test = predictions.get(row["model"])
            if ref_test is not None and other_test is not None:
                row["test_label_agreement"] = round(
                    float((ref_test["predicted_label"] == other_test["predicted_label"]).mean()), 4)
                row["test_cohen_kappa"] = round(
                    float(cohen_kappa_score(ref_test["predicted_label"], other_test["predicted_label"])), 4)
            test_agree = row.get("test_label_agreement", float("nan"))
            print(f"  {REFERENCE_MODEL} vs {row['model']}: headlines agree {row['label_agreement']:.1%} "
                  f"(kappa {row['cohen_kappa']}); test sentences agree {test_agree:.1%} "
                  f"(kappa {row.get('test_cohen_kappa')})")
        report["headline_agreement_vs_reference"] = agreement

    seeds_json = p.metrics / "seed_summary.json"
    if seeds_json.exists():
        report["seed_variation"] = json.loads(seeds_json.read_text(encoding="utf-8"))

    _write_json(report, p.metrics / "test_set_analysis.json")
    cols = ["model", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "ece", "mcnemar_p_value"]
    print(table[[c for c in cols if c in table]].to_string(index=False))


def stage_sample_headlines(p: Paths, args: argparse.Namespace) -> None:
    """Draw a blind, year-stratified sample of headlines to hand-label (never overwrites labels)."""
    from src.labelling import gold_path, sample_for_annotation

    out = gold_path(p.ticker, p.data / "labelled")
    if out.exists():
        print(f"  {out} already exists - not overwriting your labels")
        return
    news = pd.read_csv(_require(p.news_csv, "fetch-news"))
    _write_csv(sample_for_annotation(news, n=args.n_label, seed=args.seed), out)
    print("  next: python -m src.labelling   (guidelines: docs/annotation_guidelines.md)")


def stage_evaluate_headlines(p: Paths, args: argparse.Namespace) -> None:
    """Score every model against the hand-labelled headlines (in-domain evaluation)."""
    from src.labelling import gold_path, labelled_rows, load_gold

    gold_all = load_gold(_require(gold_path(p.ticker, p.data / "labelled"), "sample-headlines"))
    gold = labelled_rows(gold_all)
    if gold.empty:
        raise SystemExit("No labelled headlines yet - run: python -m src.labelling")
    if len(gold) < 100:
        print(f"  warning: only {len(gold)} labelled headlines; confidence intervals will be wide")
    gold["Date"] = pd.to_datetime(gold["Date"], utc=True)

    sources = {"DistilBERT fine-tuned": p.scored_csv,
               "DistilBERT class-weighted": _headline_scored_csv(p, "weighted"),
               "DistilBERT headline-adapted": _headline_scored_csv(p, "adapted")}
    sources.update({f"{model_slug(m)} (off-the-shelf)": _headline_scored_csv(p, model_slug(m))
                    for m in args.external_models})

    predictions = {}
    for name, path in sources.items():
        if not path.exists():
            print(f"  skipping {name}: {path.name} not found (run 'score' / 'benchmark')")
            continue
        scored = pd.read_csv(path)
        scored["Date"] = pd.to_datetime(scored["Date"], utc=True)
        merged = gold.merge(scored, on=["Date", "Article_title"], how="left", validate="one_to_one")
        if merged["predicted_label"].isna().any():
            raise ValueError(f"{path.name} is missing some labelled headlines")
        predictions[name] = merged

    y_true = gold["label"].tolist()
    predictions.update(_majority_baseline(y_true))
    table = _compare_models(y_true, predictions, REFERENCE_MODEL, args.n_boot)

    report = {
        "ticker": p.ticker,
        "labelled": len(gold),
        "skipped": int((gold_all["label"] == "skip").sum()),
        "unlabelled": int((gold_all["label"] == "").sum()),
        "label_distribution": {k: int(v) for k, v in gold["label"].value_counts().items()},
        "models": table.to_dict(orient="records"),
    }
    market_wide = gold["Article_title"].str.contains(config.MARKET_WIDE_HEADLINE_PATTERN, case=False, regex=True)
    if market_wide.any() and (~market_wide).sum() >= 30:
        keep = (~market_wide).to_numpy()
        subset = {name: df.iloc[keep] if len(df) == len(gold) else df for name, df in predictions.items()}
        y_sub = [y for y, k in zip(y_true, keep) if k]
        subset = {n: d for n, d in subset.items() if not n.startswith("Majority")}
        subset.update(_majority_baseline(y_sub))
        sens = _compare_models(y_sub, subset, REFERENCE_MODEL, args.n_boot)
        report["sensitivity_company_specific_only"] = {
            "excluded_market_wide_headlines": int(market_wide.sum()),
            "pattern": config.MARKET_WIDE_HEADLINE_PATTERN,
            "models": sens.to_dict(orient="records"),
        }
        print(f"  company-specific only (n={len(y_sub)}, {int(market_wide.sum())} market-wide excluded):")
        print(sens[["model", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high"]].to_string(index=False))
    if "annotator" in gold_all:
        report["annotators"] = sorted(set(gold_all["annotator"]) - {""})
    if "note" in gold_all:
        notes = gold_all.loc[gold_all["label"] == "skip", "note"].replace("", "unspecified")
        report["skip_reasons"] = {k: int(v) for k, v in notes.value_counts().items()}
    test_metrics = p.metrics / "finetuned_test_metrics.json"
    if test_metrics.exists() and REFERENCE_MODEL in predictions:
        phrasebank_f1 = json.loads(test_metrics.read_text(encoding="utf-8"))["macro_f1"]
        headline_f1 = float(table.loc[table["model"] == REFERENCE_MODEL, "macro_f1"].iloc[0])
        report["domain_shift"] = {"model": REFERENCE_MODEL, "phrasebank_test_macro_f1": phrasebank_f1,
                                  "headline_macro_f1": headline_f1, "drop": round(phrasebank_f1 - headline_f1, 4)}

    agreement = _headline_agreement(p, args)  # all of this ticker's headlines, labelled or not
    if agreement:
        report["headline_agreement_vs_reference"] = agreement

    slug = p.ticker.lower()
    _write_csv(table, p.metrics / f"{slug}_headline_model_comparison.csv")
    _write_json(report, p.metrics / f"{slug}_headline_evaluation.json")
    cols = ["model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "mcnemar_p_value"]
    print(table[[c for c in cols if c in table]].to_string(index=False))


def stage_adapt(p: Paths, args: argparse.Namespace) -> None:
    """Continue fine-tuning DistilBERT on annotated headlines from other companies (+ PhraseBank replay)."""
    from sklearn.model_selection import train_test_split

    from src.evaluate import calculate_metrics
    from src.inference import score_news_dataframe
    from src.labelling import labelled_rows, load_gold
    from src.train import train_distilbert_classifier

    labels_csv = p.data / "labelled" / "headline_train_labels.csv"
    heads = labelled_rows(load_gold(_require(labels_csv, "label headline_train_labels.csv"))).copy()
    if (heads["Stock_symbol"].str.upper() == p.ticker.upper()).any():
        raise ValueError(f"training headlines include the evaluation ticker {p.ticker}")
    heads = heads.rename(columns={"Article_title": "sentence"})[["sentence", "label"]]
    train_h, val_h = train_test_split(heads, test_size=0.2, stratify=heads["label"], random_state=args.seed)

    phrasebank = pd.read_csv(_require(p.processed / "train.csv", "prepare"))
    replay = phrasebank.sample(n=min(len(train_h), len(phrasebank)), random_state=args.seed)[["sentence", "label"]]
    train_df = pd.concat([train_h, replay], ignore_index=True).sample(frac=1.0, random_state=args.seed)
    print(f"  training on {len(train_h)} headlines + {len(replay)} PhraseBank sentences; "
          f"validating on {len(val_h)} headlines")

    metrics = train_distilbert_classifier(
        train_df, val_h.reset_index(drop=True), output_dir=p.adapted_model,
        base_model_name=str(_require(p.model, "train")), num_epochs=args.adapt_epochs,
        seed=args.seed, keep_checkpoints=False, max_length=64,
    )
    metrics.update({"model": f"{config.BASE_MODEL_NAME} (PhraseBank fine-tuned, then headline-adapted)",
                    "base_model": "DistilBERT fine-tuned on PhraseBank (continued fine-tuning)",
                    "train_headlines": len(train_h), "replay_phrasebank_sentences": len(replay),
                    "validation": "held-out 20% of the training headlines (other companies)",
                    "label_source": "LLM-annotated (see data/labelled/headline_train_labels.csv)"})
    _write_json(metrics, p.metrics / "adapted_validation_metrics.json")

    test_df = pd.read_csv(p.processed / "test.csv")
    preds = _evaluate_model(p.adapted_model, test_df)
    _write_csv(preds, p.predictions / "adapted_test_predictions.csv")
    test_m = calculate_metrics(preds["label"].tolist(), preds["predicted_label"].tolist())
    _write_json({"model": "DistilBERT headline-adapted", "evaluation_split": "held-out test set",
                 "num_examples": len(preds), **test_m}, p.metrics / "adapted_test_metrics.json")
    print(f"  PhraseBank test after adaptation: accuracy={test_m['accuracy']} macro_f1={test_m['macro_f1']}")

    if p.news_csv.exists():
        news = pd.read_csv(p.news_csv)
        news["Date"] = pd.to_datetime(news["Date"], utc=True)
        news = news.sort_values("Date", kind="stable").reset_index(drop=True)
        _write_csv(score_news_dataframe(news, model_path=p.adapted_model), _headline_scored_csv(p, "adapted"))


SCORING_MODELS = {"base": "model", "adapted": "adapted_model"}  # slug -> Paths attribute


def all_scored_csv(p: Paths, slug: str) -> Path:
    return p.predictions / f"nasdaq100_scored_headlines_{slug}.csv"


def stage_score_all(p: Paths, args: argparse.Namespace) -> None:
    """Score every relevant headline for every ticker with each model (resumable per-ticker cache)."""
    from src.inference import load_fine_tuned_model, predict_sentiment_batch
    from src.relevance import add_relevance

    news = pd.read_csv(_require(p.processed / "fnspid_all_2019_2023.csv", "fetch-news --ticker ALL"))
    news = add_relevance(news)
    news = news[news["relevant"]].reset_index(drop=True)
    print(f"  {len(news)} relevant headlines across {news['Stock_symbol'].nunique()} tickers")

    for slug, attr in SCORING_MODELS.items():
        model_dir = getattr(p, attr)
        if not (model_dir / "model.safetensors").exists():
            print(f"  skipping {slug}: {model_dir.name} not trained")
            continue
        cache = p.data / "cache" / "scores" / slug
        cache.mkdir(parents=True, exist_ok=True)
        model, tokenizer, device = load_fine_tuned_model(model_dir)
        for ticker, group in news.groupby("Stock_symbol"):
            out = cache / f"{ticker}.csv"
            if out.exists():
                continue
            labels, probs, confs, scores = predict_sentiment_batch(
                group["Article_title"].astype(str).tolist(), model, tokenizer, device, max_length=64)
            scored = group[["Date", "Stock_symbol", "Article_title"]].copy()
            scored["predicted_label"] = labels
            scored["sentiment_score"] = scores
            scored["confidence"] = confs
            scored.to_csv(out, index=False)
        combined = pd.concat([pd.read_csv(f) for f in sorted(cache.glob("*.csv"))], ignore_index=True)
        _write_csv(combined, all_scored_csv(p, slug))


def stage_prices(p: Paths, args: argparse.Namespace) -> None:
    """Download adjusted daily prices (Yahoo Finance) for every ticker in the news file, plus QQQ."""
    from src.market import BENCHMARK, download_prices

    news = pd.read_csv(_require(p.processed / "fnspid_all_2019_2023.csv", "fetch-news --ticker ALL"),
                       usecols=["Stock_symbol"])
    tickers = sorted(news["Stock_symbol"].unique())
    prices = download_prices([*tickers, BENCHMARK], "2018-11-01", "2024-02-15")
    _write_csv(prices, p.data / "prices" / "daily.csv")
    missing = sorted(set(tickers) - set(prices["ticker"]))
    print(f"  prices for {prices['ticker'].nunique() - 1} of {len(tickers)} tickers; missing: {missing}")


def stage_event_study(p: Paths, args: argparse.Namespace) -> None:
    """Headline sentiment vs abnormal returns and volume, all tickers (date-clustered bootstrap)."""
    from src.event_study import (
        bucket_returns,
        build_event_panel,
        daily_sentiment,
        information_coefficients,
        mean_car_paths,
    )
    from src.market import BENCHMARK, add_returns, load_prices
    from src.visualization import plot_car_paths

    returns = add_returns(load_prices(_require(p.data / "prices" / "daily.csv", "prices")))
    ic_tables, bucket_tables, runs = [], [], []
    for slug in SCORING_MODELS:
        path = all_scored_csv(p, slug)
        if not path.exists():
            print(f"  skipping {slug}: {path.name} not found (run 'score-all')")
            continue
        scored = pd.read_csv(path)
        for variant, exclude in [("all relevant headlines", False), ("excluding price-move headlines", True)]:
            panel = build_event_panel(daily_sentiment(scored, exclude_price_moves=exclude), returns)
            ic = information_coefficients(panel, n_boot=args.n_event_boot, seed=args.seed)
            buckets = bucket_returns(panel, threshold=args.event_threshold, n_boot=args.n_event_boot, seed=args.seed)
            for table in (ic, buckets):
                table.insert(0, "model", slug)
                table.insert(1, "headlines", variant)
            ic_tables.append(ic)
            bucket_tables.append(buckets)
            runs.append({"model": slug, "headlines": variant, "events": len(panel),
                         "tickers": int(panel["ticker"].nunique()),
                         "first_event": str(panel["event_date"].min().date()),
                         "last_event": str(panel["event_date"].max().date())})
            print(f"  [{slug} | {variant}] {len(panel)} ticker-days")
            print(ic[["outcome", "n_events", "value", "ci_low", "ci_high", "p_value"]].to_string(index=False))
            if exclude:
                plot_car_paths(mean_car_paths(panel, threshold=args.event_threshold),
                               title=f"Cumulative abnormal return around headline days ({slug} model, "
                                     f"price-move headlines excluded)",
                               save_path=p.figures / f"event_study_car_{slug}.png")

    if not runs:
        raise SystemExit("No scored headlines - run: python -m src.pipeline score-all")
    _write_csv(pd.concat(ic_tables, ignore_index=True), p.metrics / "event_study_information_coefficients.csv")
    _write_csv(pd.concat(bucket_tables, ignore_index=True), p.metrics / "event_study_buckets.csv")
    _write_json({
        "design": {
            "unit": "ticker x trading day with >=1 relevant headline; sentiment = mean P(pos)-P(neg)",
            "abnormal_return": f"stock return minus {BENCHMARK} return (market-adjusted model)",
            "abnormal_volume": "log volume minus mean log volume over the previous 20 trading days",
            "timing": "FNSPID dates have no times; weekend/holiday news is assigned to the next trading day",
            "inference": f"bootstrap clustered by event date, {args.n_event_boot} resamples",
            "bucket_threshold": args.event_threshold,
            "not_advice": "Association study on historical data; not a trading signal or investment advice.",
        },
        "runs": runs,
    }, p.metrics / "event_study_summary.json")


MULTI_EVAL_TICKERS = ["TSLA", "AAPL", "NVDA", "AMZN", "GILD", "SBUX"]


def _multi_eval_models(p: Paths, args: argparse.Namespace) -> Dict[str, str]:
    models = {"DistilBERT fine-tuned": str(p.model), "DistilBERT class-weighted": str(p.weighted_model),
              "DistilBERT headline-adapted": str(p.adapted_model)}
    models.update({f"{model_slug(m)} (off-the-shelf)": m for m in args.external_models})
    return {name: src for name, src in models.items() if not Path(src).is_absolute() or Path(src).exists()}


def stage_evaluate_multi(p: Paths, args: argparse.Namespace) -> None:
    """Accuracy on annotated headlines for several companies, plus per-company relevance-filter precision/recall."""
    import gc

    from src.inference import load_fine_tuned_model, predict_sentiment_batch
    from src.labelling import gold_path, load_gold
    from src.relevance import is_relevant

    gold_frames = []
    for ticker in MULTI_EVAL_TICKERS:
        path = gold_path(ticker, p.data / "labelled")
        if not path.exists():
            print(f"  skipping {ticker}: {path.name} not found")
            continue
        g = load_gold(path)
        g["ticker"] = ticker
        g["design"] = "random" if ticker == "TSLA" else "stratified by relevance filter"
        gold_frames.append(g)
    gold = pd.concat(gold_frames, ignore_index=True)
    gold = gold[gold["label"] != ""]
    gold["passes_filter"] = [is_relevant(t, k) for t, k in zip(gold["Article_title"], gold["ticker"])]
    gold["on_topic"] = gold["label"].isin(config.LABELS)
    gold["unclear"] = (gold["label"] == "skip") & (gold.get("note", "") == "unclear")

    # ---- relevance filter, per company (strata re-weighted by their true sizes)
    news = pd.read_csv(_require(p.processed / "fnspid_all_2019_2023.csv", "fetch-news --ticker ALL"),
                       usecols=["Stock_symbol", "Article_title"]).drop_duplicates()
    relevance_rows = []
    for ticker, g in gold[~gold["unclear"]].groupby("ticker", sort=False):
        pool = news.loc[news["Stock_symbol"] == ticker, "Article_title"]
        n_pass_pool = sum(is_relevant(t, ticker) for t in pool)
        n_fail_pool = len(pool) - n_pass_pool
        kept, dropped = g[g["passes_filter"]], g[~g["passes_filter"]]
        precision = kept["on_topic"].mean() if len(kept) else float("nan")
        miss_rate = dropped["on_topic"].mean() if len(dropped) else float("nan")
        found, missed = n_pass_pool * precision, n_fail_pool * miss_rate
        relevance_rows.append({
            "ticker": ticker, "design": g["design"].iloc[0], "labelled": len(g),
            "pool_headlines": len(pool), "pool_pass_share": round(n_pass_pool / len(pool), 4),
            "precision": round(float(precision), 4),
            "on_topic_rate_among_dropped": round(float(miss_rate), 4),
            "recall_estimate": round(float(found / (found + missed)), 4) if found + missed else float("nan"),
            "on_topic_share_estimate": round(float((found + missed) / len(pool)), 4),
        })
    relevance = pd.DataFrame(relevance_rows)
    _write_csv(relevance, p.metrics / "multi_company_relevance_filter.csv")
    print(relevance[["ticker", "precision", "recall_estimate", "on_topic_share_estimate"]].to_string(index=False))

    # ---- score every labelled on-topic headline with each model (one model in memory at a time)
    scored = gold[gold["on_topic"]].copy()
    titles = scored["Article_title"].tolist()
    for name, source in _multi_eval_models(p, args).items():
        model, tokenizer, device = load_fine_tuned_model(source)
        labels, _, _, scores = predict_sentiment_batch(titles, model, tokenizer, device, max_length=64)
        scored[f"pred::{name}"] = labels
        del model, tokenizer
        gc.collect()
        print(f"  scored {len(titles)} headlines with {name}")
    keep_cols = ["ticker", "item_id", "Date", "Article_title", "label", "passes_filter"]
    _write_csv(scored[keep_cols + [c for c in scored.columns if c.startswith("pred::")]],
               p.predictions / "multi_company_headline_predictions.csv")

    # ---- accuracy: deployed population (passes filter) pooled and per company
    model_names = [c.split("::", 1)[1] for c in scored.columns if c.startswith("pred::")]

    def compare(frame):
        preds = {n: pd.DataFrame({"predicted_label": frame[f"pred::{n}"].tolist()}) for n in model_names}
        y = frame["label"].tolist()
        preds.update(_majority_baseline(y))
        return _compare_models(y, preds, REFERENCE_MODEL, args.n_boot)

    deployed = scored[scored["passes_filter"]]
    new_companies = deployed[deployed["ticker"] != "TSLA"]
    tables = {
        "pooled_5_new_companies": compare(new_companies),
        "pooled_all_6_companies": compare(deployed),
    }
    per_company = []
    for ticker, g in deployed.groupby("ticker", sort=False):
        t = compare(g)
        t.insert(0, "ticker", ticker)
        per_company.append(t)
    per_company = pd.concat(per_company, ignore_index=True)

    for key, table in tables.items():
        table.insert(0, "population", key)
    pooled = pd.concat(tables.values(), ignore_index=True)
    _write_csv(pooled, p.metrics / "multi_company_pooled_accuracy.csv")
    _write_csv(per_company, p.metrics / "multi_company_per_company_accuracy.csv")

    phrasebank = json.loads((p.metrics / "finetuned_test_metrics.json").read_text(encoding="utf-8"))["macro_f1"]
    _write_json({
        "companies": MULTI_EVAL_TICKERS,
        "population": "annotated on-topic headlines that pass the relevance filter (what the system scores)",
        "labelled_on_topic_passing_filter": {k: int(v) for k, v in deployed["ticker"].value_counts().items()},
        "phrasebank_test_macro_f1": phrasebank,
        "annotators": sorted(set(gold.get("annotator", pd.Series(dtype=str))) - {""}),
        "notes": [
            "Labels are LLM-annotated (Claude) following docs/annotation_guidelines.md; not human-verified.",
            "TSLA was sampled at random; the other companies 80 filter-kept + 30 filter-dropped headlines each.",
            "The headline-adapted model saw ~5 other headlines per non-TSLA company in training (none of these).",
        ],
        "pooled": pooled.to_dict(orient="records"),
    }, p.metrics / "multi_company_evaluation.json")

    cols = ["population", "model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
            "mcnemar_p_value"]
    print(pooled[cols].to_string(index=False))
    print(per_company[["ticker", "model", "n", "accuracy", "macro_f1"]].to_string(index=False))


def stage_onnx_export(p: Paths, args: argparse.Namespace) -> None:
    """Export the headline-adapted model to ONNX (fp32) and quantize it to int8."""
    from src.onnx_backend import OnnxSentimentModel, export_onnx, model_size_mb, quantize_onnx, write_reference

    _require(p.adapted_model / "config.json", "adapt")
    export_onnx(p.adapted_model, p.adapted_onnx)
    quantize_onnx(p.adapted_onnx, p.adapted_onnx_int8)
    # fp32 labels for a fixed headline set, so every machine can check its int8 kernels (self_check)
    heads = pd.read_csv(_require(p.predictions / "multi_company_headline_predictions.csv", "evaluate-multi"))
    write_reference(p.adapted_onnx, heads["Article_title"].head(100).tolist(), [p.adapted_onnx, p.adapted_onnx_int8])
    agreement = OnnxSentimentModel(p.adapted_onnx_int8).self_check()
    for path in (p.adapted_model, p.adapted_onnx, p.adapted_onnx_int8):
        print(f"  {path.name}: {model_size_mb(path)} MB")
    print(f"  int8 self-check on this CPU: {agreement:.0%} label agreement with fp32 on 100 reference headlines")


def _cpu_name() -> str:
    import platform

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def stage_onnx_benchmark(p: Paths, args: argparse.Namespace) -> None:
    """PyTorch vs ONNX fp32 vs ONNX int8: latency, memory, size, and accuracy on PhraseBank + 6-company headlines."""
    import gc
    import os
    import subprocess
    import tempfile

    import numpy as np
    from sklearn.metrics import cohen_kappa_score

    from src.inference import load_fine_tuned_model, predict_sentiment_batch
    from src.onnx_backend import OnnxSentimentModel, predict_sentiment_batch_onnx

    variants = {"PyTorch fp32": ("torch", p.adapted_model), "ONNX fp32": ("onnx", p.adapted_onnx),
                "ONNX int8 (dynamic)": ("onnx", p.adapted_onnx_int8)}
    for _, path in variants.values():
        _require(path / "config.json", "onnx-export")
    reference = "PyTorch fp32"

    # ---- latency + memory: each variant in a fresh process so peak RSS includes only what it imports
    pool = pd.read_csv(_require(p.predictions / "nasdaq100_scored_headlines_adapted.csv", "score-all"),
                       usecols=["Article_title"])["Article_title"].dropna().drop_duplicates()
    texts = pool.sample(n=min(args.bench_headlines, len(pool)), random_state=args.seed).tolist()
    latency = {}
    with tempfile.TemporaryDirectory() as tmp:
        texts_file = Path(tmp) / "headlines.txt"
        texts_file.write_text("\n".join(t.replace("\n", " ") for t in texts), encoding="utf-8")
        for name, (backend, path) in variants.items():
            cmd = [sys.executable, "-m", "src.onnx_backend", "bench", "--backend", backend, "--model", str(path),
                   "--texts", str(texts_file), "--threads", str(args.threads)]
            # cap every thread pool (BLAS/OpenMP and the Rust tokenizer) at the same count
            env = {**os.environ, "OMP_NUM_THREADS": str(args.threads), "RAYON_NUM_THREADS": str(args.threads)}
            out = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env,
                                 cwd=str(config.PROJECT_ROOT))
            latency[name] = json.loads(out.stdout.strip().splitlines()[-1])
            lat = latency[name]["latency"]
            print(f"  {name:<20} batch1 {lat['batch_1']['ms_per_headline_median']:.2f} ms  "
                  f"batch32 {lat['batch_32']['ms_per_headline_median']:.2f} ms  "
                  f"peak RSS {latency[name]['peak_rss_mb']:.0f} MB  size {latency[name]['model_size_mb']} MB")

    # ---- accuracy: same inputs, same max_length as the committed evaluations
    test_df = pd.read_csv(_require(p.processed / "test.csv", "prepare"))
    heads = pd.read_csv(_require(p.predictions / "multi_company_headline_predictions.csv", "evaluate-multi"))
    heads = heads[heads["passes_filter"]]
    sets = {"phrasebank_test": (test_df["sentence"].tolist(), test_df["label"].tolist(), config.MAX_SEQUENCE_LENGTH),
            "headlines_6_companies": (heads["Article_title"].tolist(), heads["label"].tolist(), 64)}
    preds: Dict[str, Dict[str, tuple]] = {k: {} for k in sets}
    for name, (backend, path) in variants.items():
        if backend == "torch":
            model, tokenizer, device = load_fine_tuned_model(path)
            for key, (x, _, max_len) in sets.items():
                labels, probs, _, _ = predict_sentiment_batch(x, model, tokenizer, device, max_length=max_len)
                preds[key][name] = (labels, probs)
            del model, tokenizer
        else:
            model = OnnxSentimentModel(path)
            for key, (x, _, max_len) in sets.items():
                labels, probs, _, _ = predict_sentiment_batch_onnx(x, model, max_length=max_len)
                preds[key][name] = (labels, probs)
            del model
        gc.collect()

    accuracy_rows, agreement = [], {}
    for key, (_, y_true, _) in sets.items():
        frames = {n: pd.DataFrame({"predicted_label": lab, **{f"{c}_probability": pr[:, i]
                                   for i, c in enumerate(config.LABELS)}}) for n, (lab, pr) in preds[key].items()}
        table = _compare_models(y_true, frames, reference, args.n_boot)
        table.insert(0, "evaluation_set", key)
        accuracy_rows.append(table)
        ref_labels, ref_probs = preds[key][reference]
        agreement[key] = {
            n: {"label_agreement_with_pytorch": round(float(np.mean(np.array(lab) == np.array(ref_labels))), 4),
                "max_abs_probability_diff": round(float(np.abs(pr - ref_probs).max()), 6),
                "cohen_kappa_with_pytorch": round(float(cohen_kappa_score(ref_labels, lab)), 4)}
            for n, (lab, pr) in preds[key].items() if n != reference}
    accuracy = pd.concat(accuracy_rows, ignore_index=True)
    _write_csv(accuracy, p.metrics / "onnx_benchmark_accuracy.csv")

    ref_lat = latency[reference]["latency"]
    speedups = {n: {b: round(ref_lat[b]["ms_per_headline_median"] / v["latency"][b]["ms_per_headline_median"], 2)
                    for b in v["latency"]} for n, v in latency.items() if n != reference}
    _write_json({
        "model": p.adapted_model.name,
        "hardware": {"cpu": _cpu_name(), "threads": args.threads},
        "notes": [
            "Latency is end to end (tokenize + forward + softmax), median of 3 runs, after a warm-up.",
            "Each variant ran in a fresh process; peak RSS includes the Python libraries it imports "
            "(the ONNX path never imports torch).",
            "int8 = onnxruntime dynamic quantization (int8 weights, per-tensor scales; activations quantized at "
            "run time).",
            "Accuracy uses max_length 256 on PhraseBank and 64 on headlines, as in the committed evaluations; "
            "McNemar / paired bootstrap compare each variant with the PyTorch model.",
        ],
        "latency": latency,
        "speedup_vs_pytorch": speedups,
        "agreement_with_pytorch": agreement,
        "accuracy": accuracy.to_dict(orient="records"),
    }, p.metrics / "onnx_benchmark.json")
    print(accuracy[["evaluation_set", "model", "n", "accuracy", "macro_f1", "mcnemar_p_value"]].to_string(index=False))
    print(json.dumps(agreement, indent=1))


def _live_service(p: Paths, args: argparse.Namespace):
    """Sentiment service for the live feed: prefers the int8 ONNX model (no torch, small RAM) when it exists."""
    import os

    from src.onnx_backend import is_onnx_dir
    from src.service import SentimentService

    backend = args.backend or os.environ.get("SENTIMENT_BACKEND") or (
        "onnx" if is_onnx_dir(p.adapted_onnx_int8) else "auto")
    source = None
    if not os.environ.get("SENTIMENT_MODEL"):
        if backend in ("onnx", "auto") and is_onnx_dir(p.adapted_onnx_int8):
            source = str(p.adapted_onnx_int8)
        elif backend != "onnx" and (p.adapted_model / "config.json").exists():
            source = str(p.adapted_model)
    return SentimentService(model_source=source, backend=backend)


def stage_live_update(p: Paths, args: argparse.Namespace) -> None:
    """Fetch new headlines (Google News + Yahoo by default), relevance-filter, score, store in data/live/news.db."""
    from src.live_feed import LIVE_TICKERS, LiveStore, fetch_many
    from src.relevance import is_relevant

    started = pd.Timestamp.now(tz="UTC")
    tickers = [t.upper() for t in args.live_tickers] if args.live_tickers else LIVE_TICKERS
    errors: List[str] = []
    print(f"  fetching {len(tickers)} tickers from {', '.join(args.live_sources)} (last {args.live_days} days)")
    news = fetch_many(tickers, days=args.live_days, sources=args.live_sources, errors=errors)
    relevant = [is_relevant(t, k) for t, k in zip(news["Article_title"], news["Stock_symbol"])]
    store = LiveStore(p.live_db)
    inserted = store.insert_new(news, relevant)

    todo = store.unscored()
    if len(todo):
        service = _live_service(p, args)
        for start in range(0, len(todo), 256):
            chunk = todo.iloc[start:start + 256]
            preds = service.predict(chunk["title"].tolist())
            store.save_scores(chunk["id"].tolist(), [x.label for x in preds],
                              [[x.probabilities[c] for c in config.LABELS] for x in preds],
                              [x.confidence for x in preds], [x.sentiment_score for x in preds],
                              model=f"{service.model_name} ({service.backend})")
        print(f"  scored {len(todo)} new relevant headlines with {service.model_name} ({service.backend})")
    store.log_run(started, len(tickers), args.live_sources, len(news), inserted, len(todo), errors)
    print(f"  fetched {len(news)} headlines, {sum(relevant)} about their company; {inserted} new rows in {p.live_db}")
    if errors:
        print(f"  {len(errors)} source errors (logged in the runs table), e.g. {errors[0]}")


def stage_alerts(p: Paths, args: argparse.Namespace) -> None:
    """Flag tickers whose 7-day live sentiment or headline volume is unusual vs their 90-day baseline."""
    import os

    from src.alerts import compute_alerts, post_alerts
    from src.live_feed import LiveStore

    store = LiveStore(_require(p.live_db, "live-update"))
    scored, runs = store.scored_frame(), store.runs()
    feed_start = pd.to_datetime(runs["started_at"], utc=True).min() if len(runs) else None
    table = compute_alerts(scored, feed_start=feed_start)
    _write_csv(table, p.live_db.parent / "alerts_latest.csv")
    fired = table[table["alerts"] != ""]
    now = pd.Timestamp.now(tz="UTC")
    if feed_start is not None and (now - feed_start).days < 30:
        print(f"  the feed has {(now - feed_start).days} days of history; alerts start after 30 days of daily runs")
    log = p.live_db.parent / "alerts_log.csv"
    if len(fired):
        stamped = fired.assign(raised_at=now.isoformat())
        stamped.to_csv(log, mode="a", header=not log.exists(), index=False)
        for msg in fired["message"]:
            print(f"  ALERT {msg}")
        url = os.environ.get("ALERT_WEBHOOK_URL", "").strip()
        if url:
            print(f"  posted {post_alerts(fired['message'].tolist(), url)} alerts to the webhook")
    else:
        print(f"  no alerts across {len(table)} tickers")


def stage_live_annotate(p: Paths, args: argparse.Namespace) -> None:
    """Gemini labels a blind sample of live headlines; reports model-vs-Gemini agreement (needs GEMINI_API_KEY)."""
    from src import stats
    from src.gemini import ANNOTATOR_PREFIX, MODEL, GeminiClient, annotate_headlines
    from src.live_feed import LiveStore, company_name

    store = LiveStore(_require(p.live_db, "live-update"))
    client = GeminiClient()
    annotator = f"{ANNOTATOR_PREFIX}:{MODEL}"  # the one fixed model; labels from any other model are ignored
    sample = store.unannotated(annotator, args.annotate_n, seed=args.seed)
    if len(sample):
        guidelines = (config.PROJECT_ROOT / "docs" / "annotation_guidelines.md").read_text(encoding="utf-8")
        items = [{"id": int(r.id), "ticker": r.ticker, "company": company_name(r.ticker), "title": r.title}
                 for r in sample.itertuples(index=False)]
        labels = annotate_headlines(client, items, guidelines)
        store.save_annotations(labels, annotator)
        print(f"  {annotator} labelled {len(labels)} of {len(items)} sampled headlines")

    ann = store.annotations()
    ann = ann[ann["annotator"] == annotator]
    on_topic = ann[ann["label"].isin(config.LABELS)]
    summary = {"annotator": annotator, "labelled": int(len(ann)),
               "skipped_by_annotator": int((ann["label"] == "skip").sum()),
               "notes": ["Gemini is a second LLM annotator, not ground truth. Agreement measures consistency with "
                         "an independent labeller on live headlines; a drop over time suggests drift.",
                         "Sampling is blind: Gemini never sees the model's label."]}
    if len(on_topic) >= 5:
        y_true, y_pred = on_topic["label"].tolist(), on_topic["predicted_label"].tolist()
        summary.update({"n_on_topic": int(len(on_topic)),
                        "accuracy": stats.bootstrap_ci(y_true, y_pred, stats.accuracy, n_boot=args.n_boot),
                        "macro_f1": stats.bootstrap_ci(y_true, y_pred, stats.macro_f1, n_boot=args.n_boot),
                        "label_counts_annotator": on_topic["label"].value_counts().to_dict(),
                        "label_counts_model": on_topic["predicted_label"].value_counts().to_dict()})
        print(f"  model vs {annotator} on {len(on_topic)} live headlines: "
              f"accuracy {summary['accuracy']['value']:.3f}, macro F1 {summary['macro_f1']['value']:.3f}")
    _write_json(summary, p.live_db.parent / "live_agreement.json")


def stage_fetch_news(p: Paths, args: argparse.Namespace) -> None:
    """Stream one ticker's 2019-2023 headlines from FNSPID (or slice the all-ticker file if present)."""
    from src.data_loading import ALL_TICKERS, stream_fnspid_slice

    all_csv = p.processed / "fnspid_all_2019_2023.csv"
    if args.ticker.upper() not in ALL_TICKERS and all_csv.exists():
        news = pd.read_csv(all_csv)
        news = news[news["Stock_symbol"].str.upper() == args.ticker.upper()].reset_index(drop=True)
        print(f"  sliced {len(news)} {args.ticker.upper()} headlines from {all_csv.name} (no re-download)")
    else:
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
    "train-seeds": stage_train_seeds,
    "benchmark": stage_benchmark,
    "adapt": stage_adapt,
    "score-all": stage_score_all,
    "prices": stage_prices,
    "event-study": stage_event_study,
    "analyze": stage_analyze,
    "fetch-news": stage_fetch_news,
    "score": stage_score,
    "monitor": stage_monitor,
    "sample-headlines": stage_sample_headlines,
    "evaluate-headlines": stage_evaluate_headlines,
    "evaluate-multi": stage_evaluate_multi,
    "onnx-export": stage_onnx_export,
    "onnx-benchmark": stage_onnx_benchmark,
    "live-update": stage_live_update,
    "alerts": stage_alerts,
    "live-annotate": stage_live_annotate,
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
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46],
                        help="seeds for train-seeds (42 reuses the reported model)")
    parser.add_argument("--external-models", nargs="+", default=["ProsusAI/finbert"],
                        help="off-the-shelf HF classifiers for 'benchmark'")
    parser.add_argument("--adapt-epochs", type=int, default=3, help="epochs for 'adapt'")
    parser.add_argument("--rescore-test", action="store_true",
                        help="benchmark: re-score the PhraseBank test split even if predictions exist")
    parser.add_argument("--n-event-boot", type=int, default=500, help="bootstrap resamples for event-study")
    parser.add_argument("--event-threshold", type=float, default=0.3,
                        help="|daily sentiment| above which a day counts as positive/negative")
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for CIs")
    parser.add_argument("--n-label", type=int, default=300, help="headlines to sample for labelling")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="subsample the training set (smoke tests only)")
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for 'onnx-benchmark'")
    parser.add_argument("--bench-headlines", type=int, default=1000, help="headlines timed by 'onnx-benchmark'")
    parser.add_argument("--live-tickers", nargs="+", default=None,
                        help="tickers for 'live-update' (default: all 89 in src/company_names.py)")
    parser.add_argument("--live-days", type=int, default=2, help="look-back window of each 'live-update' fetch")
    parser.add_argument("--live-sources", nargs="+", default=["google", "yahoo"],
                        choices=["google", "yahoo", "finnhub", "gemini"], help="headline sources for 'live-update'")
    parser.add_argument("--backend", choices=["auto", "torch", "onnx"], default=None,
                        help="scoring backend for 'live-update' (default: SENTIMENT_BACKEND, else onnx if exported)")
    parser.add_argument("--annotate-n", type=int, default=25, help="live headlines Gemini labels per 'live-annotate'")
    parser.add_argument("--data-dir", type=Path, default=config.DATA_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=config.OUTPUTS_DIR)
    parser.add_argument("--models-dir", type=Path, default=config.MODELS_DIR)
    return parser


# 'all' = the automatic stages. train-seeds (hours on CPU) and the labelling stages are run explicitly.
ALL_STAGES = ["prepare", "baseline", "train", "evaluate", "fetch-news", "score", "monitor", "benchmark", "analyze"]


def resolve_stages(requested: List[str]) -> List[str]:
    if requested == ["all"]:
        return list(ALL_STAGES)
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

    from src.env import load_env

    load_env()  # API keys and SENTIMENT_BACKEND from a git-ignored .env, if present
    matplotlib.use("Agg")  # stages save figures to disk; never open windows

    paths = Paths(data=args.data_dir, outputs=args.outputs_dir, models=args.models_dir, ticker=args.ticker)
    paths.ensure()
    for name in resolve_stages(args.stages):
        print(f"[{name}]")
        STAGES[name](paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
