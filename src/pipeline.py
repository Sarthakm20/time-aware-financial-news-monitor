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

    runs = pd.DataFrame(rows)
    metric_cols = [c for c in runs.columns if c not in ("seed", "source")]
    summary = {
        "model": f"{config.BASE_MODEL_NAME} fine-tuned on Financial PhraseBank",
        "evaluation_split": "held-out test set",
        "epochs": args.epochs,
        "seeds": [int(s) for s in runs["seed"]],
        **{col: {"mean": round(float(runs[col].mean()), 4),
                 "std": round(float(runs[col].std(ddof=1)), 4) if len(runs) > 1 else 0.0,
                 "min": round(float(runs[col].min()), 4),
                 "max": round(float(runs[col].max()), 4)} for col in metric_cols},
    }
    _write_json(summary, p.metrics / "seed_summary.json")


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
        _write_csv(preds, p.predictions / f"{slug}_test_predictions.csv")
        print(f"  {name}: test accuracy={metrics['accuracy']} macro_f1={metrics['macro_f1']}")
        if news is not None:
            _write_csv(score_news_dataframe(news, model_path=name), _headline_scored_csv(p, slug))

    if news is not None and p.weighted_model.exists():
        _write_csv(score_news_dataframe(news, model_path=p.weighted_model), _headline_scored_csv(p, "weighted"))


TEST_SET_MODELS = {
    "Zero-shot BART-MNLI": "zero_shot_test_predictions.csv",
    "DistilBERT fine-tuned": "finetuned_test_predictions.csv",
    "DistilBERT class-weighted": "weighted_test_predictions.csv",
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
    others = {"DistilBERT class-weighted": _headline_scored_csv(p, "weighted")}
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
               "DistilBERT class-weighted": _headline_scored_csv(p, "weighted")}
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

    slug = p.ticker.lower()
    _write_csv(table, p.metrics / f"{slug}_headline_model_comparison.csv")
    _write_json(report, p.metrics / f"{slug}_headline_evaluation.json")
    cols = ["model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "mcnemar_p_value"]
    print(table[[c for c in cols if c in table]].to_string(index=False))


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
    "train-seeds": stage_train_seeds,
    "benchmark": stage_benchmark,
    "analyze": stage_analyze,
    "fetch-news": stage_fetch_news,
    "score": stage_score,
    "monitor": stage_monitor,
    "sample-headlines": stage_sample_headlines,
    "evaluate-headlines": stage_evaluate_headlines,
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
    parser.add_argument("--n-boot", type=int, default=2000, help="bootstrap resamples for CIs")
    parser.add_argument("--n-label", type=int, default=300, help="headlines to sample for labelling")
    parser.add_argument("--max-train-samples", type=int, default=None,
                        help="subsample the training set (smoke tests only)")
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

    matplotlib.use("Agg")  # stages save figures to disk; never open windows

    paths = Paths(data=args.data_dir, outputs=args.outputs_dir, models=args.models_dir, ticker=args.ticker)
    paths.ensure()
    for name in resolve_stages(args.stages):
        print(f"[{name}]")
        STAGES[name](paths, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
