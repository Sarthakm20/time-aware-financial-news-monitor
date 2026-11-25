"""Dashboard payloads read from the committed evaluation and event-study files in outputs/metrics/.

These results exist only for all companies and all years together, so every payload carries a scope label
and ignores the dashboard's company and period filters.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from src.dashboard_data import clean

MODEL_FILE_NAMES = {"adapted": "DistilBERT headline-adapted", "base": "DistilBERT fine-tuned"}
VARIANTS = {"excluding": "excluding price-move headlines", "all": "all relevant headlines"}
SCOPE_ALL = "All companies · 2019–2023"
POOLED = "pooled_all_6_companies"
TONES = ("positive", "neutral", "negative")
SPREAD_LABELS = {"ar_m1": "Day −1 (before)", "ar_0": "Day 0", "ar_p1": "Day +1", "car_p1_p5": "Days +1…+5",
                 "av_0": "Abnormal volume, day 0"}


class MetricsUnavailable(Exception):
    """A results file the dashboard needs is not on this server."""


@lru_cache(maxsize=64)
def _read(path: str, mtime: float) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path.endswith(".json") else pd.read_csv(path)


def _load(metrics_dir: Path, name: str) -> Any:
    path = Path(metrics_dir) / name
    if not path.exists():
        raise MetricsUnavailable(name)
    data = _read(str(path), path.stat().st_mtime)
    return data.copy() if isinstance(data, pd.DataFrame) else data


def corpus(metrics_dir: Path) -> Optional[Dict[str, Any]]:
    try:
        return dict(_load(metrics_dir, "corpus_summary.json"))
    except MetricsUnavailable:
        return None


def _run(metrics_dir: Path, model: str, variant: str) -> Dict[str, Any]:
    runs = _load(metrics_dir, "event_study_summary.json")["runs"]
    return next(r for r in runs if r["model"] == model and r["headlines"] == VARIANTS[variant])


def _buckets(metrics_dir: Path, model: str, variant: str) -> pd.DataFrame:
    b = _load(metrics_dir, "event_study_buckets.csv")
    return b[(b["model"] == model) & (b["headlines"] == VARIANTS[variant])]


def headline_f1(metrics_dir: Path, model: str, ticker: Optional[str] = None) -> Dict[str, Any]:
    """Macro F1 on LLM-annotated headlines: per company where annotated, otherwise pooled over six companies."""
    name = MODEL_FILE_NAMES[model]
    per = _load(metrics_dir, "multi_company_per_company_accuracy.csv")
    symbol = ticker.upper() if ticker else None
    if symbol and symbol in set(per["ticker"]):
        row, scope = per[(per["ticker"] == symbol) & (per["model"] == name)].iloc[0], symbol
    else:
        pooled = _load(metrics_dir, "multi_company_pooled_accuracy.csv")
        row, scope = pooled[(pooled["population"] == POOLED) & (pooled["model"] == name)].iloc[0], SCOPE_ALL
    return clean({"value": row["macro_f1"], "ci_low": row["macro_f1_ci_low"], "ci_high": row["macro_f1_ci_high"],
                  "n": int(row["n"]), "scope": scope})


def relevance(metrics_dir: Path) -> Dict[str, Any]:
    rel = _load(metrics_dir, "multi_company_relevance_filter.csv")
    c = corpus(metrics_dir) or {}
    collected, scored = c.get("collected_headlines"), c.get("scored_headlines")
    return clean({
        "kept_share": scored / collected if collected else None,
        "collected": collected,
        "scored": scored,
        "precision_min": rel["precision"].min(),
        "precision_max": rel["precision"].max(),
        "scope": SCOPE_ALL,
        "by_ticker": rel[["ticker", "precision", "recall_estimate", "on_topic_share_estimate"]].round(4)
                     .to_dict(orient="records"),
    })


def _volume_ratio(b: pd.DataFrame) -> float:
    """Day-0 abnormal log-volume on negative-headline days relative to positive ones."""
    av0 = b[b["outcome"] == "av_0"].set_index("group")
    return round(float(av0.loc["negative", "value"] / av0.loc["positive", "value"]), 1)


def market_relationship(metrics_dir: Path, model: str, variant: str = "excluding") -> Dict[str, Any]:
    """Day-0 market-adjusted return by headline tone (percent, 95% CI) and the volume ratio, negative vs positive."""
    b = _buckets(metrics_dir, model, variant)
    ar0 = b[b["outcome"] == "ar_0"].set_index("group")
    return clean({
        "variant": VARIANTS[variant],
        "events": _run(metrics_dir, model, variant)["events"],
        "scope": SCOPE_ALL,
        "by_tone": [{"group": g, "value": ar0.loc[g, "value"] * 100, "ci_low": ar0.loc[g, "ci_low"] * 100,
                     "ci_high": ar0.loc[g, "ci_high"] * 100} for g in TONES],
        "volume_ratio": _volume_ratio(b),
    })


def market_impact(metrics_dir: Path, model: str, variant: str = "excluding") -> Dict[str, Any]:
    b = _buckets(metrics_dir, model, variant)
    spread = b[b["group"] == "positive minus negative"].set_index("outcome")
    run = _run(metrics_dir, model, variant)
    try:
        paths = _load(metrics_dir, "event_study_car_paths.csv")
        paths = paths[(paths["model"] == model) & (paths["headlines"] == VARIANTS[variant])]
        paths = paths.sort_values(["group", "day"])
        car_paths = [{"group": r.group, "day": int(r.day), "mean_car": round(r.mean_car * 100, 4)}
                     for r in paths.itertuples()]
    except MetricsUnavailable:
        car_paths = []
    return clean({
        "variant": VARIANTS[variant],
        "events": run["events"],
        "tickers": run["tickers"],
        "scope": SCOPE_ALL,
        "spread": [{"outcome": o, "label": label, "value": spread.loc[o, "value"] * 100,
                    "ci_low": spread.loc[o, "ci_low"] * 100, "ci_high": spread.loc[o, "ci_high"] * 100,
                    "unit": "pts" if o == "av_0" else "pct",
                    "significant": bool(spread.loc[o, "ci_low"] > 0 or spread.loc[o, "ci_high"] < 0)}
                   for o, label in SPREAD_LABELS.items()],
        "car_paths": car_paths,
        "car_paths_available": bool(car_paths),
        "volume_ratio": _volume_ratio(b),
    })


def model_evaluation(metrics_dir: Path) -> Dict[str, Any]:
    test = _load(metrics_dir, "test_set_model_comparison.csv")
    adapted = _load(metrics_dir, "adapted_test_metrics.json")
    pooled = _load(metrics_dir, "multi_company_pooled_accuracy.csv")
    per = _load(metrics_dir, "multi_company_per_company_accuracy.csv")
    seeds = _load(metrics_dir, "seed_summary.json")

    phrasebank = [{"model": r.model, "n": int(r.n), "macro_f1": r.macro_f1, "ci_low": r.macro_f1_ci_low,
                   "ci_high": r.macro_f1_ci_high, "accuracy": r.accuracy, "ece": r.ece} for r in test.itertuples()]
    phrasebank.append({"model": MODEL_FILE_NAMES["adapted"], "n": int(adapted["num_examples"]),
                       "macro_f1": adapted["macro_f1"], "ci_low": None, "ci_high": None,
                       "accuracy": adapted["accuracy"], "ece": None, "note": "No bootstrap CI saved for this model"})
    pooled = pooled[pooled["population"] == POOLED]
    per = per[~per["model"].str.startswith("Majority")]
    return clean({
        "phrasebank": phrasebank,
        "headlines": [{"model": r.model, "n": int(r.n), "macro_f1": r.macro_f1, "ci_low": r.macro_f1_ci_low,
                       "ci_high": r.macro_f1_ci_high, "accuracy": r.accuracy, "mcnemar_p": r.mcnemar_p_value}
                      for r in pooled.itertuples()],
        "per_company": [{"ticker": r.ticker, "model": r.model, "n": int(r.n), "macro_f1": r.macro_f1,
                         "accuracy": r.accuracy} for r in per.itertuples()],
        "relevance": relevance(metrics_dir)["by_ticker"],
        "seeds": {"n": seeds["n_seeds"], "mean": seeds["macro_f1"]["mean"], "std": seeds["macro_f1"]["std"]},
        "calibration_ece": test.loc[test["model"] == MODEL_FILE_NAMES["base"], "ece"].iloc[0],
    })
