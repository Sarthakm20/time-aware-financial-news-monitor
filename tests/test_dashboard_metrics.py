import json

import pytest

from src.config import METRICS_DIR
from src.dashboard_metrics import (
    SCOPE_ALL,
    MetricsUnavailable,
    corpus,
    headline_f1,
    market_impact,
    market_relationship,
    model_evaluation,
    relevance,
)

pytestmark = pytest.mark.skipif(not (METRICS_DIR / "event_study_buckets.csv").exists(),
                                reason="committed metrics not available")


def test_market_relationship_matches_spec():
    m = market_relationship(METRICS_DIR, "adapted")
    tone = {r["group"]: r for r in m["by_tone"]}
    assert [r["group"] for r in m["by_tone"]] == ["positive", "neutral", "negative"]
    assert m["events"] == 22999 and m["scope"] == SCOPE_ALL and m["variant"] == "excluding price-move headlines"
    assert round(tone["positive"]["value"], 2) == 0.26 and round(tone["negative"]["value"], 2) == -0.58
    assert tone["negative"]["ci_low"] < tone["negative"]["value"] < tone["negative"]["ci_high"]
    assert m["volume_ratio"] == 1.9
    everything = {r["group"]: r for r in market_relationship(METRICS_DIR, "adapted", "all")["by_tone"]}
    assert round(everything["negative"]["value"], 2) == -0.81


def test_headline_f1_pooled_and_per_company():
    assert round(headline_f1(METRICS_DIR, "adapted")["value"], 3) == 0.655
    assert headline_f1(METRICS_DIR, "base")["value"] == pytest.approx(0.5685)
    tsla = headline_f1(METRICS_DIR, "adapted", "tsla")
    assert tsla["scope"] == "TSLA" and tsla["ci_low"] < tsla["value"] < tsla["ci_high"] and tsla["n"] > 0
    assert headline_f1(METRICS_DIR, "adapted", "MSFT")["scope"] == SCOPE_ALL


def test_relevance_and_corpus():
    r = relevance(METRICS_DIR)
    assert r["precision_min"] == pytest.approx(0.962, abs=1e-3) and r["precision_max"] == 1.0
    assert r["collected"] == 61818 and r["scored"] == 27146 and round(r["kept_share"], 3) == 0.439
    assert r["scope"] == SCOPE_ALL and len(r["by_ticker"]) == 6
    assert corpus(METRICS_DIR)["tickers"] == 89


def test_market_impact_spread_and_paths():
    mi = market_impact(METRICS_DIR, "adapted")
    assert [s["outcome"] for s in mi["spread"]] == ["ar_m1", "ar_0", "ar_p1", "car_p1_p5", "av_0"]
    day0 = mi["spread"][1]
    assert day0["label"] == "Day 0" and day0["unit"] == "pct" and day0["significant"] is True
    assert mi["spread"][-1]["unit"] == "pts"
    assert mi["car_paths_available"] and {p["group"] for p in mi["car_paths"]} == {"positive", "negative"}
    assert mi["events"] == 22999 and mi["tickers"] == 85
    assert market_impact(METRICS_DIR, "base", "all")["events"] == 24807
    assert mi["volume_ratio"] == 1.9 and market_impact(METRICS_DIR, "adapted", "all")["volume_ratio"] == 1.8


def test_model_evaluation_includes_adapted_without_ci():
    ev = model_evaluation(METRICS_DIR)
    adapted = [r for r in ev["phrasebank"] if r["model"] == "DistilBERT headline-adapted"][0]
    assert adapted["ci_low"] is None and round(adapted["macro_f1"], 3) == 0.847
    assert round(ev["calibration_ece"], 3) == 0.021 and json.dumps(ev, allow_nan=False)
    assert not any(r["model"].startswith("Majority") for r in ev["per_company"])
    assert ev["seeds"]["n"] >= 3 and 0 < ev["seeds"]["std"] < 0.05
    assert {"model", "n", "macro_f1", "ci_low", "ci_high", "accuracy", "mcnemar_p"} <= set(ev["headlines"][0])


def test_missing_files_raise(tmp_path):
    with pytest.raises(MetricsUnavailable):
        market_impact(tmp_path, "adapted")
    with pytest.raises(MetricsUnavailable):
        model_evaluation(tmp_path)
    assert corpus(tmp_path) is None


def _year_run(model, variant, year):
    runs = json.loads((METRICS_DIR / "event_study_summary.json").read_text(encoding="utf-8"))["runs_by_year"]
    return next(r for r in runs if r["model"] == model and r["headlines"] == variant and r["year"] == year)


def test_market_figures_follow_the_period():
    m = market_relationship(METRICS_DIR, "adapted", year=2021)
    run = _year_run("adapted", "excluding price-move headlines", 2021)
    assert m["scope"] == "All companies · 2021" and m["events"] == run["events"] and m["events"] < 22999
    assert [r["group"] for r in m["by_tone"]] == ["positive", "neutral", "negative"]
    assert m["by_tone"][0]["value"] != market_relationship(METRICS_DIR, "adapted")["by_tone"][0]["value"]
    mi = market_impact(METRICS_DIR, "adapted", "all", year=2021)
    assert mi["scope"] == "All companies · 2021" and mi["tickers"] == _year_run("adapted", "all relevant headlines", 2021)["tickers"]
    assert mi["car_paths_available"] and {p["group"] for p in mi["car_paths"]} == {"positive", "negative"}
    assert len(mi["spread"]) == 5 and mi["volume_ratio"] > 0


def test_every_dashboard_year_has_results():
    for year in (2019, 2020, 2021, 2022, 2023):
        for model in ("adapted", "base"):
            assert market_impact(METRICS_DIR, model, "excluding", year=year)["events"] > 1000


def test_year_without_results_is_unavailable():
    with pytest.raises(MetricsUnavailable):
        market_impact(METRICS_DIR, "adapted", year=2018)
