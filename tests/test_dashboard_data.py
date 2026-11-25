import json

import pandas as pd
import pytest

from src.dashboard_data import (
    YEARS,
    clean,
    company_view,
    coverage_gap,
    filter_scored,
    search_headlines,
    sentiment_distribution,
    trend_points,
)
from src.explorer import load_scored


def _label(score):
    return "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"


@pytest.fixture()
def scored(tmp_path):
    rows = []
    for t, base in [("TSLA", 0.3), ("AAPL", -0.2)]:
        for i, d in enumerate(pd.date_range("2023-01-01", periods=60, freq="3D", tz="UTC")):
            score = base + (0.1 if i % 2 else -0.1)
            rows.append({"Date": d, "Stock_symbol": t, "Article_title": f"{t} headline {i}",
                         "predicted_label": _label(score), "sentiment_score": score, "confidence": 0.8})
    for i, d in enumerate(pd.date_range("2019-01-01", periods=5, freq="40D", tz="UTC")):
        rows.append({"Date": d, "Stock_symbol": "OLDCO", "Article_title": f"Oldco update {i}",
                     "predicted_label": "neutral", "sentiment_score": 0.05, "confidence": 0.7})
    path = tmp_path / "scored.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    load_scored.cache_clear()
    return load_scored(path)


def test_years():
    assert YEARS == (2019, 2020, 2021, 2022, 2023)


def test_clean_replaces_nan_and_inf():
    assert clean({"a": float("nan"), "b": [1.0, float("inf")], "c": "x"}) == {"a": None, "b": [1.0, None], "c": "x"}


def test_distribution_sums_to_one_and_handles_empty(scored):
    d = sentiment_distribution(scored)
    assert abs(d["positive"] + d["neutral"] + d["negative"] - 1) < 1e-3 and d["n"] == len(scored)
    assert sentiment_distribution(scored.iloc[0:0]) == {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "n": 0}


def test_filter_scored_year_and_ticker(scored):
    assert filter_scored(scored, "tsla", 2023)["Stock_symbol"].unique().tolist() == ["TSLA"]
    assert filter_scored(scored, None, 2019)["Stock_symbol"].unique().tolist() == ["OLDCO"]
    assert filter_scored(scored, "OLDCO", 2023).empty
    with pytest.raises(KeyError):
        filter_scored(scored, "ZZZZ")


def test_trend_points_weekly_multi_ticker(scored):
    pts = trend_points(scored)
    assert pts and set(pts[0]) == {"date", "sentiment", "headlines"} and -1 <= pts[0]["sentiment"] <= 1
    assert pts == sorted(pts, key=lambda p: p["date"])
    assert trend_points(scored.iloc[0:0]) == []


def test_company_view_has_no_nan(scored):
    view = company_view(scored, "OLDCO")
    assert json.dumps(view, allow_nan=False)
    assert view["cards"]["headlines"] == 5 and view["windows"] and view["timeline"]
    assert set(view["windows"][0]) == {"window", "headlines", "mean", "recency_weighted", "positive_pct",
                                       "negative_pct"}
    empty = company_view(scored, "OLDCO", year=2023)
    assert empty["cards"]["headlines"] == 0 and empty["cards"]["recent_30d"] is None
    assert empty["timeline"] == [] and empty["windows"] == [] and empty["headlines"] == []
    assert json.dumps(empty, allow_nan=False)


def test_company_view_orders_headlines(scored):
    view = company_view(scored, "TSLA", order="most_negative", n=3)
    assert len(view["headlines"]) == 3
    assert view["headlines"][0]["sentiment_score"] <= view["headlines"][-1]["sentiment_score"]


def test_search_is_literal_paged_and_bounded(scored):
    assert search_headlines(scored, q="(")["total"] == 0
    r = search_headlines(scored, q="tsla headline 1", page_size=5)
    assert r["total"] == 11 and len(r["rows"]) == 5 and r["pages"] == 3
    assert set(r["rows"][0]) == {"date", "ticker", "headline", "label", "sentiment_score", "confidence"}
    far = search_headlines(scored, page=999)
    assert far["rows"] == [] and far["total"] == len(scored)
    assert search_headlines(scored, sort="most_negative")["rows"][0]["sentiment_score"] == round(float(scored["sentiment_score"].min()), 4)
    assert search_headlines(scored, ticker="aapl", label="negative")["total"] == 30
    assert search_headlines(scored, start="2019-01-01", end="2019-12-31")["total"] == 5


def test_coverage_gap(scored):
    assert coverage_gap(filter_scored(scored, "TSLA")) is None
    gappy = pd.DataFrame({"Date": pd.to_datetime(["2020-01-01", "2020-01-05", "2020-09-01"], utc=True),
                          "sentiment_score": [0.0, 0.0, 0.0]})
    assert coverage_gap(gappy) == {"start": "2020-01-05", "end": "2020-09-01", "days": 240}
