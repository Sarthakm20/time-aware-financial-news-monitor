import numpy as np
import pandas as pd
import pytest

from src.event_study import (
    bucket_returns,
    build_event_panel,
    daily_sentiment,
    information_coefficients,
    is_price_move_headline,
    mean_car_paths,
)
from src.market import add_returns


def _prices(n_days=40, tickers=("AAA", "BBB")):
    dates = pd.bdate_range("2021-01-04", periods=n_days)  # Mon-Fri only
    rows = []
    for t in (*tickers, "QQQ"):
        close = 100 * np.cumprod(1 + np.full(n_days, 0.01 if t == "QQQ" else 0.02))
        rows += [{"Date": d, "ticker": t, "close": c, "volume": 1000.0} for d, c in zip(dates, close)]
    return pd.DataFrame(rows)


def test_add_returns_market_adjusts_and_drops_benchmark():
    r = add_returns(_prices())
    assert set(r["ticker"]) == {"AAA", "BBB"}
    valid = r.dropna(subset=["abnormal_return"])
    np.testing.assert_allclose(valid["abnormal_return"], 0.01, atol=1e-12)  # 2% - 1%
    assert np.allclose(r["abnormal_volume"].dropna(), 0.0)  # constant volume


def test_add_returns_requires_benchmark():
    with pytest.raises(ValueError):
        add_returns(_prices().query("ticker != 'QQQ'"))


@pytest.mark.parametrize("title, expected", [
    ("Why Tesla Stock Popped Today", True),
    ("Netflix shares are trading higher after Credit Suisse maintained an Outperform rating", True),
    ("ASML (ASML) Stock Sinks As Market Gains: Here's Why", True),
    ("Why Datadog Stock Is Falling Today", True),
    ("Tesla Recalls 120,000 Vehicles", False),
    ("Barclays Upgrades JD On Continued Strength", False),
    ("Costco Wholesale (COST) Q2 2023 Earnings Call Transcript", False),
])
def test_price_move_pattern(title, expected):
    assert bool(is_price_move_headline(pd.Series([title])).iloc[0]) is expected


def test_daily_sentiment_aggregates_and_can_exclude_price_moves():
    scored = pd.DataFrame({
        "Date": ["2021-01-05 00:00:00+00:00", "2021-01-05 00:00:00+00:00", "2021-01-06 00:00:00+00:00"],
        "Stock_symbol": ["AAA", "AAA", "AAA"],
        "Article_title": ["AAA wins contract", "Why AAA Stock Jumped Today", "AAA recall"],
        "sentiment_score": [0.8, 0.4, -0.6],
    })
    d = daily_sentiment(scored)
    assert d["sentiment"].round(4).tolist() == [0.6, -0.6] and d["n_headlines"].tolist() == [2, 1]
    d2 = daily_sentiment(scored, exclude_price_moves=True)
    assert d2["sentiment"].tolist() == [0.8, -0.6]


def _returns_with(ar_by_day, ticker="AAA"):
    dates = pd.bdate_range("2021-01-04", periods=len(ar_by_day))
    return pd.DataFrame({"Date": dates, "ticker": ticker, "abnormal_return": ar_by_day,
                         "abnormal_volume": np.arange(len(ar_by_day), dtype=float)})


def test_event_panel_aligns_horizons_and_maps_weekends():
    ar = np.arange(30, dtype=float) / 100  # AR on trading day i = i/100
    returns = _returns_with(ar)
    dates = returns["Date"]
    daily = pd.DataFrame({
        "ticker": ["AAA", "AAA", "AAA"],
        "date": [dates[10], pd.Timestamp("2021-01-23"), pd.Timestamp("2021-01-24")],  # Sat+Sun -> Mon (day 15)
        "sentiment": [0.5, 1.0, 0.0], "n_headlines": [1, 3, 1],
    })
    panel = build_event_panel(daily, returns, path_days=range(-2, 3)).set_index("event_date")
    e = panel.loc[dates[10]]
    assert (e["ar_m1"], e["ar_0"], e["ar_p1"]) == pytest.approx((0.09, 0.10, 0.11))
    assert e["car_p1_p5"] == pytest.approx(sum(range(11, 16)) / 100)
    assert e["av_0"] == 10
    assert e["car_path_-2"] == pytest.approx(0.08) and e["car_path_2"] == pytest.approx(0.08 + 0.09 + 0.10 + 0.11 + 0.12)

    monday = dates[dates >= pd.Timestamp("2021-01-23")].iloc[0]
    w = panel.loc[monday]
    assert w["n_headlines"] == 4 and w["sentiment"] == pytest.approx(0.75)  # headline-weighted merge


def test_event_panel_drops_events_without_full_window():
    returns = _returns_with(np.zeros(10))
    daily = pd.DataFrame({"ticker": ["AAA"] * 2, "date": [returns["Date"][0], returns["Date"][8]],
                          "sentiment": [0.1, 0.2], "n_headlines": [1, 1]})
    assert build_event_panel(daily, returns).empty


def _planted_panel(n=400, effect=0.02, seed=0):
    rng = np.random.default_rng(seed)
    sentiment = rng.uniform(-1, 1, n)
    return pd.DataFrame({
        "event_date": pd.bdate_range("2021-01-04", periods=n),
        "sentiment": sentiment,
        "ar_p1": effect * sentiment + rng.normal(0, 0.01, n),   # planted signal
        "ar_0": rng.normal(0, 0.01, n),                          # pure noise
    })


def test_information_coefficients_find_planted_signal_not_noise():
    ic = information_coefficients(_planted_panel(), horizons=["ar_p1", "ar_0"], n_boot=200).set_index("outcome")
    assert ic.loc["ar_p1", "value"] > 0.5 and ic.loc["ar_p1", "ci_low"] > 0 and ic.loc["ar_p1", "p_value"] < 0.01
    assert ic.loc["ar_0", "ci_low"] < 0 < ic.loc["ar_0", "ci_high"]


def test_bucket_spread_and_car_paths():
    panel = _planted_panel()
    b = bucket_returns(panel, threshold=0.3, horizons=["ar_p1"], n_boot=200)
    spread = b[b["group"] == "positive minus negative"].iloc[0]
    assert spread["value"] > 0.02 and spread["ci_low"] > 0

    panel["car_path_0"] = panel["ar_p1"]
    paths = mean_car_paths(panel, threshold=0.3)
    pos = paths[(paths.group == "positive") & (paths.day == 0)]["mean_car"].iloc[0]
    neg = paths[(paths.group == "negative") & (paths.day == 0)]["mean_car"].iloc[0]
    assert pos > 0 > neg


def test_constant_outcome_gives_nan_instead_of_crashing():
    panel = _planted_panel(50)
    panel["av_0"] = 0.0
    ic = information_coefficients(panel, horizons=["av_0"], n_boot=50).iloc[0]
    assert np.isnan(ic["value"]) and np.isnan(ic["ci_low"])
