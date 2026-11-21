import numpy as np
import pandas as pd
import pytest

from src.aggregation import (
    compute_recency_weights,
    compute_rolling_sentiment,
    summarize_sentiment_windows,
)
from src.config import PREDICTIONS_DIR


def _ts(day: str) -> pd.Timestamp:
    return pd.Timestamp(day, tz="UTC")


def _scored(rows):
    return pd.DataFrame(rows, columns=["Date", "sentiment_score", "predicted_label"])


def test_recency_weights_halve_every_half_life():
    as_of = _ts("2023-12-31")
    dates = pd.Series([as_of, as_of - pd.Timedelta(days=30), as_of - pd.Timedelta(days=60)])
    weights = compute_recency_weights(dates, as_of_date=as_of, half_life_days=30)
    np.testing.assert_allclose(weights.to_numpy(), [1.0, 0.5, 0.25])


def test_recency_weights_default_to_latest_date():
    dates = pd.Series([_ts("2023-01-01"), _ts("2023-01-11")])
    weights = compute_recency_weights(dates, half_life_days=10)
    np.testing.assert_allclose(weights.to_numpy(), [0.5, 1.0])


def test_recency_weights_reject_non_positive_half_life():
    with pytest.raises(ValueError):
        compute_recency_weights(pd.Series([_ts("2023-01-01")]), half_life_days=0)


def test_window_summary_counts_means_and_weighted_mean():
    news = _scored([
        ("2023-12-31", 1.0, "positive"),
        ("2023-12-01", -1.0, "negative"),   # 30 days old -> weight 0.5
        ("2023-01-01", 0.0, "neutral"),     # outside the 30-day window
    ])
    summary = summarize_sentiment_windows(
        news, half_life_days=30, windows={"30d": 30, "all": None}
    ).set_index("window")

    assert summary.loc["30d", "headline_count"] == 2
    assert summary.loc["30d", "unweighted_sentiment"] == pytest.approx(0.0)
    # (1*1 + 0.5*-1) / 1.5
    assert summary.loc["30d", "recency_weighted_sentiment"] == pytest.approx(1 / 3, abs=1e-4)
    assert summary.loc["30d", "positive_percent"] == pytest.approx(50.0)
    assert summary.loc["30d", "neutral_percent"] == pytest.approx(0.0)

    assert summary.loc["all", "headline_count"] == 3
    assert summary.loc["all", "neutral_percent"] == pytest.approx(100 / 3, abs=1e-3)


def test_window_summary_skips_empty_windows():
    news = _scored([("2023-12-31", 0.5, "positive")])
    summary = summarize_sentiment_windows(
        news, as_of_date=_ts("2024-06-30"), windows={"30d": 30, "all": None}
    )
    assert summary["window"].tolist() == ["all"]


def test_rolling_sentiment_is_headline_weighted_and_fills_gaps():
    news = _scored([
        ("2023-01-01 09:00", 1.0, "positive"),
        ("2023-01-01 15:00", 1.0, "positive"),
        ("2023-01-03 10:00", -1.0, "negative"),
    ])
    rolling = compute_rolling_sentiment(news, window_days=3).set_index("Date")

    assert len(rolling) == 3                       # Jan 2 is filled in
    assert rolling["article_count"].tolist() == [2, 0, 1]
    assert np.isnan(rolling["daily_sentiment"].iloc[1])
    # Jan 3 window holds 3 headlines: (1 + 1 - 1) / 3, not the mean of daily means (0.0)
    assert rolling["rolling_sentiment"].iloc[-1] == pytest.approx(1 / 3)
    assert rolling["rolling_article_count"].iloc[-1] == 3


@pytest.mark.skipif(
    not (PREDICTIONS_DIR / "tsla_scored_headlines_2019_2023.csv").exists(),
    reason="committed scored headlines not available",
)
def test_rolling_sentiment_matches_published_tsla_timeline():
    scored = pd.read_csv(PREDICTIONS_DIR / "tsla_scored_headlines_2019_2023.csv")
    rolling = compute_rolling_sentiment(scored, window_days=30)
    # Last value shown in notebooks/time_aware_monitor.ipynb for 2023-12-16
    assert rolling["Date"].iloc[-1] == _ts("2023-12-16")
    assert rolling["rolling_sentiment"].iloc[-1] == pytest.approx(0.123766, abs=1e-6)
