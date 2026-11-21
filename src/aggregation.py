from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.config import HALF_LIFE_DAYS, MONITORING_WINDOWS, ROLLING_WINDOW_DAYS


def compute_recency_weights(
    dates: pd.Series,
    as_of_date: Optional[pd.Timestamp] = None,
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.Series:
    """Exponential decay weights ``w = 2 ** (-age_days / half_life_days)``."""
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    if as_of_date is None:
        as_of_date = dates.max()

    age_days = (as_of_date - dates).dt.total_seconds() / 86400.0
    return 2.0 ** (-age_days / half_life_days)


def summarize_sentiment_windows(
    scored_news: pd.DataFrame,
    date_col: str = "Date",
    sentiment_col: str = "sentiment_score",
    label_col: str = "predicted_label",
    as_of_date: Optional[pd.Timestamp] = None,
    half_life_days: float = HALF_LIFE_DAYS,
    windows: Dict[str, Optional[int]] = MONITORING_WINDOWS,
) -> pd.DataFrame:
    """Unweighted and recency-weighted sentiment for each lookback window.

    A window of ``N`` days covers headlines dated on or after ``as_of_date - N days``;
    ``None`` means all history. Empty windows are skipped.
    """
    df = scored_news.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)
    df = df.sort_values(date_col).reset_index(drop=True)

    if as_of_date is None:
        as_of_date = df[date_col].max()

    df["recency_weight"] = compute_recency_weights(
        df[date_col], as_of_date=as_of_date, half_life_days=half_life_days
    )

    results = []
    for window_name, days in windows.items():
        if days is None:
            sub_df = df
        else:
            window_start = as_of_date - pd.Timedelta(days=days)
            sub_df = df[df[date_col] >= window_start]

        if len(sub_df) == 0:
            continue

        weighted_score = float(np.average(sub_df[sentiment_col], weights=sub_df["recency_weight"]))
        unweighted_score = float(sub_df[sentiment_col].mean())
        label_share = sub_df[label_col].value_counts(normalize=True) * 100.0

        results.append({
            "window": window_name,
            "headline_count": len(sub_df),
            "unweighted_sentiment": round(unweighted_score, 4),
            "recency_weighted_sentiment": round(weighted_score, 4),
            "positive_percent": round(float(label_share.get("positive", 0.0)), 4),
            "neutral_percent": round(float(label_share.get("neutral", 0.0)), 4),
            "negative_percent": round(float(label_share.get("negative", 0.0)), 4),
        })

    return pd.DataFrame(results)


def compute_rolling_sentiment(
    scored_news: pd.DataFrame,
    date_col: str = "Date",
    sentiment_col: str = "sentiment_score",
    window_days: int = ROLLING_WINDOW_DAYS,
) -> pd.DataFrame:
    """Headline-weighted rolling sentiment over calendar days.

    Each day's value is (sum of scores in the last ``window_days`` days) /
    (number of headlines in that span), so busy days count in proportion to their
    headlines and days without news do not dilute the average. Spans with no
    headlines are NaN.
    """
    df = scored_news.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    daily = (
        df.assign(_day=df[date_col].dt.floor("D"))
        .groupby("_day")
        .agg(
            daily_sentiment_sum=(sentiment_col, "sum"),
            article_count=(sentiment_col, "size"),
        )
        .sort_index()
    )

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range, fill_value=0)
    daily.index.name = date_col

    with np.errstate(invalid="ignore", divide="ignore"):
        daily["daily_sentiment"] = np.where(
            daily["article_count"] > 0,
            daily["daily_sentiment_sum"] / daily["article_count"],
            np.nan,
        )

    rolling_sum = daily["daily_sentiment_sum"].rolling(window_days, min_periods=1).sum()
    daily["rolling_article_count"] = daily["article_count"].rolling(window_days, min_periods=1).sum()
    with np.errstate(invalid="ignore", divide="ignore"):
        daily["rolling_sentiment"] = np.where(
            daily["rolling_article_count"] > 0,
            rolling_sum / daily["rolling_article_count"],
            np.nan,
        )

    return daily.reset_index()
