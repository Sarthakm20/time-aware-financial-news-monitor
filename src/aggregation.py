from typing import Dict, Optional
import numpy as np
import pandas as pd

from src.config import HALF_LIFE_DAYS, MONITORING_WINDOWS


def compute_recency_weights(
    dates: pd.Series,
    as_of_date: Optional[pd.Timestamp] = None,
    half_life_days: float = HALF_LIFE_DAYS,
) -> pd.Series:
    if as_of_date is None:
        as_of_date = dates.max()

    age_days = (as_of_date - dates).dt.total_seconds() / 86400.0
    weights = 2.0 ** (-age_days / half_life_days)
    return weights


def summarize_sentiment_windows(
    scored_news: pd.DataFrame,
    date_col: str = "Date",
    sentiment_col: str = "sentiment_score",
    label_col: str = "predicted_label",
    as_of_date: Optional[pd.Timestamp] = None,
    half_life_days: float = HALF_LIFE_DAYS,
    windows: Dict[str, Optional[int]] = MONITORING_WINDOWS,
) -> pd.DataFrame:
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
            sub_df = df.copy()
        else:
            window_start = as_of_date - pd.Timedelta(days=days)
            sub_df = df[df[date_col] >= window_start].copy()

        if len(sub_df) == 0:
            continue

        weighted_score = float(np.average(sub_df[sentiment_col], weights=sub_df["recency_weight"]))
        unweighted_score = float(sub_df[sentiment_col].mean())

        label_counts = sub_df[label_col].value_counts(normalize=True) * 100.0

        results.append({
            "window": window_name,
            "headline_count": len(sub_df),
            "unweighted_sentiment": round(unweighted_score, 4),
            "recency_weighted_sentiment": round(weighted_score, 4),
            "positive_percent": round(float(label_counts.get("positive", 0.0)), 4),
            "neutral_percent": round(float(label_counts.get("neutral", 0.0)), 4),
            "negative_percent": round(float(label_counts.get("negative", 0.0)), 4),
        })

    return pd.DataFrame(results)


def compute_rolling_sentiment(
    scored_news: pd.DataFrame,
    date_col: str = "Date",
    sentiment_col: str = "sentiment_score",
    window_days: int = 30,
) -> pd.DataFrame:
    df = scored_news.copy()
    df[date_col] = pd.to_datetime(df[date_col], utc=True)

    daily = (
        df.set_index(date_col)
        .resample("D")
        .agg(
            daily_sentiment=(sentiment_col, "mean"),
            article_count=(sentiment_col, "count"),
        )
    )

    daily["rolling_sentiment"] = daily["daily_sentiment"].rolling(window_days, min_periods=1).mean()
    daily["rolling_article_count"] = daily["article_count"].rolling(window_days, min_periods=1).sum()

    return daily.reset_index()

