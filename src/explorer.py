"""Read-only access to the precomputed, relevance-filtered Nasdaq-100 headline scores."""

from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from src.aggregation import compute_rolling_sentiment, summarize_sentiment_windows
from src.config import HALF_LIFE_DAYS, PREDICTIONS_DIR

DEFAULT_SCORED_CSV = PREDICTIONS_DIR / "nasdaq100_scored_headlines_adapted.csv"


@lru_cache(maxsize=4)
def load_scored(path: Path = DEFAULT_SCORED_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Stock_symbol"] = df["Stock_symbol"].str.upper()
    return df.sort_values(["Stock_symbol", "Date"], kind="stable").reset_index(drop=True)


def ticker_overview(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker: headline count, date range, mean sentiment, label shares."""
    g = scored.groupby("Stock_symbol")
    out = pd.DataFrame({
        "headlines": g.size(),
        "first_date": g["Date"].min().dt.date.astype(str),
        "last_date": g["Date"].max().dt.date.astype(str),
        "mean_sentiment": g["sentiment_score"].mean().round(4),
        "positive_share": g["predicted_label"].apply(lambda s: (s == "positive").mean()).round(4),
        "negative_share": g["predicted_label"].apply(lambda s: (s == "negative").mean()).round(4),
    })
    return out.reset_index().rename(columns={"Stock_symbol": "ticker"})


def ticker_frame(scored: pd.DataFrame, ticker: str) -> pd.DataFrame:
    sub = scored[scored["Stock_symbol"] == ticker.upper()]
    if sub.empty:
        raise KeyError(ticker)
    return sub


def ticker_windows(scored: pd.DataFrame, ticker: str, half_life_days: float = HALF_LIFE_DAYS) -> pd.DataFrame:
    return summarize_sentiment_windows(ticker_frame(scored, ticker), half_life_days=half_life_days)


def ticker_timeline(scored: pd.DataFrame, ticker: str, window_days: int = 30,
                    freq: Optional[str] = "W") -> pd.DataFrame:
    """Rolling headline-weighted sentiment; optionally downsampled (last value per period) to keep payloads small."""
    rolling = compute_rolling_sentiment(ticker_frame(scored, ticker), window_days=window_days)
    if freq:
        rolling = (rolling.set_index("Date")
                   .resample(freq)
                   .agg({"rolling_sentiment": "last", "rolling_article_count": "last", "article_count": "sum"})
                   .dropna(subset=["rolling_sentiment"])
                   .reset_index())
    return rolling


def ticker_headlines(scored: pd.DataFrame, ticker: str, n: int = 10, order: str = "latest") -> List[Dict]:
    """Latest headlines, or the most positive / negative ones."""
    sub = ticker_frame(scored, ticker)
    if order == "latest":
        sub = sub.sort_values("Date", ascending=False)
    elif order == "most_positive":
        sub = sub.sort_values("sentiment_score", ascending=False)
    elif order == "most_negative":
        sub = sub.sort_values("sentiment_score", ascending=True)
    else:
        raise ValueError("order must be latest, most_positive or most_negative")
    cols = ["Date", "Article_title", "predicted_label", "sentiment_score", "confidence"]
    out = sub[cols].head(n).copy()
    out["Date"] = out["Date"].dt.date.astype(str)
    return out.rename(columns={"Date": "date", "Article_title": "headline", "predicted_label": "label"}
                      ).to_dict(orient="records")
