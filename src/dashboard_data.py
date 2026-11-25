"""Dashboard payloads computed from the scored, relevance-filtered headlines.

Everything returned here is JSON-ready: plain Python types, and NaN/inf replaced by None (see ``clean``).
"""

import math
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from src.aggregation import compute_rolling_sentiment
from src.explorer import ticker_headlines, ticker_timeline, ticker_windows

YEARS = (2019, 2020, 2021, 2022, 2023)
SORTS = ("newest", "oldest", "most_positive", "most_negative")
GAP_DAYS = 90


def clean(obj: Any) -> Any:
    """Recursively convert numpy scalars to Python and NaN/inf to None, so payloads are strict JSON."""
    if isinstance(obj, dict):
        return {k: clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    return obj


def filter_scored(scored: pd.DataFrame, ticker: Optional[str] = None, year: Optional[int] = None) -> pd.DataFrame:
    """Rows for one ticker and/or one calendar year. Unknown ticker -> KeyError; an empty year is allowed."""
    sub = scored
    if ticker:
        symbol = ticker.upper()
        sub = sub[sub["Stock_symbol"] == symbol]
        if sub.empty:
            raise KeyError(symbol)
    if year:
        sub = sub[sub["Date"].dt.year == int(year)]
    return sub


def sentiment_distribution(df: pd.DataFrame) -> Dict[str, Any]:
    n = int(len(df))
    if n == 0:
        return {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "n": 0}
    shares = df["predicted_label"].value_counts(normalize=True)
    return {label: round(float(shares.get(label, 0.0)), 4) for label in ("positive", "neutral", "negative")} | {"n": n}


def trend_points(df: pd.DataFrame, window_days: int = 30) -> List[Dict[str, Any]]:
    """Weekly points of the headline-weighted rolling sentiment (last value per week), across all rows given."""
    if df.empty:
        return []
    weekly = (compute_rolling_sentiment(df, window_days=window_days)
              .set_index("Date")
              .resample("W")
              .agg({"rolling_sentiment": "last", "rolling_article_count": "last"})
              .dropna(subset=["rolling_sentiment"]))
    return clean([{"date": str(d.date()), "sentiment": round(float(r.rolling_sentiment), 4),
                   "headlines": int(r.rolling_article_count)} for d, r in weekly.iterrows()])


def coverage_gap(df: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """The longest stretch without headlines, if it exceeds 90 days (missing data, not quiet news)."""
    days = pd.Series(sorted(pd.to_datetime(df["Date"], utc=True).dt.normalize().unique()))
    if len(days) < 2:
        return None
    gaps = days.diff().dt.days
    i = int(gaps.idxmax())
    if gaps.max() <= GAP_DAYS:
        return None
    return {"start": str(days[i - 1].date()), "end": str(days[i].date()), "days": int(gaps.max())}


def _empty_company(symbol: str) -> Dict[str, Any]:
    cards = dict.fromkeys(["first_date", "last_date", "recent_30d", "recent_delta", "history", "positive_share",
                           "negative_share"])
    return {"ticker": symbol, "cards": {"headlines": 0, **cards}, "coverage_gap": None, "timeline": [],
            "windows": [], "headlines": []}


def company_view(scored: pd.DataFrame, ticker: str, year: Optional[int] = None, window_days: int = 30,
                 half_life_days: float = 30.0, order: str = "latest", n: int = 12) -> Dict[str, Any]:
    symbol = ticker.upper()
    sub = filter_scored(scored, symbol, year)
    if sub.empty:
        return _empty_company(symbol)

    windows = ticker_windows(sub, symbol, half_life_days=half_life_days).set_index("window")
    history = windows.loc["All available history", "unweighted_sentiment"]
    recent = windows.loc["Last 30 days", "unweighted_sentiment"] if "Last 30 days" in windows.index else None
    labels = sub["predicted_label"]

    timeline = ticker_timeline(sub, symbol, window_days=window_days, freq="W")
    return clean({
        "ticker": symbol,
        "cards": {
            "headlines": len(sub),
            "first_date": str(sub["Date"].min().date()),
            "last_date": str(sub["Date"].max().date()),
            "recent_30d": recent,
            "recent_delta": None if recent is None else round(recent - history, 4),
            "history": history,
            "positive_share": round(float((labels == "positive").mean()), 4),
            "negative_share": round(float((labels == "negative").mean()), 4),
        },
        "coverage_gap": coverage_gap(sub),
        "timeline": [{"date": str(r.Date.date()), "sentiment": round(float(r.rolling_sentiment), 4),
                      "headlines_in_window": int(r.rolling_article_count)} for r in timeline.itertuples()],
        "windows": [{"window": name, "headlines": int(w.headline_count), "mean": w.unweighted_sentiment,
                     "recency_weighted": w.recency_weighted_sentiment, "positive_pct": w.positive_percent,
                     "negative_pct": w.negative_percent} for name, w in windows.iterrows()],
        "headlines": ticker_headlines(sub, symbol, n=n, order=order),
    })


def search_headlines(scored: pd.DataFrame, q: Optional[str] = None, ticker: Optional[str] = None,
                     label: Optional[str] = None, start: Optional[str] = None, end: Optional[str] = None,
                     sort: str = "newest", page: int = 1, page_size: int = 25) -> Dict[str, Any]:
    """Literal, case-insensitive headline search with filters and pagination (page past the end -> no rows)."""
    if sort not in SORTS:
        raise ValueError(f"sort must be one of {', '.join(SORTS)}")
    sub = scored
    if q:
        sub = sub[sub["Article_title"].str.contains(q, case=False, regex=False, na=False)]
    if ticker:
        sub = sub[sub["Stock_symbol"] == ticker.upper()]
    if label:
        sub = sub[sub["predicted_label"] == label]
    if start:
        sub = sub[sub["Date"] >= pd.Timestamp(start, tz="UTC")]
    if end:
        sub = sub[sub["Date"] < pd.Timestamp(end, tz="UTC") + pd.Timedelta(days=1)]

    by, ascending = {"newest": ("Date", False), "oldest": ("Date", True), "most_positive": ("sentiment_score", False),
                     "most_negative": ("sentiment_score", True)}[sort]
    sub = sub.sort_values(by, ascending=ascending, kind="stable")
    total = int(len(sub))
    page_rows = sub.iloc[(page - 1) * page_size: page * page_size]
    return clean({
        "total": total,
        "page": page,
        "page_size": page_size,
        "pages": math.ceil(total / page_size),
        "rows": [{"date": str(r.Date.date()), "ticker": r.Stock_symbol, "headline": r.Article_title,
                  "label": r.predicted_label, "sentiment_score": round(float(r.sentiment_score), 4),
                  "confidence": round(float(r.confidence), 4)} for r in page_rows.itertuples()],
    })
