"""Daily prices, returns, abnormal returns and abnormal volume.

Prices come from Yahoo Finance via ``yfinance`` (adjusted closes). They are cached under
``data/prices/`` and not committed; ``python -m src.pipeline prices`` re-downloads them.
"""

from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd

BENCHMARK = "QQQ"  # Nasdaq-100 ETF: the market model for a Nasdaq-100 universe


def download_prices(tickers: Iterable[str], start: str, end: str) -> pd.DataFrame:
    """Long frame: Date, ticker, close (split/dividend adjusted), volume. Missing tickers are skipped."""
    import yfinance as yf

    tickers = sorted(set(tickers))
    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False,
                      group_by="ticker", threads=True)
    frames: List[pd.DataFrame] = []
    for t in tickers:
        if t not in raw.columns.get_level_values(0):
            continue
        sub = raw[t][["Close", "Volume"]].dropna(subset=["Close"])
        if sub.empty:
            continue
        frames.append(pd.DataFrame({"Date": sub.index.tz_localize(None).normalize(), "ticker": t,
                                    "close": sub["Close"].to_numpy(), "volume": sub["Volume"].to_numpy()}))
    if not frames:
        return pd.DataFrame(columns=["Date", "ticker", "close", "volume"])
    return pd.concat(frames, ignore_index=True).sort_values(["ticker", "Date"]).reset_index(drop=True)


def load_prices(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.sort_values(["ticker", "Date"]).reset_index(drop=True)


def add_returns(prices: pd.DataFrame, benchmark: str = BENCHMARK, volume_lookback: int = 20) -> pd.DataFrame:
    """Add daily return, market-adjusted abnormal return and abnormal log volume.

    abnormal_return = r_stock - r_benchmark (same day).
    abnormal_volume = log(volume) - mean(log volume over the previous ``volume_lookback`` days).
    """
    df = prices.sort_values(["ticker", "Date"]).copy()
    df["ret"] = df.groupby("ticker")["close"].pct_change()
    bench = df.loc[df["ticker"] == benchmark, ["Date", "ret"]].rename(columns={"ret": "bench_ret"})
    if bench.empty:
        raise ValueError(f"benchmark {benchmark} not in prices")
    df = df.merge(bench, on="Date", how="left")
    df["abnormal_return"] = df["ret"] - df["bench_ret"]

    logv = np.log(df["volume"].where(df["volume"] > 0))
    baseline = logv.groupby(df["ticker"]).transform(
        lambda s: s.shift(1).rolling(volume_lookback, min_periods=volume_lookback // 2).mean())
    df["abnormal_volume"] = logv - baseline
    return df[df["ticker"] != benchmark].reset_index(drop=True)
