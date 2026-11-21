"""Does headline sentiment line up with abnormal returns and trading volume?

Design
------
* Unit of observation: one (ticker, trading day) with at least one relevant headline.
  Its sentiment is the mean P(pos) - P(neg) of that day's headlines.
* A headline dated on a weekend/holiday is assigned to the next trading day ("day 0").
* FNSPID gives dates without times, so day 0 mixes reaction and reporting. Many headlines
  *describe* a move that already happened ("Why X Stock Popped Today"), so a day-0 or
  day -1 correlation is not evidence of information. The test that matters is days +1..+5.
* Returns are market-adjusted: abnormal return = stock return - QQQ return.
* Uncertainty: bootstrap clustered by date (whole trading days are resampled), because
  events on the same day share the same market shock.
"""

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Headlines that report a price move rather than news about the company.
PRICE_MOVE_PATTERN = (
    r"(?i)(?:\b(?:stock|shares|ADRs?)\b.{0,40}?\b(?:jump|soar|surg|pop|rise|rose|rall|climb|gain|fall|fell|"
    r"drop|slid|slide|sink|sank|plung|tumbl|crash|spik|trad(?:e|es|ing|ed) (?:higher|lower)|"
    r"moves?\b|(?:is|are) (?:up|down)\b))"
    r"|^why\b.{0,60}\b(?:stock|shares)\b"
    r"|\b(?:stock|shares)\b.{0,20}\b(?:today|this week|this morning|on (?:monday|tuesday|wednesday|thursday|friday))\b"
)

HORIZONS = {
    "ar_m1": "Day -1 abnormal return (before the headline date)",
    "ar_0": "Day 0 abnormal return",
    "ar_p1": "Day +1 abnormal return",
    "car_p1_p5": "Cumulative abnormal return, days +1..+5",
    "av_0": "Day 0 abnormal log volume",
}


def is_price_move_headline(titles: pd.Series) -> pd.Series:
    return titles.astype(str).str.contains(PRICE_MOVE_PATTERN, regex=True)


def daily_sentiment(scored: pd.DataFrame, exclude_price_moves: bool = False) -> pd.DataFrame:
    """Mean headline sentiment per (ticker, calendar date)."""
    df = scored.copy()
    if exclude_price_moves:
        df = df[~is_price_move_headline(df["Article_title"])]
    df["date"] = pd.to_datetime(df["Date"], utc=True).dt.tz_localize(None).dt.normalize()
    return (df.groupby(["Stock_symbol", "date"])
              .agg(sentiment=("sentiment_score", "mean"), n_headlines=("sentiment_score", "size"))
              .reset_index()
              .rename(columns={"Stock_symbol": "ticker"}))


def build_event_panel(daily: pd.DataFrame, returns: pd.DataFrame, path_days: Iterable[int] = range(-5, 11)
                      ) -> pd.DataFrame:
    """Attach abnormal returns/volume around each (ticker, day) event.

    ``returns`` needs Date, ticker, abnormal_return, abnormal_volume (see ``src.market.add_returns``).
    Adds ar_m1, ar_0, ar_p1, car_p1_p5, av_0 and ``car_path_<d>``: the cumulative abnormal return
    from the first day of ``path_days`` through day d (NaN when the full window is not available).
    """
    path_days = list(path_days)
    rows: List[dict] = []
    for ticker, ev in daily.groupby("ticker"):
        r = returns[returns["ticker"] == ticker].sort_values("Date")
        if r.empty:
            continue
        dates = r["Date"].to_numpy(dtype="datetime64[ns]")
        ar = r["abnormal_return"].to_numpy(dtype=float)
        av = r["abnormal_volume"].to_numpy(dtype=float)
        pos = np.searchsorted(dates, ev["date"].to_numpy(dtype="datetime64[ns]"), side="left")
        n = len(dates)
        for (_, e), k in zip(ev.iterrows(), pos):
            if k < 1 or k + 5 >= n:  # need day -1 and days +1..+5
                continue
            row = {"ticker": ticker, "event_date": pd.Timestamp(dates[k]), "sentiment": e["sentiment"],
                   "n_headlines": e["n_headlines"], "ar_m1": ar[k - 1], "ar_0": ar[k], "ar_p1": ar[k + 1],
                   "car_p1_p5": float(np.nansum(ar[k + 1:k + 6])), "av_0": av[k]}
            start, end = path_days[0], path_days[-1]
            if k + start >= 0 and k + end < n:  # full window available
                cum = np.nancumsum(ar[k + start:k + end + 1])
                for d, value in zip(path_days, cum):
                    row[f"car_path_{d}"] = float(value)
            rows.append(row)
    panel = pd.DataFrame(rows)
    if panel.empty:
        return panel
    # several calendar dates can map to the same trading day (weekend news): merge them
    agg = {c: "first" for c in panel.columns if c not in ("ticker", "event_date", "sentiment", "n_headlines")}
    weights = panel["n_headlines"]
    panel = panel.assign(_w=weights, _ws=panel["sentiment"] * weights)
    grouped = panel.groupby(["ticker", "event_date"], as_index=False).agg(
        {**agg, "_w": "sum", "_ws": "sum"})
    grouped["sentiment"] = grouped["_ws"] / grouped["_w"]
    grouped["n_headlines"] = grouped["_w"].astype(int)
    return grouped.drop(columns=["_w", "_ws"]).sort_values(["event_date", "ticker"]).reset_index(drop=True)


def _date_cluster_bootstrap(panel: pd.DataFrame, stat, n_boot: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(panel["event_date"])
    order = np.argsort(codes, kind="stable")
    counts = np.bincount(codes)
    clusters = np.split(order, np.cumsum(counts)[:-1])  # row positions for each date
    out = np.empty(n_boot)
    for b in range(n_boot):
        pick = rng.integers(0, len(clusters), size=len(clusters))
        out[b] = stat(panel.iloc[np.concatenate([clusters[i] for i in pick])])
    return out


def _summarise(point: float, boots: np.ndarray) -> Dict[str, float]:
    boots = boots[np.isfinite(boots)]
    if not np.isfinite(point) or boots.size == 0:  # e.g. a constant outcome: correlation undefined
        return {"value": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    lo, hi = np.quantile(boots, [0.025, 0.975])
    p = min(1.0, 2 * min((boots <= 0).mean(), (boots >= 0).mean()))
    return {"value": round(float(point), 5), "ci_low": round(float(lo), 5), "ci_high": round(float(hi), 5),
            "p_value": round(float(p), 4)}


def information_coefficients(panel: pd.DataFrame, horizons: Optional[Iterable[str]] = None,
                             n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Spearman correlation between event-day sentiment and each outcome, with date-clustered CIs."""
    rows = []
    for h in (horizons or HORIZONS):
        sub = panel[["event_date", "sentiment", h]].dropna()

        def stat(df, h=h):
            return spearmanr(df["sentiment"], df[h]).statistic if len(df) > 2 else np.nan

        res = _summarise(stat(sub), _date_cluster_bootstrap(sub, stat, n_boot, seed))
        rows.append({"outcome": h, "description": HORIZONS.get(h, h), "n_events": len(sub), **res})
    return pd.DataFrame(rows)


def bucket_returns(panel: pd.DataFrame, threshold: float = 0.3, horizons: Optional[Iterable[str]] = None,
                   n_boot: int = 500, seed: int = 0) -> pd.DataFrame:
    """Mean outcome for positive (> threshold) vs negative (< -threshold) sentiment days, and the spread."""
    df = panel.copy()
    df["bucket"] = np.select([df["sentiment"] > threshold, df["sentiment"] < -threshold],
                             ["positive", "negative"], "neutral")
    rows = []
    for h in (horizons or HORIZONS):
        sub = df[["event_date", "bucket", h]].dropna()
        for bucket in ["positive", "neutral", "negative"]:
            b = sub[sub["bucket"] == bucket]

            def mean_stat(d, h=h):
                return d[h].mean()

            if len(b) > 2:
                rows.append({"outcome": h, "group": bucket, "n_events": len(b),
                             **_summarise(mean_stat(b), _date_cluster_bootstrap(b, mean_stat, n_boot, seed))})

        def spread(d, h=h):
            pos, neg = d.loc[d["bucket"] == "positive", h], d.loc[d["bucket"] == "negative", h]
            return pos.mean() - neg.mean() if len(pos) and len(neg) else np.nan

        pn = sub[sub["bucket"] != "neutral"]
        if (pn["bucket"] == "positive").sum() > 2 and (pn["bucket"] == "negative").sum() > 2:
            rows.append({"outcome": h, "group": "positive minus negative", "n_events": len(pn),
                         **_summarise(spread(pn), _date_cluster_bootstrap(pn, spread, n_boot, seed))})
    return pd.DataFrame(rows)


def mean_car_paths(panel: pd.DataFrame, threshold: float = 0.3) -> pd.DataFrame:
    """Mean cumulative abnormal return by event day for positive / negative sentiment days."""
    cols = sorted([c for c in panel.columns if c.startswith("car_path_")], key=lambda c: int(c.split("_")[-1]))
    out = []
    for name, mask in [("positive", panel["sentiment"] > threshold), ("negative", panel["sentiment"] < -threshold)]:
        means = panel.loc[mask, cols].mean()
        out.extend({"group": name, "day": int(c.split("_")[-1]), "mean_car": float(v), "n_events": int(mask.sum())}
                   for c, v in means.items())
    return pd.DataFrame(out)
