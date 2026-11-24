"""Sentiment and headline-volume alerts on the live feed.

For each ticker, the last ``recent_days`` are compared with the ``baseline_days`` before them:

- **sentiment drop**: z = (recent mean - baseline mean) / (baseline sd / sqrt(n recent headlines)) <= z_threshold
- **volume spike**: recent headlines per day >= volume_ratio x baseline headlines per day

Phase 3 found that bad news comes with about 2x abnormal trading volume, which makes volume
spikes the best-supported alert. An alert is raised only once the feed has ``min_history_days``
of history, so the first days of collection cannot look like a spike.
"""

import math
from typing import List, Optional

import pandas as pd

DISCLAIMER = "Summarises headline tone; not a trading signal or investment advice."


def compute_alerts(scored: pd.DataFrame, asof: Optional[pd.Timestamp] = None, recent_days: int = 7,
                   baseline_days: int = 90, z_threshold: float = -2.0, volume_ratio: float = 3.0,
                   min_recent: int = 3, min_baseline: int = 10, min_history_days: int = 30,
                   feed_start: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """One row per ticker with the statistics and the alerts it triggers (``alerts`` is '' when none)."""
    cols = ["ticker", "recent_headlines", "baseline_headlines", "recent_mean", "baseline_mean", "baseline_sd",
            "sentiment_z", "recent_per_day", "baseline_per_day", "volume_ratio", "alerts", "message"]
    if scored.empty:
        return pd.DataFrame(columns=cols)
    asof = pd.Timestamp(asof or pd.Timestamp.now(tz="UTC"))
    asof = asof.tz_localize("UTC") if asof.tzinfo is None else asof.tz_convert("UTC")
    recent_start = asof - pd.Timedelta(days=recent_days)
    base_start = recent_start - pd.Timedelta(days=baseline_days)
    feed_start = pd.Timestamp(feed_start) if feed_start is not None else scored["Date"].min()
    feed_start = feed_start.tz_localize("UTC") if feed_start.tzinfo is None else feed_start.tz_convert("UTC")
    history_days = (asof - feed_start).total_seconds() / 86400
    baseline_span = max(min(baseline_days, (recent_start - max(base_start, feed_start)).total_seconds() / 86400), 1)

    # Volume is only comparable for days the feed was actually collected (a backfill returns a capped sample).
    volume_start = max(base_start, feed_start)

    rows = []
    for ticker, g in scored.groupby("Stock_symbol"):
        recent = g.loc[(g["Date"] > recent_start) & (g["Date"] <= asof), "sentiment_score"]
        base = g.loc[(g["Date"] > base_start) & (g["Date"] <= recent_start), "sentiment_score"]
        n_volume_base = int(((g["Date"] > volume_start) & (g["Date"] <= recent_start)).sum())
        r_mean = recent.mean() if len(recent) else float("nan")
        b_mean = base.mean() if len(base) else float("nan")
        b_sd = base.std(ddof=1) if len(base) > 1 else float("nan")
        z = float("nan")
        if len(recent) >= min_recent and len(base) >= min_baseline and b_sd and not math.isnan(b_sd):
            z = (r_mean - b_mean) / (b_sd / math.sqrt(len(recent)))
        r_rate, b_rate = len(recent) / recent_days, n_volume_base / baseline_span
        ratio = r_rate / b_rate if b_rate > 0 else float("nan")

        alerts: List[str] = []
        if history_days >= min_history_days:
            if not math.isnan(z) and z <= z_threshold:
                alerts.append("sentiment_drop")
            if len(recent) >= min_recent and n_volume_base >= min_baseline and ratio >= volume_ratio:
                alerts.append("volume_spike")
        message = ""
        if alerts:
            parts = []
            if "sentiment_drop" in alerts:
                parts.append(f"{recent_days}-day headline sentiment {r_mean:+.2f} vs {baseline_days}-day baseline "
                             f"{b_mean:+.2f} (z = {z:.1f}, {len(recent)} headlines)")
            if "volume_spike" in alerts:
                parts.append(f"{r_rate:.1f} headlines/day vs {b_rate:.1f} normally ({ratio:.1f}x)")
            message = f"{ticker}: " + "; ".join(parts) + f". {DISCLAIMER}"
        rows.append({"ticker": ticker, "recent_headlines": len(recent), "baseline_headlines": len(base),
                     "recent_mean": round(r_mean, 4), "baseline_mean": round(b_mean, 4), "baseline_sd": round(b_sd, 4),
                     "sentiment_z": round(z, 2), "recent_per_day": round(r_rate, 2),
                     "baseline_per_day": round(b_rate, 2), "volume_ratio": round(ratio, 2),
                     "alerts": ",".join(alerts), "message": message})
    return pd.DataFrame(rows, columns=cols)


def webhook_payload(url: str, text: str) -> dict:
    """Discord wants ``content``; Slack (and most others) want ``text``."""
    return {"content": text[:1990]} if "discord.com" in url or "discordapp.com" in url else {"text": text}


def post_alerts(messages: List[str], url: str, session=None) -> int:
    if not messages:
        return 0
    import requests

    text = "News sentiment alerts\n" + "\n".join(f"- {m}" for m in messages)
    resp = (session or requests).post(url, json=webhook_payload(url, text), timeout=20)
    resp.raise_for_status()
    return len(messages)
