"""Dashboard payloads for the live news feed (data/live/news.db) and Yahoo share prices."""

from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from src.dashboard_data import clean

LIVE_COMMAND = "python -m src.pipeline live-update"
MAX_LIVE_HEADLINES = 50


def _unavailable() -> Dict[str, Any]:
    return {"available": False, "command": LIVE_COMMAND, "docs": "docs/deployment.md"}


def _alerts(folder: Path) -> List[str]:
    path = folder / "alerts_latest.csv"
    if not path.exists():
        return []
    fired = pd.read_csv(path).fillna({"alerts": "", "message": ""})
    return [m for m in fired.loc[fired["alerts"] != "", "message"] if m]


def _agreement(folder: Path) -> Optional[Dict[str, Any]]:
    import json

    path = folder / "live_agreement.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return data if "macro_f1" in data else None


def live_summary(live_db: Path, ticker: Optional[str] = None, now: Optional[pd.Timestamp] = None) -> Dict[str, Any]:
    live_db = Path(live_db)
    if not live_db.exists():  # never construct LiveStore here: it would create an empty database
        return _unavailable()
    from src.live_feed import LiveStore

    store = LiveStore(live_db)
    live, runs = store.scored_frame(), store.runs()
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    symbol = ticker.upper() if ticker else None
    sub = live[live["Stock_symbol"] == symbol] if symbol else live

    out: Dict[str, Any] = {
        "available": True,
        "last_run_utc": runs["started_at"].iloc[0].replace("T", " ").rstrip("Z") if len(runs) else None,
        "runs": int(len(runs)),
        "sources": runs["sources"].iloc[0] if len(runs) else None,
        "total_headlines": int(len(live)),
        "tickers": int(live["Stock_symbol"].nunique()),
        "ticker": symbol,
        "ticker_has_data": not sub.empty,
        "alerts": _alerts(live_db.parent),
        "agreement": _agreement(live_db.parent),
    }
    if sub.empty:
        out.update(cards={"headlines_7d": 0, "mean_7d": None, "delta_vs_all": None, "positive_share": None,
                          "negative_share": None, "history_days": None}, daily=[], headlines=[])
        return clean(out)

    last7 = sub[sub["Date"] > now - pd.Timedelta(days=7)]
    mean_7d = float(last7["sentiment_score"].mean()) if len(last7) else None
    daily = sub.set_index("Date").resample("D")["sentiment_score"].agg(["mean", "count"])
    daily = daily[daily["count"] > 0]
    latest = sub.sort_values("Date", ascending=False).head(MAX_LIVE_HEADLINES)
    out.update(
        cards={
            "headlines_7d": int(len(last7)),
            "mean_7d": mean_7d,
            "delta_vs_all": (mean_7d - float(sub["sentiment_score"].mean())
                             if mean_7d is not None and len(sub) > len(last7) else None),
            "positive_share": float((sub["predicted_label"] == "positive").mean()),
            "negative_share": float((sub["predicted_label"] == "negative").mean()),
            "history_days": int((now - sub["Date"].min()).days),
        },
        daily=[{"day": str(d.date()), "mean": round(float(r["mean"]), 4), "count": int(r["count"])}
               for d, r in daily.iterrows()],
        headlines=[{"published_at": r.Date.isoformat(), "headline": r.Article_title, "label": r.predicted_label,
                    "sentiment_score": r.sentiment_score, "publisher": r.Publisher, "source": r.Source, "url": r.Url}
                   for r in latest.itertuples(index=False)],
    )
    return clean(out)


@lru_cache(maxsize=256)
def _fetch_close_cached(ticker: str, start: str, end: str, day: str) -> tuple:
    try:
        import yfinance as yf  # not installed in the API-only image; imported only when prices are asked for

        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if px is None or px.empty:
            return ()
        close = px["Close"].squeeze()
        return tuple({"date": str(pd.Timestamp(d).date()), "close": round(float(c), 4)}
                     for d, c in close.items() if pd.notna(c))
    except Exception:  # noqa: BLE001 - an optional overlay must never break the page
        return ()


def fetch_close(ticker: str, start: str, end: str) -> List[Dict[str, Any]]:
    """Adjusted daily close from Yahoo Finance; [] when unavailable (delisted, offline, ...). Cached per day."""
    return [dict(p) for p in _fetch_close_cached(ticker.upper(), start, end, date.today().isoformat())]


fetch_close.cache_clear = _fetch_close_cached.cache_clear
