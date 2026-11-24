"""Live headlines: fetch from free sources, de-duplicate, and keep them in a local SQLite file.

Sources (``--live-sources``):

- ``google``: Google News RSS search for the company name. No key. Has publish times.
- ``yahoo``: Yahoo Finance ticker news through ``yfinance``. No key. About 10 recent items.
- ``finnhub``: Finnhub company-news. Needs ``FINNHUB_API_KEY``.
- ``gemini``: Gemini with Google Search grounding. Needs ``GEMINI_API_KEY``. Opt-in, because an
  LLM can paraphrase a title (see ``src.gemini.search_headlines``).

Rows use the FNSPID column names (``Date, Article_title, Stock_symbol, Url, Publisher``), so the
relevance filter, scoring and aggregation code is reused unchanged. Unlike FNSPID they also have
``Published_at``, a real UTC publish time.
"""

import email.utils
import re
import sqlite3
import time
import xml.etree.ElementTree as ET
from contextlib import closing
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence
from urllib.parse import quote_plus

import pandas as pd

from src.company_names import COMPANY_TERMS
from src.config import DATA_DIR

LIVE_DIR = DATA_DIR / "live"
LIVE_DB = LIVE_DIR / "news.db"
DEFAULT_SOURCES = ("google", "yahoo")
USER_AGENT = "Mozilla/5.0 (compatible; financial-news-monitor/1.0; research prototype)"
FEED_COLUMNS = ["Published_at", "Article_title", "Stock_symbol", "Url", "Publisher", "Source"]

LIVE_TICKERS: List[str] = sorted(COMPANY_TERMS)


def company_name(ticker: str) -> str:
    terms = COMPANY_TERMS.get(ticker.upper())
    return terms[0] if terms else ticker.upper()


def title_key(title: str) -> str:
    """Normalised title for de-duplication across sources (case, punctuation, whitespace)."""
    return re.sub(r"[^a-z0-9]+", " ", str(title).lower()).strip()


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=FEED_COLUMNS)


def _http_get(url: str, session=None, timeout: float = 20.0, headers: Optional[dict] = None,
              retry_waits: Sequence[float] = (5.0, 20.0), **kwargs):
    """GET with a polite back-off on 429/503 (free feeds rate-limit bursts)."""
    import requests

    for wait in (*retry_waits, None):
        resp = (session or requests).get(url, timeout=timeout, headers={"User-Agent": USER_AGENT, **(headers or {})},
                                         **kwargs)
        if resp.status_code not in (429, 503) or wait is None:
            break
        time.sleep(wait)
    resp.raise_for_status()
    return resp


# --------------------------------------------------------------------------- Google News RSS


def google_news_url(query: str, days: int) -> str:
    return (f"https://news.google.com/rss/search?q={quote_plus(f'{query} when:{max(1, int(days))}d')}"
            "&hl=en-US&gl=US&ceid=US:en")


def parse_google_news_rss(xml_text: str, ticker: str) -> pd.DataFrame:
    root = ET.fromstring(xml_text)
    rows = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        source = (item.findtext("source") or "").strip()
        if source and title.endswith(f" - {source}"):
            title = title[: -len(f" - {source}")].rstrip()
        pub = item.findtext("pubDate")
        try:
            published = email.utils.parsedate_to_datetime(pub).astimezone(timezone.utc) if pub else None
        except (TypeError, ValueError):
            published = None
        if not title or published is None:
            continue
        rows.append({"Published_at": pd.Timestamp(published), "Article_title": title, "Stock_symbol": ticker,
                     "Url": (item.findtext("link") or "").strip(), "Publisher": source, "Source": "google"})
    return pd.DataFrame(rows, columns=FEED_COLUMNS)


def fetch_google_news(ticker: str, days: int = 2, session=None) -> pd.DataFrame:
    query = f'"{company_name(ticker)}" stock'
    return parse_google_news_rss(_http_get(google_news_url(query, days), session).text, ticker)


# --------------------------------------------------------------------------- Yahoo Finance (yfinance)


def parse_yahoo_news(items: Iterable[dict], ticker: str) -> pd.DataFrame:
    """Handles both yfinance layouts: flat (``title``, ``providerPublishTime``) and nested (``content``)."""
    rows = []
    for raw in items or []:
        item = raw.get("content", raw) if isinstance(raw, dict) else {}
        title = str(item.get("title", "")).strip()
        if "providerPublishTime" in item:
            published = pd.to_datetime(item["providerPublishTime"], unit="s", utc=True, errors="coerce")
        else:
            published = pd.to_datetime(item.get("pubDate") or item.get("displayTime"), utc=True, errors="coerce")
        provider = item.get("provider")
        publisher = provider.get("displayName", "") if isinstance(provider, dict) else item.get("publisher", "")
        link = item.get("link") or (item.get("canonicalUrl") or {}).get("url", "") or \
            (item.get("clickThroughUrl") or {}).get("url", "")
        if title and not pd.isna(published):
            rows.append({"Published_at": published, "Article_title": title, "Stock_symbol": ticker,
                         "Url": link or "", "Publisher": publisher or "", "Source": "yahoo"})
    return pd.DataFrame(rows, columns=FEED_COLUMNS)


def fetch_yahoo(ticker: str, days: int = 2, session=None) -> pd.DataFrame:
    import yfinance as yf

    return parse_yahoo_news(yf.Ticker(ticker).news, ticker)


# --------------------------------------------------------------------------- Finnhub (optional key)


def fetch_finnhub(ticker: str, days: int = 2, session=None) -> pd.DataFrame:
    from src.env import get_secret

    key = get_secret("FINNHUB_API_KEY")
    if not key:
        raise RuntimeError("FINNHUB_API_KEY is not set (see .env.example)")
    today = datetime.now(timezone.utc).date()
    params = {"symbol": ticker, "from": str(today - timedelta(days=days)), "to": str(today)}
    data = _http_get("https://finnhub.io/api/v1/company-news", session, params=params,
                     headers={"X-Finnhub-Token": key}).json()
    rows = [{"Published_at": pd.to_datetime(d.get("datetime"), unit="s", utc=True),
             "Article_title": d.get("headline", ""), "Stock_symbol": ticker, "Url": d.get("url", ""),
             "Publisher": d.get("source", ""), "Source": "finnhub"}
            for d in data if isinstance(d, dict) and d.get("headline")]
    return pd.DataFrame(rows, columns=FEED_COLUMNS)


# --------------------------------------------------------------------------- Gemini (optional key)

_GEMINI_CLIENT = None


def fetch_gemini(ticker: str, days: int = 2, session=None) -> pd.DataFrame:
    global _GEMINI_CLIENT
    from src.gemini import GeminiClient, search_headlines

    if _GEMINI_CLIENT is None:
        _GEMINI_CLIENT = GeminiClient()
    return search_headlines(_GEMINI_CLIENT, ticker, company_name(ticker), days=days).reindex(columns=FEED_COLUMNS)


SOURCES: Dict[str, Callable[..., pd.DataFrame]] = {
    "google": fetch_google_news,
    "yahoo": fetch_yahoo,
    "finnhub": fetch_finnhub,
    "gemini": fetch_gemini,
}


def fetch_company_news(ticker: str, days: int = 2, sources: Sequence[str] = DEFAULT_SOURCES,
                       errors: Optional[List[str]] = None, now: Optional[datetime] = None) -> pd.DataFrame:
    """Headlines from every source, newest first, de-duplicated on the normalised title (first source wins)."""
    now = now or datetime.now(timezone.utc)
    frames = []
    for name in sources:
        try:
            frames.append(SOURCES[name](ticker, days=days))
        except Exception as exc:  # one failing source must not stop the run
            if errors is not None:
                errors.append(f"{ticker}/{name}: {type(exc).__name__}: {str(exc)[:150]}")
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        return _empty()
    df = pd.concat(frames, ignore_index=True)
    df["Published_at"] = pd.to_datetime(df["Published_at"], utc=True)
    df = df[(df["Published_at"] >= now - timedelta(days=days)) & (df["Published_at"] <= now + timedelta(hours=1))]
    df["title_key"] = df["Article_title"].map(title_key)
    df = df[df["title_key"] != ""].drop_duplicates("title_key", keep="first")
    return df.sort_values("Published_at", ascending=False).drop(columns="title_key").reset_index(drop=True)


def fetch_many(tickers: Sequence[str], days: int = 2, sources: Sequence[str] = DEFAULT_SOURCES,
               pause: float = 0.5, errors: Optional[List[str]] = None) -> pd.DataFrame:
    frames = []
    for i, ticker in enumerate(tickers):
        frames.append(fetch_company_news(ticker, days=days, sources=sources, errors=errors))
        if pause and i < len(tickers) - 1:
            time.sleep(pause)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else _empty()


# --------------------------------------------------------------------------- storage

SCHEMA = """
CREATE TABLE IF NOT EXISTS headlines (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    title TEXT NOT NULL,
    title_key TEXT NOT NULL,
    published_at TEXT NOT NULL,
    publisher TEXT,
    url TEXT,
    source TEXT,
    fetched_at TEXT NOT NULL,
    relevant INTEGER,
    predicted_label TEXT,
    negative_probability REAL,
    neutral_probability REAL,
    positive_probability REAL,
    confidence REAL,
    sentiment_score REAL,
    model TEXT,
    UNIQUE (ticker, title_key)
);
CREATE INDEX IF NOT EXISTS idx_headlines_ticker_time ON headlines (ticker, published_at);
CREATE TABLE IF NOT EXISTS annotations (
    headline_id INTEGER NOT NULL REFERENCES headlines (id),
    annotator TEXT NOT NULL,
    label TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (headline_id, annotator)
);
CREATE TABLE IF NOT EXISTS runs (
    started_at TEXT NOT NULL,
    finished_at TEXT,
    tickers INTEGER,
    sources TEXT,
    fetched INTEGER,
    inserted INTEGER,
    scored INTEGER,
    errors TEXT
);
"""


def _iso(ts) -> str:
    return pd.Timestamp(ts).tz_convert("UTC").strftime("%Y-%m-%dT%H:%M:%SZ")


class LiveStore:
    """SQLite file of live headlines. A headline is unique per (ticker, normalised title)."""

    def __init__(self, path: Path = LIVE_DB):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.executescript(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)

    def insert_new(self, df: pd.DataFrame, relevant: Optional[Sequence[bool]] = None) -> int:
        if df.empty:
            return 0
        now = _iso(pd.Timestamp.now(tz="UTC"))
        rel = list(relevant) if relevant is not None else [None] * len(df)
        rows = [(r.Stock_symbol.upper(), r.Article_title, title_key(r.Article_title), _iso(r.Published_at),
                 r.Publisher or "", r.Url or "", r.Source or "", now, None if f is None else int(bool(f)))
                for r, f in zip(df.itertuples(index=False), rel)]
        with closing(self._connect()) as con, con:
            before = con.total_changes
            con.executemany(
                "INSERT OR IGNORE INTO headlines (ticker, title, title_key, published_at, publisher, url, source, "
                "fetched_at, relevant) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
            return con.total_changes - before

    def unscored(self, relevant_only: bool = True) -> pd.DataFrame:
        where = "predicted_label IS NULL" + (" AND relevant = 1" if relevant_only else "")
        with closing(self._connect()) as con:
            return pd.read_sql_query(f"SELECT id, ticker, title FROM headlines WHERE {where}", con)

    def save_scores(self, ids: Sequence[int], labels, probs, confidences, scores, model: str) -> None:
        rows = [(lab, float(p[0]), float(p[1]), float(p[2]), float(c), float(s), model, int(i))
                for i, lab, p, c, s in zip(ids, labels, probs, confidences, scores)]
        with closing(self._connect()) as con, con:
            con.executemany(
                "UPDATE headlines SET predicted_label = ?, negative_probability = ?, neutral_probability = ?, "
                "positive_probability = ?, confidence = ?, sentiment_score = ?, model = ? WHERE id = ?", rows)

    def log_run(self, started_at, tickers: int, sources: Sequence[str], fetched: int, inserted: int, scored: int,
                errors: Sequence[str]) -> None:
        with closing(self._connect()) as con, con:
            con.execute("INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (_iso(started_at), _iso(pd.Timestamp.now(tz="UTC")), tickers, ",".join(sources), fetched,
                         inserted, scored, "\n".join(errors)[:20000]))

    def save_annotations(self, labels: Dict[int, str], annotator: str) -> int:
        now = _iso(pd.Timestamp.now(tz="UTC"))
        with closing(self._connect()) as con, con:
            before = con.total_changes
            con.executemany("INSERT OR REPLACE INTO annotations VALUES (?, ?, ?, ?)",
                            [(int(i), annotator, lab, now) for i, lab in labels.items()])
            return con.total_changes - before

    def unannotated(self, annotator_prefix: str, limit: int, seed: int = 42) -> pd.DataFrame:
        """A random sample of scored, relevant headlines no annotator starting with ``annotator_prefix`` has
        labelled. Only id, ticker and title are returned, so the annotator never sees the model's output."""
        with closing(self._connect()) as con:
            df = pd.read_sql_query(
                "SELECT h.id, h.ticker, h.title FROM headlines h WHERE h.relevant = 1 AND h.predicted_label IS NOT "
                "NULL AND NOT EXISTS (SELECT 1 FROM annotations a WHERE a.headline_id = h.id AND a.annotator LIKE ?)",
                con, params=(annotator_prefix.replace("%", "") + "%",))
        return df.sample(n=min(limit, len(df)), random_state=seed) if len(df) else df

    def annotations(self) -> pd.DataFrame:
        with closing(self._connect()) as con:
            return pd.read_sql_query(
                "SELECT a.headline_id, a.annotator, a.label, a.created_at, h.ticker, h.title, h.published_at, "
                "h.predicted_label FROM annotations a JOIN headlines h ON h.id = a.headline_id", con)

    def runs(self) -> pd.DataFrame:
        with closing(self._connect()) as con:
            return pd.read_sql_query("SELECT * FROM runs ORDER BY started_at DESC, rowid DESC", con)

    def scored_frame(self, ticker: Optional[str] = None, relevant_only: bool = True) -> pd.DataFrame:
        """Scored headlines in the FNSPID/explorer layout (``Date`` = publish time, UTC)."""
        where = ["predicted_label IS NOT NULL"] + (["relevant = 1"] if relevant_only else [])
        params: tuple = ()
        if ticker:
            where.append("ticker = ?")
            params = (ticker.upper(),)
        with closing(self._connect()) as con:
            df = pd.read_sql_query(f"SELECT * FROM headlines WHERE {' AND '.join(where)}", con, params=params)
        df["Date"] = pd.to_datetime(df["published_at"], utc=True)
        df = df.rename(columns={"ticker": "Stock_symbol", "title": "Article_title", "publisher": "Publisher",
                                "url": "Url", "source": "Source"})
        return df.sort_values(["Stock_symbol", "Date"], kind="stable").reset_index(drop=True)
