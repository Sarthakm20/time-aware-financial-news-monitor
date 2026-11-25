import json

import pandas as pd
import pytest

from src.dashboard_live import fetch_close, live_summary
from src.live_feed import LiveStore

NOW = pd.Timestamp("2026-11-24 12:00", tz="UTC")


@pytest.fixture()
def live_db(tmp_path):
    db = tmp_path / "live" / "news.db"
    store = LiveStore(db)
    news = pd.DataFrame([
        {"Published_at": NOW - pd.Timedelta(hours=2), "Article_title": "Apple beats estimates", "Stock_symbol": "AAPL",
         "Url": "https://a", "Publisher": "Reuters", "Source": "google"},
        {"Published_at": NOW - pd.Timedelta(days=30), "Article_title": "Apple old news", "Stock_symbol": "AAPL",
         "Url": "https://b", "Publisher": "CNBC", "Source": "yahoo"},
        {"Published_at": NOW - pd.Timedelta(days=1), "Article_title": "Nvidia misses", "Stock_symbol": "NVDA",
         "Url": "https://c", "Publisher": "AP", "Source": "google"},
    ])
    store.insert_new(news, relevant=[True, True, True])
    todo = store.unscored()
    store.save_scores(todo["id"].tolist(), ["positive", "negative", "negative"],
                      [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1], [0.7, 0.2, 0.1]], [0.8, 0.8, 0.7], [0.7, -0.7, -0.6],
                      model="m")
    store.log_run(started_at="2026-11-24T11:17:48Z", tickers=89, sources=["google", "yahoo"], fetched=3,
                  inserted=3, scored=3, errors=[])
    pd.DataFrame([{"ticker": "AAPL", "alerts": "", "message": ""},
                  {"ticker": "NVDA", "alerts": "sentiment", "message": "NVDA: 7-day sentiment z = -2.4"}]
                 ).to_csv(db.parent / "alerts_latest.csv", index=False)
    (db.parent / "live_agreement.json").write_text(json.dumps(
        {"annotator": "gemini:x", "n_on_topic": 50, "accuracy": {"value": 0.78, "ci_low": 0.66, "ci_high": 0.88},
         "macro_f1": {"value": 0.78, "ci_low": 0.66, "ci_high": 0.88}}), encoding="utf-8")
    return db


def test_live_summary_without_db(tmp_path):
    missing = tmp_path / "none.db"
    assert live_summary(missing) == {"available": False, "command": "python -m src.pipeline live-update",
                                     "docs": "docs/deployment.md"}
    assert not missing.exists()  # reading must never create the database


def test_live_summary_with_ticker_and_without_data(live_db):
    s = live_summary(live_db, "aapl", now=NOW)
    assert s["available"] and s["ticker"] == "AAPL" and s["ticker_has_data"]
    assert s["cards"]["headlines_7d"] == 1 and s["cards"]["mean_7d"] == pytest.approx(0.7)
    assert s["cards"]["delta_vs_all"] == pytest.approx(0.7)  # 7-day mean 0.7 vs all-live mean 0.0
    assert s["cards"]["history_days"] == 30
    assert s["last_run_utc"] == "2026-11-24 11:17:48" and s["runs"] == 1 and s["sources"] == "google,yahoo"
    assert s["total_headlines"] == 3 and s["tickers"] == 2
    assert s["headlines"][0]["headline"] == "Apple beats estimates" and s["headlines"][0]["url"] == "https://a"
    assert [d["count"] for d in s["daily"]] == [1, 1]
    assert s["alerts"] == ["NVDA: 7-day sentiment z = -2.4"]
    assert s["agreement"]["macro_f1"]["value"] == 0.78
    assert json.dumps(s, allow_nan=False)

    other = live_summary(live_db, "MSFT", now=NOW)
    assert other["ticker_has_data"] is False and other["daily"] == [] and other["headlines"] == []
    assert json.dumps(other, allow_nan=False)


def test_live_summary_all_companies(live_db):
    s = live_summary(live_db, None, now=NOW)
    assert s["ticker"] is None and s["ticker_has_data"] and len(s["headlines"]) == 3
    assert s["cards"]["headlines_7d"] == 2


def test_fetch_close_never_raises(monkeypatch):
    import yfinance

    def boom(*args, **kwargs):
        raise RuntimeError("offline")

    monkeypatch.setattr(yfinance, "download", boom)
    fetch_close.cache_clear()
    assert fetch_close("TSLA", "2023-01-01", "2023-02-01") == []


def test_fetch_close_returns_points(monkeypatch):
    import yfinance

    frame = pd.DataFrame({"Close": [100.0, 101.5]}, index=pd.to_datetime(["2023-01-03", "2023-01-04"]))
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: frame)
    fetch_close.cache_clear()
    assert fetch_close("TSLA", "2023-01-01", "2023-01-05") == [{"date": "2023-01-03", "close": 100.0},
                                                               {"date": "2023-01-04", "close": 101.5}]
