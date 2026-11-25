import json

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.config import METRICS_DIR
from src.explorer import load_scored
from src.service import Prediction, SentimentService

pytestmark = pytest.mark.skipif(not (METRICS_DIR / "event_study_buckets.csv").exists(),
                                reason="committed metrics not available")


class FakeService(SentimentService):
    def __init__(self):
        super().__init__(model_source="fake-model", backend="auto")

    @property
    def loaded(self):
        return True

    def predict(self, headlines, ticker=None):
        probs = {"negative": 0.05, "neutral": 0.9, "positive": 0.05}
        return [Prediction(h, "neutral", probs, 0.0, 0.9) for h in headlines]


def _label(score):
    return "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral"


@pytest.fixture()
def scored_csv(tmp_path):
    rows = []
    for t, base in [("TSLA", 0.3), ("AAPL", -0.2)]:
        for i, d in enumerate(pd.date_range("2023-01-01", periods=60, freq="3D", tz="UTC")):
            score = base + (0.1 if i % 2 else -0.1)
            rows.append({"Date": d, "Stock_symbol": t, "Article_title": f"{t} headline {i}",
                         "predicted_label": _label(score), "sentiment_score": score, "confidence": 0.8})
    for i, d in enumerate(pd.date_range("2019-01-01", periods=5, freq="40D", tz="UTC")):
        rows.append({"Date": d, "Stock_symbol": "OLDCO", "Article_title": f"Oldco update {i}",
                     "predicted_label": "neutral", "sentiment_score": 0.05, "confidence": 0.7})
    path = tmp_path / "scored.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    load_scored.cache_clear()
    return path


@pytest.fixture()
def web_dir(tmp_path):
    web = tmp_path / "web"
    (web / "static" / "js").mkdir(parents=True)
    (web / "index.html").write_text("<!doctype html><html><body>shell</body></html>", encoding="utf-8")
    (web / "static" / "js" / "main.js").write_text("export {};", encoding="utf-8")
    return web


@pytest.fixture()
def client(scored_csv, web_dir, tmp_path):
    return TestClient(create_app(service=FakeService(), scored_paths={"adapted": scored_csv, "base": scored_csv},
                                 metrics_dir=METRICS_DIR, live_db=tmp_path / "none.db", web_dir=web_dir,
                                 price_fetcher=lambda *a: []))


def test_meta_and_overview(client):
    meta = client.get("/api/meta").json()
    assert [m["id"] for m in meta["models"]] == ["adapted", "base"] and meta["years"] == [2019, 2020, 2021, 2022, 2023]
    assert [t["ticker"] for t in meta["tickers"]] == ["AAPL", "OLDCO", "TSLA"]
    assert meta["corpus"]["collected_headlines"] == 61818
    ov = client.get("/api/overview").json()
    assert ov["cards"]["company_days"] == {"value": 24807, "scope": "All companies · 2019–2023"}
    assert ov["cards"]["coverage"] == {"kind": "universe", "companies": 3}
    assert ov["cards"]["headlines"] == {"scored": 125, "collected": 61818}
    assert ov["market"]["events"] == 22999 and ov["trend"] and ov["distribution"]["n"] == 125
    assert ov["relevance"]["precision_max"] == 1.0 and ov["cards"]["f1"]["scope"] == "All companies · 2019–2023"


def test_overview_for_company_and_empty_year(client):
    ov = client.get("/api/overview?ticker=oldco&period=2023").json()
    assert ov["cards"]["headlines"]["scored"] == 0 and ov["trend"] == [] and ov["distribution"]["n"] == 0
    assert ov["cards"]["coverage"] == {"kind": "company", "ticker": "OLDCO", "first_date": "2019-01-01",
                                       "last_date": "2019-06-10"}
    assert ov["cards"]["headlines"]["collected"] is None and ov["mean_sentiment"] is None
    assert client.get("/api/overview?ticker=TSLA").json()["cards"]["f1"]["scope"] == "TSLA"


def test_unknown_ticker_and_bad_params(client):
    r = client.get("/api/company/ZZZZ")
    assert r.status_code == 404 and r.json()["detail"] == "No company called ZZZZ in the dataset."
    assert client.get("/api/overview?ticker=ZZZZ").status_code == 404
    assert client.get("/api/overview?period=2018").status_code == 422
    assert client.get("/api/overview?model=finbert").status_code == 422
    assert client.get("/api/headlines?page_size=101").status_code == 422
    assert client.get("/api/headlines?start=not-a-date").status_code == 422
    far = client.get("/api/headlines?page=999").json()
    assert far["rows"] == [] and far["total"] == 125


def test_headlines_search(client):
    body = client.get("/api/headlines", params={"q": "S&P 500 (QQQ)?", "ticker": "tsla"}).json()
    assert body["total"] == 0
    body = client.get("/api/headlines", params={"label": "negative", "sort": "most_negative", "page_size": 5}).json()
    assert body["total"] == 30 and len(body["rows"]) == 5 and body["rows"][0]["ticker"] == "AAPL"


def test_company_payload_is_valid_json(client):
    r = client.get("/api/company/OLDCO?period=2023")
    assert r.status_code == 200 and json.loads(r.text)
    assert r.json()["accuracy"] is None and r.json()["cards"]["recent_30d"] is None
    tsla = client.get("/api/company/tsla?window_days=14&order=most_positive&n=3").json()
    assert tsla["ticker"] == "TSLA" and tsla["accuracy"]["scope"] == "TSLA" and len(tsla["headlines"]) == 3
    assert "TSLA" in tsla["accuracy_available_for"] and len(tsla["accuracy_available_for"]) == 6
    assert client.get("/api/company/TSLA?window_days=500").status_code == 422


def test_market_model_price_live(client):
    assert client.get("/api/market-impact?headlines=all").json()["variant"] == "all relevant headlines"
    assert client.get("/api/market-impact").json()["car_paths_available"] is True
    assert client.get("/api/model-evaluation").json()["calibration_ece"] is not None
    assert client.get("/api/price/tsla?start=2023-01-01&end=2023-02-01").json() == {"ticker": "TSLA", "points": []}
    assert client.get("/api/live/summary").json()["available"] is False


def test_missing_metrics_gives_503(scored_csv, tmp_path):
    c = TestClient(create_app(service=FakeService(), scored_paths={"adapted": scored_csv, "base": scored_csv},
                              metrics_dir=tmp_path / "empty", web_dir=tmp_path / "noweb"))
    r = c.get("/api/market-impact")
    assert r.status_code == 503 and "not available on this server" in r.json()["detail"]
    assert c.get("/api/overview").status_code == 503
    assert c.get("/").status_code == 404


def test_missing_scores_gives_503(tmp_path):
    c = TestClient(create_app(service=FakeService(), scored_paths={"adapted": tmp_path / "x.csv",
                                                                   "base": tmp_path / "y.csv"},
                              metrics_dir=METRICS_DIR, web_dir=tmp_path / "noweb"))
    assert c.get("/api/headlines").status_code == 503


def test_site_served_and_api_not_shadowed(client):
    assert "<html" in client.get("/").text.lower()
    assert client.get("/static/js/main.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok" and client.get("/tickers").status_code == 200
