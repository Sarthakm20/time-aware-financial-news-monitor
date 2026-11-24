import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.api import create_app
from src.explorer import load_scored
from src.service import MAX_HEADLINES_PER_REQUEST, Prediction, SentimentService


class FakeService(SentimentService):
    """Positive if the headline contains 'beat', negative if 'miss', else neutral."""

    def __init__(self):
        super().__init__(model_source="fake-model", backend="auto")  # independent of any local .env
        self.calls = 0

    @property
    def loaded(self):
        return True

    def predict(self, headlines, ticker=None):
        self.calls += 1
        if len(headlines) > MAX_HEADLINES_PER_REQUEST:
            raise ValueError("too many")
        out = []
        for h in headlines:
            label = "positive" if "beat" in h.lower() else "negative" if "miss" in h.lower() else "neutral"
            probs = {k: (0.9 if k == label else 0.05) for k in ["negative", "neutral", "positive"]}
            out.append(Prediction(h, label, probs, probs["positive"] - probs["negative"], 0.9,
                                  relevant=("tesla" in h.lower()) if ticker else None))
        return out


@pytest.fixture()
def scored_csv(tmp_path):
    rows = []
    for t, base in [("TSLA", 0.3), ("AAPL", -0.2)]:
        for i, d in enumerate(pd.date_range("2023-01-01", periods=60, freq="3D", tz="UTC")):
            score = base + (0.1 if i % 2 else -0.1)
            rows.append({"Date": d, "Stock_symbol": t, "Article_title": f"{t} headline {i}",
                         "predicted_label": "positive" if score > 0.25 else "negative" if score < -0.25 else "neutral",
                         "sentiment_score": score, "confidence": 0.8})
    path = tmp_path / "scored.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    load_scored.cache_clear()
    return path


@pytest.fixture()
def client(scored_csv):
    return TestClient(create_app(service=FakeService(), scored_csv=scored_csv))


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model": "fake-model", "backend": "torch", "model_loaded": True}


def test_predict_returns_scores_and_relevance(client):
    r = client.post("/predict", json={"headlines": ["Tesla deliveries beat estimates", "Intel results miss"],
                                      "ticker": "TSLA"})
    assert r.status_code == 200
    body = r.json()
    assert body["labels"] == ["negative", "neutral", "positive"]
    first, second = body["predictions"]
    assert first["label"] == "positive" and first["relevant"] is True
    assert second["label"] == "negative" and second["relevant"] is False
    assert first["sentiment_score"] == pytest.approx(0.85)
    assert "not a trading signal" in body["disclaimer"]


@pytest.mark.parametrize("payload", [
    {"headlines": []},                                          # empty list
    {"headlines": ["x" * 501]},                                 # too long
    {"headlines": ["ok"] * (MAX_HEADLINES_PER_REQUEST + 1)},    # too many
    {"nope": ["a"]},                                            # wrong field
])
def test_predict_validates_input(client, payload):
    assert client.post("/predict", json=payload).status_code == 422


def test_tickers_and_sentiment(client):
    tickers = client.get("/tickers").json()["tickers"]
    assert {t["ticker"] for t in tickers} == {"TSLA", "AAPL"}
    assert all(t["headlines"] == 60 for t in tickers)

    r = client.get("/tickers/tsla/sentiment", params={"n_headlines": 3})
    assert r.status_code == 200
    body = r.json()
    assert body["ticker"] == "TSLA"
    assert [w["window"] for w in body["windows"]][-1] == "All available history"
    assert len(body["latest_headlines"]) == 3
    assert body["latest_headlines"][0]["headline"] == "TSLA headline 59"


def test_timeline_is_weekly_and_bounded(client):
    body = client.get("/tickers/AAPL/timeline", params={"window_days": 30}).json()
    assert body["window_days"] == 30
    dates = pd.to_datetime([p["date"] for p in body["points"]])
    assert len(dates) > 10 and (dates.to_series().diff().dropna() == pd.Timedelta(days=7)).all()
    assert all(-1 <= p["rolling_sentiment"] <= 1 for p in body["points"])


def test_unknown_ticker_and_bad_params(client):
    assert client.get("/tickers/ZZZZ/sentiment").status_code == 404
    assert client.get("/tickers/ZZZZ/timeline").status_code == 404
    assert client.get("/tickers/TSLA/sentiment", params={"half_life_days": 0}).status_code == 422
    assert client.get("/tickers/TSLA/timeline", params={"window_days": 1000}).status_code == 422


def test_missing_precomputed_scores_gives_503(tmp_path):
    load_scored.cache_clear()
    c = TestClient(create_app(service=FakeService(), scored_csv=tmp_path / "missing.csv"))
    assert c.get("/tickers").status_code == 503
    assert c.get("/health").status_code == 200


def test_service_rejects_bad_batches():
    svc = SentimentService(model_source="unused")
    with pytest.raises(ValueError):
        svc.predict([""])
    with pytest.raises(ValueError):
        svc.predict(["ok"] * (MAX_HEADLINES_PER_REQUEST + 1))
    assert svc.predict([]) == []
    assert not svc.loaded  # validation happens before the model is loaded


def test_model_display_name_never_exposes_local_paths(tmp_path):
    from src.service import model_display_name

    local = tmp_path / "distilbert_headline_adapted"
    local.mkdir()
    assert model_display_name(str(local)) == "distilbert_headline_adapted"
    assert model_display_name("someone/finbert") == "someone/finbert"


def test_live_endpoint(scored_csv, tmp_path):
    from src.live_feed import LiveStore

    db = tmp_path / "live" / "news.db"
    missing = TestClient(create_app(service=FakeService(), scored_csv=scored_csv, live_db=db))
    assert missing.get("/live/AAPL").status_code == 503

    store = LiveStore(db)
    now = pd.Timestamp.now(tz="UTC")
    news = pd.DataFrame([
        {"Published_at": now - pd.Timedelta(hours=2), "Article_title": "Apple beats estimates", "Stock_symbol": "AAPL",
         "Url": "https://a", "Publisher": "Reuters", "Source": "google"},
        {"Published_at": now - pd.Timedelta(days=30), "Article_title": "Apple old news", "Stock_symbol": "AAPL",
         "Url": "https://b", "Publisher": "CNBC", "Source": "yahoo"},
    ])
    store.insert_new(news, relevant=[True, True])
    todo = store.unscored()
    store.save_scores(todo["id"].tolist(), ["positive", "negative"], [[0.1, 0.1, 0.8], [0.8, 0.1, 0.1]],
                      [0.8, 0.8], [0.7, -0.7], model="m")

    c = TestClient(create_app(service=FakeService(), scored_csv=scored_csv, live_db=db))
    body = c.get("/live/aapl?days=7").json()
    assert body["headlines_in_window"] == 1 and body["positive_share"] == 1.0
    assert body["headlines"][0]["headline"] == "Apple beats estimates" and body["headlines"][0]["source"] == "google"
    assert "investment advice" in body["disclaimer"]
    assert c.get("/live/MSFT").status_code == 404
