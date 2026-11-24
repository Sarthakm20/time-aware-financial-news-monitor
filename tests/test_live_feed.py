"""Live feed: parsing, de-duplication, storage, alerts, Gemini client and .env handling (no network)."""

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

from src import live_feed
from src.alerts import compute_alerts, webhook_payload
from src.env import get_secret, read_env_file
from src.gemini import GeminiClient, GeminiError, annotate_headlines, parse_json, search_headlines
from src.live_feed import LiveStore, fetch_company_news, parse_google_news_rss, parse_yahoo_news, title_key

RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0"><channel><title>Google News</title>
<item><title>Apple Shares Rise After Record iPhone Sales - Reuters</title>
  <link>https://news.google.com/rss/articles/abc</link>
  <pubDate>Mon, 23 Nov 2026 14:30:00 GMT</pubDate><source url="https://www.reuters.com">Reuters</source></item>
<item><title>Apple faces EU fine - The Verge</title><link>https://x/2</link>
  <pubDate>Mon, 23 Nov 2026 09:00:00 -0500</pubDate><source url="https://www.theverge.com">The Verge</source></item>
<item><title>No date here - Somewhere</title><link>https://x/3</link></item>
</channel></rss>"""

NOW = datetime(2026, 11, 24, 12, 0, tzinfo=timezone.utc)


def test_parse_google_news_rss_strips_publisher_and_parses_time():
    df = parse_google_news_rss(RSS, "AAPL")
    assert df["Article_title"].tolist() == ["Apple Shares Rise After Record iPhone Sales", "Apple faces EU fine"]
    assert df["Publisher"].tolist() == ["Reuters", "The Verge"]
    assert df["Published_at"].iloc[1] == pd.Timestamp("2026-11-23 14:00", tz="UTC")  # -0500 converted to UTC
    assert set(df["Source"]) == {"google"}


def test_parse_yahoo_news_both_layouts():
    flat = [{"title": "Nvidia beats", "providerPublishTime": 1795000000, "publisher": "Motley Fool", "link": "u1"}]
    nested = [{"content": {"title": "Nvidia slips", "pubDate": "2026-11-23T10:00:00Z",
                           "provider": {"displayName": "Yahoo Finance"}, "canonicalUrl": {"url": "u2"}}}]
    df = parse_yahoo_news(flat + nested, "NVDA")
    assert df["Article_title"].tolist() == ["Nvidia beats", "Nvidia slips"]
    assert df["Publisher"].tolist() == ["Motley Fool", "Yahoo Finance"]
    assert df["Url"].tolist() == ["u1", "u2"]


def test_fetch_company_news_dedupes_across_sources_and_survives_errors(monkeypatch):
    def google(ticker, days):
        return parse_google_news_rss(RSS, ticker)

    def yahoo(ticker, days):
        return pd.DataFrame([{"Published_at": pd.Timestamp("2026-11-23 15:00", tz="UTC"),
                              "Article_title": "apple shares rise after record iPhone sales!", "Stock_symbol": ticker,
                              "Url": "y", "Publisher": "Yahoo", "Source": "yahoo"}])

    def broken(ticker, days):
        raise ConnectionError("offline")

    monkeypatch.setattr(live_feed, "SOURCES", {"google": google, "yahoo": yahoo, "broken": broken})
    errors = []
    df = fetch_company_news("AAPL", days=3, sources=["google", "yahoo", "broken"], errors=errors, now=NOW)
    assert len(df) == 2 and df["Source"].tolist().count("google") == 2  # yahoo duplicate dropped
    assert len(errors) == 1 and "broken" in errors[0]
    assert fetch_company_news("AAPL", days=3, sources=["broken"], now=NOW).empty
    assert title_key("Apple's  Q3: beats!") == "apple s q3 beats"


def test_store_dedupes_scores_and_samples(tmp_path):
    store = LiveStore(tmp_path / "news.db")
    df = parse_google_news_rss(RSS, "AAPL")
    assert store.insert_new(df, relevant=[True, False]) == 2
    assert store.insert_new(df, relevant=[True, False]) == 0  # same headlines again: ignored
    todo = store.unscored()
    assert len(todo) == 1  # only the relevant one is scored
    store.save_scores(todo["id"].tolist(), ["positive"], [[0.1, 0.2, 0.7]], [0.7], [0.6], model="m")
    frame = store.scored_frame("AAPL")
    assert frame["predicted_label"].tolist() == ["positive"]
    assert str(frame["Date"].dt.tz) == "UTC" and frame["Stock_symbol"].iloc[0] == "AAPL"

    sample = store.unannotated("gemini:x", limit=5)
    assert len(sample) == 1
    store.save_annotations({int(sample["id"].iloc[0]): "negative"}, "gemini:x")
    assert store.unannotated("gemini:x", limit=5).empty
    ann = store.annotations()
    assert ann[["label", "predicted_label"]].values.tolist() == [["negative", "positive"]]

    store.log_run(pd.Timestamp.now(tz="UTC"), 1, ["google"], 2, 2, 1, ["x"])
    assert store.runs()["inserted"].tolist() == [2]


def _series(ticker, start, days, per_day, score_fn):
    rows = []
    for d in range(days):
        for k in range(per_day(d)):
            rows.append({"Stock_symbol": ticker, "Date": start + pd.Timedelta(days=d, hours=k % 20),
                         "sentiment_score": score_fn(d, k)})
    return rows


def test_alerts_flag_sentiment_drop_and_volume_spike_only():
    asof = pd.Timestamp("2026-11-24", tz="UTC")
    start = asof - pd.Timedelta(days=97)
    rng = np.random.default_rng(0)
    rows = (_series("BAD", start, 97, lambda d: 3, lambda d, k: (-0.6 if d >= 90 else 0.1) + rng.normal(0, 0.2))
            + _series("BUSY", start, 97, lambda d: 12 if d >= 90 else 2, lambda d, k: rng.normal(0, 0.2))
            + _series("CALM", start, 97, lambda d: 3, lambda d, k: rng.normal(0, 0.2)))
    table = compute_alerts(pd.DataFrame(rows), asof=asof).set_index("ticker")
    assert table.loc["BAD", "alerts"] == "sentiment_drop"
    assert "volume_spike" in table.loc["BUSY", "alerts"]
    assert table.loc["CALM", "alerts"] == ""
    assert "not a trading signal or investment advice" in table.loc["BAD", "message"]


def test_alerts_wait_for_history():
    asof = pd.Timestamp("2026-11-24", tz="UTC")
    rows = _series("BAD", asof - pd.Timedelta(days=20), 20, lambda d: 5, lambda d, k: -0.9 if d >= 13 else 0.5)
    table = compute_alerts(pd.DataFrame(rows), asof=asof)
    assert table["alerts"].tolist() == [""]
    assert compute_alerts(pd.DataFrame(columns=["Stock_symbol", "Date", "sentiment_score"])).empty


def test_webhook_payload_shapes():
    assert webhook_payload("https://discord.com/api/webhooks/1", "hi") == {"content": "hi"}
    assert webhook_payload("https://hooks.slack.com/services/x", "hi") == {"text": "hi"}


# --------------------------------------------------------------------------- Gemini (mocked HTTP)


class FakeResponse:
    def __init__(self, status, payload):
        self.status_code, self._payload = status, payload
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), []

    def post(self, url, json, timeout, headers):
        self.calls.append({"url": url, "body": json, "headers": headers})
        return self.responses.pop(0)


def _answer(text):
    return FakeResponse(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


def test_gemini_annotation_validates_labels_and_sends_key_in_header():
    session = FakeSession([_answer('```json\n{"labels": [{"id": 1, "label": "Positive"}, {"id": 2, "label": "bullish"},'
                                   ' {"id": 99, "label": "neutral"}, {"id": 3, "label": "skip"}]}\n```')])
    client = GeminiClient(api_key="secret-key", min_interval=0, session=session)
    items = [{"id": i, "ticker": "AAPL", "company": "Apple", "title": f"h{i}"} for i in (1, 2, 3)]
    assert annotate_headlines(client, items, "guidelines") == {1: "positive", 3: "skip"}
    call = session.calls[0]
    assert call["headers"]["x-goog-api-key"] == "secret-key" and "secret-key" not in call["url"]
    assert call["body"]["generationConfig"]["responseMimeType"] == "application/json"


def test_gemini_retries_on_rate_limit(monkeypatch):
    monkeypatch.setattr("src.gemini.time.sleep", lambda s: None)
    limited = FakeResponse(429, {"error": {"message": "quota", "details": [{"retryDelay": "2s"}]}})
    session = FakeSession([limited, _answer('{"labels": []}')])
    client = GeminiClient(api_key="k", min_interval=0, session=session)
    assert client.generate("x")["candidates"]
    assert len(session.calls) == 2
    with pytest.raises(GeminiError):
        GeminiClient(api_key="k", min_interval=0, session=FakeSession([FakeResponse(400, {"error": {
            "message": "bad"}})])).generate("x")


def test_gemini_search_headlines_keeps_only_dated_linked_rows_in_window():
    text = json.dumps([
        {"title": "Apple unveils chip", "publisher": "CNBC", "published": "2026-11-23", "url": "https://cnbc.com/a"},
        {"title": "No link", "published": "2026-11-23"},
        {"title": "Too old", "published": "2025-01-01", "url": "https://x.com/b"},
    ])
    session = FakeSession([_answer(text)])
    client = GeminiClient(api_key="k", min_interval=0, session=session)
    df = search_headlines(client, "AAPL", "Apple", days=7, now=NOW)
    assert df["Article_title"].tolist() == ["Apple unveils chip"] and df["Source"].iloc[0] == "gemini"
    assert session.calls[0]["body"]["tools"] == [{"google_search": {}}]


def test_gemini_requires_key(monkeypatch, tmp_path):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr("src.gemini.get_secret", lambda name: None)
    with pytest.raises(GeminiError, match=".env"):
        GeminiClient()
    assert parse_json('Here: [{"a": 1}] done') == [{"a": 1}]


def test_env_file_parsing(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# comment\nGEMINI_API_KEY="abc123"\nexport FINNHUB_API_KEY=xyz  # note\nEMPTY=\n',
                   encoding="utf-8")
    assert read_env_file(env) == {"GEMINI_API_KEY": "abc123", "FINNHUB_API_KEY": "xyz", "EMPTY": ""}
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert get_secret("GEMINI_API_KEY", env) == "abc123"
    monkeypatch.setenv("GEMINI_API_KEY", "from-env")
    assert get_secret("GEMINI_API_KEY", env) == "from-env"
    assert get_secret("MISSING", env) is None


def test_env_example_contains_no_secrets():
    """.env.example is committed: every key/URL in it must stay empty (real values belong in the ignored .env)."""
    from pathlib import Path

    example = Path(__file__).resolve().parents[1] / ".env.example"
    values = read_env_file(example)
    assert "GEMINI_API_KEY" in values
    leaked = {k: v for k, v in values.items() if v and (k.endswith(("_KEY", "_TOKEN", "_URL", "_SECRET")))}
    assert not leaked, f"secret values in .env.example: {sorted(leaked)}"


def test_http_get_backs_off_on_rate_limit(monkeypatch):
    class Resp:
        def __init__(self, code):
            self.status_code, self.text = code, "ok"

        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError(self.status_code)

    class Session:
        def __init__(self, codes):
            self.codes = list(codes)

        def get(self, url, timeout, headers, **kw):
            return Resp(self.codes.pop(0))

    waits = []
    monkeypatch.setattr(live_feed.time, "sleep", waits.append)
    assert live_feed._http_get("u", Session([429, 200])).text == "ok"
    assert waits == [5.0]
    with pytest.raises(RuntimeError):
        live_feed._http_get("u", Session([429, 429, 429]))


def test_gemini_always_uses_the_fixed_model(monkeypatch):
    """Every call goes to gemini-3.1-flash-lite: no override, no fallback, even when it is overloaded or gone."""
    import inspect

    import requests

    from src import gemini

    assert gemini.MODEL == "gemini-3.1-flash-lite"
    assert "model" not in inspect.signature(GeminiClient).parameters
    monkeypatch.setenv("GEMINI_MODEL", "some-other-model")  # must be ignored
    monkeypatch.setattr("src.gemini.time.sleep", lambda s: None)
    busy = FakeResponse(503, {"error": {"message": "high demand"}})

    class TimeoutThenBusy(FakeSession):
        def post(self, url, json, timeout, headers):
            self.calls.append({"url": url})
            if len(self.calls) == 1:
                raise requests.Timeout("slow")
            return self.responses.pop(0)

    session = TimeoutThenBusy([busy, busy, _answer('{"labels": []}')])
    client = GeminiClient(api_key="k", min_interval=0, max_retries=3, session=session)
    assert client.generate("x")["candidates"]
    assert client.model == "gemini-3.1-flash-lite"
    assert {c["url"].split("/models/")[1].split(":")[0] for c in session.calls} == {"gemini-3.1-flash-lite"}

    stuck = GeminiClient(api_key="k", min_interval=0, max_retries=1, session=FakeSession([busy, busy]))
    with pytest.raises(GeminiError, match="gemini-3.1-flash-lite unavailable"):
        stuck.generate("x")
    gone = GeminiClient(api_key="k", min_interval=0, session=FakeSession([FakeResponse(404, {"error": {
        "message": "not found"}})]))
    with pytest.raises(GeminiError, match="404"):
        gone.generate("x")
