"""Minimal Google Gemini REST client: live-headline annotation and (opt-in) search-grounded headline discovery.

The key comes from ``GEMINI_API_KEY`` (environment or ``.env``) and is sent only in the
``x-goog-api-key`` header to generativelanguage.googleapis.com. It is never logged.

The free tier has a low request rate, so calls are spaced out (``min_interval``) and retried
on 429/503 with back-off.
"""

import json
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Sequence

import pandas as pd

from src.env import get_secret

API_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
# The one model this project uses: fixed on purpose (no env override, no fallbacks), so every live label
# comes from the same annotator. If it is unavailable the call fails loudly instead of switching models.
# ("gemini-3.1-flash" itself does not exist for this key; -lite is the 3.1 Flash text model. Checked 2026-11-24.)
MODEL = "gemini-3.1-flash-lite"
TRANSIENT_STATUS = (429, 500, 503, 504)
ANNOTATION_LABELS = ("negative", "neutral", "positive", "skip")
ANNOTATOR_PREFIX = "gemini"


class GeminiError(RuntimeError):
    pass


class GeminiClient:
    """Always calls ``MODEL``. Transient errors (429/5xx, timeouts) are retried with back-off on the same model."""

    def __init__(self, api_key: Optional[str] = None, min_interval: float = 6.0, max_retries: int = 3,
                 timeout: float = 60.0, session=None):
        self.api_key = api_key or get_secret("GEMINI_API_KEY")
        if not self.api_key:
            raise GeminiError("GEMINI_API_KEY is not set: put it in the project's .env file (see .env.example).")
        self.model = MODEL
        self.min_interval = min_interval
        self.max_retries = max_retries
        self.timeout = timeout
        self._last_call = 0.0
        if session is None:
            import requests

            session = requests.Session()
        self.session = session

    def generate(self, prompt: str, search: bool = False, json_output: bool = False,
                 temperature: float = 0.0) -> dict:
        body: Dict = {"contents": [{"role": "user", "parts": [{"text": prompt}]}],
                      "generationConfig": {"temperature": temperature}}
        if search:
            body["tools"] = [{"google_search": {}}]
        elif json_output:
            body["generationConfig"]["responseMimeType"] = "application/json"
        return self._post_with_retries(body)

    def _post_with_retries(self, body: dict) -> dict:
        import requests

        url = API_URL.format(model=MODEL)
        last = ""
        for attempt in range(self.max_retries + 1):
            wait = self.min_interval - (time.monotonic() - self._last_call)
            if wait > 0:
                time.sleep(wait)
            self._last_call = time.monotonic()
            try:
                resp = self.session.post(url, json=body, timeout=self.timeout,
                                         headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"})
            except (requests.Timeout, requests.ConnectionError) as exc:
                last, resp = type(exc).__name__, None
            if resp is not None:
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code not in TRANSIENT_STATUS:
                    raise GeminiError(f"Gemini API error {resp.status_code}: {_error_message(resp)}")
                last = f"HTTP {resp.status_code}"
            if attempt < self.max_retries:
                time.sleep(_retry_delay(resp, attempt) if resp is not None else min(10.0 * 2**attempt, 60.0))
        raise GeminiError(f"{MODEL} unavailable after {self.max_retries + 1} attempts ({last})")


def _error_message(resp) -> str:
    try:
        return resp.json().get("error", {}).get("message", "")[:300]
    except ValueError:
        return resp.text[:300]


def _retry_delay(resp, attempt: int) -> float:
    try:
        for detail in resp.json().get("error", {}).get("details", []):
            delay = str(detail.get("retryDelay", ""))
            if delay.endswith("s"):
                return min(float(delay[:-1]) + 1, 120.0)
    except (ValueError, AttributeError):
        pass
    return min(10.0 * 2**attempt, 120.0)


def response_text(resp: dict) -> str:
    candidates = resp.get("candidates") or []
    if not candidates:
        reason = resp.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise GeminiError(f"Gemini returned no answer ({reason})")
    parts = candidates[0].get("content", {}).get("parts", [])
    return "".join(p.get("text", "") for p in parts)


def parse_json(text: str):
    """JSON from a model answer, tolerating ```json fences and text around the array/object."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, flags=re.DOTALL)
        if not match:
            raise GeminiError("Gemini answer contained no JSON") from None
        return json.loads(match.group(1))


# --------------------------------------------------------------------------- annotation


def annotation_prompt(items: Sequence[dict], guidelines: str) -> str:
    lines = "\n".join(f'{i["id"]}\t{i["ticker"]} ({i["company"]})\t{i["title"]}' for i in items)
    return (
        "You are annotating financial news headlines for a research evaluation set. Follow these guidelines "
        "exactly.\n\n<guidelines>\n" + guidelines.strip() + "\n</guidelines>\n\n"
        "Label each headline below for the company named on its line. Use only the headline text. Allowed "
        "labels: negative, neutral, positive, skip (skip = about a different company, or impossible to tell).\n"
        'Answer with JSON only: {"labels": [{"id": <id>, "label": "<label>"}, ...]}, one entry per id.\n\n'
        "id\tcompany\theadline\n" + lines
    )


def annotate_headlines(client: GeminiClient, items: Sequence[dict], guidelines: str,
                       batch_size: int = 25) -> Dict[int, str]:
    """items: dicts with id, ticker, company, title. Returns {id: label} for the ids Gemini answered validly."""
    out: Dict[int, str] = {}
    for start in range(0, len(items), batch_size):
        batch = list(items[start:start + batch_size])
        data = parse_json(response_text(client.generate(annotation_prompt(batch, guidelines), json_output=True)))
        entries = data.get("labels", []) if isinstance(data, dict) else data
        valid_ids = {int(i["id"]) for i in batch}
        for entry in entries:
            try:
                idx, label = int(entry["id"]), str(entry["label"]).strip().lower()
            except (KeyError, TypeError, ValueError):
                continue
            if idx in valid_ids and label in ANNOTATION_LABELS:
                out[idx] = label
    return out


# --------------------------------------------------------------------------- search-grounded headlines (opt-in)


def search_headlines(client: GeminiClient, ticker: str, company: str, days: int = 7,
                     now: Optional[datetime] = None) -> pd.DataFrame:
    """Recent headlines found by Gemini with Google Search grounding.

    Lower trust than an RSS feed: a model can paraphrase a title or get a date wrong, so rows
    without a URL or outside the window are dropped, and every row is tagged ``Source=gemini``.
    """
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(days=days)).date()
    prompt = (
        f"Use Google Search to find news articles published between {since} and {now.date()} about "
        f"{company} (stock ticker {ticker}). Return up to 15 of them as a JSON array only, each item "
        '{"title": exact article headline, "publisher": outlet name, "published": "YYYY-MM-DD", "url": article URL}. '
        "Copy headlines exactly as published; do not paraphrase or invent articles."
    )
    resp = client.generate(prompt, search=True)
    try:
        items = parse_json(response_text(resp))
    except (GeminiError, json.JSONDecodeError):
        items = []
    if isinstance(items, dict):
        items = items.get("articles") or items.get("items") or []
    rows = []
    for item in items if isinstance(items, list) else []:
        if not isinstance(item, dict):
            continue
        title, url = str(item.get("title", "")).strip(), str(item.get("url", "")).strip()
        published = pd.to_datetime(item.get("published"), errors="coerce", utc=True)
        if not title or not url.startswith("http") or pd.isna(published):
            continue
        if not (since <= published.date() <= now.date()):
            continue
        rows.append({"Published_at": published, "Article_title": title, "Stock_symbol": ticker,
                     "Url": url, "Publisher": str(item.get("publisher", "")).strip(), "Source": "gemini"})
    return pd.DataFrame(rows)
