# Apps & Deployment


## Dashboard (`app/streamlit_app.py`)

```bash
streamlit run app/streamlit_app.py
```

| Tab | What it shows |
|:---|:---|
| 🔎 Company explorer | Any of 89 Nasdaq-100 tickers: rolling sentiment timeline (optional live share-price overlay), lookback windows with recency weighting, and the latest / most positive / most negative headlines |
| 📈 Market reaction | Event-study results: the CAR chart and positive-minus-negative spreads with CIs |
| 🧪 Model quality | PhraseBank vs real-headline performance, seed variation and calibration, with the domain-shift caveat always visible |
| ✍️ Score a headline | Live scoring with the headline-adapted model, plus the relevance check |
| 📡 Live feed | This week's headlines for the selected company (publish times, links), daily sentiment, alerts and the Gemini agreement check; reads `data/live/news.db` |

Everything except live scoring and the live feed runs from the committed `outputs/` files.

## REST API (`src/api.py`)

```bash
uvicorn src.api:app --port 8000      # interactive docs: http://localhost:8000/docs
```

| Endpoint | Purpose |
|:---|:---|
| `GET /health` | Status, model name and backend (`torch` / `onnx`) |
| `POST /predict` | Score up to 256 headlines: `{"headlines": [...], "ticker": "TSLA"}` → label, class probabilities, P(pos)−P(neg), relevance |
| `GET /tickers` | Tickers with precomputed scores |
| `GET /tickers/{symbol}/sentiment` | Lookback-window summary and latest headlines |
| `GET /tickers/{symbol}/timeline` | Weekly rolling-sentiment series |
| `GET /live/{symbol}?days=7` | Live headlines and their sentiment from the live feed (503 if the feed isn't set up) |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"headlines": ["Tesla Deliveries Top Estimates"], "ticker": "TSLA"}'
```

## Docker

```bash
docker build -t news-sentiment-api .
# model from the Hugging Face Hub ...
docker run -p 8000:8000 -e SENTIMENT_MODEL=<hf-user>/distilbert-financial-headline-sentiment news-sentiment-api
# ... or a locally trained one
docker run -p 8000:8000 -v "$PWD/models/distilbert_headline_adapted:/app/models/distilbert_headline_adapted:ro" news-sentiment-api
```

The image runs as a non-root user and has a health check. Model weights are not baked in, so the image stays small and CI can build it.

## Live news feed (Phase 5B)

The live feed needs **no API key**: Google News RSS and Yahoo Finance are free and include publish *times*,
which FNSPID lacks. Headlines go through the same relevance filter and model, and are stored in
`data/live/news.db` (SQLite, de-duplicated per ticker and normalised title, git-ignored).

```bash
python -m src.pipeline live-update                 # all 89 tickers, last 2 days (~3 min); --live-days 7 for a first run
python -m src.pipeline live-update --live-tickers AAPL NVDA --live-sources google yahoo finnhub
python -m src.pipeline alerts                      # sentiment-drop / volume-spike alerts (after 30 days of daily runs)
python -m src.pipeline live-annotate               # Gemini labels 25 new live headlines blind (needs GEMINI_API_KEY)
```

**API keys** go in a `.env` file in the project root (copy `.env.example`; `.env` is git-ignored):

```ini
GEMINI_API_KEY=...        # https://aistudio.google.com/apikey (free tier)
FINNHUB_API_KEY=          # optional extra source
ALERT_WEBHOOK_URL=        # optional Slack/Discord incoming webhook
SENTIMENT_BACKEND=onnx    # score with the int8 ONNX model, without PyTorch
```

The Gemini model is fixed to **`gemini-3.1-flash-lite`** (`MODEL` in `src/gemini.py`). There is no setting to
change it and no fallback to another model, so every live label comes from the same annotator. If that model is
unavailable, the stage fails with a clear error after retrying, instead of switching models.

What Gemini is used for:
- **`live-annotate` (main use):** an independent LLM labels a random daily sample of live headlines, following
  `docs/annotation_guidelines.md`, without seeing the model's output. Model–Gemini agreement over time is a live
  accuracy and drift check. Gemini is a second annotator, not ground truth. The existing labels came from Claude,
  so this also gives a two-LLM agreement figure.
- **`--live-sources ... gemini` (opt-in):** extra headlines found with Google Search grounding. Rows without a URL
  or outside the date window are dropped, and every row is tagged `Source=gemini`, because an LLM can paraphrase
  a title.

**Alerts** compare each ticker's last 7 days with its previous 90:
- a *sentiment drop* is z ≤ −2, using the standard error of the 7-day mean;
- a *volume spike* is headlines per day ≥ 3× normal. Phase 3 found bad news brings about 2× abnormal trading
  volume, so volume spikes are the best-supported kind.

Alerts start only after 30 days of daily collection, because backfilled history is a capped sample and would
make volume look abnormal. Every message carries the "not investment advice" disclaimer.

**Run it daily** with Windows Task Scheduler (uses the project's `.venv`; logs in `data/live/logs/`):

```powershell
schtasks /Create /SC DAILY /ST 08:00 /TN "NewsSentimentLiveUpdate" `
  /TR "powershell -NoProfile -ExecutionPolicy Bypass -File \"$PWD\scripts\live_update.ps1\""
```

Limitations of the free sources:
- Google News returns at most ~100 items per query, so very heavily covered companies saturate the volume measure.
- Yahoo returns only the ~10 latest items per ticker.
- Coverage starts when you start collecting.
