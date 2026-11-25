# Apps & Deployment


## Dashboard (`web/` + `src/dashboard_api.py`)

```bash
uvicorn src.api:app --reload      # open http://localhost:8000
```

The FastAPI app serves the dashboard site (`web/`, static HTML/CSS/JS on AdminLTE 4 and Apache ECharts, no build
step) and the JSON it reads (`/api/...`). Company, period and model filters sit in the top bar and are kept in the
URL, so a copied link reopens the same view. Every chart shows exact values under the cursor and can be shown as a table.

| Page | What it shows |
|:---|:---|
| Overview | Coverage, the 30-day rolling sentiment trend, the sentiment split, the relevance filter and day-0 return by news tone |
| Company Analysis | Any of 89 tickers: rolling sentiment with a synced share-price chart, lookback windows with recency weighting, and the latest / most positive / most negative headlines |
| Market Impact | Event-study results, for all years or one year: positive-minus-negative return gaps with CIs and the cumulative abnormal return paths |
| News Explorer | Search and filter all 27,146 scored headlines by text, company, sentiment and date |
| Model Evaluation | PhraseBank vs real-headline performance per model and company, relevance filter, seed variation and calibration, with the domain-shift caveat always visible |
| Score a Headline | Live scoring of your own headlines with the deployed model |
| Live Feed | Recent live headlines (publish times, links), daily sentiment, alerts and the Gemini agreement check; reads `data/live/news.db` |

Everything except Score a Headline and the Live Feed runs from the committed `outputs/` files. Filters a page
cannot use are hidden there. Event-study figures exist for all companies together, overall and per year, so they
follow the Period filter and are labelled "All companies · {period}"; relevance and pooled accuracy figures exist only
for all companies and all years, and are labelled "All companies · 2019–2023".

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
| `GET /api/...` | The dashboard's data: `meta`, `overview`, `company/{symbol}`, `headlines`, `market-impact`, `model-evaluation`, `price/{symbol}`, `live/summary` (see `/docs`) |

```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" \
     -d '{"headlines": ["Tesla Deliveries Top Estimates"], "ticker": "TSLA"}'
```

## Docker

The root Docker image is API-only; the Hugging Face Space (`python deploy/publish.py space --user <hf-user>`) serves the dashboard.

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
