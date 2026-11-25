# Dashboard Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the Streamlit dashboard with a seven-page analytics site (static HTML/JS on AdminLTE 4 + ECharts) served by the existing FastAPI app, showing only numbers that come from the project's files.

**Architecture:** Python computes every payload: three small data modules (`dashboard_data`, `dashboard_metrics`, `dashboard_live`) feed one FastAPI router (`dashboard_api`) mounted under `/api/` by `create_app`, which also serves `web/index.html` at `/` and `web/static/` at `/static` when `web/` exists. The browser side is a hash-routed single page: `state.js` keeps company/period/model in the URL, `charts.js` holds the shared ECharts builders, and one module per page renders into `<main>`.

**Tech Stack:** Python 3.11, FastAPI, pandas, pytest; vanilla ES modules; vendored AdminLTE 4.10.2, Bootstrap 5.3.9, Bootstrap Icons 1.13.2, Apache ECharts 5.6.0, Libre Franklin (Fontsource 5.3.0). No Node build step.

**Spec:** `docs/superpowers/specs/2026-11-25-dashboard-redesign-design.md` (read §6–§10 before any task; the reference image is `docs/superpowers/specs/assets/2026-11-25-overview-reference.png`).

## Global Constraints

- Every number on screen comes from `outputs/`, `data/live/`, or Yahoo prices. Never copy figures from the reference image (spec §6).
- Existing endpoints `/health`, `/predict`, `/tickers`, `/tickers/{s}/sentiment`, `/tickers/{s}/timeline`, `/live/{s}` keep their paths and response shapes; `tests/test_api.py` must stay green.
- No dual-axis charts. Share price and headline counts get their own chart beneath, with the x-axis shared and the crosshair synced.
- Sentiment is never colour-only: always the label word or a signed number beside the colour, in the order positive → neutral → negative.
- Colours: sidebar `#1c2740`; active nav `#2d4a85` with a 3 px `#7fb0f0` bar; page `#f3f5f9`; card `#ffffff`, 1 px border `#e3e7ee`, radius 10 px, no shadow; ink `#141a26` / `#4b5565` / muted `#7a8394`; grid `#e6e9ef`; axis `#c5cbd5`; series accent `#2a78d6` (area at 10 % alpha); positive `#087a5a`; neutral `#b9bec8`; negative `#ec6a50`; caveat text `#7a4b00` on `#fdf3e1`, always with an icon.
- Type: Libre Franklin 400/500/600/700, self-hosted. Sizes (px): page title 34/700, card title 17/600, stat value 30/700, body 15/400, caption 13/400, axis 12/400. `tabular-nums` only on tables and axis ticks. Sentence case for controls; card titles in the reference's title case; no all-caps eyebrow labels.
- Motion: no load animations; chart transitions ≤ 200 ms; every animation off under `prefers-reduced-motion: reduce`.
- Every chart has a "Show as table" toggle; every control is keyboard-reachable with a visible focus ring; layout works from 390 px to 1440 px; sidebar becomes an off-canvas drawer below 992 px.
- Scope labels: event-study, relevance and pooled-F1 figures always show "All companies · 2019–2023" (spec §8.0) and ignore the company and period filters.
- Model ids are `adapted` and `base`. UI labels: "Headline-adapted DistilBERT" and "PhraseBank DistilBERT". Names in the metrics files: `DistilBERT headline-adapted` and `DistilBERT fine-tuned`.
- Every JSON payload is NaN-free: missing numbers serialise as `null`.
- Lint: `ruff check src tests deploy` must pass. After Task 12, `app` is no longer in the ruff path.

## Review Focus

1. **NaN in payloads.** A lookback window or live ticker with no headlines gives a pandas NaN. The endpoint must return `null`, not a 500 or invalid JSON. Tested in Tasks 2, 4 and 5.
2. **A company with no headlines in the chosen year** (for example a ticker whose coverage ends in 2021, with period 2023). Expect 200 with zero counts and empty series, and the page shows "No {TICKER} headlines in {YEAR}." rather than a blank chart. Tested in Tasks 2 and 5; page copy in Task 8.
3. **Ticker case and unknown tickers in a hand-edited URL** (`ticker=nvda`, `ticker=ZZZZ`). Lowercase is accepted. An unknown ticker gives a 404, and the page shows "No company called ZZZZ in the dataset." and resets the filter to All companies. Tested in Task 5; page behaviour in Tasks 6 and 8.
4. **Search text with regex characters** (`S&P 500 (QQQ)?`) must match literally and never error. Tested in Task 2.
5. **Pagination beyond range.** `page=999` returns `rows: []` with the real total. `page_size=101` returns 422. Tested in Tasks 2 and 5.

---

### Task 1: Save CAR paths and a corpus summary from the pipeline

**Files:**
- Modify: `src/pipeline.py` (`stage_event_study` ≈ lines 675–735; `STAGES` ≈ line 1141)
- Test: `tests/test_pipeline.py`
- Generated (committed): `outputs/metrics/event_study_car_paths.csv`, `outputs/metrics/corpus_summary.json`

**Interfaces:**
- Produces: `event_study_car_paths.csv` with columns `model, headlines, group, day, mean_car, n_events`. There is one row per (model, variant, group ∈ {positive, negative}, day −5…10), and `mean_car` is a fraction, not a percent.
- Produces: `corpus_summary.json` = `{"collected_headlines": int, "scored_headlines": int, "tickers": int, "first_date": "YYYY-MM-DD", "last_date": "YYYY-MM-DD", "source": "FNSPID 2019-2023, relevance-filtered"}`.
- Produces: new pipeline stage `corpus-summary`, function `stage_corpus_summary(p: Paths, args) -> None`.

- [ ] **Step 1: Extend the end-to-end event-study test.** In `test_event_study_stage_end_to_end` add:

```python
    paths = pd.read_csv(tmp_path / "outputs" / "metrics" / "event_study_car_paths.csv")
    assert list(paths.columns) == ["model", "headlines", "group", "day", "mean_car", "n_events"]
    assert set(paths["headlines"]) == {"all relevant headlines", "excluding price-move headlines"}
    assert set(paths["group"]) == {"positive", "negative"} and paths["day"].min() == -5 and paths["day"].max() == 10
```

- [ ] **Step 2: Add `test_corpus_summary_stage(tmp_path)`.** Build a `data/processed/fnspid_all_2019_2023.csv` with 5 rows (columns `Date, Article_title, Stock_symbol`, 2 tickers) and an `outputs/predictions/nasdaq100_scored_headlines_adapted.csv` with 3 of those rows. Run `pipeline.main(["corpus-summary", "--data-dir", …, "--outputs-dir", …, "--models-dir", …])`. Assert the return is `0`, and that the JSON equals `collected_headlines == 5`, `scored_headlines == 3`, `tickers == 2`, plus the scored rows' first and last dates.

- [ ] **Step 3: Run both tests and confirm they fail.** Run `pytest tests/test_pipeline.py -k "event_study_stage or corpus_summary" -v`. Expected: FAIL (the CSV is missing; the stage is unknown).

- [ ] **Step 4: Implement.**
  - In `stage_event_study`, compute `mean_car_paths(panel, threshold=args.event_threshold)` for **every** (slug, variant), insert the `model` and `headlines` columns, collect the results, and `_write_csv` them to `p.metrics / "event_study_car_paths.csv"`. Keep the existing PNG plot.
  - Add `stage_corpus_summary`. It reads `p.processed / "fnspid_all_2019_2023.csv"` (`usecols=["Date", "Stock_symbol"]`) and `all_scored_csv(p, "adapted")` (both through `_require`), then writes the JSON with `_write_json`. Register it as `"corpus-summary"` in `STAGES`.

- [ ] **Step 5: Run the tests and confirm they pass.** Run the Step 3 command. Expected: PASS.

- [ ] **Step 6: Regenerate the real files.**
  - Run `python -m src.pipeline corpus-summary`, then `python -m src.pipeline event-study`.
  - If the laptop runs out of memory, run the same command on the Plaksha HPC (`hpc_submit_job`, preset `cpu`) and download the four `event_study_*` outputs.
  - Expected: `corpus_summary.json` shows 61818 / 27146 / 89. Then run `git diff --stat outputs/metrics`. Expected: only the two new files, plus an unchanged or byte-identical `event_study_buckets.csv`, `event_study_information_coefficients.csv` and `event_study_summary.json` (same seed, same data). If any existing number changes, stop and report it.

- [ ] **Step 7: Commit.**

```bash
git add src/pipeline.py tests/test_pipeline.py outputs/metrics/event_study_car_paths.csv outputs/metrics/corpus_summary.json
git commit -m "Pipeline: save CAR paths and a corpus summary for the dashboard"
```

---

### Task 2: Scored-headline data layer (`src/dashboard_data.py`)

**Files:**
- Create: `src/dashboard_data.py`
- Test: `tests/test_dashboard_data.py`

**Interfaces:**
- Consumes: `src.explorer.load_scored`, `ticker_frame`, `ticker_windows`, `ticker_headlines`; `src.aggregation.compute_rolling_sentiment`.
- Produces (every function returns JSON-ready Python, NaN → `None`):
  - `YEARS: tuple[int, ...] = (2019, 2020, 2021, 2022, 2023)`
  - `clean(obj) -> obj`: recursively replaces float NaN and inf with `None`. It is used by every module below.
  - `filter_scored(scored: pd.DataFrame, ticker: str | None = None, year: int | None = None) -> pd.DataFrame`: uppercases the ticker. Raises `KeyError(ticker)` if the ticker is absent from the **unfiltered** frame. An empty result after the year filter is allowed.
  - `sentiment_distribution(df) -> dict`: `{"positive": float, "neutral": float, "negative": float, "n": int}`. The shares are rounded to 4 decimal places and are all 0.0 when `n == 0`.
  - `trend_points(df, window_days: int = 30) -> list[dict]`: `[{"date": "YYYY-MM-DD", "sentiment": float, "headlines": int}]`, weekly, taking the last rolling value per week (same resampling as `ticker_timeline`). Works on a multi-ticker frame. Returns `[]` for an empty frame.
  - `coverage_gap(df) -> dict | None`: `{"start": date, "end": date, "days": int}` for the largest gap between headline days, if it exceeds 90 days.
  - `company_view(scored, ticker, year=None, window_days=30, half_life_days=30.0, order="latest", n=12) -> dict`: keys `ticker, cards, coverage_gap, timeline, windows, headlines`.
    - `cards = {"headlines", "first_date", "last_date", "recent_30d", "recent_delta", "history", "positive_share", "negative_share"}`.
    - `windows` rows: `{"window", "headlines", "mean", "recency_weighted", "positive_pct", "negative_pct"}`.
    - `timeline` rows: `{"date", "sentiment", "headlines_in_window"}`.
  - `search_headlines(scored, q=None, ticker=None, label=None, start=None, end=None, sort="newest", page=1, page_size=25) -> dict`: `{"total", "page", "page_size", "pages", "rows": [{"date", "ticker", "headline", "label", "sentiment_score", "confidence"}]}`. `sort` is one of `newest | oldest | most_positive | most_negative`.

- [ ] **Step 1: Write the failing tests.** Reuse the `scored_csv` fixture shape from `tests/test_api.py` (TSLA and AAPL, 60 rows each from 2023-01-01 every 3 days). Add a third ticker, `OLDCO`, with 5 rows in 2019 only.

```python
def test_distribution_sums_to_one_and_handles_empty(scored):
    d = sentiment_distribution(scored)
    assert abs(d["positive"] + d["neutral"] + d["negative"] - 1) < 1e-3 and d["n"] == len(scored)
    assert sentiment_distribution(scored.iloc[0:0]) == {"positive": 0.0, "neutral": 0.0, "negative": 0.0, "n": 0}

def test_filter_scored_year_and_ticker(scored):
    assert filter_scored(scored, "tsla", 2023)["Stock_symbol"].unique().tolist() == ["TSLA"]
    assert filter_scored(scored, "OLDCO", 2023).empty                       # Review Focus 2
    with pytest.raises(KeyError):
        filter_scored(scored, "ZZZZ")

def test_trend_points_weekly_multi_ticker(scored):
    pts = trend_points(scored)
    assert pts and set(pts[0]) == {"date", "sentiment", "headlines"} and -1 <= pts[0]["sentiment"] <= 1
    assert trend_points(scored.iloc[0:0]) == []

def test_company_view_has_no_nan(scored):                                    # Review Focus 1
    view = company_view(scored, "OLDCO")
    assert json.dumps(view, allow_nan=False)
    assert company_view(scored, "OLDCO", year=2023)["cards"]["headlines"] == 0

def test_search_is_literal_paged_and_bounded(scored):
    assert search_headlines(scored, q="(")["total"] == 0                     # Review Focus 4
    r = search_headlines(scored, q="tsla headline 1", page_size=5)
    assert r["total"] == 11 and len(r["rows"]) == 5 and r["pages"] == 3      # headline 1, 10..19
    far = search_headlines(scored, page=999)
    assert far["rows"] == [] and far["total"] == len(scored)                 # Review Focus 5
    assert search_headlines(scored, sort="most_negative")["rows"][0]["sentiment_score"] == scored["sentiment_score"].min()

def test_coverage_gap(scored):
    assert coverage_gap(filter_scored(scored, "TSLA")) is None
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_dashboard_data.py -v`. Expected: FAIL (`ModuleNotFoundError: src.dashboard_data`).

- [ ] **Step 3: Implement the module.**
  - Search uses `str.contains(q, case=False, regex=False)`.
  - `company_view` calls `filter_scored` once and builds every part from that frame. When the frame is empty, it returns zero and `None` cards and empty lists, without calling `ticker_windows`, which would fail on an empty frame.

- [ ] **Step 4: Run the tests and confirm they pass.** Run `pytest tests/test_dashboard_data.py -v`. Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/dashboard_data.py tests/test_dashboard_data.py
git commit -m "Dashboard data layer: filters, distribution, trend, company view, search"
```

---

### Task 3: Metrics layer (`src/dashboard_metrics.py`)

**Files:**
- Create: `src/dashboard_metrics.py`
- Test: `tests/test_dashboard_metrics.py`

**Interfaces:**
- Consumes: `src.dashboard_data.clean`; files in `outputs/metrics/` (committed).
- Produces:
  - `class MetricsUnavailable(Exception)`: raised when a needed file is missing.
  - `MODEL_FILE_NAMES = {"adapted": "DistilBERT headline-adapted", "base": "DistilBERT fine-tuned"}`
  - `VARIANTS = {"excluding": "excluding price-move headlines", "all": "all relevant headlines"}`
  - `SCOPE_ALL = "All companies · 2019–2023"`
  - `corpus(metrics_dir) -> dict | None`: the contents of `corpus_summary.json`, or `None` if absent.
  - `headline_f1(metrics_dir, model, ticker=None) -> dict`: `{"value", "ci_low", "ci_high", "n", "scope"}`. `scope` is the ticker when that ticker is in `multi_company_per_company_accuracy.csv`; otherwise it is `SCOPE_ALL`, with the value taken from `multi_company_pooled_accuracy.csv` (population `pooled_all_6_companies`).
  - `relevance(metrics_dir) -> dict`: `{"kept_share", "collected", "scored", "precision_min", "precision_max", "scope": SCOPE_ALL, "by_ticker": [{"ticker", "precision", "recall_estimate", "on_topic_share_estimate"}]}`.
  - `market_relationship(metrics_dir, model, variant="excluding") -> dict`: `{"variant", "events", "scope", "by_tone": [{"group", "value", "ci_low", "ci_high"}], "volume_ratio"}`. Returns and CIs are in **percent** (outcome `ar_0` × 100). `volume_ratio` is av_0 negative / av_0 positive, rounded to 1 decimal place.
  - `market_impact(metrics_dir, model, variant="excluding") -> dict`: `{"variant", "events", "tickers", "scope", "spread": [{"outcome", "label", "value", "ci_low", "ci_high", "unit", "significant"}], "car_paths": [{"group", "day", "mean_car"}], "car_paths_available": bool}`.
    - Spread labels: `ar_m1` "Day −1 (before)", `ar_0` "Day 0", `ar_p1` "Day +1", `car_p1_p5` "Days +1…+5", `av_0` "Abnormal volume, day 0".
    - `unit` is `"pct"`, or `"pts"` for `av_0`. Values are × 100.
    - `significant = ci_low > 0 or ci_high < 0`.
    - `mean_car` is in percent.
  - `model_evaluation(metrics_dir) -> dict`: `{"phrasebank": [...], "headlines": [...], "per_company": [...], "relevance": [...], "seeds": {"n", "mean", "std"}, "calibration_ece"}`.
    - `phrasebank` comes from `test_set_model_comparison.csv`, plus a row `{"model": "DistilBERT headline-adapted", "macro_f1": <adapted_test_metrics.json macro_f1>, "ci_low": None, "ci_high": None, "note": "No bootstrap CI saved for this model"}`.
    - `headlines` is the pooled six-company file, with `mcnemar_p`.
    - `per_company` drops the "Majority" rows.
    - `calibration_ece` is the `ece` of `DistilBERT fine-tuned` in the PhraseBank file.

- [ ] **Step 1: Write the failing tests against the committed metrics.** Mark the module with `pytestmark = pytest.mark.skipif(not (METRICS_DIR / "event_study_buckets.csv").exists(), …)`.

```python
def test_market_relationship_matches_spec():
    m = market_relationship(METRICS_DIR, "adapted")
    tone = {r["group"]: r for r in m["by_tone"]}
    assert m["events"] == 22999 and m["scope"] == SCOPE_ALL
    assert round(tone["positive"]["value"], 2) == 0.26 and round(tone["negative"]["value"], 2) == -0.58
    assert m["volume_ratio"] == 1.9
    assert round({r["group"]: r for r in market_relationship(METRICS_DIR, "adapted", "all")["by_tone"]}
                 ["negative"]["value"], 2) == -0.81

def test_headline_f1_pooled_and_per_company():
    assert round(headline_f1(METRICS_DIR, "adapted")["value"], 3) == 0.655
    assert round(headline_f1(METRICS_DIR, "base")["value"], 3) == 0.568
    assert headline_f1(METRICS_DIR, "adapted", "TSLA")["scope"] == "TSLA"
    assert headline_f1(METRICS_DIR, "adapted", "MSFT")["scope"] == SCOPE_ALL

def test_relevance_and_corpus():
    r = relevance(METRICS_DIR)
    assert r["precision_min"] == pytest.approx(0.962, abs=1e-3) and r["precision_max"] == 1.0
    assert r["collected"] == 61818 and r["scored"] == 27146 and round(r["kept_share"], 3) == 0.439

def test_market_impact_spread_and_paths():
    mi = market_impact(METRICS_DIR, "adapted")
    assert [s["outcome"] for s in mi["spread"]] == ["ar_m1", "ar_0", "ar_p1", "car_p1_p5", "av_0"]
    assert mi["car_paths_available"] and {p["group"] for p in mi["car_paths"]} == {"positive", "negative"}

def test_model_evaluation_includes_adapted_without_ci():
    ev = model_evaluation(METRICS_DIR)
    adapted = [r for r in ev["phrasebank"] if r["model"] == "DistilBERT headline-adapted"][0]
    assert adapted["ci_low"] is None and round(adapted["macro_f1"], 3) == 0.847
    assert round(ev["calibration_ece"], 3) == 0.021 and json.dumps(ev, allow_nan=False)

def test_missing_files_raise(tmp_path):
    with pytest.raises(MetricsUnavailable):
        market_impact(tmp_path, "adapted")
    assert corpus(tmp_path) is None
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_dashboard_metrics.py -v`. Expected: FAIL (module missing).

- [ ] **Step 3: Implement the module.**
  - Read through one cached helper, `_csv(metrics_dir, name)` / `_json(...)`, which uses `lru_cache` keyed on `(path, mtime)` and raises `MetricsUnavailable(name)` on `FileNotFoundError`.
  - Pass every return value through `clean`.
  - When the CAR-paths file is missing, `market_impact` returns `car_paths: []` with `car_paths_available: False`. It does not raise.

- [ ] **Step 4: Run the tests and confirm they pass.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/dashboard_metrics.py tests/test_dashboard_metrics.py
git commit -m "Dashboard metrics layer: event study, F1, relevance, model evaluation"
```

---

### Task 4: Live summary and price fetch

**Files:**
- Create: `src/dashboard_live.py`
- Test: `tests/test_dashboard_live.py`

**Interfaces:**
- Consumes: `src.live_feed.LiveStore` (`scored_frame(ticker)`, `runs()`); `src.dashboard_data.clean`.
- Produces:
  - `live_summary(live_db: Path, ticker: str | None = None, now: pd.Timestamp | None = None) -> dict`.
    - With no DB: `{"available": False, "command": "python -m src.pipeline live-update", "docs": "docs/deployment.md"}`.
    - Otherwise: `{"available": True, "last_run_utc", "runs", "sources", "total_headlines", "tickers", "ticker", "ticker_has_data", "cards": {"headlines_7d", "mean_7d", "delta_vs_all", "positive_share", "negative_share", "history_days"}, "daily": [{"day", "mean", "count"}], "headlines": [{"published_at", "headline", "label", "sentiment_score", "publisher", "source", "url"}], "alerts": [str], "agreement": dict | None}`.
    - `headlines` holds at most 50 items, newest first. With no ticker, it covers all tickers.
    - `alerts` holds the non-empty `message` values from `alerts_latest.csv`, next to the DB.
    - `agreement` is `live_agreement.json`, but only if it has `macro_f1`.
  - `fetch_close(ticker: str, start: str, end: str) -> list[dict]`: `[{"date": "YYYY-MM-DD", "close": float}]` from `yfinance.download(..., auto_adjust=True, progress=False)`. Returns `[]` on any exception or an empty frame. Cached with `lru_cache(maxsize=256)` on `(ticker, start, end, date.today())`.

- [ ] **Step 1: Write the failing tests.** Build the DB as `test_live_endpoint` does.

```python
def test_live_summary_without_db(tmp_path):
    assert live_summary(tmp_path / "none.db") == {"available": False, "command": "python -m src.pipeline live-update",
                                                  "docs": "docs/deployment.md"}

def test_live_summary_with_ticker_and_without_data(live_db):
    s = live_summary(live_db, "aapl")
    assert s["available"] and s["ticker_has_data"] and s["cards"]["headlines_7d"] == 1
    assert json.dumps(s, allow_nan=False)                                    # Review Focus 1
    other = live_summary(live_db, "MSFT")
    assert other["ticker_has_data"] is False and other["daily"] == []

def test_fetch_close_never_raises(monkeypatch):
    import yfinance
    monkeypatch.setattr(yfinance, "download", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("offline")))
    fetch_close.cache_clear()
    assert fetch_close("TSLA", "2023-01-01", "2023-02-01") == []
```

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_dashboard_live.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement the module.**
  - The daily series is `resample("D")` mean and count of `sentiment_score`, dropping days with no headlines.
  - `history_days` = `(now - first Date).days`.
  - `delta_vs_all` is `None` unless the ticker has more headlines overall than in the last 7 days.
  - Import `yfinance` inside `fetch_close`, never at module top. The root API image (`requirements-api.txt`) does not install it, and `src.api` must still import there.

- [ ] **Step 4: Run the tests and confirm they pass.** Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add src/dashboard_live.py tests/test_dashboard_live.py
git commit -m "Dashboard live summary and Yahoo price fetch"
```

---

### Task 5: Dashboard router and app wiring

**Files:**
- Create: `src/dashboard_api.py`
- Modify: `src/api.py` (`create_app` signature and the module docstring's endpoint list), `src/config.py` (add `WEB_DIR = PROJECT_ROOT / "web"`)
- Test: `tests/test_dashboard_api.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `build_dashboard_router(scored_paths: Dict[str, Path], metrics_dir: Path, live_db: Path, price_fetcher: Callable = fetch_close) -> APIRouter`, with `prefix="/api"`.
  - `create_app(service=None, scored_csv=DEFAULT_SCORED_CSV, live_db=LIVE_DB, scored_paths: Optional[Dict[str, Path]] = None, metrics_dir: Path = METRICS_DIR, web_dir: Path = WEB_DIR)`. When `scored_paths` is `None`, it is `{"adapted": scored_csv, "base": PREDICTIONS_DIR / "nasdaq100_scored_headlines_base.csv"}`.
  - Endpoints (`model: Literal["adapted", "base"] = "adapted"`, `period: Optional[int] = Query(None, ge=2019, le=2023)`):
    - `GET /api/meta`: `{"tickers": [{"ticker", "headlines", "first_date", "last_date"}], "years": YEARS, "models": [{"id": "adapted", "label": "Headline-adapted DistilBERT"}, {"id": "base", "label": "PhraseBank DistilBERT"}], "corpus": dict | None, "disclaimer"}`.
    - `GET /api/overview?ticker=&period=&model=`: `{"scope": {"ticker", "period", "model"}, "cards": {"coverage", "headlines", "f1", "company_days"}, "distribution", "mean_sentiment", "trend", "relevance", "market", "disclaimer"}`.
      - `coverage` = `{"kind": "universe", "companies": 89}` or `{"kind": "company", "ticker", "first_date", "last_date"}`.
      - `headlines` = `{"scored": int, "collected": int | None}`, where `collected` is only set when there is no ticker and no period.
      - `company_days` = `{"value": 24807, "scope": SCOPE_ALL}`, from the "all relevant headlines" run of the selected model.
    - `GET /api/company/{ticker}?period=&model=&window_days=30 (7–90)&half_life_days=30 (7–180)&order=latest&n=12 (1–100)`: `company_view(...)` plus `"accuracy": headline_f1(...) if scope == ticker else None` and `"accuracy_available_for": [6 tickers]`.
    - `GET /api/headlines?q=&ticker=&label=&start=&end=&sort=newest&page=1 (≥1)&page_size=25 (1–100)&model=`: `search_headlines(...)`.
    - `GET /api/market-impact?model=&headlines=excluding|all`: `market_impact(...)`.
    - `GET /api/model-evaluation`: `model_evaluation(...)`.
    - `GET /api/price/{ticker}?start=&end=`: `{"ticker", "points": fetch_close(...)}`.
    - `GET /api/live/summary?ticker=`: `live_summary(...)`, always 200.
  - Errors:
    - An unknown ticker (`KeyError`) gives 404 `{"detail": "No company called ZZZZ in the dataset."}`.
    - `MetricsUnavailable` or a missing scored CSV gives 503 `{"detail": "… not available on this server"}`.
  - Serving: only if `(web_dir / "index.html").exists()`, register `GET /` (returns a `FileResponse` of `index.html`) and mount `/static` as `StaticFiles(directory=web_dir / "static")`. Both go after all API routes.

- [ ] **Step 1: Write the failing tests.**
  - The fixture writes a tmp scored CSV (the TSLA/AAPL/OLDCO shape from Task 2) and uses it for both models.
  - `metrics_dir` is the real `METRICS_DIR`, skipped if absent.
  - The `FakeService` is copied from `tests/test_api.py`.
  - `web_dir` is a tmp dir containing `index.html` and `static/js/main.js`.

```python
def test_meta_and_overview(client):
    meta = client.get("/api/meta").json()
    assert [m["id"] for m in meta["models"]] == ["adapted", "base"] and meta["years"] == [2019, 2020, 2021, 2022, 2023]
    ov = client.get("/api/overview").json()
    assert ov["cards"]["company_days"] == {"value": 24807, "scope": "All companies · 2019–2023"}
    assert ov["market"]["events"] == 22999 and ov["trend"]

def test_overview_for_company_and_empty_year(client):
    ov = client.get("/api/overview?ticker=oldco&period=2023").json()             # Review Focus 2 + 3
    assert ov["cards"]["headlines"]["scored"] == 0 and ov["trend"] == [] and ov["distribution"]["n"] == 0
    assert ov["cards"]["coverage"]["kind"] == "company"

def test_unknown_ticker_and_bad_params(client):
    r = client.get("/api/company/ZZZZ")
    assert r.status_code == 404 and r.json()["detail"] == "No company called ZZZZ in the dataset."
    assert client.get("/api/overview?period=2018").status_code == 422
    assert client.get("/api/overview?model=finbert").status_code == 422
    assert client.get("/api/headlines?page_size=101").status_code == 422          # Review Focus 5
    assert client.get("/api/headlines?page=999").json()["rows"] == []

def test_company_payload_is_valid_json(client):
    r = client.get("/api/company/OLDCO?period=2023")                             # Review Focus 1 + 2
    assert r.status_code == 200 and json.loads(r.text)                           # strict JSON, no NaN
    assert r.json()["accuracy"] is None and r.json()["cards"]["recent_30d"] is None
    assert client.get("/api/company/TSLA").json()["accuracy"] is not None        # TSLA is annotated

def test_market_model_price_live(client, monkeypatch):
    assert client.get("/api/market-impact?headlines=all").json()["variant"] == "all relevant headlines"
    assert client.get("/api/model-evaluation").json()["calibration_ece"] is not None
    assert client.get("/api/price/TSLA?start=2023-01-01&end=2023-02-01").json() == {"ticker": "TSLA", "points": []}
    assert client.get("/api/live/summary").json()["available"] is False

def test_missing_metrics_gives_503(scored_csv, tmp_path):
    c = TestClient(create_app(service=FakeService(), scored_paths={"adapted": scored_csv, "base": scored_csv},
                              metrics_dir=tmp_path / "empty", web_dir=tmp_path / "noweb"))
    assert c.get("/api/market-impact").status_code == 503
    assert c.get("/").status_code == 404                                         # site not mounted without web/

def test_site_served_and_api_not_shadowed(client):
    assert "<html" in client.get("/").text.lower()
    assert client.get("/static/js/main.js").status_code == 200
    assert client.get("/health").json()["status"] == "ok" and client.get("/tickers").status_code == 200
```

  The `client` fixture is `create_app(service=FakeService(), scored_paths={"adapted": csv, "base": csv}, metrics_dir=METRICS_DIR, live_db=tmp_path / "none.db", web_dir=<tmp web>, price_fetcher=lambda *a: [])`. To make that possible, `create_app` gets one more keyword, `price_fetcher: Callable = fetch_close`, which it forwards to the router. The tmp scored CSV must include TSLA, so that TSLA's measured accuracy resolves.

- [ ] **Step 2: Run the tests and confirm they fail.** Run `pytest tests/test_dashboard_api.py -v`. Expected: FAIL.

- [ ] **Step 3: Implement the router.** Implement `build_dashboard_router` and wire it into `create_app`. Scored frames load through the existing `load_scored(path)` (already `lru_cache`d). Add `WEB_DIR` to `src/config.py`.

- [ ] **Step 4: Run the whole API suite.** Run `pytest tests/test_dashboard_api.py tests/test_api.py -v`. Expected: all PASS, including every existing test unchanged.

- [ ] **Step 5: Commit.**

```bash
git add src/dashboard_api.py src/api.py src/config.py tests/test_dashboard_api.py
git commit -m "Dashboard API: /api endpoints and static site mount"
```

---

### Task 6: Site shell (vendor, layout, tokens, routing)

**Files:**
- Create:
  - `web/index.html`
  - `web/static/css/app.css`
  - `web/static/js/main.js`
  - `web/static/js/state.js`
  - `web/static/js/api.js`
  - `web/static/js/ui.js`
  - `web/static/vendor/` (AdminLTE `adminlte.min.css` and `adminlte.min.js`, `bootstrap.bundle.min.js`, `bootstrap-icons.min.css` plus `fonts/`, `echarts.min.js`, and `VERSIONS.txt`)
  - `web/static/fonts/libre-franklin-latin-{400,500,600,700}-normal.woff2`
- Test: `tests/test_dashboard_api.py` (one added test against the real `web/`)

**Interfaces:**
- Produces (JS, ES modules):
  - `state.js`:
    - `readRoute() -> {page: string, params: {ticker?, period?, model?, ...}}`, where `page` is one of `overview | company | market | news | models | score | live` (default `overview`).
    - `go(page, params)`.
    - `setParams(patch)`: merges into the current hash; `null` deletes a key.
    - `onRoute(callback)`.
    - Tickers are uppercased on read.
  - `api.js`: `getJSON(path, params) -> Promise<object>` drops null params. It throws `ApiError {status, detail}` when the response is not ok. `postJSON(path, body)`.
  - `ui.js`:
    - `card({title, info, body}) -> HTMLElement`. `info` is the source/definition text shown in a Bootstrap popover from the title's ⓘ button.
    - `statCard({icon, tone, value, label, sub, scope})`. `tone` is one of `blue | green | violet | amber` (the icon-circle tints from the reference).
    - `note({kind: "caveat" | "info" | "error", text})`.
    - `emptyState(text, action?)`.
    - `sentimentBadge(label, score)`: renders e.g. "Positive +0.62", word plus signed number.
    - `fmt` = `{pct(x, digits), signed(x, digits), int(n), date(iso)}`.
  - `main.js`: registry `{overview: () => import("./pages/overview.js"), …}`. Each page module exports `render(root, params, meta) -> Promise<void>` and optionally `destroy()`.
    - `main.js` loads `/api/meta` once, fills the Company combobox (All companies + tickers), Period (All years + `years`) and Model selects, and keeps them in sync with the URL.
    - It sets `aria-current="page"` on the active nav link and sets `document.title = "<Page> · Sentiment Monitor"`.
    - On `ApiError` 404 for a ticker, it shows `note({kind: "error", text: detail})` and calls `setParams({ticker: null})`.

- [ ] **Step 1: Write the failing test.**

```python
@pytest.mark.skipif(not (WEB_DIR / "index.html").exists(), reason="web/ not built yet")
def test_real_shell_lists_seven_pages_and_assets(real_client):
    html = real_client.get("/").text
    for label in ["Overview", "Company Analysis", "Market Impact", "News Explorer", "Model Evaluation",
                  "Score a Headline", "Live Feed"]:
        assert label in html
    for asset in ["/static/js/main.js", "/static/css/app.css", "/static/vendor/echarts.min.js",
                  "/static/vendor/adminlte.min.css", "/static/fonts/libre-franklin-latin-400-normal.woff2"]:
        assert real_client.get(asset).status_code == 200, asset
```

  `real_client` is `TestClient(create_app(service=FakeService()))` with the default paths. Remove the `skipif` once `web/index.html` exists, so the test cannot silently skip.

- [ ] **Step 2: Run the test and confirm it fails.** Run `pytest tests/test_dashboard_api.py::test_real_shell_lists_seven_pages_and_assets -v`. Expected: SKIP or FAIL. Delete the skip and confirm it FAILS.

- [ ] **Step 3: Vendor the pinned files.**
  - Download from `https://cdn.jsdelivr.net/npm/<pkg>@<version>/<path>` for: `admin-lte@4.10.2/dist/css/adminlte.min.css`, `admin-lte@4.10.2/dist/js/adminlte.min.js`, `bootstrap@5.3.9/dist/js/bootstrap.bundle.min.js`, `bootstrap-icons@1.13.2/font/bootstrap-icons.min.css` (plus its `fonts/*.woff2`), `echarts@5.6.0/dist/echarts.min.js`, and `@fontsource/libre-franklin@5.3.0/files/libre-franklin-latin-{400,500,600,700}-normal.woff2`.
  - Record the URLs and versions in `web/static/vendor/VERSIONS.txt`.
  - Check `tnum`: `python -c "from fontTools.ttLib import TTFont; print('tnum' in {r.FeatureTag for r in TTFont('web/static/fonts/libre-franklin-latin-400-normal.woff2')['GSUB'].table.FeatureList.FeatureRecord})"`. If it prints `False`, use `@fontsource/public-sans@5` files instead and note it in `VERSIONS.txt` (spec §7).

- [ ] **Step 4: Build `index.html` and `app.css`.**
  - `index.html` uses AdminLTE's `app-wrapper`, `app-header`, `app-sidebar` and `app-main`.
  - The sidebar contains, in order:
    - the product name "Sentiment Monitor" with "Nasdaq-100 Research" beneath;
    - seven nav links with Bootstrap Icons (`house`, `buildings`, `bar-chart`, `newspaper`, `shield-check`, `pencil-square`, `broadcast`);
    - a "Dataset context" block (2019 – 2023 · 89 Companies · 61,818 collected headlines, filled from `/api/meta`);
    - a status footer that reads "Historical data · not live", or on the Live Feed page "Live · updated <time>".
  - The top bar is one right-aligned row holding Company, Period and Model selects, with visible `<label>`s.
  - `app.css` defines the Global Constraints tokens as CSS custom properties on `:root`. It overrides AdminLTE's sidebar and card variables, sets `@font-face` for Libre Franklin, gives `:focus-visible` a 2 px `#2a78d6` outline, and adds the `prefers-reduced-motion` block.

- [ ] **Step 5: Implement `state.js`, `api.js`, `ui.js` and `main.js`.** For now, each page module is a placeholder that renders its title.

- [ ] **Step 6: Run the test and confirm it passes.** Run the Step 2 command. Expected: PASS.

- [ ] **Step 7: Check it in a browser.**
  - Run `uvicorn src.api:app --reload` and open `http://localhost:8000`.
  - Clicking each of the seven nav links changes the hash and the active item. Back and Forward work. Changing Company, Period and Model rewrites the hash, and reloading keeps them.
  - At 390 px wide, the sidebar opens from the menu button and Tab reaches every control with a visible ring.
  - Take a screenshot at 1440 px and compare it with the reference image's sidebar and top bar.

- [ ] **Step 8: Commit.**

```bash
git add web tests/test_dashboard_api.py
git commit -m "Dashboard shell: vendored AdminLTE/ECharts, layout, tokens, URL state"
```

---

### Task 7: Chart builders and the Overview page

**Files:**
- Create: `web/static/js/charts.js`, `web/static/js/pages/overview.js`
- Modify: `web/static/css/app.css` (card grid for Overview)

**Interfaces:**
- Consumes: `/api/overview`, `/api/meta`; `ui.js`; `state.js`.
- Produces (`charts.js`; every builder returns `{chart, dispose()}` and attaches a "Show as table" toggle under the chart):
  - `lineChart(el, {points, x, y, name, yMin, yMax, color, area = true, zeroLine = true, zoomPresets = ["1Y", "3Y", "5Y"], format})`.
    - Crosshair: `tooltip.trigger = "axis"` with `axisPointer.type = "cross"`, and axis-pointer labels on **both** axes showing the date and the value with `format`.
    - Zoom: `dataZoom` with `inside` plus a slider. Each preset sets the window that many years back from the last point.
  - `barsWithCI(el, {rows, category, value, low, high, colorFor, unit, labelFor})`: bars plus a `custom` series drawing whiskers. Hovering a bar shows the value and its 95 % CI. Bars get 4 px rounded ends at the data end.
  - `donut(el, {slices: [{name, value, color}], centerLabel})`: 2 px surface gap between slices, legend with percentages, centre text showing the largest slice.
  - `ring(el, {value, color, label})`: single-value progress ring.
  - `heatmap(el, {cells, x, y, value, min, max})`: one-hue blue ramp, value printed in each cell.
  - `connect(...charts)`: calls `echarts.connect` to sync crosshairs.
  - Shared theme: Libre Franklin, axis colours, gridlines `#e6e9ef`, animation duration 200 ms, or 0 under reduced motion.

- [ ] **Step 1: Build `overview.js` against spec §8.1.**
  - Four `statCard`s:
    - Companies Covered (89, "Nasdaq-100 universe"), or the company's date range when a company is selected.
    - Headlines Scored ("of 61,818 collected" when unfiltered).
    - Real-Headline Macro F1 (signed to 2 decimal places, "vs 0.84 on PhraseBank sentences", scope label).
    - Company-Days (24,807, "All companies · 2019–2023").
  - Row 2: "Market Sentiment Trend" `lineChart` (8 columns), titled "30-day rolling sentiment", plus "Sentiment Distribution" `donut` (4 columns).
  - Row 3: "Headline Relevance Filter" (`ring` 44 %, "96–100 % precision on retained headlines", caption "The filter keeps headlines about the company and drops lists that only mention it.") and "Market Relationship Insights" (`barsWithCI` of `market.by_tone` in %, caption "Day-0 market-adjusted return, excluding price-move headlines · 22,999 company-days").
  - Two findings: "News often describes price moves after they happen, rather than predicting them." and "Negative-news days see about 1.9× the trading volume of positive ones."
  - A "See Market Impact" link.
  - Empty company/year: replace the trend and the donut with `emptyState("No {TICKER} headlines in {YEAR}.")`.

- [ ] **Step 2: Check it in a browser.**
  - At `#/overview`, hovering the trend chart shows a vertical and horizontal crosshair with the date on the x-axis label and the value on the y-axis label at every mouse position, not only on the line. 1Y/3Y/5Y zoom. "Show as table" lists the weekly points.
  - The numbers match spec §6: distribution 26.6 / 60.6 / 12.9 %; bars +0.26 / +0.02 / −0.58 %.
  - Selecting TSLA re-cuts the cards and trend. F1 shows a TSLA-scoped value. Company-Days keeps its "All companies" label.
  - `#/overview?ticker=<ticker absent in 2023>&period=2023` shows the empty-state sentence.
  - Compare a 1440 px screenshot with the reference image.

- [ ] **Step 3: Commit.**

```bash
git add web/static/js/charts.js web/static/js/pages/overview.js web/static/css/app.css
git commit -m "Overview page and shared ECharts builders with crosshair"
```

---

### Task 8: Company Analysis page

**Files:**
- Create: `web/static/js/pages/company.js`

**Interfaces:**
- Consumes: `/api/company/{ticker}`, `/api/price/{ticker}`; `lineChart`, `connect`, `ui.js`.

- [ ] **Step 1: Build `company.js` against spec §8.2.**
  - With no ticker in the URL, call `setParams({ticker: "TSLA"})`.
  - Cards:
    - "Headlines" (with "first → last date").
    - "Last 30 Days" (signed to 3 decimal places, with delta "vs all history").
    - "All-History Sentiment".
    - "Positive / Negative Share".
  - Caveat `note`s:
    - coverage gap: "No {TICKER} headlines between {start} and {end} ({days} days). Treat this stretch as missing data, not quiet news."
    - measured accuracy: "Measured accuracy for {TICKER}: macro F1 {f1} [{lo}, {hi}] on {n} LLM-annotated headlines."
  - Above the chart, inline range inputs for "Rolling window" (7–90 days) and "Recency half-life" (7–180 days). These write `window_days` and `half_life_days` into the URL, debounced by 300 ms.
  - Full-width `lineChart` of `timeline` (y from −1 to 1).
  - A "Show share price" switch. When on, it adds a second `lineChart` beneath, sharing the x range, joined with `connect`. If there are no points, it shows `note({kind: "info", text: "No price data for {TICKER} (it may have been delisted)."})`.
  - The "Lookback windows" table at full width, with the column header "Recency-weighted (t½ = {h} d)".
  - "Headlines" at full width: a segmented control (Latest | Most positive | Most negative) and rows of `sentimentBadge`, date and headline. 12 rows, then a "Show more" button that sets `n` to 24, 36 and so on.
  - Empty year: cards show "–" and the chart area shows the empty-state sentence.

- [ ] **Step 2: Check it in a browser.**
  - The crosshair shows the date and value anywhere over the chart. With price on, moving over one chart moves the crosshair on the other.
  - The windows table and headlines list no longer sit side by side at any width ≥ 390 px.
  - `ticker=nvda` resolves to NVDA. `ticker=ZZZZ` shows "No company called ZZZZ in the dataset." and resets to All companies.

- [ ] **Step 3: Commit.**

```bash
git add web/static/js/pages/company.js
git commit -m "Company Analysis page: crosshair timeline, synced price chart, unstacked tables"
```

---

### Task 9: Market Impact and Model Evaluation pages

**Files:**
- Create: `web/static/js/pages/market.js`, `web/static/js/pages/models.js`

**Interfaces:**
- Consumes: `/api/market-impact`, `/api/model-evaluation`; `barsWithCI`, `lineChart`, `heatmap`, `ui.js`.

- [ ] **Step 1: Build `market.js` against spec §8.3.**
  - A segmented control "Excluding price-move headlines | All relevant headlines" writes `headlines=excluding|all`.
  - `barsWithCI` of `spread` (units "%" or "pts"). Significant bars get a "significant" text marker; colour is not the signal.
  - CAR paths: a `lineChart` with two series, legend plus direct end labels "Positive days" and "Negative days", x as day −5…10, no area. If `car_paths_available` is false, show `note({kind: "info", text: "Event-window paths are not saved on this server."})`.
  - The three findings, as text from the current app:
    - "Mostly after the fact: much of the gap already exists the day before the headline date, and there is no significant drift over the following week."
    - "Bad news draws more trading: negative-sentiment days carry about 1.9× the abnormal volume."
    - "FNSPID has dates but no times, so a day +1 effect may simply be after-hours news."
  - A scope label "All companies · 2019–2023 · {events} company-days". The page ignores the company and period filters and says so in a caption.

- [ ] **Step 2: Build `models.js` against spec §8.5.**
  - Horizontal `barsWithCI` for "Financial PhraseBank test (N = 726)" and "Real headlines, six companies (N = 554)". The adapted-model PhraseBank bar has no whiskers and carries the note "No bootstrap CI saved for this model".
  - Caveats: "FinBERT was trained on PhraseBank, so its score there is likely inflated." and "Headline labels come from an LLM annotator and are not human-verified."
  - Domain-shift `note`: "Macro F1 falls from 0.84 on PhraseBank sentences to about 0.55–0.70 on real headlines. Read the timelines as a rough indicator of tone."
  - A `heatmap` of per-company × model macro F1. The selected company's row is outlined and the selected model's column header is bold.
  - The relevance table.
  - The line "Seed variation ({n} seeds): macro F1 {mean} ± {std}. Calibration: ECE {ece}, so probabilities are used as is."

- [ ] **Step 3: Check it in a browser.**
  - Hovering any bar shows the value and CI. Toggling the variant updates the bars and paths. The CAR crosshair shows the day and the % value.
  - Heatmap cells show their values in text. Every "Show as table" toggle works.

- [ ] **Step 4: Commit.**

```bash
git add web/static/js/pages/market.js web/static/js/pages/models.js
git commit -m "Market Impact and Model Evaluation pages"
```

---

### Task 10: News Explorer page

**Files:**
- Create: `web/static/js/pages/news.js`

**Interfaces:**
- Consumes: `/api/headlines`; `ui.js`; `state.js` (`q`, `label`, `sort`, `page` live in the URL).

- [ ] **Step 1: Build `news.js` against spec §8.4.**
  - A search input ("Search headlines", debounced by 300 ms; resets `page` to 1).
  - Filters: sentiment (All, Positive, Neutral, Negative). Company comes from the top bar. Date range comes from Period, or from two date inputs `start`/`end` that override it.
  - Sort: Newest, Oldest, Most positive, Most negative.
  - The result count line reads "{total} headlines". Under it, once: "Every headline here already passed the relevance filter."
  - Rows: ticker, `sentimentBadge`, confidence (as %), date, headline.
  - Numbered pagination with Previous/Next, which sets `page` in the URL.
  - Zero results: `emptyState("No headlines match. Clear the search or widen the date range.")`.

- [ ] **Step 2: Check it in a browser.**
  - Searching `(` returns the empty state, not an error. `page=999` in the URL shows the empty state with the true total.
  - Back after paging returns to the previous page of results.

- [ ] **Step 3: Commit.**

```bash
git add web/static/js/pages/news.js
git commit -m "News Explorer page: search, filters, pagination"
```

---

### Task 11: Score a Headline and Live Feed pages

**Files:**
- Create: `web/static/js/pages/score.js`, `web/static/js/pages/live.js`

**Interfaces:**
- Consumes: `POST /predict` (existing; body `{headlines, ticker}`), `/api/live/summary`; `lineChart`, `barsWithCI`, `connect`, `ui.js`.

- [ ] **Step 1: Build `score.js` against spec §8.6.**
  - A textarea "One headline per line" pre-filled with the three current examples, an optional ticker input defaulting to the top-bar company, and a "Score headlines" button. While a request runs, the button is disabled and reads "Scoring…".
  - The client trims and drops blank lines and caps the list at 50.
  - Results: headline, `sentimentBadge`, and a three-part probability bar (negative, neutral, positive) with the % in each part. When a ticker was given, a badge reads "About {TICKER}" or "Not about {TICKER}".
  - Errors:
    - 503: "The sentiment model isn't available on this server. Set SENTIMENT_MODEL to a Hugging Face Hub id."
    - 422: the API's `detail`.
  - Caption: "Known weakness: product recalls are often scored neutral rather than negative."

- [ ] **Step 2: Build `live.js` against spec §8.7.**
  - If `available` is false, show `emptyState` with "No live feed on this server yet. Start collecting with `python -m src.pipeline live-update`, then reload. See docs/deployment.md for running it daily."
  - Otherwise show:
    - "Last update {time} UTC · {runs} runs · {total} headlines across {tickers} companies".
    - Four cards.
    - The daily-mean `lineChart`, with a separate daily-count bar chart beneath, joined by `connect`.
    - The latest 50 headlines with publisher and an "Open article" link (`rel="noopener"`, new tab).
    - Alerts as caveat notes, or "No alerts in the latest run. Alerts need 30 days of daily collection."
    - The agreement line from `agreement`.
  - If `ticker_has_data` is false: "No live headlines for {TICKER} yet. Pick another company."
  - Caption: "Live headlines are scored once by the configured model; the Period and Model filters don't apply here."
  - The sidebar status footer switches to "Live · updated {time}".

- [ ] **Step 3: Check it in a browser.**
  - Score: three headlines show labels and probability bars. Stopping the model (`SENTIMENT_MODEL=missing/model`) shows the 503 sentence.
  - Live: with the local `data/live/news.db` present, the charts show crosshairs. With `live_db` pointed at a missing file, the empty state shows.

- [ ] **Step 4: Commit.**

```bash
git add web/static/js/pages/score.js web/static/js/pages/live.js
git commit -m "Score a Headline and Live Feed pages"
```

---

### Task 12: Retire Streamlit, ship the site to the Space, update docs and CI

**Files:**
- Delete: `app/streamlit_app.py`, `tests/test_dashboard.py`
- Modify:
  - `tests/test_deploy.py`
  - `deploy/publish.py` (`SPACE_FILES`, plus a new `SPACE_TREES = ["web"]`)
  - `deploy/space/Dockerfile`, `deploy/space/README.md`, `deploy/space/requirements.txt`
  - `requirements.txt`
  - `.github/workflows/ci.yml` (ruff paths)
  - `README.md`, `docs/deployment.md`, `docs/setup.md`, `docs/architecture.md`

**Interfaces:**
- Consumes: everything above.
- Produces: `stage_space` copies `SPACE_FILES` and each tree in `SPACE_TREES`.

- [ ] **Step 1: Rewrite the bundle test.** Replace `test_space_bundle_runs_the_dashboard_on_its_own` with `test_space_bundle_serves_the_site_on_its_own`.
  - It keeps the staging, README and `sys.path`/`sys.modules` isolation exactly as now.
  - It then runs `import src.api` from the bundle and builds `TestClient(src.api.create_app())`. With `src` imported from the bundle, every default path (`PROJECT_ROOT`, `WEB_DIR`, `METRICS_DIR`, `LIVE_DB`) resolves inside the bundle. The default `SentimentService` is constructed but never loads a model on these routes.
  - It asserts that `src.api.__file__` is under `tmp_path`.
  - It asserts that `/` contains "Market Impact", that `/static/js/pages/overview.js` returns 200, that `/api/overview` returns 200 with `cards.company_days.value == 24807`, and that `/api/live/summary` returns `{"available": False, …}`.
  - It also asserts `"app_port: 7860"` in the README and `"uvicorn"` in the Dockerfile.

- [ ] **Step 2: Run the test and confirm it fails.** Run `pytest tests/test_deploy.py -v`. Expected: FAIL (the bundle lacks `web/` and the dashboard modules).

- [ ] **Step 3: Update the bundle and the Space files.**
  - `SPACE_FILES`:
    - add `src/api.py`, `src/dashboard_api.py`, `src/dashboard_data.py`, `src/dashboard_metrics.py`, `src/dashboard_live.py`;
    - add `outputs/metrics/event_study_car_paths.csv`, `outputs/metrics/corpus_summary.json`, `outputs/metrics/adapted_test_metrics.json`;
    - remove `app/streamlit_app.py` and both CAR PNGs.
  - Space `Dockerfile`: `EXPOSE 7860`, `CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "7860"]`, and drop the `STREAMLIT_*` env vars.
  - Space README: `app_port: 7860`. Its bullet list describes the seven pages.
  - Space `requirements.txt`: remove `streamlit` and `altair`; add `fastapi>=0.110.0` and `uvicorn>=0.29.0`.
  - Root `requirements.txt`: remove `streamlit>=1.40.0`.
  - `ci.yml`: change `ruff check src tests app deploy` to `ruff check src tests deploy`.

- [ ] **Step 4: Delete the Streamlit app and its test.** Run `git rm app/streamlit_app.py tests/test_dashboard.py`. Remove `app/` if it is now empty.

- [ ] **Step 5: Update the docs.**
  - Replace every `streamlit run app/streamlit_app.py` with `uvicorn src.api:app --reload`, then open `http://localhost:8000`.
  - `docs/deployment.md` gains one line: "The root Docker image is API-only; the Hugging Face Space serves the dashboard."
  - `docs/architecture.md` describes `web/`, the `dashboard_*` modules and the `/api/` routes.
  - Leave `sarthak.md` and `answer.md` untouched.
  - Then run `grep -rni streamlit --exclude-dir=.venv --exclude-dir=.git --exclude-dir=notebooks --exclude-dir=.ruff_cache .`. Expected hits: only `.gitignore`, `sarthak.md` and `answer.md`.

- [ ] **Step 6: Run the full suite and lint.** Run `pytest -q && ruff check src tests deploy`. Expected: all tests pass and ruff prints `All checks passed!`. Run `python deploy/publish.py space --user someone` (dry run). Expected: lists the `web/` files and no Streamlit file.

- [ ] **Step 7: Final browser pass over all seven pages.**
  - Run at 1440 px and 390 px. Hover every chart, Tab through every control, and toggle `prefers-reduced-motion` in DevTools.
  - Check every Overview number against spec §6, and every chart against the dataviz anti-patterns: no dual axis, legend present for 2+ series, never colour-only.

- [ ] **Step 8: Commit.**

```bash
git add -A tests deploy requirements.txt .github/workflows/ci.yml README.md docs app
git commit -m "Retire Streamlit: Space serves the FastAPI dashboard; docs and CI updated"
```
