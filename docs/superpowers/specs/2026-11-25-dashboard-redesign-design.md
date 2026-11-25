# Dashboard redesign: from Streamlit to a FastAPI-served analytics site

Date: 2026-11-25 · Branch: `dashboard-redesign` (cut from `docs-split`) · Status: approved 2026-11-25

## 1. Intent

The dashboard (`app/streamlit_app.py`) works, but its owner finds it hard to use and generic-looking:

- **Hovering a chart tells you nothing.** In the Company explorer, the Altair line only shows a tooltip when the
  cursor sits exactly on the 2 px stroke. The owner wants live coordinates (date and value) wherever the mouse is.
- **The Company explorer is congested.** "Lookback windows" and "Headlines" are squeezed side by side
  (`st.columns([2, 3])`).
- **It looks like every other Streamlit app.** The owner called the current look "AI slop" and asked to move off
  Streamlit.

The owner supplied a reference image for the look:
[`assets/2026-11-25-overview-reference.png`](assets/2026-11-25-overview-reference.png) (Overview page only). The
image is a mock-up: its layout and style are the brief, **its numbers are placeholders and must not be copied**
(see §6).

Success: a visitor can open any page, point at any chart and read exact values, move between companies and pages
without losing their place, and every number on screen traces back to a file in `outputs/` or the live database.

## 2. Goals, non-goals, anti-goals

**Goals**

- Seven pages in a fixed left sidebar: Overview, Company Analysis, Market Impact, News Explorer, Model Evaluation,
  Score a Headline, Live Feed.
- Every line/area chart has a crosshair with live axis labels and a tooltip; every bar/cell has a hover tooltip.
- A global filter row (company, period, model) that every page honours, held in the URL so a view can be shared.
- Visual language from the reference image (§7).
- Feature parity with today's five Streamlit tabs, plus a new cross-company News Explorer.
- The Hugging Face Space serves the new site instead of Streamlit.

**Non-goals**

- Dark mode (the reference is light; can come later).
- User accounts, saved views, or any write path other than scoring headlines.
- New analysis. The site presents results that already exist; the one data change is saving the CAR paths that the
  pipeline already computes (§9).
- ONNX benchmark or training-curve pages.

**Anti-goals (a failure even if it works)**

- Any number that does not come from the project's files, including the image's placeholders and an invented
  "last updated" time.
- Dual-axis charts (sentiment and price on two y-scales, as the current price overlay and live-feed chart do).
- Sentiment shown by colour alone.
- Template motion: fade-and-slide on every card, hover lift on every card.
- Losing the caveats (domain shift, LLM-annotated labels, "not a trading signal", coverage gaps). They are content,
  not footnotes.

## 3. Constraints

- Python stays the data layer; the existing FastAPI app (`src/api.py`) is extended, not replaced. Existing endpoint
  paths and response shapes stay as they are (`tests/test_api.py` covers them).
- No Node build step. The front end is static HTML, CSS and ES-module JavaScript.
- The Space bundle (`deploy/publish.py` `SPACE_FILES`) ships no raw data, models or secrets, and no `data/live/`.
  The Live Feed page must work (as an empty state) without the live database.
- The owner's laptop has 8 GB RAM; any heavy recompute goes to the Plaksha HPC.
- Licences: AdminLTE 4 (MIT), Bootstrap 5.3 (MIT), Apache ECharts (Apache-2.0). All vendored with pinned versions.

## 4. Approaches considered

| | Approach | Trade-off | Outcome |
|---|---|---|---|
| A | Static site served by the existing FastAPI app; AdminLTE 4 / Bootstrap 5 for layout; a JS chart library | Most work, and adds JavaScript to maintain; but full control over layout and hover, one container serves page and API | **Chosen** |
| B | Keep Streamlit; fix the hover with an Altair nearest-point rule, stack the tables, add a theme | Smallest change, Space and tests untouched; but Streamlit's sidebar/tabs/tables cannot take the reference layout, so it misses the main complaint | Rejected |
| C | React (Vite) SPA over FastAPI | Same result as A, plus Node and a build step in the Docker image; overkill for seven views | Rejected |
| – | Dash / Plotly / Panel | Another Python framework with its own look that fights AdminLTE's | Dropped early |

Chart library within A: **Apache ECharts** (owner was told ApexCharts earlier; changed because ECharts' `axisPointer`
draws the crosshair *with live labels on both axes*, which is literally the requested behaviour, and `echarts.connect`
syncs the crosshair across the stacked sentiment/price charts). Error bars use a small `custom` series.

## 5. Architecture

```
browser ── GET /            → web/index.html  (app shell: sidebar, top bar, <main>)
        ── GET /static/...  → web/ (css, js modules, vendored libs, fonts)
        ── GET /api/...     → JSON from outputs/ and data/live/news.db   (new)
        ── GET /tickers..., /live/{s}, POST /predict                    (existing, unchanged)
```

- **Single page, hash routes:** `#/overview`, `#/company`, `#/market`, `#/news`, `#/models`, `#/score`, `#/live`,
  with query state, e.g. `#/company?ticker=NVDA&period=2022&model=adapted`. Browser back/forward works; a copied
  URL reopens the same view.
- **Front-end layout:** `web/index.html`, `web/css/app.css` (tokens + overrides on AdminLTE), `web/js/main.js`
  (router + filter state), `web/js/api.js` (fetch wrapper, error/empty handling), `web/js/charts.js` (ECharts
  theme and shared builders: line+crosshair, bars+CI, donut, heatmap), one module per page in `web/js/pages/`.
  `web/vendor/` holds pinned AdminLTE, Bootstrap bundle and ECharts files; `web/fonts/` holds the self-hosted font.
- **Serving:** in `create_app`, mount `StaticFiles(directory=web, html=True)` **after** all API routes so API paths
  win, and **only when the `web/` directory exists**. FastAPI's own `/docs` stays.
- **Root Docker image stays API-only.** The root `Dockerfile` (built and smoke-tested by CI's `docker-api` job)
  ships `src/` and the adapted CSV only; it gets no `web/` and no metrics. There the site is not mounted and the
  new `/api/*` endpoints return 503 "not available on this server", like `/tickers` does without scores. The Space
  is the dashboard deployment. `docs/deployment.md` says so in one line.
- **Data loading:** scored CSVs (27,146 rows each) and metrics files are read once and cached in-process
  (`functools.lru_cache` keyed by path + mtime), the same way the Streamlit app cached them.

## 6. Numbers: what the image says vs what the data says

Every figure is computed from these sources (adapted model unless the Model filter says otherwise):

| Card | Reference image | Real value and source |
|---|---|---|
| Companies covered | 89 | 89 tickers in `nasdaq100_scored_headlines_*.csv` |
| Headlines | "62K+ analysed" | **61,818 collected** (`data/processed/fnspid_all_2019_2023.csv`), **27,146 kept and scored**. The card shows 27,146 scored, with "of 61,818 collected" beneath. 61,818 is not in the Space bundle: add it to a small `outputs/metrics/corpus_summary.json` written by the pipeline (or by a one-off script) |
| Real-headline macro F1 | ~0.57 vs 0.84 | Follows the Model filter: **0.655** adapted / **0.568** PhraseBank, pooled over 6 companies, N=554 (`multi_company_pooled_accuracy.csv`); 0.84 is the PhraseBank test score (`test_set_model_comparison.csv`) |
| Company-days | 24,807 | `event_study_summary.json` → `runs[*].events` (matches) |
| Sentiment distribution | 52 / 34 / 14 % | **26.6 % positive / 60.6 % neutral / 12.9 % negative** (adapted, all years). Recomputed for the selected company/period/model |
| Relevance filter | 44 %, 96–100 % precision | 27,146 / 61,818 = 43.9 %; precision 0.962–1.000 across the six audited tickers (`multi_company_relevance_filter.csv`) |
| Return by news tone | +0.62 / +0.01 / −0.47 % | Day-0 market-adjusted abnormal return, **excluding price-move headlines** (the conservative variant): **+0.26 % / +0.02 % / −0.58 %** with 95 % CIs (`event_study_buckets.csv`, outcome `ar_0`). Market Impact offers the "all headlines" variant (+0.39 / +0.04 / −0.81 %) |
| "~2× volume" | ~2× | Abnormal log-volume day 0: negative 0.081 vs positive 0.043 → **about 1.9×** (excluding price moves) |
| Last updated | Oct 27, 2023 4:32 PM | Invented. Historical pages show "Historical data · 2019–2023"; Live Feed shows the real last run time from the `runs` table |

Also fix the image's typo "Helevant Headlines".

## 7. Visual design

### Tokens

| Role | Value | Note |
|---|---|---|
| Sidebar | `#1c2740` | Deep navy from the reference |
| Sidebar active item | `#2d4a85` fill, white text, 3 px left bar `#7fb0f0` | |
| Page plane | `#f3f5f9` | |
| Card surface | `#ffffff`, 1 px border `#e3e7ee`, radius 10 px, no shadow | Border, not shadow, separates cards |
| Ink | primary `#141a26`, secondary `#4b5565`, muted (axes, captions) `#7a8394` | |
| Gridline / axis | `#e6e9ef` / `#c5cbd5` | |
| Series accent (single-series lines, area at 10 % alpha) | `#2a78d6` | Validated |
| Positive | `#087a5a` | Validated with negative and accent: lightness, chroma, contrast ≥ 3:1 pass; CVD ΔE 6.4 is in the 6–8 band, **legal only with secondary encoding** |
| Neutral | `#b9bec8` | Grey midpoint by design |
| Negative | `#ec6a50` | |
| Heatmap ramp | single-hue blue, dataviz reference steps 100→650 | |
| Caveat notes | text `#7a4b00` on `#fdf3e1`, with an icon | Status colour never alone |

Secondary encoding for sentiment, everywhere: the label word ("Positive"), the signed number (+0.26 %), and fixed
order positive → neutral → negative. Never a coloured dot by itself.

### Type

- **Libre Franklin** (self-hosted, weights 400/500/600/700). Franklin Gothic is the face of American financial news
  pages; it fits a news-sentiment tool without being the default Inter. One family throughout.
- Scale (px): page title 34/700, card title 17/600, stat value 30/700, body 15/400, caption 13/400, axis 12/400.
- `font-variant-numeric: tabular-nums` on tables and axis ticks only; stat values keep proportional figures. If the
  vendored Libre Franklin lacks `tnum`, the builder substitutes Public Sans (its tabular-figure derivative) and notes it.
- Sentence case for buttons and controls; card titles follow the reference's title case. No all-caps eyebrow labels.

### Layout

- Sidebar 248 px fixed: product name, seven nav items with icons, "Dataset context" block, data-status footer
  ("Historical data · not live" or "Live · updated <time>"). Collapses to an off-canvas drawer below 992 px
  (AdminLTE `data-lte-toggle="sidebar"`).
- Top bar, one row, right-aligned: Company combobox (All companies + 89 tickers, searchable), Period (All years,
  2019 … 2023), Model (Headline-adapted / PhraseBank). On mobile it wraps under the page title.
- Content: page title + one-line description, then a 12-column grid. Left-aligned throughout.

### Principles

1. **The trend chart is the one bold thing.** Every page's main time-series spans the width, with the crosshair as
   the signature interaction. Everything around it stays quiet.
2. **Numbers are sourced.** Each card's info icon opens a popover naming the file and definition.
3. **Caveats sit where the claim is made**, not in a footer.
4. **Motion only answers the user.** No load animations; chart transitions ≤ 200 ms on filter change; all motion off
   under `prefers-reduced-motion`.
5. **Every chart has "Show as table"** (dataviz accessibility rule), keyboard-focusable controls, visible focus ring.

## 8. Pages

"Period" filters data; chart 1Y/3Y/5Y buttons only zoom.

**8.0 Filter scope.** Only figures computed from the scored headlines can be re-cut by company and period. The
event study, relevance audit and F1 evaluation exist only as overall results (no per-ticker or per-year breakdown
in `outputs/`), and producing one would be new analysis (§2). Those figures therefore always show their overall
value with a visible scope label, "All companies · 2019–2023", whatever the filters say. They are never hidden and
never relabelled as the selected company's.

| Figure | Company filter | Period filter | Model filter |
|---|---|---|---|
| Headline counts, sentiment, distribution, trend, headline lists, News Explorer | yes | yes | yes |
| Real-headline F1 | per company for TSLA, AAPL, NVDA, AMZN, GILD, SBUX; otherwise pooled, labelled | no | yes |
| Relevance filter (kept share, precision) | no (labelled) | no | no |
| Event study: company-days, return by tone, volume ratio, CAR paths | no (labelled) | no | yes |
| Model Evaluation page | no; highlights the selected company's heatmap row if annotated | no | highlights the selected model |
| Live Feed | yes | no | no (caption explains) |

**8.1 Overview** (follows the reference image)
- Stat cards ×4 (§6). With a company selected: Companies covered becomes that company's headline count and date
  range; Headlines and sentiment follow the filters; F1 and Company-days follow §8.0.
- Company-days shows **24,807** (all relevant headlines, the full event set). The market-relationship bars use the
  excluding-price-moves run, so their caption states its own count, **22,999 company-days**.
- *Sentiment trend*: 30-day rolling mean sentiment across all selected headlines, weekly points, area fill,
  zero line dashed, 1Y/3Y/5Y zoom, crosshair. Named "30-day rolling sentiment", not "weighted", unless weighting
  is defined.
- *Sentiment distribution*: donut, centre shows the largest class, direct labels with percentages.
- *Headline relevance filter*: 44 % ring + "96–100 % precision on retained headlines" + one-line explanation.
- *Market relationship*: day-0 abnormal return by tone as bars with CI whiskers, plus two key findings in plain
  text (reacting vs predicting; ~1.9× volume). Links to Market Impact.

**8.2 Company Analysis** (replaces Company explorer; requires a company — if "All companies", default to TSLA)
- Stat cards: headlines (with first/last date), last-30-days sentiment with delta vs history, all-history sentiment,
  positive/negative share.
- Coverage-gap warning (gap > 90 days) and measured-accuracy note, as today.
- Rolling sentiment chart, full width; window slider (7–90 days) and recency half-life (7–180 days) as inline
  controls above it. Toggle "Show share price": a **second chart directly below** sharing the x-axis, crosshair
  synced (`echarts.connect`), prices from Yahoo via a new endpoint, "unavailable" note for delisted tickers.
- Lookback windows table, full width.
- Headlines, full width, as rows (ticker, label, score, date), segmented control Latest / Most positive / Most
  negative, 12 rows with "Show more".

**8.3 Market Impact** (replaces Market reaction)
- Toggle: excluding price-move headlines (default) / all relevant headlines.
- Positive-minus-negative spread for day −1, day 0, day +1, days +1…+5, abnormal volume: bars with 95 % CIs and
  a "significant" marker.
- Mean CAR path, day −5…+10, positive vs negative days: two lines with legend + direct end labels, crosshair.
  Needs new data (§9).
- Three findings as today ("mostly after the fact", "bad news draws more trading", "dates have no times").

**8.4 News Explorer** (new)
- Search box (case-insensitive substring on headline), filters company / sentiment / date range (Period by
  default), sort newest first or by score.
- Rows: ticker, label, score, confidence, date, headline. 25 per page, numbered pagination, result count.
- No relevance column: every stored headline already passed the filter (state this once under the results).

**8.5 Model Evaluation** (replaces Model quality)
- PhraseBank test (N=726) and pooled six-company headlines (N=554): horizontal bars of macro F1 with CI whiskers per
  model; accuracy and McNemar p in the table view. The headline-adapted model has no CI row in
  `test_set_model_comparison.csv`: show its 0.847 from `adapted_test_metrics.json` without whiskers, noted. FinBERT inflation note and "LLM-annotated, not
  human-verified" note.
- Domain-shift callout (0.84 → ~0.55–0.70).
- Per-company × model macro-F1 heatmap with values in cells.
- Relevance filter table (precision, recall estimate, on-topic share).
- Seed variation and calibration line.
- Model filter does not filter here (all models shown); it highlights the selected model's row.

**8.6 Score a Headline**
- Textarea (one per line, max 50), optional ticker, "Score headlines" button → `POST /predict`.
- Each result: headline, label, score, and a three-part probability bar (neg/neu/pos) with numbers; relevance
  badge when a ticker is given.
- Errors in the interface's voice: "The sentiment model isn't available on this server. Set SENTIMENT_MODEL to a
  Hugging Face Hub id." Loading state on the button while the model loads.

**8.7 Live Feed**
- Empty state when `data/live/news.db` is missing: the command to start collecting
  (`python -m src.pipeline live-update`) and a pointer to `docs/deployment.md`.
- Otherwise: last update, run count, sources; stat cards as today; daily mean sentiment line (crosshair) with a
  separate daily-headline-count bar strip beneath (no dual axis); latest 50 headlines with publisher and link;
  alerts list; live agreement check.
- Uses the global Company filter; Period and Model do not apply (live data is scored once, by the configured model;
  say so in a caption).

## 9. Data and API changes

New read-only endpoints (prefix `/api/`), all taking `model=adapted|base` where relevant, all returning plain JSON
with the disclaimer string where the existing endpoints do:

| Endpoint | Returns |
|---|---|
| `GET /api/overview?ticker=&period=&model=` | stat cards, distribution, weekly trend points, relevance figures, day-0 return by tone with CIs, volume ratio |
| `GET /api/company/{ticker}?period=&model=&window_days=&half_life_days=` | cards, timeline, windows table, headlines by order, coverage gap, measured accuracy |
| `GET /api/headlines?q=&ticker=&label=&start=&end=&sort=&page=&page_size=&model=` | paginated rows + total |
| `GET /api/market-impact?model=&headlines=` | spread rows with CIs, CAR paths |
| `GET /api/model-evaluation` | the comparison tables, heatmap cells, relevance table, seed/calibration |
| `GET /api/price/{ticker}?start=&end=` | adjusted close from yfinance, cached 24 h; `[]` when unavailable |
| `GET /api/live/summary?ticker=` | runs, daily series, latest headlines, alerts, agreement; 200 with `available:false` when no DB |

Existing `/tickers`, `/tickers/{s}/sentiment`, `/tickers/{s}/timeline`, `/live/{s}`, `/predict`, `/health` are
unchanged.

**CAR paths.** `src.event_study.mean_car_paths` is computed in `src/pipeline.py` (≈ line 711) but only plotted.
Change the event-study stage to also write `outputs/metrics/event_study_car_paths.csv` (model, headlines variant,
group, day, mean_car, n_events) and rerun that stage once for both models and both variants. It needs
`data/prices/daily.csv` (present locally). If it exceeds laptop memory, run it on the HPC.

**Corpus summary.** Write `outputs/metrics/corpus_summary.json` (`collected_headlines: 61818`, tickers, date
range) so the Space can show the collected count without raw data.

## 10. Retiring Streamlit and deployment

- Delete `app/streamlit_app.py` and `tests/test_dashboard.py`.
- `tests/test_deploy.py`: the bundle test boots the FastAPI app from the staged bundle with `TestClient` and checks
  `/`, one page asset, and `/api/overview`.
- `deploy/publish.py` `SPACE_FILES`: drop the Streamlit app and the two CAR PNGs; add `src/api.py`, the `web/`
  tree, `event_study_car_paths.csv`, `corpus_summary.json`, and any metrics files the new endpoints read
  (`event_study_information_coefficients.csv` only if used).
- `deploy/space/Dockerfile`: `CMD uvicorn src.api:app --host 0.0.0.0 --port 7860`; `deploy/space/README.md`:
  `app_port: 7860`, description updated; `deploy/space/requirements.txt`: drop streamlit/altair, add
  fastapi/uvicorn.
- Root `requirements.txt`: drop streamlit (altair goes with it).
- Docs: `README.md`, `docs/deployment.md`, `docs/setup.md`, `docs/architecture.md` — replace
  `streamlit run app/streamlit_app.py` with `uvicorn src.api:app --reload` and open `http://localhost:8000`.
  `sarthak.md` and `answer.md` are the owner's local notes: leave them.

## 11. Testing

- API tests in `tests/test_api.py` style for every new endpoint: shape, filters, pagination bounds, 404 for an
  unknown ticker, `available:false` live state, 503 from `/api/*` when metrics are absent, the site not mounted
  when `web/` is absent, numbers in §6 asserted against the committed files (e.g.
  distribution sums to 1, overview headline count 27,146).
- A static-serving test: `/` returns the shell, `/static/js/main.js` is served, API routes are not shadowed.
- Visual check during the build: run locally, review each page in a browser at 1440 px and 390 px wide, hover
  every chart, tab through every control. No JavaScript test framework.

## 12. Left to the builder

- Icon set (Bootstrap Icons, which AdminLTE uses, is the default).
- Exact ECharts option objects, tooltip formatting, and the error-bar implementation.
- Caching details, pagination component markup, and how the router lazy-loads page modules.
- Splitting `src/api.py` into a router module (e.g. `src/dashboard_api.py`) if it grows past ~400 lines.
- Font substitution per §7 if `tnum` is missing.

## 13. Calls made without the owner (check these)

- ECharts instead of ApexCharts (§4).
- Overview's return-by-tone uses the conservative "excluding price-move headlines" variant.
- Headlines card shows 27,146 scored with 61,818 collected beneath, instead of "62K+ analysed".
- Price overlay becomes a synced second chart, not a second y-axis.
- Libre Franklin as the typeface; light mode only.
- Branch cut from `docs-split` so the split docs are in place.
- Event-study, relevance and pooled-F1 figures stay overall (labelled) when a company or period is selected (§8.0),
  rather than adding a per-company event study.
- The root Docker image stays API-only; the Space serves the site (§5).
