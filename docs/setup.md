# Setup & Reproduction

```bash
git clone https://github.com/Sarthakm20/time-aware-financial-news-monitor.git
cd time-aware-financial-news-monitor

# Create and activate environment (Windows PowerShell shown)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (requirements-dev.txt adds pytest + ruff)
pip install -r requirements-dev.txt
```

## Command-Line Pipeline

Each stage reads the previous stage's files from disk, so stages can be run on their own:

| Stage | What it does | Writes |
|:---|:---|:---|
| `prepare` | Clean PhraseBank, drop conflicting/duplicate sentences, stratified 70/15/15 split | `data/processed/*.csv` |
| `baseline` | Zero-shot BART-MNLI on the test split (**hours on CPU**) | `zero_shot_test_*` |
| `train` | Fine-tune DistilBERT (`--weighted` also trains the class-weighted model) | `models/`, validation metrics |
| `evaluate` | Score the untouched test split | `finetuned_test_*`, `weighted_test_metrics.json`, figures |
| `fetch-news` | Stream one ticker's 2019–2023 headlines from FNSPID | `data/processed/fnspid_<ticker>_2019_2023.csv` |
| `score` | Label each headline and compute P(pos) − P(neg) | `outputs/predictions/<ticker>_scored_headlines_*.csv` |
| `monitor` | Window summaries and 30-day rolling timeline | `<ticker>_sentiment_window_summary.csv`, figure |
| `benchmark` | Off-the-shelf HF classifiers (default FinBERT; `--external-models`) on the test split and headlines | `<model>_test_*`, `<ticker>_scored_headlines_<model>.csv` |
| `analyze` | Bootstrap CIs, McNemar / paired-bootstrap tests, calibration, headline agreement | `test_set_model_comparison.csv`, `test_set_analysis.json`, reliability figure |
| `train-seeds` | Retrain DistilBERT per `--seeds` and report mean ± std (resumable) | `seed_runs.csv`, `seed_summary.json` |
| `sample-headlines` | Blind, year-stratified sample of headlines to hand-label (never overwrites) | `data/labelled/<ticker>_headlines_gold.csv` |
| `evaluate-headlines` | Score every model against the hand labels | `<ticker>_headline_evaluation.json`, comparison CSV |
| `adapt` | Continue fine-tuning on annotated headlines from other companies (+ PhraseBank replay) | `models/distilbert_headline_adapted`, `adapted_*` |
| `score-all` | Relevance-filter and score every ticker's headlines with each model (resumable) | `nasdaq100_scored_headlines_<model>.csv` |
| `prices` | Adjusted daily prices for all tickers + QQQ (Yahoo Finance; cached, not committed) | `data/prices/daily.csv` |
| `event-study` | Sentiment vs abnormal returns/volume, date-clustered bootstrap | `event_study_*.csv/json`, CAR figures |
| `evaluate-multi` | 6-company headline accuracy + relevance-filter precision/recall | `multi_company_*.csv/json` |
| `onnx-export` | Export the headline-adapted model to ONNX fp32 and quantize it to int8 | `models/distilbert_headline_adapted_onnx[_int8]` |
| `onnx-benchmark` | PyTorch vs ONNX fp32 vs int8: latency, memory, size, accuracy (McNemar) | `onnx_benchmark.json`, `onnx_benchmark_accuracy.csv` |
| `live-update` | Fetch current headlines (Google News RSS + Yahoo; optional Finnhub/Gemini), relevance-filter, score | `data/live/news.db` (not committed) |
| `alerts` | 7-day sentiment z-score and headline-volume spikes vs a 90-day baseline; optional Slack/Discord webhook | `data/live/alerts_latest.csv`, `alerts_log.csv` |
| `live-annotate` | Gemini labels a blind daily sample of live headlines → live model-agreement check | `data/live/live_agreement.json` |

```bash
python -m src.pipeline prepare                    # seconds; reproduces the committed splits exactly
python -m src.pipeline train --weighted evaluate  # retrain both models and re-score the test split
python -m src.pipeline benchmark analyze          # FinBERT + CIs, significance, calibration
python -m src.pipeline fetch-news score monitor --ticker AAPL
python -m src.pipeline all                        # every automatic stage, in order

# In-domain evaluation (needs a person)
python -m src.pipeline sample-headlines
python -m src.labelling                           # n / u / p / s, b = undo, q = quit; resumable
python -m src.pipeline evaluate-headlines

# Nasdaq-100 market study (~45 min on CPU, mostly scoring)
python -m src.pipeline fetch-news --ticker ALL
python -m src.pipeline adapt score-all prices event-study
```

`--data-dir`, `--outputs-dir` and `--models-dir` redirect inputs and outputs, which is useful for checking a re-run against the committed results without overwriting them.

## Tests & Linting

```bash
pytest              # unit tests + regression checks against the committed data and metrics
ruff check src tests deploy
```

The test suite covers:
- the split logic, recency weights, window and rolling aggregations, and metric calculations;
- inference post-processing (with a fake model, so no download is needed) and the pipeline CLI;
- the API and dashboard endpoints, the static site mount, and the Space bundle serving the site on its own.

It also verifies that `src/` reproduces the committed train/validation/test splits exactly and the published TSLA rolling-sentiment value. GitHub Actions runs lint and tests on every push, then **builds the API Docker image and smoke-tests the running container**.

The models (`models/`, ~4 GB with checkpoints) are not committed; run `python -m src.pipeline train` to recreate them.
