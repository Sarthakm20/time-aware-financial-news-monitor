# Time-Aware Financial News Sentiment Monitor for Company Research

> **Important Limitation & Disclaimer:**
> This project is a research and news‑triage prototype. Its sentiment scores summarize the tone of dated financial‑news coverage; they **do not predict stock returns, provide trading signals, or constitute investment advice**. FNSPID’s historical dataset ends in 2023, demonstrating the system’s historical research workflow. Monitoring live headlines requires an external real‑time financial news API.

---

## Quick Overview – Four Core Questions

1. **What does this do?**
   *It fine‑tunes a lightweight DistilBERT model on the Financial PhraseBank, evaluates it against a zero‑shot NLI baseline, and then applies the model to historical company news to produce time‑aware sentiment summaries.*
2. **How do I run it?**
   ```bash
   # Clone the repo
   git clone https://github.com/Sarthakm20/time-aware-financial-news-monitor.git
   cd time-aware-financial-news-monitor

   # Set up Python environment (Windows PowerShell shown)
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   pip install -r requirements.txt

   # Run the notebook pipeline in the Notebook Execution Order given below
   ```
   *All notebooks import functions from `src/`; no manual path edits are required as long as you run them from the project root.*
3. **What did you find?**
   *The fine‑tuned DistilBERT achieves **85.4 % accuracy** (Macro F1 0.8388) versus the zero‑shot BART‑MNLI baseline at **50.0 % accuracy** (Macro F1 0.5299). In a case‑study on Tesla headlines (2019‑2023) the most recent 30‑day window shows a net‑positive weighted sentiment of **+0.1474**, whereas the five‑year historical average is near neutral (+0.0238).*
4. **What would you do next with more time?**
   *   • Expand the training data beyond the 6 k‑sentence Financial PhraseBank.
   *   • Explore larger transformer architectures and longer fine‑tuning schedules to test diminishing returns.
   *   • Integrate a real‑time news API and implement an online monitoring dashboard with alert thresholds.
   *   • Conduct a thorough statistical analysis to assess significance of sentiment shifts across windows.

---

## Honest Accounting of Fine‑Tuning Gains

The zero‑shot BART‑MNLI baseline performed at **50 % accuracy**, essentially random for a three‑class problem. Fine‑tuning DistilBERT on domain‑specific data lifted accuracy by **35.4 percentage points** and macro F1 by **0.31**, a substantial and practically meaningful improvement. The compute cost was modest: a single 3‑epoch training run on a consumer‑grade GPU (~5‑7 minutes, ≈0.1 GPU‑hour). Therefore, the fine‑tuning effort was **well‑justified**; the performance uplift far exceeds the tiny compute investment.

---

## Reported Failures & Limitations

- **Dataset size:** The Financial PhraseBank contains only ~6 k sentences, limiting the model’s exposure to diverse financial language. This small size is a known constraint and explains why further gains may plateau without additional data.
- No major code‑level failures were encountered after refactoring notebooks to import from `src/`.
- The open-source FNSPID dataset covers historical news up through late 2023. Consequently, all demonstrated timelines reflect historical data rather than live real-time data. Therefore it is outdated in that regards
- It is meant to summarize the tone in financial reporting. It does not predict the stock prices, asset returns.It is not an investment advice or constitute trading signals
- This implementation classifies article headlines rather than full article bodies.
---

## Key Results & Evidence

| Metric | Zero‑Shot Baseline  | Fine‑Tuned DistilBERT | Improvement |
|-------|----------------------------------------|----------------------|------------|
| **Accuracy** | 50.00 % | **85.40 %** | **+35.40 %** |
| **Macro F1** | 0.5299 | **0.8388** | **+0.3089** |
| **Negative F1** | 0.7063 | **0.8394** | +0.1331 |
| **Neutral F1** | 0.3250 | **0.8871** | **+0.5621** |
| **Positive F1** | 0.5583 | **0.7900** | +0.2317 |

*Fine‑tuning dramatically improves the model’s ability to recognise neutral financial sentiment, which zero‑shot NLI struggles with.*

---

## Project Architecture

```
┌──────────────────────────────────────┐
│ 1. Financial PhraseBank                │
│ Human‑labelled financial statements    │
│ Labels: positive / neutral / negative │
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ 2. Data Preparation                    │
│ Clean text • Stratified split (70/15/15)│
│ train / val / test                     │
└──────────────────┬───────────────────┘
                   │
          ┌─────────┴─────────┐
          │                   │
┌────────▼──────────┐ ┌──────▼───────────┐
│ 3A. Zero‑Shot     │ │ 3B. Fine‑Tuned   │
│ facebook/bart‑mnli│ │ distilbert‑base │
│ No domain training│ │ 3 epochs, lr=2e‑5│
└────────┬──────────┘ └──────┬───────────┘
          │                   │
          └─────────┬─────────┘
                    │
┌──────────────────▼───────────────────┐
│ 4. Fair Model Evaluation             │
│ Untouched test split (N=726)         │
│ Accuracy: 85.4 % vs 50.0 %            │
│ Macro F1: 0.8388 vs 0.5299           │
└──────────────────┬───────────────────┘
                    │
            Saved Trained Model
                    │
┌──────────────────▼───────────────────┐
│ 5. FNSPID Historical News            │
│ Filter: Ticker (TSLA), 2019–2023     │
│ 872 unique dated headlines           │
└──────────────────┬───────────────────┘
                    │
┌──────────────────▼───────────────────┐
│ 6. Batch Inference                   │
│ Calculate P(pos), P(neu), P(neg)     │
│ Continuous Score = P(pos) - P(neg)   │
└──────────────────┬───────────────────┘
                    │
┌──────────────────▼───────────────────┐
│ 7. Time‑Aware Aggregation            │
│ Compare 30d / 180d / 365d / All‑Time │
│ Apply 30‑day half‑life recency decay │
└──────────────────┬───────────────────┘
                    │
┌──────────────────▼───────────────────┐
│ 8. Visualizations & Outputs          │
│ 30‑day rolling sentiment timeline    │
│ Weighted vs unweighted comparison    │
│ Confusion matrices & loss curves     │
└──────────────────────────────────────┘
```

---

## Notebook Execution Order

1. `notebooks/phrasebank_eda.ipynb`
2. `notebooks/data_splitting.ipynb`
3. `notebooks/zero_shot_baseline.ipynb`
4. `notebooks/finetuning_smoke_test.ipynb`
5. `notebooks/finetune_distilbert.ipynb`
6. `notebooks/final_test_evaluation.ipynb`
7. `notebooks/download_fnspid_slice.ipynb`
8. `notebooks/score_historical_headlines.ipynb`
9. `notebooks/time_aware_monitor.ipynb`

---

## Codebase Modules (`src/`)

* `src/config.py` – paths, constants, hyper‑parameters.
* `src/preprocessing.py` – cleaning, stratified split, tokenisation.
* `src/data_loading.py` – PhraseBank download, FNSPID streaming.
* `src/baseline.py` – zero‑shot classification runner.
* `src/train.py` – DistilBERT fine‑tuning pipeline.
* `src/evaluate.py` – accuracy, macro F1, confusion matrix, error export.
* `src/inference.py` – batch inference and continuous sentiment score.
* `src/aggregation.py` – exponential recency weighting, rolling aggregations.
* `src/visualization.py` – plots and confusion‑matrix visualisation.

---

## Setup & Reproduction

```bash
git clone https://github.com/Sarthakm20/time-aware-financial-news-monitor.git
cd time-aware-financial-news-monitor

# Create and activate environment (Windows PowerShell shown)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt
```

Run the notebooks in the order listed above