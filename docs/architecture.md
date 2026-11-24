# Architecture, Notebooks & Code


## Project Architecture

```
┌──────────────────────────────────────┐
│ 1. Financial PhraseBank              │
│ Human‑labelled financial statements  │
│ Labels: positive / neutral / negative│
└──────────────────┬───────────────────┘
                   │
┌──────────────────▼───────────────────┐
│ 2. Data Preparation                  │
│ Clean text • Stratified split (70/15/15)
│ train (3,385) / val (725) / test (726)
└──────────────────┬───────────────────┘
                   │
          ┌────────┴────────┐
          │                 │
┌─────────▼─────────┐ ┌─────▼───────────────────────────────┐
│ 3A. Zero‑Shot     │ │ 3B. DistilBERT Fine‑Tuning          │
│ facebook/bart‑mnli│ │ • Standard Loss (Cross-Entropy)     │
│ No domain training│ │ • Class-Weighted Loss (Imbalance)   │
└─────────┬─────────┘ └─────┬───────────────────────────────┘
          │                 │
          └────────┬────────┘
                   │
┌──────────────────▼───────────────────┐
│ 4. Fair Model & Per-Class Evaluation │
│ Untouched test split (N=726)         │
│ Standard FT: 85.4% acc, 0.8388 F1    │
│ Negative class boost via weights     │
└──────────────────┬───────────────────┘
                   │
           Saved Best Model
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
│ Per-class F1 comparisons             │
│ Confusion matrices & loss curves     │
└──────────────────────────────────────┘
```

---

## Notebook Execution Order & Research Journey

The pipeline is split into two logical phases that reflect the complete research progression: first building and validating a trustworthy classifier, then deploying it into a real-world, time-aware monitoring workflow.

---

### Phase 1: Model Development, Evaluation & Imbalance Analysis (Notebooks 1–7)

> **Thought Process & Rationale:**  
> Before analyzing real market news, we need a reliable sentiment engine. Generic NLP models struggle with financial language. Phase 1 rigorously establishes baseline performance, fine-tunes domain awareness, and thoroughly evaluates edge cases and dataset skew.

1. `notebooks/phrasebank_eda.ipynb`  
   * **Why we do it:** Downloads and cleans human-annotated sentences from Financial PhraseBank. Establishes the ground truth and reveals initial challenges.
2. `notebooks/data_splitting.ipynb`  
   * **Why we do it:** Performs stratified 70/15/15 splitting. Stratification guarantees all three splits reflect identical class proportions, preventing evaluation bias.
3. `notebooks/zero_shot_baseline.ipynb`  
   * **Why we do it:** Measures zero-shot performance using `facebook/bart-large-mnli` on the untouched test split. Serves as our benchmark to answer: *"Do we actually need fine-tuning, or is an off-the-shelf LLM good enough?"*
4. `notebooks/finetuning_smoke_test.ipynb`  
   * **Why we do it:** Rapid sanity run on 100 samples to verify CUDA setup, tokenization pipelines, and Hugging Face `Trainer` logging before dedicating compute to full training.
5. `notebooks/finetune_distilbert.ipynb`  
   * **Why we do it:** Fine-tunes `distilbert-base-uncased` on the full training set. Checkpoints the best model based on validation macro F1.
6. `notebooks/final_test_evaluation.ipynb`  
   * **Why we do it:** Conducts side-by-side evaluation of the baseline vs fine-tuned model on the untouched test split. Computes confusion matrices, macro F1, and exports false predictions for error analysis.
7. `notebooks/per_class_and_weighted_training.ipynb`  
   * **Why we do it:** Directly investigates the label imbalance problem. Breaks down F1 performance per individual class (Negative, Neutral, Positive) and trains a third model using balanced class-weighted loss to honestly evaluate whether penalizing minority errors improves or degrades overall system utility.

---

### Phase 2: Time-Aware Historical News Monitoring Case Study (Notebooks 8–10)

> **Thought Process & Rationale:**  
> Financial markets are dynamic; a headline from yesterday matters far more than one from two years ago. Phase 2 applies the validated model to an unlabelled, dated historical news stream (Tesla, 2019–2023) and implements exponential recency weighting to answer: *"Is news coverage turning more positive or negative over recent windows?"*

8. `notebooks/download_fnspid_slice.ipynb`  
   * **Why we do it:** Streams and extracts dated historical headlines for a specific ticker (`TSLA`) from the large-scale FNSPID dataset, filtering for 2019–2023.
9. `notebooks/score_historical_headlines.ipynb`  
   * **Why we do it:** Deploys our saved fine-tuned DistilBERT model in high-throughput batch inference over 872 unlabelled headlines, generating class probabilities and a continuous tone score:  
     $$\text{Score} = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
10. `notebooks/time_aware_monitor.ipynb`  
    * **Why we do it:** Applies time-decay math ($w_i = 2^{-\Delta t / 30}$) and calculates weighted vs unweighted sentiment across multi-lookback windows (30d, 180d, 365d, 5-Year). Generates rolling timelines that capture shifts in media narrative over time.

---

## Codebase Modules (`src/`)

* `src/config.py`: Directory paths, model constants, hyperparameters, and window definitions.
* `src/preprocessing.py`: Text cleaning, stratified dataset splitting, and tokenization.
* `src/data_loading.py`: Financial PhraseBank downloading and FNSPID streaming.
* `src/baseline.py`: Zero-shot classification runner.
* `src/train.py`: DistilBERT fine-tuning pipeline with Hugging Face Trainer.
* `src/evaluate.py`: Accuracy, macro F1, confusion matrix, and error export.
* `src/inference.py`: PyTorch batch inference and continuous tone calculation.
* `src/aggregation.py`: Exponential recency weighting and rolling aggregations.
* `src/visualization.py`: Confusion matrices and rolling sentiment timeline plots.
* `src/per_class_evaluation.py`: Per-class F1, precision, and recall calculators with visualization.
* `src/weighted_trainer.py`: Custom `WeightedLossTrainer` implementing balanced class-weight CrossEntropyLoss.
* `src/pipeline.py`: Command-line entry point that runs the stages in [setup.md](setup.md) end to end.
* `src/stats.py`: Bootstrap CIs, paired bootstrap and McNemar tests, expected calibration error, temperature scaling.
* `src/labelling.py`: Blind headline sampling and a resumable keyboard labelling tool.
* `src/relevance.py` + `src/company_names.py`: Is a ticker-tagged headline actually about that company?
* `src/market.py`: Price download (yfinance), market-adjusted returns, abnormal volume.
* `src/event_study.py`: Event panel, date-clustered bootstrap, rank ICs, positive/negative spreads, CAR paths.
* `src/onnx_backend.py`: ONNX export, int8 quantization, torch-free inference (`onnxruntime` + `tokenizers`), latency benchmark.
* `src/labels.py`: Label-order mapping and probability summaries shared by the PyTorch and ONNX backends.
* `src/live_feed.py`: Live headline sources (Google News RSS, Yahoo, Finnhub, Gemini) and the SQLite store.
* `src/alerts.py`: Sentiment-drop and volume-spike alerts, webhook posting.
* `src/gemini.py`: Minimal Gemini REST client: blind headline annotation, search-grounded headline discovery.
* `src/env.py`: Reads API keys from the environment or a git-ignored `.env` file.
