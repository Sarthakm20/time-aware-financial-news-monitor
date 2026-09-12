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
   *The fine‑tuned DistilBERT achieves **85.40% accuracy** (Macro F1 **0.8388**) versus the zero‑shot BART‑MNLI baseline at **45.18%–50.00% accuracy** (Macro F1 **0.4747–0.5299**). In a case‑study on Tesla headlines (2019–2023), the most recent 30‑day window shows a net‑positive weighted sentiment of **+0.1474**, whereas the five‑year historical average is near neutral (**+0.0238**).*

4. **What would you do next with more time?**  
   * • Expand the training data beyond the 6k‑sentence Financial PhraseBank.  
   * • Explore larger transformer architectures and longer fine‑tuning schedules to test diminishing returns.  
   * • Integrate a real‑time news API and implement an online monitoring dashboard with alert thresholds.  
   * • Conduct a thorough statistical analysis to assess significance of sentiment shifts across windows.

---

## Honest Accounting of Fine‑Tuning Gains

The zero‑shot BART‑MNLI baseline performed at **~45–50% accuracy**, essentially random for a three‑class financial problem (struggling severely with neutral text at 0.18–0.32 F1). Fine‑tuning DistilBERT on domain‑specific data lifted accuracy by **35–40 percentage points** and macro F1 by **over 0.30**, a massive, practically meaningful improvement. 

The compute cost was modest: a single 3‑epoch training run on a standard setup took ~5–7 minutes (or ~50 minutes on CPU). Therefore, the fine‑tuning effort was **well‑justified**; the performance uplift far exceeds the tiny compute investment.

---

## Reported Failures, Trade-offs & Limitations

- **Dataset size:** The Financial PhraseBank contains only ~6k sentences, limiting the model’s exposure to diverse financial language. This small size is a known constraint and explains why further gains may plateau without additional data.
- **Class Imbalance & Weighted-Loss Trade-Off:** The PhraseBank is heavily skewed toward neutral sentences (59% neutral, 28% positive, 13% negative). On implementing balanced class-weighted loss training to address this skew, the **minority negative class F1 improved from 0.8394 to 0.8601**. However, this came at the expense of overall accuracy (dropping from 85.40% to 80.44%) and neutral F1 (dropping from 0.8871 to 0.8282). This honest negative trade-off confirmed that standard unweighted fine-tuning remains the superior all-round model for general news triage.
- **Data Freshness:** The open-source FNSPID dataset covers historical news up through late 2023. Consequently, all demonstrated timelines reflect historical research rather than live real-time feeds.
- **Scope & Non-Advisory:** It is meant to summarize tone in financial reporting. It does not predict stock prices or asset returns, and does not constitute trading signals or investment advice.
- **Input Scope:** This implementation classifies article headlines rather than full article bodies.

---

## Key Results & Evidence

### 1. Overall Model Comparison (Held-out Test Split, N=726)

| Metric | Zero‑Shot Baseline  | Fine‑Tuned DistilBERT | Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | 45.18% – 50.00% | **85.40%** | **+35.40% to +40.22%** |
| **Macro F1** | 0.4747 – 0.5299 | **0.8388** | **+0.3089 to +0.3641** |
| **Negative F1** | 0.6980 | **0.8394** | +0.1414 |
| **Neutral F1** | 0.1845 | **0.8871** | **+0.7026** |
| **Positive F1** | 0.5417 | **0.7900** | +0.2483 |

*General-domain NLI zero-shot models heavily fail on financial neutrality (down to 0.1845 F1). Fine-tuning DistilBERT on domain text pushes neutral F1 to 0.8871.*

### 2. Per-Class F1 & Class-Weighted Loss Experiment

To investigate the label imbalance (neutral dominance vs sparse negatives), we trained a third model using balanced class weights (`weights = [2.67, 0.56, 1.18]` for negative, neutral, and positive respectively):

| Class | Zero-Shot F1 | Fine-Tuned F1 (Standard) | Weighted-Loss Fine-Tuned F1 |
|:---|:---:|:---:|:---:|
| **Negative** | 0.6980 | 0.8394 | **0.8601** |
| **Neutral** | 0.1845 | **0.8871** | 0.8282 |
| **Positive** | 0.5417 | **0.7900** | 0.7432 |
| **Overall Accuracy** | 0.4518 | **0.8540** | 0.8044 |
| **Macro F1** | 0.4747 | **0.8388** | 0.8105 |

* Visualizations generated in `outputs/figures/`:
  - `per_class_f1_baseline_vs_finetuned.png`
  - `per_class_f1_standard_vs_weighted.png`

### 3. Time-Aware Monitoring Case Study: Tesla (`TSLA`, 2019–2023)

Evaluated across 872 dated news headlines from FNSPID:

$$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
$$\text{Recency Weight} = 2^{-\Delta t / 30} \quad (\text{Half-life} = 30 \text{ days})$$

| Lookback Window | Headline Count | Unweighted Sentiment | Recency-Weighted Sentiment | % Positive | % Neutral | % Negative |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Last 30 days** | 30 | **+0.1238** | **+0.1474** | 16.67% | 76.67% | 6.67% |
| **Last 180 days** | 178 | +0.0655 | +0.1019 | 15.17% | 73.60% | 11.24% |
| **Last 365 days** | 362 | +0.0400 | +0.1017 | 16.02% | 69.34% | 14.64% |
| **All History (5 Years)** | 872 | +0.0238 | +0.1017 | 14.45% | 69.38% | 16.17% |

---

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

Run the notebooks in the order listed above. All logic imports modularly from `src/`.


---

## Link for briefing about the idea for the project

https://drive.google.com/file/d/1cAkBFAwn92-9G91cO5rYZCoCQIeb6KHW/view?usp=drive_link
