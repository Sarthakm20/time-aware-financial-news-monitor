# Methodology: Time-Aware Financial News Sentiment Monitor

This document details the data preparation, modeling, evaluation, and time-aware aggregation methodology implemented in this repository.

---

## 1. Dataset Selection & Separation of Concerns

To guarantee experimental validity and avoid data contamination, the project strictly separates classifier training from product deployment:

1. **Training & Evaluation Layer:**  
   * **Dataset:** [Financial PhraseBank](https://huggingface.co/datasets/takala/financial_phrasebank) (50% agreement configuration).  
   * **Nature:** 4,840 human-annotated sentences from English financial news.  
   * **Labels:** Three mutually exclusive categories: `negative`, `neutral`, and `positive`.  
   * **Purpose:** Proving model capability, generalization, and domain adaptation against a general-purpose NLP baseline.

2. **Downstream Application Layer:**  
   * **Dataset:** [FNSPID (Financial News and Stock Price Integration Dataset)](https://github.com/Zdong104/FNSPID_Financial_News_Dataset).  
   * **Nature:** Dated news headlines linked to stock ticker symbols (1999–2023).  
   * **Company Evaluated:** Tesla (`TSLA`), 872 unique headlines spanning July 2019 to December 2023.  
   * **Rule:** FNSPID's automated sentiment labels are **never** used to evaluate or train the classifier. The classifier is applied as a frozen model over dated text.

---

## 2. Data Preparation & Stratified Splitting

Financial PhraseBank exhibits natural class imbalance:
* `neutral`: 59.36% (2,873 statements)
* `positive`: 28.16% (1,363 statements)
* `negative`: 12.48% (604 statements)

Before splitting, sentences that appear with **conflicting labels** are removed entirely and remaining repeated sentences are kept once, so no text can appear in more than one split (4,840 → 4,836 statements; `src.preprocessing.prepare_model_dataframe`).

To preserve identical class distributions across all subsets, a two-stage stratified split was performed with fixed seed `42`:
* **Train Split (70%):** 3,385 statements
* **Validation Split (15%):** 725 statements
* **Test Split (15%):** 726 statements

The held-out test split remained completely untouched during model selection and hyperparameter tuning.

---

## 3. Modeling & Fine-Tuning Setup

### Zero-Shot Baseline
* **Model:** `facebook/bart-large-mnli` (407M parameters).
* **Framework:** Multi-class zero-shot Natural Language Inference (NLI).
* **Hypothesis Template:** `"The financial sentiment of this statement is {}."` (`ZERO_SHOT_HYPOTHESIS_TEMPLATE` in `src/config.py`). This gives the reported 50.00% accuracy / 0.5299 macro F1.
* **Prompt sensitivity:** the alternative template `"This text expresses {} financial sentiment."` scored 45.18% / 0.4747 on the same split. It is reported here for transparency and not used elsewhere.

### Fine-Tuned Model
* **Base Architecture:** `distilbert-base-uncased` (66M parameters).
* **Sequence Classification Head:** Linear layer mapping the pooled `[CLS]` token to 3 logits.
* **Hyperparameters:**
  * Optimizer: AdamW
  * Learning Rate: $2 \times 10^{-5}$
  * Batch Size: 16
  * Epochs: 2 (212 optimizer steps per epoch; the saved model is checkpoint 424)
  * Weight Decay: 0.01
  * Max sequence length: 256 tokens
  * Best Model Selection: Monitored on validation macro F1 at every epoch.

### Class-Weighted Variant
* Same setup with balanced cross-entropy weights `[2.67, 0.56, 1.18]` (negative, neutral, positive) via `src.weighted_trainer.WeightedLossTrainer`.
* The reported run was scheduled for 3 epochs. Its best validation checkpoint (and the saved model) is epoch 2, so both models saw the same number of epochs, but their learning-rate schedules differ slightly. Re-running with the current config (2 epochs) gives an exactly matched comparison.

---

## 4. Headline Scoring Formulation

Given model logits, class probabilities are computed via the softmax function:

$$P(\text{negative}) + P(\text{neutral}) + P(\text{positive}) = 1.0$$

To summarize tone as a scalar rather than discrete categorical labels, each headline receives a **continuous sentiment score**:

$$\text{Sentiment Score} = P(\text{positive}) - P(\text{negative}) \in [-1.0, +1.0]$$

* A strongly positive statement ($P(\text{pos}) = 0.95, P(\text{neg}) = 0.01$) yields $+0.94$.
* A factual neutral announcement ($P(\text{neu}) = 0.96$) yields $\approx 0.00$.
* A negative statement ($P(\text{neg}) = 0.88, P(\text{pos}) = 0.02$) yields $-0.86$.

---

## 5. Time-Aware Recency Decay & Aggregation

News impact decays over time. A negative headline from yesterday carries immediate informational relevance, whereas a headline from two years ago belongs to historical baseline context.

### Exponential Recency Weighting
For a headline published at date $t_i$ relative to reference as-of date $t_{\text{as-of}}$:

$$\Delta t_i = t_{\text{as-of}} - t_i \quad (\text{in days})$$

$$w_i = 2^{-\frac{\Delta t_i}{\lambda}}$$

Where $\lambda$ is the half-life parameter, set to **30 days**:
* $\Delta t = 0$ days $\implies w = 1.000$
* $\Delta t = 30$ days $\implies w = 0.500$
* $\Delta t = 60$ days $\implies w = 0.250$
* $\Delta t = 180$ days $\implies w = 0.0156$

### Weighted Window Average
For any lookback window $W$:

$$\bar{S}_{\text{weighted}} = \frac{\sum_{i \in W} w_i \cdot S_i}{\sum_{i \in W} w_i}$$

Comparing $\bar{S}_{\text{weighted}}$ against the simple arithmetic mean $\bar{S}_{\text{unweighted}}$ within a window shows whether the most recent part of that window differs from the window as a whole.

**Caveat:** because $w < 0.016$ beyond 180 days, $\bar{S}_{\text{weighted}}$ is almost the same for every window longer than ~6 months. To compare recent coverage with the long-term average, compare **unweighted** means across windows (e.g. last 30 days vs all history).

### Rolling Timeline
The 30-day rolling series (`src.aggregation.compute_rolling_sentiment`) is headline-weighted: for each calendar day it divides the sum of scores in the trailing 30 days by the number of headlines in that span. Days without news are kept in the calendar but do not dilute the average.
