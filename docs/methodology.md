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
* **Hypothesis Template:** `"This text expresses {} financial sentiment."`

### Fine-Tuned Model
* **Base Architecture:** `distilbert-base-uncased` (66M parameters).
* **Sequence Classification Head:** Linear layer mapping the pooled `[CLS]` token to 3 logits.
* **Hyperparameters:**
  * Optimizer: AdamW
  * Learning Rate: $2 \times 10^{-5}$
  * Batch Size: 16
  * Epochs: 3
  * Weight Decay: 0.01
  * Best Model Selection: Monitored on validation macro F1 at every epoch.

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

Comparing $\bar{S}_{\text{weighted}}$ against the simple arithmetic mean $\bar{S}_{\text{unweighted}}$ reveals whether recent coverage is diverging from the company's long-term historical average.
