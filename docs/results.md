# Key Results & Evidence


## 1. Overall Model Comparison (Held-out Test Split, N=726)

| Metric | Zero‑Shot Baseline  | Fine‑Tuned DistilBERT | Improvement |
|:---|:---:|:---:|:---:|
| **Accuracy** | 50.00% | **85.40%** | **+35.40 pp** |
| **Macro F1** | 0.5299 | **0.8388** | **+0.3089** |
| **Negative F1** | 0.7063 | **0.8394** | +0.1331 |
| **Neutral F1** | 0.3250 | **0.8871** | **+0.5621** |
| **Positive F1** | 0.5583 | **0.7900** | +0.2317 |

*Sources: `outputs/metrics/zero_shot_test_metrics.json` and `outputs/metrics/finetuned_test_metrics.json`. The general-domain NLI zero-shot model fails most on financial neutrality (0.325 F1); fine-tuning DistilBERT on domain text pushes neutral F1 to 0.8871.*

## 2. Per-Class F1 & Class-Weighted Loss Experiment

To investigate the label imbalance (neutral dominance vs sparse negatives), we trained a third model using balanced class weights (`weights = [2.67, 0.56, 1.18]` for negative, neutral, and positive respectively):

| Class | Zero-Shot F1 | Fine-Tuned F1 (Standard) | Weighted-Loss Fine-Tuned F1 |
|:---|:---:|:---:|:---:|
| **Negative** | 0.7063 | 0.8394 | **0.8601** |
| **Neutral** | 0.3250 | **0.8871** | 0.8282 |
| **Positive** | 0.5583 | **0.7900** | 0.7432 |
| **Overall Accuracy** | 0.5000 | **0.8540** | 0.8044 |
| **Macro F1** | 0.5299 | **0.8388** | 0.8105 |

*Weighted-loss numbers: `outputs/metrics/weighted_test_metrics.json`.*

* Visualizations generated in `outputs/figures/`:
  - `per_class_f1_baseline_vs_finetuned.png`
  - `per_class_f1_standard_vs_weighted.png`

## 3. Time-Aware Monitoring Case Study: Tesla (`TSLA`, 2019–2023)

Evaluated across 872 dated news headlines from FNSPID:

$$\text{Sentiment Score} = P(\text{Positive}) - P(\text{Negative}) \in [-1.0, +1.0]$$
$$\text{Recency Weight} = 2^{-\Delta t / 30} \quad (\text{Half-life} = 30 \text{ days})$$

| Lookback Window | Headline Count | Unweighted Sentiment | Recency-Weighted Sentiment | % Positive | % Neutral | % Negative |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Last 30 days** | 30 | **+0.1238** | **+0.1474** | 16.67% | 76.67% | 6.67% |
| **Last 180 days** | 178 | +0.0655 | +0.1019 | 15.17% | 73.60% | 11.24% |
| **Last 365 days** | 362 | +0.0400 | +0.1017 | 16.02% | 69.34% | 14.64% |
| **All History (5 Years)** | 872 | +0.0238 | +0.1017 | 14.45% | 69.38% | 16.17% |

*Source: `outputs/metrics/tsla_sentiment_window_summary.csv`. Read the unweighted column across windows: coverage in the latest 30 days (+0.12) is more positive than the five-year average (+0.02). The weighted column converges to ~+0.10 for long windows because of the 30-day half-life (see Limitations).*

## 4. Rigorous Evaluation: Uncertainty, Significance, Calibration, Stronger Baseline

Single accuracy numbers on 726 sentences hide how much they could move. Every model below was re-scored with **95% bootstrap confidence intervals** (2,000 resamples), compared with the fine-tuned DistilBERT on the *same* sentences using **McNemar's exact test**, and checked for **calibration** (expected calibration error, ECE).

| Model (test split, N=726) | Macro F1 [95% CI] | Accuracy [95% CI] | ECE | vs DistilBERT (McNemar *p*) |
|:---|:---:|:---:|:---:|:---:|
| **DistilBERT fine-tuned** (ours) | **0.839** [0.807, 0.866] | **85.4%** [82.6, 87.7] | **0.021** | (reference) |
| DistilBERT, class-weighted loss | 0.811 [0.779, 0.840] | 80.4% [77.4, 83.2] | 0.069 | **0.0001**: significantly worse |
| FinBERT (`ProsusAI/finbert`, off-the-shelf)† | 0.894 [0.867, 0.917] | 89.9% [87.7, 92.2] | 0.027 | 0.0002† |
| Zero-shot BART-MNLI | 0.530 [0.496, 0.566] | 50.0% [46.3, 53.7] | 0.113 | < 10⁻⁶ |
| Always predict "neutral" | 0.248 | 59.4% [55.8, 63.0] | – | < 10⁻⁶ |

† **FinBERT was itself fine-tuned on Financial PhraseBank**, so it has very likely seen many of these test sentences. Its higher score here is not a fair win. The fair comparison is on data neither model trained on (the headline evaluation below).

**Findings**
1. **Fine-tuning gain is real and large:** +0.31 macro F1 over zero-shot, with non-overlapping CIs. Zero-shot BART-MNLI is *below* the trivial always-neutral baseline in accuracy (50.0% vs 59.4%).
2. **Class-weighted loss is significantly worse** overall (*p* = 0.0001). The earlier conclusion holds statistically, not just numerically.
3. **DistilBERT is already well calibrated** (ECE 0.021). Temperature scaling fitted on the validation split gives T = 1.07 and does not improve the test split (ECE 0.0216, NLL 0.365 → 0.365), so no recalibration is applied. The probabilities behind the P(pos) − P(neg) score can be read at face value, at least on in-domain sentences. Reliability diagram: `outputs/figures/finetuned_reliability_diagram.png`.
4. **Evidence of domain shift, before any hand labels:** DistilBERT and FinBERT agree on **89.0%** of PhraseBank test sentences (Cohen's κ = 0.81) but only **77.5%** of the 872 real TSLA headlines (κ = 0.57). FinBERT also labels 24.5% of headlines negative vs DistilBERT's 16.2%. At least one model degrades on headlines; the hand-labelled set (below) measures which, and by how much.

*Sources: `outputs/metrics/test_set_model_comparison.csv`, `outputs/metrics/test_set_analysis.json`, `outputs/metrics/finbert_test_metrics.json`.*

## 5. In-Domain Evaluation on Real Headlines: Large Domain Shift

PhraseBank sentences are not what the monitor actually scores. To measure performance where it matters, 300 TSLA headlines were sampled **blind** (no model output shown) and stratified by year, then annotated following `docs/annotation_guidelines.md` ("good, bad or neither news *for Tesla*?"). 230 received a sentiment label (133 neutral, 56 positive, 41 negative); 65 were **off-topic** (not about Tesla) and 5 unclear.

> ⚠️ **Annotation provenance:** these labels were produced by an **LLM annotator (Claude)**, not a human, and are recorded as such in `data/labelled/tsla_headlines_gold.csv` (`annotator` column). They have not yet been human-verified, so treat the numbers below as provisional.

| Model (N=230 annotated headlines) | Macro F1 [95% CI] | Accuracy | vs DistilBERT (McNemar *p*) | *Company-specific only (N=196)*: Macro F1 |
|:---|:---:|:---:|:---:|:---:|
| DistilBERT fine-tuned | 0.496 [0.421, 0.564] | 57.4% | (reference) | 0.551 [0.470, 0.631] |
| DistilBERT class-weighted | 0.546 [0.477, 0.613] | 58.3% | 0.86 | 0.606 [0.531, 0.680] |
| FinBERT (off-the-shelf) | 0.508 [0.435, 0.576] | 55.7% | 0.67 | 0.572 [0.498, 0.644] |
| Always predict "neutral" | 0.244 | **57.8%** | 1.00 | 0.225 |

**Findings**
1. **Headline performance is far below benchmark performance.** DistilBERT drops from **0.839 → 0.496 macro F1 (−0.34)**, and on accuracy no model beats always predicting "neutral". The fall is mostly on the polar classes: positive recall is just 0.27, e.g. *"Here's Why Tesla Stock Popped Today"* → neutral, and *"Tesla Is 'Crushing The Auto Industry'"* → negative.
2. **The drop is not an artefact of the labelling convention.** The guidelines label market-wide headlines ("Wall Street surges…") neutral *for Tesla*, while models score their general tone. Removing those 34 headlines raises DistilBERT only to 0.551, still **~0.29 below** its benchmark score. On company-specific headlines all models do beat the majority class (61% vs 51% accuracy).
3. **FinBERT's benchmark lead disappears on unseen data** (0.894 → 0.508, no better than DistilBERT, *p* = 0.67). This supports the contamination explanation for its test-split score.
4. **Model ranking flips:** the class-weighted model, significantly worse on PhraseBank, is the best of the three on headlines (0.546; 0.606 company-specific), because headlines contain proportionally more polar news. The difference is not significant at N=230, but it suggests weighting is worth revisiting for the deployed use case.
5. **Data quality:** 21.7% of sampled "TSLA" headlines are not about Tesla. A relevance filter would remove noise from the monitoring timeline.

*Sources: `outputs/metrics/tsla_headline_evaluation.json`, `outputs/metrics/tsla_headline_model_comparison.csv`. Reproduce: `python -m src.pipeline evaluate-headlines`.*

## 6. Seed Variation: The Result Is Not a Lucky Seed

DistilBERT was retrained from scratch with different random seeds (same data, same 2-epoch recipe) and each run was scored on the same test split:

| Seed | Accuracy | Macro F1 | Negative F1 | Neutral F1 | Positive F1 |
|:---:|:---:|:---:|:---:|:---:|:---:|
| 42 (reported model) | 85.40% | 0.8388 | 0.8394 | 0.8871 | 0.7900 |
| 43 | 85.54% | 0.8444 | 0.8602 | 0.8886 | 0.7845 |
| 44 | 85.26% | 0.8394 | 0.8478 | 0.8853 | 0.7852 |
| **Mean ± std (n = 3)** | **85.40% ± 0.14** | **0.841 ± 0.003** | 0.849 ± 0.010 | 0.887 ± 0.002 | 0.787 ± 0.003 |

Run-to-run variation (±0.003 macro F1) is **an order of magnitude smaller** than the bootstrap CI width (±0.03) and two orders smaller than the PhraseBank → headline drop (−0.34), so none of the conclusions above depend on the seed. The reported model is a typical run, not a favourable one.

*n = 3: two further seeds (45, 46) were planned, but the CPU runs were stopped because the machine ran low on memory. `python -m src.pipeline train-seeds` resumes from the finished seeds and adds them. Sources: `outputs/metrics/seed_runs.csv`, `outputs/metrics/seed_summary.json`.*

## 7. Does Headline Sentiment Line Up With the Stock? Nasdaq-100 Event Study

**Scale-up.** All 61,818 FNSPID headlines for **89 Nasdaq-100 tickers (2019–2023)** were filtered for relevance (below), leaving 27,146 headlines. These were scored by both the PhraseBank model and a **headline-adapted** model, then matched to adjusted daily prices for 85 tickers (Yahoo Finance; ANSS, EA, SGEN and WBA have since been delisted).

**Step 1: Relevance filter.** FNSPID's ticker tags are noisy: only **43.9%** of all headlines mention the tagged company. `src/relevance.py` is a transparent keyword rule (symbol, company name, products, executives; ambiguous symbols such as `ON` and `FAST` only as `$ON` / `(ON)`; "Musk" only when the headline is not about Twitter/X or SpaceX). On the 300 annotated TSLA headlines it reaches **precision 1.00** and keeps **93 of 97** headlines with a clear sentiment. The 67 annotated headlines it drops are almost all neutral listicles ("5 Stocks to Buy…").

**Step 2: Headline-adapted model.** The PhraseBank model was fine-tuned further on 318 annotated headlines from **87 other companies** (never Tesla), mixed with 318 PhraseBank sentences so it would not forget what it had learned:

| Model | PhraseBank test | Other-company headlines (val, N=80) | Tesla headlines (unseen company, N=230) |
|:---|:---:|:---:|:---:|
| DistilBERT (PhraseBank only) | 0.839 | – | 0.496 |
| **DistilBERT headline-adapted** | **0.847** (no forgetting) | **0.740** | 0.523 (+0.03, *p* = 0.73) |

It learns the headline style it was trained on, but the gain does not transfer significantly to Tesla's unusual coverage. Section 7's price data gives an independent check, below.

**Step 3: Event study design.**
- **Unit:** one (stock, trading day) with ≥1 relevant headline, scored as the mean P(pos) − P(neg).
- **Returns:** market-adjusted (stock − QQQ).
- **Volume:** log volume vs its prior 20-day average.
- **Uncertainty:** 95% CIs from a bootstrap **clustered by date** (500 resamples).
- **Timing:** FNSPID has **dates without times**, so day −1 and day 0 can't separate "news caused the move" from "news reported the move". Days +1…+5 are the test of whether sentiment carries information about what comes next.

Results for the adapted model, excluding headlines that merely report price moves ("Why X Stock Popped Today", 7% of days). Spread = positive-sentiment days minus negative-sentiment days, with |sentiment| > 0.3:

| Window | Positive days (n=6,037) | Negative days (n=2,517) | **Spread** [95% CI] | Rank IC (all days) |
|:---|:---:|:---:|:---:|:---:|
| Day −1 (before) | +0.28% | −0.18% | **+0.46%** [+0.34, +0.58] | 0.061* |
| Day 0 | +0.26% | −0.58% | **+0.83%** [+0.71, +0.96] | 0.079* |
| Day +1 | +0.09% | −0.12% | +0.21% [+0.07, +0.33] | 0.011 (n.s.) |
| Days +1…+5 | +0.21% | +0.06% | +0.15% [−0.10, +0.41] (n.s.) | 0.008 (n.s.) |
| Abnormal volume, day 0 | +4.3% | **+8.1%** | −3.8 pts [−5.8, −1.7] | −0.043* |

<sub>*95% CI excludes zero. Rank IC = Spearman correlation between day sentiment and the outcome across all 22,999 company-days. Full tables for both models and both headline sets: `outputs/metrics/event_study_information_coefficients.csv` and `event_study_buckets.csv`.</sub>

![Cumulative abnormal return around headline days](../outputs/figures/event_study_car_adapted.png)

**Findings**
1. **Sentiment tracks price moves strongly, but mostly after the fact.** Positive-news days beat negative-news days by about 0.8–1.2% on the day, yet a large part of the gap (+0.46%) is already there the *day before*. The figure shows most of the move happening up to day 0. Headlines largely report and explain moves.
2. **No evidence of a lasting signal.** The five-day cumulative spread is not significant for either model or headline set (*p* = 0.06–0.83), consistent with prices absorbing the news quickly. The +0.2% day-+1 spread is small. Without timestamps it may simply be news published after the close, and with ~40 tests it should be treated as suggestive only.
3. **Bad news draws more trading.** Negative-sentiment days have about twice the abnormal volume of positive ones (+8–14% vs +4–6% above normal), significantly in every specification. Attention is asymmetric.
4. **External validation of the headline adaptation.** Without using any labels, the adapted model's scores line up better with same-day market reactions than the PhraseBank model's: rank IC **0.133 vs 0.081** on all headlines, 0.079 vs 0.049 excluding price-move headlines. The model adapted to headlines measures market-relevant tone better, even though the small Tesla label set could not show a significant difference.
5. **Price-move headlines inflate the association.** Excluding them cuts the same-day IC by about 40% (0.133 → 0.079). Any news-sentiment study should control for headlines that merely narrate returns.

> This is an association study on historical data. It does not show that sentiment can be traded profitably, and it is not investment advice.

## 8. Accuracy Beyond Tesla: Six-Company Headline Evaluation

The headline accuracy check (§5) and the relevance-filter check (§7) originally covered Tesla only. They now cover **five more companies from different sectors**: Apple (hardware), Nvidia (semiconductors), Amazon (retail/cloud), Gilead (biotech) and Starbucks (consumer).

**Sampling (blind, per company):**
- 80 headlines the relevance filter keeps, which is what the system actually scores;
- 30 headlines it drops, so the filter's recall can be measured;
- the two strata are re-weighted by their true sizes;
- headlines used in training or already labelled are excluded;
- 550 new labels in total, LLM-annotated under `docs/annotation_guidelines.md` and **not human-verified**.

**Sentiment accuracy on headlines the system scores** (on-topic and passing the filter):

| Model | 5 new companies (N=391) | All 6 incl. Tesla (N=554) | vs PhraseBank model (McNemar *p*, all 6) |
|:---|:---:|:---:|:---:|
| DistilBERT (PhraseBank only) | 0.578 [0.517, 0.634] | 0.569 [0.518, 0.615] | (reference) |
| DistilBERT class-weighted | 0.653 [0.594, 0.702] | 0.643 [0.597, 0.686] | 0.028 |
| **DistilBERT headline-adapted** | **0.688** [0.634, 0.737] | **0.655** [0.608, 0.696] | **0.005** |
| FinBERT (off-the-shelf) | 0.609 [0.549, 0.661] | 0.600 [0.551, 0.644] | 1.00 |
| Always "neutral" | 0.241 | 0.230 | < 10⁻⁶ |

<sub>Macro F1 [95% bootstrap CI]. Accuracy, all 6: 65.0% (PhraseBank model) / 69.9% (adapted) vs 52.7% always-neutral.</sub>

| Company | PhraseBank model | Headline-adapted | FinBERT | N |
|:---|:---:|:---:|:---:|:---:|
| TSLA | 0.537 | 0.575 | 0.565 | 163 |
| AAPL | 0.558 | 0.619 | 0.648 | 77 |
| NVDA | 0.548 | **0.709** | 0.633 | 80 |
| AMZN | 0.510 | 0.588 | 0.513 | 76 |
| GILD | 0.581 | **0.705** | 0.543 | 80 |
| SBUX | 0.568 | **0.706** | 0.613 | 78 |

**Relevance filter, per company:**

| | TSLA | AAPL | NVDA | AMZN | GILD | SBUX |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Precision | 1.00 | 0.96 | 1.00 | 0.96 | 1.00 | 1.00 |
| Recall (re-weighted) | 0.67 | 0.41 | 0.39 | 0.57 | 0.62 | 0.63 |

**Findings**
1. **The domain shift is general, not a Tesla quirk.** The PhraseBank model scores 0.51–0.58 macro F1 on every company's headlines, against 0.84 on PhraseBank sentences.
2. **Headline adaptation helps significantly once more companies are measured** (+0.09 macro F1 pooled, *p* = 0.005). The gain is largest where headlines resemble the training headlines (NVDA, GILD, SBUX ≈ 0.70) and smallest for Tesla, the one company held out of training entirely. Caveat: the training and evaluation labels come from the same LLM annotator, so part of the gain may be learning that annotator's conventions. Human labels would settle this.
3. **The relevance filter is precise everywhere (0.96–1.00).** Its recall is modest, but **96% of the on-topic headlines it drops are neutral** generic listicles. It keeps 262 of the 268 sampled headlines that carry sentiment (98%), so it removes noise while losing almost no signal.
4. **Every model beats the always-neutral baseline** on the filtered headlines (65–70% vs 53% accuracy), unlike on Tesla's unfiltered sample (§5). Filtering out off-topic headlines is part of why.

*Sources: `outputs/metrics/multi_company_*.csv`, `multi_company_evaluation.json`, per-company `<ticker>_headline_evaluation.json`. Reproduce: `python -m src.pipeline evaluate-multi`.*

**The same per-company treatment as Tesla.** Each of the five companies went through the Tesla workflow (`python -m src.pipeline fetch-news score monitor benchmark evaluate-headlines --ticker <T>`). Each now has:
- a time-aware monitor: 30/180/365-day and all-history windows with recency weighting, plus a 30-day rolling chart (`outputs/metrics/<t>_sentiment_window_summary.csv`, `outputs/figures/<t>_30day_rolling_sentiment.png`);
- headlines scored by all four models;
- a headline evaluation with the domain-shift figure, the company-specific sensitivity check and model agreement (`<t>_headline_evaluation.json`).

| Latest 30 days vs all history (unweighted, PhraseBank model) | TSLA | AAPL | NVDA | AMZN | GILD | SBUX |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Last 30 days | +0.124 | +0.148 | +0.149 | −0.009 | +0.084 | −0.009 |
| All history | +0.024 | +0.056 | +0.083 | +0.082 | +0.113 | +0.082 |

## 9. Phase 5: A 6.9× Faster Model and a Live News Feed

**5A. ONNX Runtime export and int8 quantization** (`python -m src.pipeline onnx-export onnx-benchmark`).
The headline-adapted model on 4 CPU threads (AMD EPYC 9335, Plaksha HPC). Latency is end to end: tokenize,
forward pass and softmax, median of 3 runs over 1,000 headlines. Each variant ran in a fresh process.

| | PyTorch fp32 | ONNX fp32 | **ONNX int8** |
|:---|:---:|:---:|:---:|
| ms per headline, batch 1 | 20.2 | 6.6 (3.1×) | **2.9 (6.9×)** |
| ms per headline, batch 32 | 23.5 | 10.1 | **3.5 (6.7×)** |
| Peak RAM (whole process) | 1,240 MB | 519 MB | **245 MB** |
| Model file | 255 MB | 255 MB | **64 MB** |
| Load time | 6.6 s | 0.24 s | **0.17 s** |
| PhraseBank test macro F1 (n=726) | 0.847 | 0.847 | 0.846 (McNemar *p* = 0.74) |
| 6-company headlines macro F1 (n=554) | 0.655 | 0.655 | 0.659 (McNemar *p* = 0.47) |

- **ONNX fp32 is exact:** identical labels to PyTorch, with probabilities within 2×10⁻⁶.
- **int8 keeps accuracy but changes some individual labels.** It agrees with fp32 on 92.6% of the real headlines.
  Of the 41 disagreements, 36 soften a positive or negative headline to neutral. A neutral headline never changes.
  Only **5 (0.9%) reverse the polarity**, and the sentiment scores correlate at 0.965 (`onnx_int8_vs_fp32.json`).
  For an aggregate tone monitor this is a good trade. Use ONNX fp32 if every single label matters.
- **The ONNX path never imports PyTorch.** That is most of the memory saving, and it is why the live feed runs
  comfortably on an 8 GB laptop. The first full run scored 3,643 headlines with a 321 MB peak.
- Batching does not help per-headline latency on this CPU, because headlines are short and each batch is padded
  to its longest headline.
- **A portability lesson.** Per-channel weight scales agreed better with fp32 on the AMD server: 97.3% vs 94.9% of
  PhraseBank labels (`scripts/onnx_quant_sweep.py`). On an Intel laptop, though, the same file returned
  near-uniform probabilities, labelling every headline neutral. The default is therefore per-tensor. Every export
  also saves fp32 labels for 100 reference headlines, and `OnnxSentimentModel.self_check()` refuses to load an
  int8 model that disagrees on more than 20% of them on the current CPU (92% on the laptop, 89% on the server).

**5B. Live news feed** (see [Live news feed](deployment.md#live-news-feed-phase-5b) for how to run it).
- 89 tickers, from Google News RSS and Yahoo Finance. No API key is needed, and every headline has a real
  publish *time*, which FNSPID lacks.
- First week: **5,608 headlines collected, 4,379 about their company and scored.** The label mix is 47% neutral,
  31% positive and 22% negative.
- **Live accuracy check:** `gemini-3.1-flash-lite` labelled 50 random live headlines blind, following the same
  guidelines. The model agrees on **78% (95% CI 66–88%), macro F1 0.78 [0.66, 0.88]**.
  - Gemini is a second LLM annotator, not ground truth; the interval narrows as daily runs add labels.
  - This is in line with the 0.65–0.70 macro F1 measured against the Claude labels on 2019–2023 headlines (§8).

*Sources: `outputs/metrics/onnx_benchmark.json`, `onnx_benchmark_accuracy.csv`, `onnx_int8_vs_fp32.json`,
`onnx_quantization_sweep.json`; live figures from `data/live/` (not committed; regenerated by `live-update` and
`live-annotate`).*
