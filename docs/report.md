# Report: Fine-Tuning Gains, Failures & Limitations


## Honest Accounting of Fine‑Tuning Gains

The zero‑shot BART‑MNLI baseline reached **50.00% accuracy**, below the 59% a model would get by always predicting the majority class (neutral), and it struggled most with neutral text (0.325 F1). Fine‑tuning DistilBERT on domain‑specific data lifted accuracy by **35.4 percentage points** and macro F1 by **0.31**, a large, practically meaningful improvement.

The zero‑shot baseline is also **prompt‑sensitive**: an alternative hypothesis template (`"This text expresses {} financial sentiment."`) scored only 45.18% accuracy / 0.4747 macro F1 on the same split. All reported baseline numbers use the single template in `src/config.py` (`ZERO_SHOT_HYPOTHESIS_TEMPLATE`).

The compute cost was modest: a single 2‑epoch training run took ~5–7 minutes on a GPU (or ~50 minutes on CPU). Therefore, the fine‑tuning effort was **well‑justified**; the performance uplift far exceeds the tiny compute investment.

---

## Reported Failures, Trade-offs & Limitations

- **Dataset size:** The Financial PhraseBank (50% agreement) contains only ~4.8k sentences, limiting the model’s exposure to diverse financial language. This small size is a known constraint and explains why further gains may plateau without additional data.
- **Class Imbalance & Weighted-Loss Trade-Off:** The PhraseBank is heavily skewed toward neutral sentences (59% neutral, 28% positive, 13% negative). On implementing balanced class-weighted loss training to address this skew, the **minority negative class F1 improved from 0.8394 to 0.8601**. However, this came at the expense of overall accuracy (dropping from 85.40% to 80.44%) and neutral F1 (dropping from 0.8871 to 0.8282). This honest negative trade-off confirmed that standard unweighted fine-tuning remains the superior all-round model for general news triage. (Both saved models are epoch-2 checkpoints; the weighted run's learning-rate schedule spanned 3 epochs, so the comparison is close to, but not exactly, like-for-like. The config now trains both for 2 epochs.)
- **Domain shift (measured, large):** The classifier is trained on PhraseBank sentences but applied to news *headlines*. On 230 annotated TSLA headlines its macro F1 is 0.496 vs 0.839 on PhraseBank ([Results §5](results.md)), so **the TSLA sentiment timeline should be read as a rough tone indicator, not a precise measurement.**
- **Annotation provenance:** The headline labels were produced by an LLM annotator (Claude), following `docs/annotation_guidelines.md`, and have **not yet been verified by a human**. Treat the headline results as provisional; a human pass (or a second annotator for inter-annotator agreement) would strengthen them.
- **Ticker tagging noise:** 65 of 300 sampled "TSLA" headlines (21.7%) are not about Tesla at all (other companies, crypto, personal finance). They are excluded from the headline evaluation but *are* included in the monitoring timeline.
- **FNSPID coverage:**
  - The Nasdaq-100 slice keeps **exactly one headline per company per day**, so a "daily sentiment" is one headline's score.
  - **13 of 89 tickers have gaps longer than 90 days, almost all starting 10 June 2020.** The large caps are hit hardest: AMZN is missing 1,002 days, AAPL 723, TSLA 691 (Jun 2020 – May 2022) and NVDA 433. GILD and SBUX are essentially complete.
  - Treat these stretches as missing data, not quiet news. The dashboard warns when a selected company has one.
- **Recency weighting on long windows:** With a 30-day half-life, headlines older than ~6 months carry under 2% weight, so the recency-weighted score is nearly identical for the 180-day, 365-day and all-history windows (~+0.10). Compare unweighted means across windows, and weighted vs unweighted within a window.
- **Data Freshness:** The open-source FNSPID dataset covers historical news up through late 2023. Consequently, all demonstrated timelines reflect historical research rather than live real-time feeds.
- **Scope & Non-Advisory:** It is meant to summarize tone in financial reporting. It does not predict stock prices or asset returns, and does not constitute trading signals or investment advice.
- **Input Scope:** This implementation classifies article headlines rather than full article bodies.
