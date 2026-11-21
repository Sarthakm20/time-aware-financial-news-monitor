---
license: cc-by-nc-sa-3.0
language:
- en
library_name: transformers
pipeline_tag: text-classification
base_model: distilbert/distilbert-base-uncased
datasets:
- takala/financial_phrasebank
tags:
- finance
- sentiment-analysis
- financial-news
- distilbert
widget:
- text: "Tesla Deliveries Top Estimates as Demand Surges"
- text: "Intel Cuts Full-Year Guidance as PC Demand Slumps"
- text: "Apple To Report Q3 Earnings On Thursday"
---

# DistilBERT Financial Headline Sentiment

A 66M-parameter DistilBERT that labels financial news headlines **negative / neutral / positive**. It was fine-tuned on Financial PhraseBank, then adapted to news headlines. It is part of the [Time-Aware Financial News Sentiment Monitor](https://github.com/Sarthakm20/time-aware-financial-news-monitor), whose README has the full method and evaluation.

> Research prototype. The scores summarise the tone of headlines; they do not predict returns and are not trading signals or investment advice.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="<hf-user>/distilbert-financial-headline-sentiment", top_k=None)
scores = {d["label"]: d["score"] for d in clf("Tesla Deliveries Top Estimates as Demand Surges")[0]}
sentiment = scores["positive"] - scores["negative"]   # continuous score in [-1, 1]
```

## Training

| Stage | Data | Details |
|:---|:---|:---|
| 1. Fine-tune | Financial PhraseBank (50% agreement), 3,385 train sentences | 2 epochs, lr 2e-5, batch 16, best epoch by validation macro F1 |
| 2. Headline adaptation | 318 FNSPID headlines from 87 Nasdaq-100 companies (**LLM-annotated**) + 318 PhraseBank sentences (replay) | 3 epochs, max length 64, best epoch by macro F1 on 80 held-out headlines |

No Tesla headlines were used in training. Tesla is held out as an unseen-company test.

## Evaluation

| Test set | Macro F1 [95% CI] | Accuracy |
|:---|:---:|:---:|
| Financial PhraseBank test split (N=726) | 0.847 | 85.3% |
| Held-out headlines from training companies (N=80) | 0.740 | 78.8% |
| Tesla headlines, unseen company (N=230, LLM-annotated) | 0.523 [0.449, 0.591] | 56.1% |

- **Domain shift is the main limitation.** On headlines from a company it never saw, performance falls sharply, and no model tested beat an always-"neutral" baseline on accuracy there.
- **Market alignment:** across 24,807 company-days (85 Nasdaq-100 stocks, 2019–2023), this model's daily scores correlate more strongly with same-day market-adjusted returns than the PhraseBank-only model's (rank IC 0.133 vs 0.081).
- **Calibration:** the PhraseBank stage is well calibrated (ECE 0.021).

## Limitations and biases

- **Annotation provenance:** the headline labels used for adaptation and for the Tesla evaluation come from an LLM annotator (Claude) following written guidelines. They are **not human-verified**.
- **Known errors:** product recalls are often labelled neutral, and some strongly positive headlines ("Why X Stock Popped Today") are labelled neutral.
- **Scope:** English headlines only, US large caps, 2019–2023. Many headlines describe price moves after the fact.

## License

The weights are released under **CC BY-NC-SA 3.0**, inherited from Financial PhraseBank, the training data (Malo et al., 2014). That means **non-commercial use only**, with attribution and share-alike. The base model (DistilBERT) is Apache-2.0. The project code is MIT.

## Citation

Malo, P., Sinha, A., Korhonen, P., Wallenius, J., & Takala, P. (2014). *Good debt or bad debt: Detecting semantic orientations in economic texts.* JASIST 65(4).
