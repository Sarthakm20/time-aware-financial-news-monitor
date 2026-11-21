---
title: News Sentiment Monitor
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
license: cc-by-nc-sa-3.0
short_description: Nasdaq-100 headline sentiment, market reaction and model quality
---

# Time-Aware Financial News Sentiment Monitor

An interactive dashboard for headline sentiment across 89 Nasdaq-100 companies (2019–2023):
- per-company sentiment timelines, with an optional share-price overlay;
- an event study of how sentiment lines up with market-adjusted returns and trading volume;
- an honest model-quality report, including the domain-shift caveat;
- live scoring of your own headlines with [`<hf-user>/distilbert-financial-headline-sentiment`](https://huggingface.co/<hf-user>/distilbert-financial-headline-sentiment).

Code, method and full evaluation: https://github.com/Sarthakm20/time-aware-financial-news-monitor

> Research prototype. Sentiment summarises the tone of dated headlines; it is not a trading signal or investment advice.
