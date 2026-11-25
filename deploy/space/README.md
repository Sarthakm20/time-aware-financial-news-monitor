---
title: News Sentiment Monitor
emoji: 📰
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: cc-by-nc-sa-3.0
short_description: Nasdaq-100 headline sentiment, market reaction and model quality
---

# Time-Aware Financial News Sentiment Monitor

An interactive dashboard for headline sentiment across 89 Nasdaq-100 companies (2019–2023), in seven pages:
- **Overview**: coverage, the market-wide sentiment trend, the sentiment split, the relevance filter and the market relationship;
- **Company Analysis**: one company's sentiment over time (hover for exact values), with a synced share-price chart;
- **Market Impact**: how headline tone lines up with market-adjusted returns and trading volume;
- **News Explorer**: search and filter every scored headline;
- **Model Evaluation**: an honest model-quality report, including the domain-shift caveat;
- **Score a Headline**: score your own headlines with [`<hf-user>/distilbert-financial-headline-sentiment`](https://huggingface.co/<hf-user>/distilbert-financial-headline-sentiment);
- **Live Feed**: recent headlines, when the live feed runs on the server.

The same server also answers JSON requests: see `/docs` for the API.

Code, method and full evaluation: https://github.com/Sarthakm20/time-aware-financial-news-monitor

> Research prototype. Sentiment summarises the tone of dated headlines; it is not a trading signal or investment advice.
