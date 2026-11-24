# Live Financial News Sentiment Monitor for the Nasdaq-100

[![CI](https://github.com/Sarthakm20/time-aware-financial-news-monitor/actions/workflows/ci.yml/badge.svg)](https://github.com/Sarthakm20/time-aware-financial-news-monitor/actions/workflows/ci.yml)

> **Important Limitation & Disclaimer:**  
> This project is a research and news‑triage prototype. Its sentiment scores summarize the tone of dated financial‑news coverage; they **do not predict stock returns, provide trading signals, or constitute investment advice**. FNSPID’s historical dataset ends in 2023; the live feed (Phase 5, see [Live news feed](docs/deployment.md#live-news-feed-phase-5b)) adds current, timestamped headlines from free sources.

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

   # Reproduce the pipeline from the command line (see docs/setup.md)
   python -m src.pipeline --list
   python -m src.pipeline prepare evaluate score monitor

   # Interactive dashboard and REST API (see docs/deployment.md)
   streamlit run app/streamlit_app.py
   uvicorn src.api:app --port 8000          # docs at http://localhost:8000/docs
   ```
   *The notebooks (execution order in [docs/architecture.md](docs/architecture.md)) document the research step by step; `python -m src.pipeline` is the reproducible path built on the same `src/` code.*

3. **What did you find?**  
   *The fine‑tuned DistilBERT achieves **85.40% accuracy** (Macro F1 **0.8388**) versus **50.00% accuracy** (Macro F1 **0.5299**) for the zero‑shot BART‑MNLI baseline. In a case study on Tesla headlines (2019–2023), headlines in the most recent 30 days average **+0.1238**, against **+0.0238** across all five years (both unweighted).*  
   *The most important finding: on 230 annotated real TSLA headlines, macro F1 falls from **0.839 to 0.496**, and no model beats always predicting "neutral" on accuracy. Sentence-level benchmark scores substantially overstate performance on headlines (see [Results §5](docs/results.md)).*  
   *Across **24,807 company-days for 85 Nasdaq-100 stocks**, positive-headline days outperform negative ones by **+1.2% on the same day**, but the gap is already **+0.5% the day before**, and there is no significant drift over the following week. The headlines largely *describe* price moves rather than predict them ([Results §7](docs/results.md)).*
   *Phase 5: an int8 ONNX export runs **6.9× faster (2.9 vs 20.2 ms per headline) in a fifth of the RAM** with unchanged accuracy, and a key-free live feed now scores this week's headlines for all 89 tickers ([Results §9](docs/results.md)).*  

4. **What would you do next with more time?**  
   * • Collect *timestamped* news (FNSPID has dates only), to separate after-hours news from genuine next-day predictability.  
   * • Have a human verify the LLM-annotated headline labels, and grow the headline training set.  
   * • Expand the training data beyond the ~4.8k‑sentence Financial PhraseBank.  
   * • Explore larger transformer architectures and longer fine‑tuning schedules to test diminishing returns.  
   * • Integrate a real-time news API and an online monitoring dashboard with alert thresholds .  
   * • Conduct a thorough statistical analysis to assess significance of sentiment shifts across windows.

---

## Documentation

| Document | What's in it |
|:---|:---|
| [docs/report.md](docs/report.md) | Honest accounting of the fine-tuning gains; failures, trade-offs and limitations |
| [docs/results.md](docs/results.md) | Key results §1–§9: model comparison, Tesla case study, rigorous evaluation, domain shift, seeds, event study, six-company evaluation, Phase 5 |
| [docs/architecture.md](docs/architecture.md) | Architecture diagram, notebook execution order, `src/` modules |
| [docs/setup.md](docs/setup.md) | Installation, the command-line pipeline and its stages, tests and linting |
| [docs/deployment.md](docs/deployment.md) | Dashboard, REST API, Docker, and the live news feed |
| [docs/methodology.md](docs/methodology.md) | Methodology notes |
| [docs/annotation_guidelines.md](docs/annotation_guidelines.md) | How the headline labels were made |
