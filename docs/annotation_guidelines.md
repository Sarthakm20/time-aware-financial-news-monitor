# Headline Annotation Guidelines

These labels form the **in-domain evaluation set**: they measure how well models trained on
Financial PhraseBank *sentences* work on real news *headlines*. They are never used for training.

## The question to ask

> *If I were an investor in this company, is this headline good news, bad news, or neither?*

Label the headline's implication **for the company in the ticker** (e.g. Tesla for `TSLA`), as
written, using only the headline text. Don't look anything up, and don't use what you know
happened afterwards.

| Label | Use when | Examples |
|:---|:---|:---|
| `positive` | Good news for the company: beats, upgrades, record deliveries, new contracts, price-target raises, strong demand | "Tesla Deliveries Top Estimates", "Analyst Raises Tesla Price Target To $300" |
| `negative` | Bad news for the company: misses, downgrades, recalls, lawsuits, investigations, production problems, price-target cuts | "Tesla Recalls 120,000 Vehicles", "Tesla Shares Slide After Delivery Miss" |
| `neutral` | Factual or mixed with no clear direction: schedules, previews, lists, market-wide moves, questions, explainers | "Tesla To Report Q2 Earnings On Wednesday", "5 Stocks Moving In Pre-Market Trading" |
| `skip` | You genuinely can't tell (unreadable, needs context you don't have), or the headline is **about a different company / not about the ticker at all** | |

When skipping, optionally fill the `note` column with `off-topic` (not about the company) or `unclear`, so ticker-tagging noise can be measured. Record who labelled each row in the `annotator` column (e.g. your name, or the model used if an LLM annotated it).

## Rules of thumb

- **Direction words decide it.** "Shares jump" → positive; "shares fall" → negative. "Shares move" / "big move" → neutral.
- **Questions and previews are neutral** unless they clearly lean one way ("Is Tesla's Rally Over?" → neutral).
- **Several companies in one headline:** judge Tesla only. If Tesla is just one name in a market list, use neutral.
- **Mixed news** (good and bad in the same headline) → neutral.
- **Opinion pieces count:** "Why Tesla Stock Is A Buy" → positive.
- **Don't overthink it.** About 5–10 seconds per headline. If you're unsure after that, use `skip`.

## Process

1. `python -m src.pipeline sample-headlines`: draws 300 headlines at random, stratified by year (no model output shown).
2. `python -m src.labelling`: keyboard labelling (`n` / `u` / `p` / `s`, `b` = undo, `q` = quit). Progress is saved after every label.
3. `python -m src.pipeline evaluate-headlines`: scores every model on the labelled headlines, with 95% bootstrap CIs.

If a second person labels the same file independently, inter-annotator agreement (Cohen's κ) can be reported too.
