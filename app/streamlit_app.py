"""Interactive dashboard: Nasdaq-100 headline sentiment, market reaction and model quality.

Run locally:  streamlit run app/streamlit_app.py
Everything except the "Score a headline" tab works from the committed outputs/ files.
"""

import json
import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import FIGURES_DIR, METRICS_DIR, PREDICTIONS_DIR  # noqa: E402
from src.explorer import ticker_headlines, ticker_overview, ticker_timeline, ticker_windows  # noqa: E402

DISCLAIMER = ("Research prototype. Sentiment summarises the tone of dated news headlines; it does not predict "
              "returns and is not a trading signal or investment advice. Data: FNSPID (2019-2023).")
SCORED = {"Headline-adapted DistilBERT": "nasdaq100_scored_headlines_adapted.csv",
          "PhraseBank DistilBERT": "nasdaq100_scored_headlines_base.csv"}
COLORS = {"positive": "#2E7D32", "neutral": "#9E9E9E", "negative": "#C62828"}

st.set_page_config(page_title="News Sentiment Monitor", page_icon="📰", layout="wide")


# ----------------------------------------------------------------------------- data


@st.cache_data(show_spinner=False)
def load_scored(filename: str) -> pd.DataFrame:
    df = pd.read_csv(PREDICTIONS_DIR / filename)
    df["Date"] = pd.to_datetime(df["Date"], utc=True)
    df["Stock_symbol"] = df["Stock_symbol"].str.upper()
    return df


@st.cache_data(show_spinner=False)
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(METRICS_DIR / name)


@st.cache_data(show_spinner=False)
def load_json(name: str) -> dict:
    return json.loads((METRICS_DIR / name).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_price(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Adjusted close from Yahoo Finance; empty frame if unavailable (delisted, offline, ...)."""
    try:
        import yfinance as yf

        px = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        if px.empty:
            return pd.DataFrame()
        close = px["Close"].squeeze()
        return pd.DataFrame({"date": pd.to_datetime(close.index).tz_localize(None), "close": close.to_numpy()})
    except Exception:  # noqa: BLE001 - optional overlay must never break the page
        return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def get_service():
    from src.service import SentimentService

    return SentimentService()


# ----------------------------------------------------------------------------- header

st.title("📰 Time-Aware Financial News Sentiment Monitor")
st.caption(DISCLAIMER)

with st.sidebar:
    st.header("Settings")
    model_choice = st.radio("Sentiment model", list(SCORED), index=0,
                            help="The headline-adapted model lines up better with same-day market reactions "
                                 "(rank IC 0.133 vs 0.081).")
    scored = load_scored(SCORED[model_choice])
    overview = ticker_overview(scored)
    tickers = overview.sort_values("headlines", ascending=False)["ticker"].tolist()
    ticker = st.selectbox("Company (ticker)", tickers, index=tickers.index("TSLA") if "TSLA" in tickers else 0)
    window_days = st.slider("Rolling window (days)", 7, 90, 30, step=1)
    half_life = st.slider("Recency half-life (days)", 7, 180, 30, step=1)
    st.divider()
    st.markdown(f"**{len(scored):,}** relevance-filtered headlines · **{len(tickers)}** Nasdaq-100 tickers · "
                f"2019–2023")
    st.markdown("[Source code on GitHub](https://github.com/Sarthakm20/time-aware-financial-news-monitor)")

tab_explore, tab_market, tab_quality, tab_try = st.tabs(
    ["🔎 Company explorer", "📈 Market reaction", "🧪 Model quality", "✍️ Score a headline"])

# ----------------------------------------------------------------------------- explorer

with tab_explore:
    windows = ticker_windows(scored, ticker, half_life_days=half_life).set_index("window")
    row = overview.set_index("ticker").loc[ticker]
    recent = windows.loc["Last 30 days"] if "Last 30 days" in windows.index else None
    history = windows.loc["All available history"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Headlines", f"{int(row['headlines']):,}", help=f"{row['first_date']} → {row['last_date']}")
    c2.metric("Last 30 days (unweighted)", f"{recent['unweighted_sentiment']:+.3f}" if recent is not None else "–",
              delta=(f"{recent['unweighted_sentiment'] - history['unweighted_sentiment']:+.3f} vs all history"
                     if recent is not None else None))
    c3.metric("All-history sentiment", f"{history['unweighted_sentiment']:+.3f}")
    c4.metric("Positive / negative share", f"{row['positive_share']:.0%} / {row['negative_share']:.0%}")

    days = pd.Series(sorted(scored.loc[scored["Stock_symbol"] == ticker, "Date"].dt.normalize().unique()))
    if len(days) > 1:
        gaps = days.diff().dt.days
        i = int(gaps.idxmax())
        if gaps.max() > 90:
            st.warning(f"**Coverage gap:** the FNSPID data has no {ticker} headlines between "
                       f"{days[i - 1].date()} and {days[i].date()} ({int(gaps.max())} days). Treat that stretch as "
                       "missing data, not as quiet news. FNSPID also keeps at most one headline per company per day.")

    per_company_csv = METRICS_DIR / "multi_company_per_company_accuracy.csv"
    if per_company_csv.exists():
        acc = load_csv("multi_company_per_company_accuracy.csv")
        mine = acc[(acc["ticker"] == ticker) & (acc["model"] == ("DistilBERT headline-adapted"
                   if model_choice.startswith("Headline") else "DistilBERT fine-tuned"))]
        if not mine.empty:
            r = mine.iloc[0]
            st.info(f"**Measured accuracy for {ticker}:** macro F1 {r['macro_f1']:.2f} "
                    f"[{r['macro_f1_ci_low']:.2f}, {r['macro_f1_ci_high']:.2f}], accuracy {r['accuracy']:.0%} on "
                    f"{int(r['n'])} annotated headlines (LLM-annotated). See the Model quality tab.")
        else:
            st.caption(f"No annotated headlines for {ticker} yet; accuracy has been measured for "
                       f"{', '.join(sorted(acc['ticker'].unique()))}.")

    timeline = ticker_timeline(scored, ticker, window_days=window_days, freq="W")
    timeline["Date"] = timeline["Date"].dt.tz_localize(None)
    show_price = st.toggle("Overlay share price (Yahoo Finance)", value=False,
                           help="Fetched live; unavailable for delisted tickers.")

    base = alt.Chart(timeline).encode(x=alt.X("Date:T", title=None))
    sentiment_line = base.mark_line(color="#1565C0", strokeWidth=2).encode(
        y=alt.Y("rolling_sentiment:Q", title=f"{window_days}-day rolling sentiment", scale=alt.Scale(domain=[-1, 1])),
        tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("rolling_sentiment:Q", format="+.3f", title="sentiment"),
                 alt.Tooltip("rolling_article_count:Q", title="headlines in window")])
    zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#BDBDBD", strokeDash=[4, 4]).encode(y="y:Q")
    chart = sentiment_line + zero

    if show_price:
        start = str(timeline["Date"].min().date())
        end = str((timeline["Date"].max() + pd.Timedelta(days=7)).date())
        price = load_price(ticker, start, end)
        if price.empty:
            st.info(f"No price data available for {ticker} (it may have been delisted).")
        else:
            price_line = alt.Chart(price).mark_line(color="#FB8C00", opacity=0.7).encode(
                x="date:T", y=alt.Y("close:Q", title=f"{ticker} adjusted close (USD)"),
                tooltip=[alt.Tooltip("date:T"), alt.Tooltip("close:Q", format=",.2f")])
            chart = alt.layer(sentiment_line + zero, price_line).resolve_scale(y="independent")
    st.altair_chart(chart.properties(height=340), width="stretch")

    left, right = st.columns([2, 3])
    with left:
        st.subheader("Lookback windows")
        table = windows.reset_index()[["window", "headline_count", "unweighted_sentiment",
                                       "recency_weighted_sentiment", "positive_percent", "negative_percent"]]
        table.columns = ["Window", "Headlines", "Mean", f"Recency-weighted (t½={half_life}d)", "% pos", "% neg"]
        st.dataframe(table, hide_index=True, width="stretch",
                     column_config={"% pos": st.column_config.NumberColumn(format="%.1f"),
                                    "% neg": st.column_config.NumberColumn(format="%.1f")})
        st.caption("Sentiment = P(positive) − P(negative), averaged over headlines; range −1 to +1.")
    with right:
        st.subheader("Headlines")
        order = st.segmented_control("Show", ["latest", "most_positive", "most_negative"], default="latest",
                                     format_func=lambda s: s.replace("_", " ").capitalize())
        heads = pd.DataFrame(ticker_headlines(scored, ticker, n=12, order=order or "latest"))
        st.dataframe(heads[["date", "headline", "label", "sentiment_score"]], hide_index=True,
                     width="stretch",
                     column_config={"sentiment_score": st.column_config.ProgressColumn(
                         "score", min_value=-1, max_value=1, format="%.2f"),
                         "headline": st.column_config.TextColumn(width="large")})

# ----------------------------------------------------------------------------- market reaction

with tab_market:
    st.subheader("Does headline sentiment line up with the stock?")
    summary = load_json("event_study_summary.json")
    n_events = summary["runs"][0]["events"]
    st.markdown(
        f"Event study over **{n_events:,} company-days** for **{summary['runs'][0]['tickers']} Nasdaq-100 stocks**. "
        "Returns are market-adjusted (stock − QQQ); 95% CIs come from a bootstrap clustered by date.")
    slug = "adapted" if model_choice.startswith("Headline") else "base"
    variant = st.radio("Headlines", ["excluding price-move headlines", "all relevant headlines"], horizontal=True,
                       help="Price-move headlines ('Why X Stock Popped Today') describe a return that already "
                            "happened, which inflates the association.")
    fig = FIGURES_DIR / f"event_study_car_{slug}.png"
    if fig.exists():
        st.image(str(fig), caption="Mean cumulative abnormal return around positive vs negative headline days "
                                   "(price-move headlines excluded)")

    buckets = load_csv("event_study_buckets.csv")
    spread = buckets[(buckets["model"] == slug) & (buckets["headlines"] == variant)
                     & (buckets["group"] == "positive minus negative")].copy()
    labels = {"ar_m1": "Day −1 (before)", "ar_0": "Day 0", "ar_p1": "Day +1", "car_p1_p5": "Days +1…+5",
              "av_0": "Abnormal volume, day 0"}
    spread["Window"] = spread["outcome"].map(labels)
    for col in ["value", "ci_low", "ci_high"]:
        spread[col] = spread[col] * 100  # % (returns) or percentage points of log-volume
    spread["Positive − negative days"] = spread.apply(
        lambda r: f"{r['value']:+.2f}{' pts' if r['outcome'] == 'av_0' else '%'}  "
                  f"[{r['ci_low']:+.2f}, {r['ci_high']:+.2f}]", axis=1)
    spread["Significant?"] = ((spread["ci_low"] > 0) | (spread["ci_high"] < 0)).map({True: "yes", False: "no"})
    st.dataframe(spread[["Window", "Positive − negative days", "Significant?"]], hide_index=True,
                 width="stretch")
    st.markdown(
        "- **Mostly after the fact:** much of the positive-vs-negative gap already exists the day *before* the "
        "headline date, and there is no significant drift over the following week. Headlines largely report moves.\n"
        "- **Bad news draws more trading:** negative-sentiment days carry about twice the abnormal volume.\n"
        "- **Caveat:** FNSPID has dates but no times, so a day +1 effect may simply be after-hours news.")

# ----------------------------------------------------------------------------- model quality

with tab_quality:
    st.subheader("How good is the classifier?")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Financial PhraseBank test split (N=726)**")
        t = load_csv("test_set_model_comparison.csv")
        st.dataframe(t[["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy"]].round(3),
                     hide_index=True, width="stretch")
        st.caption("FinBERT was trained on PhraseBank, so its score here is likely inflated.")
    with col2:
        st.markdown("**Real TSLA headlines (N=230, LLM-annotated)**")
        h = load_csv("tsla_headline_model_comparison.csv")
        st.dataframe(h[["model", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high", "accuracy"]].round(3),
                     hide_index=True, width="stretch")
        st.caption("Labels were produced by an LLM annotator and are not human-verified.")
    st.warning("**Domain shift:** macro F1 falls from 0.84 on PhraseBank sentences to about 0.55–0.70 on real "
               "headlines across six companies. Read the timelines as a rough indicator of tone, not a precise "
               "measurement.")

    if (METRICS_DIR / "multi_company_pooled_accuracy.csv").exists():
        st.markdown("#### Accuracy across six companies (TSLA, AAPL, NVDA, AMZN, GILD, SBUX)")
        st.caption("Annotated headlines that pass the relevance filter, i.e. what the dashboard actually scores. "
                   "LLM-annotated, not human-verified. 95% bootstrap CIs; McNemar test vs the PhraseBank model.")
        pooled = load_csv("multi_company_pooled_accuracy.csv")
        pooled = pooled[pooled["population"] == "pooled_all_6_companies"]
        st.dataframe(pooled[["model", "n", "accuracy", "macro_f1", "macro_f1_ci_low", "macro_f1_ci_high",
                             "mcnemar_p_value"]].round(3), hide_index=True, width="stretch")

        per = load_csv("multi_company_per_company_accuracy.csv")
        per = per[~per["model"].str.startswith("Majority")]
        heat = alt.Chart(per).mark_rect().encode(
            x=alt.X("model:N", title=None), y=alt.Y("ticker:N", title=None),
            color=alt.Color("macro_f1:Q", scale=alt.Scale(scheme="blues", domain=[0.4, 0.8]), title="macro F1"),
            tooltip=["ticker", "model", "n", alt.Tooltip("macro_f1:Q", format=".3f"),
                     alt.Tooltip("accuracy:Q", format=".1%")])
        labels = heat.mark_text(fontSize=12).encode(text=alt.Text("macro_f1:Q", format=".2f"),
                                                    color=alt.value("black"))
        st.altair_chart((heat + labels).properties(height=260), width="stretch")

        rel = load_csv("multi_company_relevance_filter.csv")
        st.markdown("**Relevance filter (is the headline really about the company?)**")
        st.dataframe(rel[["ticker", "precision", "recall_estimate", "on_topic_share_estimate"]].round(2),
                     hide_index=True, width="stretch")
        st.caption("Near-perfect precision. Recall is lower, but 96% of the on-topic headlines it drops are neutral "
                   "listicles, so it keeps about 98% of headlines that carry sentiment.")
    seeds = load_json("seed_summary.json")
    st.markdown(f"Seed variation ({seeds['n_seeds']} seeds): macro F1 **{seeds['macro_f1']['mean']:.3f} ± "
                f"{seeds['macro_f1']['std']:.3f}**. Calibration: ECE 0.021, so probabilities are used as is.")

# ----------------------------------------------------------------------------- try it

with tab_try:
    st.subheader("Score your own headlines")
    text = st.text_area("One headline per line", height=140,
                        value="Tesla Deliveries Top Estimates as Demand Surges\n"
                              "Intel Cuts Full-Year Guidance as PC Demand Slumps\n"
                              "Apple To Report Q3 Earnings On Thursday")
    company = st.text_input("Optional ticker for the relevance check", value="TSLA", max_chars=10)
    if st.button("Score", type="primary"):
        lines = [line.strip() for line in text.splitlines() if line.strip()][:50]
        try:
            with st.spinner("Loading model and scoring…"):
                preds = get_service().predict(lines, ticker=company.strip() or None)
        except OSError:  # no local model, or the Hub model could not be downloaded
            st.error("The sentiment model is not available on this server right now. "
                     "Set SENTIMENT_MODEL to a Hugging Face Hub id.")
        except ValueError as exc:
            st.error(str(exc))
        else:
            out = pd.DataFrame([p.to_dict() for p in preds])
            out["P(neg)"] = out["probabilities"].map(lambda p: p["negative"])
            out["P(neu)"] = out["probabilities"].map(lambda p: p["neutral"])
            out["P(pos)"] = out["probabilities"].map(lambda p: p["positive"])
            cols = ["headline", "label", "sentiment_score", "P(neg)", "P(neu)", "P(pos)"]
            if company.strip():
                cols.append("relevant")
            st.dataframe(out[cols], hide_index=True, width="stretch",
                         column_config={"sentiment_score": st.column_config.ProgressColumn(
                             "score", min_value=-1, max_value=1, format="%.2f")})
            st.caption(f"Model: {get_service().model_name}. Known weakness: product recalls are often scored "
                       "neutral rather than negative.")
