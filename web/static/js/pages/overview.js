// Overview (spec 8.1): stat cards, sentiment trend + distribution, relevance filter, market relationship.
import { getJSON } from "../api.js";
import { C, TONE_COLORS, barsWithCI, donut, lineChart, ring, sentimentRange, withTable } from "../charts.js";
import { card, emptyState, fmt, h, segmented, statCard } from "../ui.js";

const PHRASEBANK_NAME = { adapted: "DistilBERT headline-adapted", base: "DistilBERT fine-tuned" };
const TONE_NAMES = { positive: "Positive", neutral: "Neutral", negative: "Negative" };
let charts = [];

export function destroy() {
  charts.forEach((c) => c.dispose());
  charts = [];
}

export function noHeadlines({ ticker, period }) {
  if (ticker && period) return `No ${ticker} headlines in ${period}.`;
  if (ticker) return `No ${ticker} headlines.`;
  return period ? `No headlines in ${period}.` : "No headlines.";
}

function statRow(data, params, phrasebankF1) {
  const { coverage, headlines, f1, company_days: days } = data.cards;
  const coverageCard = coverage.kind === "universe"
    ? statCard({ icon: "buildings", tone: "blue", value: fmt.int(coverage.companies), label: "Companies Covered",
      sub: "Nasdaq-100 universe" })
    : statCard({ icon: "buildings", tone: "blue", value: coverage.ticker, label: "Company Coverage",
      sub: `${fmt.date(coverage.first_date)} – ${fmt.date(coverage.last_date)}` });
  const scope = params.ticker && params.period ? `${params.ticker} in ${params.period}`
    : params.ticker ? `for ${params.ticker}` : params.period ? `in ${params.period}` : null;
  const headlineCard = statCard({ icon: "newspaper", tone: "green", value: fmt.int(headlines.scored),
    label: "Headlines Scored", sub: headlines.collected ? `of ${fmt.int(headlines.collected)} collected` : scope });
  const f1Card = statCard({ icon: "bar-chart", tone: "violet", value: fmt.num(f1.value, 2),
    label: "Real-Headline Macro F1",
    sub: phrasebankF1 ? `vs ${fmt.num(phrasebankF1, 2)} on PhraseBank sentences` : `on ${f1.n} annotated headlines`,
    scope: f1.scope === "All companies · 2019–2023" ? f1.scope : `${f1.scope} · ${f1.n} annotated headlines`,
    info: "Macro F1 on LLM-annotated real headlines: per company where annotated, otherwise pooled over the six "
      + "annotated companies. Source: multi_company_*_accuracy.csv." });
  const daysCard = statCard({ icon: "calendar-week", tone: "amber", value: fmt.int(days.value), label: "Company-Days",
    sub: "in the market-reaction study", scope: days.scope });
  return h("div", { class: "grid" }, [coverageCard, headlineCard, f1Card, daysCard].map((c) => {
    c.classList.add("span-3");
    return c;
  }));
}

function trendCard(data, params) {
  const name = params.ticker ? `${params.ticker} Sentiment Trend` : "Market Sentiment Trend";
  const info = "30-day rolling sentiment, weekly points. Sentiment = P(positive) − P(negative) for each headline, "
    + "averaged over the headlines in the window; it runs from −1 to +1. Source: scored FNSPID headlines.";
  if (!data.trend.length) {
    return card({ title: name, info, className: "span-8", body: emptyState(noHeadlines(params)) });
  }
  const chartEl = h("div", { class: "chart", role: "img", "aria-label": `${name}: 30-day rolling sentiment over time` });
  const points = data.trend.map((p) => [new Date(`${p.date}T00:00:00Z`).getTime(), p.sentiment, p.headlines]);
  const years = new Set(data.trend.map((p) => p.date.slice(0, 4))).size;
  let handle;
  const presets = [["1", "1Y"], ["3", "3Y"], ["all", years > 3 ? `${years}Y` : "All"]];
  const seg = years > 1 ? segmented({ label: "Zoom the trend chart", options: presets, value: "all",
    onChange: (key) => handle.zoomToYears(key === "all" ? 0 : Number(key)) }) : null;
  const panel = card({ title: name, info, className: "span-8", tools: seg,
    body: [withTable(chartEl, { caption: name, rows: data.trend, columns: [
      { key: "date", label: "Week ending", format: fmt.date },
      { key: "sentiment", label: "30-day sentiment", num: true, format: (v) => fmt.signed(v, 3) },
      { key: "headlines", label: "Headlines in window", num: true, format: fmt.int },
    ] }), h("p", { class: "caption" }, "Weekly points of the 30-day rolling sentiment. 0 is neutral; the scale runs from −1 to +1.")] });
  requestAnimationFrame(() => {
    const [yMin, yMax] = sentimentRange(data.trend.map((p) => p.sentiment));
    handle = lineChart(chartEl, { series: [{ name: "30-day sentiment", data: points }], yMin, yMax });
    charts.push(handle);
  });
  return panel;
}

function distributionCard(data) {
  const d = data.distribution;
  const info = "Share of scored headlines the model labels positive, neutral or negative.";
  if (!d.n) return card({ title: "Sentiment Distribution", info, className: "span-4", body: emptyState("No headlines to count.") });
  const order = ["positive", "neutral", "negative"];
  const top = order.reduce((a, b) => (d[b] > d[a] ? b : a));
  const chartEl = h("div", { class: "chart chart-donut", role: "img",
    "aria-label": `Sentiment distribution: ${order.map((k) => `${TONE_NAMES[k]} ${fmt.pct(d[k], 0)}`).join(", ")}` });
  requestAnimationFrame(() => charts.push(donut(chartEl, {
    slices: order.map((k) => ({ name: TONE_NAMES[k], value: d[k], color: TONE_COLORS[k] })),
    center: { value: fmt.pct(d[top], 0), label: TONE_NAMES[top] },
  })));
  return card({ title: "Sentiment Distribution", info, className: "span-4",
    body: [withTable(chartEl, { caption: "Sentiment distribution", rows: order.map((k) => ({ label: TONE_NAMES[k],
      share: d[k], count: Math.round(d[k] * d.n) })), columns: [
      { key: "label", label: "Label" },
      { key: "share", label: "Share", num: true, format: (v) => fmt.pct(v, 1) },
      { key: "count", label: "Headlines", num: true, format: fmt.int },
    ] }), h("p", { class: "caption" }, `${fmt.int(d.n)} headlines.`)] });
}

function relevanceCard(rel) {
  const ringEl = h("div", { class: "chart chart-xs", role: "img",
    "aria-label": `${fmt.pct(rel.kept_share, 0)} of collected headlines retained` });
  requestAnimationFrame(() => charts.push(ring(ringEl, { value: rel.kept_share, color: C.positive })));
  const precision = rel.precision_min === rel.precision_max ? fmt.pct(rel.precision_max, 0)
    : `${Math.floor(rel.precision_min * 100)}–${Math.round(rel.precision_max * 100)}%`;
  return card({ title: "Headline Relevance Filter", scope: rel.scope, className: "span-5",
    info: "Is the headline really about the company? Precision was measured on LLM-annotated samples for six "
      + "companies. Source: multi_company_relevance_filter.csv and corpus_summary.json.",
    body: [
      h("div", { class: "relevance" },
        h("figure", { class: "relevance-ring" }, ringEl,
          h("figcaption", {}, h("strong", {}, "Relevant headlines"), h("span", {}, `${fmt.int(rel.scored)} of ${fmt.int(rel.collected)} kept`))),
        h("div", { class: "relevance-precision" },
          h("div", { class: "stat-value" }, precision),
          h("div", { class: "stat-label" }, "Precision"),
          h("div", { class: "stat-sub" }, "on retained headlines"))),
      h("p", { class: "caption" }, "The filter keeps headlines about the company and drops lists that only mention it."),
    ] });
}

function marketCard(market) {
  const rows = market.by_tone.map((r) => ({ category: TONE_NAMES[r.group], value: r.value, low: r.ci_low,
    high: r.ci_high, color: TONE_COLORS[r.group] }));
  const chartEl = h("div", { class: "chart chart-sm", role: "img",
    "aria-label": `Average day-0 abnormal return: ${rows.map((r) => `${r.category} ${fmt.signed(r.value, 2)}%`).join(", ")}` });
  requestAnimationFrame(() => charts.push(barsWithCI(chartEl, { rows,
    categoryLabel: (r) => `${r.category}\n(${fmt.signed(r.value, 2)}%)` })));
  return card({ title: "Market Relationship Insights", scope: market.scope, className: "span-7",
    info: "Day-0 market-adjusted return (stock minus QQQ) on days with positive, neutral or negative headlines, "
      + "excluding headlines that only report a price move. Whiskers are 95% bootstrap CIs clustered by date. "
      + "Source: event_study_buckets.csv.",
    body: h("div", { class: "market-split" },
      h("div", {},
        h("h3", { class: "sub-head" }, "Average day-0 return by news tone"),
        withTable(chartEl, { caption: "Average day-0 return by news tone", rows, columns: [
          { key: "category", label: "Tone" },
          { key: "value", label: "Return", num: true, format: (v) => `${fmt.signed(v, 2)}%` },
          { key: "low", label: "95% CI", num: true, format: (v, r) => `${fmt.signed(r.low, 2)}% to ${fmt.signed(r.high, 2)}%` },
        ] }),
        h("p", { class: "caption" }, `Excluding price-move headlines · ${fmt.int(market.events)} company-days`)),
      h("div", { class: "findings" },
        h("h3", { class: "sub-head" }, "Key findings"),
        h("ul", {},
          h("li", {}, "News often describes price moves after they happen, rather than predicting them."),
          h("li", {}, `Negative-news days see about ${market.volume_ratio}× the abnormal trading volume of positive ones.`)),
        h("a", { href: "#/market" }, "See Market Impact"))) });
}

export async function render(root, params, ctx) {
  const model = params.model || "adapted";
  const [data, evaluation] = await Promise.all([
    getJSON("/api/overview", { ticker: params.ticker, period: params.period, model }),
    getJSON("/api/model-evaluation").catch(() => null),
  ]);
  if (!ctx.isCurrent()) return;
  destroy();
  const pb = evaluation?.phrasebank.find((r) => r.model === PHRASEBANK_NAME[model])?.macro_f1;
  root.replaceChildren(
    statRow(data, params, pb),
    h("div", { class: "grid" }, trendCard(data, params), distributionCard(data)),
    h("div", { class: "grid" }, relevanceCard(data.relevance), marketCard(data.market)),
  );
}
