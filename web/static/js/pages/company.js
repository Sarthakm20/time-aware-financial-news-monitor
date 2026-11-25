// Company Analysis (spec 8.2): one company's sentiment over time, lookback windows and headlines.
import { getJSON } from "../api.js";
import { C, connect, lineChart, withTable } from "../charts.js";
import { card, emptyState, fmt, h, note, segmented, sentimentBadge, statCard } from "../ui.js";
import { setParams } from "../state.js";
import { noHeadlines } from "./overview.js";

const ORDERS = [["latest", "Latest"], ["most_positive", "Most positive"], ["most_negative", "Most negative"]];
const PAGE = 12;
const MAX_ROWS = 100; // /api/company caps n at 100
let charts = [];

export function destroy() {
  charts.forEach((c) => c.dispose());
  charts = [];
}

const toMs = (iso) => new Date(`${iso}T00:00:00Z`).getTime();

function cards(view) {
  const c = view.cards;
  const empty = !c.headlines;
  const list = [
    statCard({ icon: "newspaper", tone: "blue", value: fmt.int(c.headlines), label: "Headlines",
      sub: empty ? "none in this period" : `${fmt.date(c.first_date)} – ${fmt.date(c.last_date)}` }),
    statCard({ icon: "clock-history", tone: "green", value: fmt.signed(c.recent_30d, 3), label: "Last 30 Days",
      sub: c.recent_delta === null ? "no headlines in the last 30 days" : `${fmt.signed(c.recent_delta, 3)} vs all history`,
      info: "Mean sentiment of headlines in the 30 days up to the latest headline in view." }),
    statCard({ icon: "graph-up", tone: "violet", value: fmt.signed(c.history, 3), label: "All-History Sentiment",
      sub: "mean over every headline in view" }),
    statCard({ icon: "pie-chart", tone: "amber",
      value: empty ? "–" : `${fmt.pct(c.positive_share)} / ${fmt.pct(c.negative_share)}`,
      label: "Positive / Negative Share", sub: "of headlines in view" }),
  ];
  return h("div", { class: "grid" }, list.map((x) => {
    x.classList.add("span-3");
    return x;
  }));
}

function notes(view) {
  const out = [];
  const gap = view.coverage_gap;
  if (gap) {
    out.push(note({ kind: "caveat", text: `No ${view.ticker} headlines between ${fmt.date(gap.start)} and `
      + `${fmt.date(gap.end)} (${gap.days} days). Treat this stretch as missing data, not quiet news.` }));
  }
  const acc = view.accuracy;
  if (acc) {
    out.push(note({ kind: "info", text: `Measured accuracy for ${view.ticker}: macro F1 ${fmt.num(acc.value)} `
      + `[${fmt.num(acc.ci_low)}, ${fmt.num(acc.ci_high)}] on ${acc.n} LLM-annotated headlines.` }));
  }
  return out;
}

function rangeControl({ id, label, min, max, value, unit, onInput }) {
  const output = h("output", { for: id, class: "range-value" }, `${value} ${unit}`);
  const input = h("input", { id, type: "range", class: "form-range", min, max, step: 1, value });
  // Update the label while dragging; reload only when the value is committed (release, or a key press).
  input.addEventListener("input", () => {
    output.textContent = `${input.value} ${unit}`;
  });
  input.addEventListener("change", () => onInput(Number(input.value)));
  return h("div", { class: "range-control" }, h("label", { for: id }, label), input, output);
}

function chartCard(view, params) {
  const windowDays = Number(params.window_days || 30);
  const halfLife = Number(params.half_life_days || 30);
  const showPrice = params.price === "1";
  const controls = h("div", { class: "chart-controls" },
    rangeControl({ id: "c-window", label: "Rolling window", min: 7, max: 90, value: windowDays, unit: "days",
      onInput: (v) => setParams({ window_days: v === 30 ? null : v }, { replace: true }) }),
    rangeControl({ id: "c-halflife", label: "Recency half-life", min: 7, max: 180, value: halfLife, unit: "days",
      onInput: (v) => setParams({ half_life_days: v === 30 ? null : v }, { replace: true }) }),
    h("div", { class: "form-check form-switch" },
      h("input", { class: "form-check-input", type: "checkbox", role: "switch", id: "c-price", checked: showPrice,
        onchange: (e) => setParams({ price: e.target.checked ? "1" : null }) }),
      h("label", { class: "form-check-label", for: "c-price" }, "Show share price")));

  const title = `${view.ticker} Sentiment Over Time`;
  const info = `${windowDays}-day rolling sentiment, weekly points: P(positive) − P(negative) averaged over the `
    + "headlines in the window, from −1 to +1. Share prices are Yahoo Finance adjusted closes.";
  if (!view.timeline.length) {
    return card({ title, info, className: "span-12", body: [controls, emptyState(noHeadlines(params))] });
  }
  const sentEl = h("div", { class: "chart", role: "img", "aria-label": `${title}, ${windowDays}-day rolling` });
  const priceEl = h("div", { class: "chart chart-sm", role: "img", "aria-label": `${view.ticker} share price` });
  const priceSlot = h("div", { class: "price-slot" });
  const body = [controls,
    withTable(sentEl, { caption: title, rows: view.timeline, columns: [
      { key: "date", label: "Week ending", format: fmt.date },
      { key: "sentiment", label: `${windowDays}-day sentiment`, num: true, format: (v) => fmt.signed(v, 3) },
      { key: "headlines_in_window", label: "Headlines in window", num: true, format: fmt.int },
    ] }),
    priceSlot];

  requestAnimationFrame(async () => {
    const points = view.timeline.map((p) => [toMs(p.date), p.sentiment, p.headlines_in_window]);
    const sentiment = lineChart(sentEl, { series: [{ name: `${windowDays}-day sentiment`, data: points }],
      yMin: -1, yMax: 1 });
    charts.push(sentiment);
    if (!showPrice) return;
    priceSlot.replaceChildren(h("p", { class: "loading" }, "Loading share prices…"));
    const first = view.timeline[0].date;
    const last = new Date(toMs(view.timeline.at(-1).date) + 7 * 864e5).toISOString().slice(0, 10);
    let price;
    try {
      price = await getJSON(`/api/price/${view.ticker}`, { start: first, end: last });
    } catch {
      price = { points: [] };
    }
    if (!document.body.contains(priceSlot)) return;
    if (!price.points.length) {
      priceSlot.replaceChildren(note({ kind: "info", text: `No price data for ${view.ticker} (it may have been delisted).` }));
      return;
    }
    priceSlot.replaceChildren(h("h3", { class: "sub-head mt-3" }, `${view.ticker} adjusted close (USD)`),
      withTable(priceEl, { caption: `${view.ticker} adjusted close`, rows: price.points, columns: [
        { key: "date", label: "Date", format: fmt.date },
        { key: "close", label: "Close (USD)", num: true, format: (v) => fmt.num(v, 2) },
      ] }));
    const priceChart = lineChart(priceEl, {
      series: [{ name: "Adjusted close", color: C.ink2, data: price.points.map((p) => [toMs(p.date), p.close]) }],
      yFormat: (v) => `$${fmt.num(v, 0)}`, zeroLine: false, area: false, zoom: true, slider: false });
    priceChart.chart.setOption({ yAxis: { scale: true },
      xAxis: { min: points[0][0], max: points.at(-1)[0] } });
    charts.push(priceChart);
    connect(sentiment, priceChart);
  });
  return card({ title, info, className: "span-12", body });
}

function windowsCard(view, params) {
  const halfLife = Number(params.half_life_days || 30);
  if (!view.windows.length) return null;
  const columns = [
    ["window", "Window"], ["headlines", "Headlines", fmt.int], ["mean", "Mean", (v) => fmt.signed(v, 3)],
    ["recency_weighted", `Recency-weighted (t½ = ${halfLife} d)`, (v) => fmt.signed(v, 3)],
    ["positive_pct", "% positive", (v) => fmt.num(v, 1)], ["negative_pct", "% negative", (v) => fmt.num(v, 1)],
  ];
  const table = h("table", { class: "data-table" },
    h("thead", {}, h("tr", {}, columns.map(([, label], i) => h("th", { class: i ? "num" : null, scope: "col" }, label)))),
    h("tbody", {}, view.windows.map((w) => h("tr", {}, columns.map(([key, , f], i) =>
      h(i ? "td" : "th", { class: i ? "num" : null, scope: i ? null : "row" }, f ? f(w[key]) : w[key]))))));
  return card({ title: "Lookback Windows", className: "span-12",
    info: "Each window ends at the latest headline in view. Recency-weighted gives a headline half the weight "
      + `every ${halfLife} days of age.`,
    body: [h("div", { class: "table-responsive" }, table),
      h("p", { class: "caption" }, "Sentiment = P(positive) − P(negative), averaged over headlines; range −1 to +1.")] });
}

function headlinesCard(view, params) {
  const order = params.order || "latest";
  const n = Math.min(Number(params.n || PAGE), MAX_ROWS);
  const seg = segmented({ label: "Order headlines", options: ORDERS, value: order,
    onChange: (key) => setParams({ order: key === "latest" ? null : key, n: null }) });
  if (!view.headlines.length) return card({ title: "Headlines", className: "span-12", tools: seg, body: emptyState(noHeadlines(params)) });
  const list = h("ul", { class: "rows" }, view.headlines.map((r) => h("li", {},
    h("div", { class: "row-meta" }, sentimentBadge(r.label, r.sentiment_score), h("span", {}, fmt.date(r.date)),
      h("span", {}, `confidence ${fmt.pct(r.confidence)}`)),
    h("p", { class: "row-title" }, r.headline))));
  const more = view.headlines.length >= n && n < view.cards.headlines && n < MAX_ROWS
    ? h("button", { type: "button", class: "btn btn-outline-secondary btn-sm mt-2",
      onclick: () => setParams({ n: Math.min(n + PAGE, MAX_ROWS) }, { replace: true }) }, "Show more") : null;
  return card({ title: "Headlines", className: "span-12", tools: seg, body: [list, more] });
}

export async function render(root, params, ctx) {
  if (!params.ticker) {
    setParams({ ticker: "TSLA" }, { replace: true });
    return;
  }
  const view = await getJSON(`/api/company/${encodeURIComponent(params.ticker)}`, {
    period: params.period, model: params.model, window_days: params.window_days,
    half_life_days: params.half_life_days, order: params.order, n: Math.min(Number(params.n || PAGE), MAX_ROWS),
  });
  if (!ctx.isCurrent()) return;
  destroy();
  root.replaceChildren(
    cards(view),
    ...notes(view),
    h("div", { class: "grid" }, chartCard(view, params)),
    h("div", { class: "grid" }, windowsCard(view, params)),
    h("div", { class: "grid" }, headlinesCard(view, params)),
  );
}
