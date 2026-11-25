// Live Feed (spec 8.7): headlines collected since the live feed started, from data/live/news.db.
import { getJSON } from "../api.js";
import { C, connect, lineChart, timeBars, withTable } from "../charts.js";
import { card, emptyState, fmt, h, note, sentimentBadge, statCard } from "../ui.js";

let charts = [];

export function destroy() {
  charts.forEach((c) => c.dispose());
  charts = [];
}

const toMs = (iso) => new Date(`${iso}T00:00:00Z`).getTime();
const safeUrl = (url) => (typeof url === "string" && /^https?:\/\//i.test(url) ? url : null);
const when = (iso) => new Date(iso).toLocaleString("en-US", { month: "short", day: "numeric", hour: "2-digit",
  minute: "2-digit", timeZone: "UTC", hour12: false });

function notCollecting(data) {
  return card({ className: "span-12", body: emptyState(
    h("span", {}, "No live feed on this server yet. Start collecting with ", h("code", {}, data.command),
      ", then reload. See ", h("code", {}, data.docs), " for running it daily."),
  ) });
}

function cards(data) {
  const c = data.cards;
  const list = [
    statCard({ icon: "newspaper", tone: "blue", value: fmt.int(c.headlines_7d), label: "Headlines, Last 7 Days" }),
    statCard({ icon: "activity", tone: "green", value: fmt.signed(c.mean_7d, 3), label: "Mean Sentiment, 7 Days",
      sub: c.delta_vs_all === null ? "no earlier live headlines to compare" : `${fmt.signed(c.delta_vs_all, 3)} vs all live headlines` }),
    statCard({ icon: "pie-chart", tone: "violet", value: `${fmt.pct(c.positive_share)} / ${fmt.pct(c.negative_share)}`,
      label: "Positive / Negative Share" }),
    statCard({ icon: "calendar-week", tone: "amber", value: `${fmt.int(c.history_days)} days`, label: "Live History",
      sub: "since the first live headline" }),
  ];
  return h("div", { class: "grid" }, list.map((x) => {
    x.classList.add("span-3");
    return x;
  }));
}

function dailyCard(data) {
  const who = data.ticker || "All companies";
  const lineEl = h("div", { class: "chart chart-sm", role: "img", "aria-label": `${who}: daily mean live sentiment` });
  const barEl = h("div", { class: "chart chart-xs", role: "img", "aria-label": `${who}: live headlines per day` });
  requestAnimationFrame(() => {
    const days = data.daily.map((d) => [toMs(d.day), d.mean, d.count]);
    const xMin = days[0][0] - 864e5 / 2;
    const xMax = days.at(-1)[0] + 864e5 / 2;
    const line = lineChart(lineEl, { series: [{ name: "Daily mean sentiment", data: days }], yMin: -1, yMax: 1,
      zoom: false, maxGapDays: 3 });
    line.chart.setOption({ xAxis: { min: xMin, max: xMax }, series: [{ showSymbol: true }] });
    const bars = timeBars(barEl, { points: data.daily.map((d) => [toMs(d.day), d.count]), name: "Headlines", xMin, xMax });
    charts.push(line, bars);
    connect(line, bars);
  });
  return card({ title: `${who}: Daily Sentiment`, className: "span-12",
    info: "Mean sentiment of the live headlines published each day (UTC), with the number of headlines beneath.",
    body: [withTable(lineEl, { caption: "Daily live sentiment", rows: data.daily, columns: [
      { key: "day", label: "Day", format: fmt.date },
      { key: "mean", label: "Mean sentiment", num: true, format: (v) => fmt.signed(v, 3) },
      { key: "count", label: "Headlines", num: true, format: fmt.int },
    ] }), h("h3", { class: "sub-head mt-3" }, "Headlines per day"), barEl] });
}

const FIRST_ROWS = 12;

function headlinesCard(data) {
  const list = h("ul", { class: "rows" }, data.headlines.map((r) => {
      const url = safeUrl(r.url);
      return h("li", {},
        h("div", { class: "row-meta" }, sentimentBadge(r.label, r.sentiment_score), h("span", {}, `${when(r.published_at)} UTC`),
          r.publisher ? h("span", {}, r.publisher) : null, h("span", {}, `via ${r.source}`),
          url ? h("a", { href: url, target: "_blank", rel: "noopener noreferrer" }, "Open article") : null),
        h("p", { class: "row-title" }, r.headline));
  }));
  [...list.children].forEach((li, i) => { li.hidden = i >= FIRST_ROWS; });
  const more = data.headlines.length > FIRST_ROWS ? h("button", { type: "button",
    class: "btn btn-outline-secondary btn-sm mt-2", onclick: (e) => {
      [...list.children].forEach((li) => { li.hidden = false; });
      e.currentTarget.remove();
    } }, `Show all ${data.headlines.length}`) : null;
  return card({ title: "Latest Live Headlines", className: "span-12", body: [list, more] });
}

function checksCard(data) {
  const parts = [];
  if (data.alerts.length) {
    parts.push(...data.alerts.map((text) => note({ kind: "caveat", text })));
  } else {
    parts.push(h("p", { class: "m-0" }, "No alerts in the latest run. Alerts need 30 days of daily collection."));
  }
  const a = data.agreement;
  if (a) {
    parts.push(h("p", { class: "mt-3 mb-0" }, `Live accuracy check: on ${a.n_on_topic} live headlines labelled blind by `
      + `${a.annotator}, the model agrees ${fmt.pct(a.accuracy.value)} of the time (macro F1 ${fmt.num(a.macro_f1.value)} `
      + `[${fmt.num(a.macro_f1.ci_low)}, ${fmt.num(a.macro_f1.ci_high)}]). The second annotator is another LLM, not ground truth.`));
  }
  return card({ title: "Alerts and Checks", className: "span-12",
    info: "Alerts fire when 7-day sentiment is at least 2 standard deviations below a company's 90-day baseline, "
      + "or headline volume is at least 3× the baseline.", body: parts });
}

export async function render(root, params, ctx) {
  const data = await getJSON("/api/live/summary", { ticker: params.ticker });
  if (!ctx.isCurrent()) return;
  destroy();
  if (!data.available) {
    root.replaceChildren(h("div", { class: "grid" }, notCollecting(data)));
    return;
  }
  ctx.setStatus(`Live · updated ${fmt.date(data.last_run_utc.slice(0, 10))}`, true);
  const summary = h("p", { class: "scope" }, `Last update ${data.last_run_utc} UTC · ${data.runs} runs · `
    + `${fmt.int(data.total_headlines)} headlines across ${data.tickers} companies · sources: ${data.sources}. `
    + "Live headlines are scored once by the configured model; the Period and Model filters don't apply here.");
  if (!data.ticker_has_data) {
    root.replaceChildren(summary, note({ kind: "info", text: `No live headlines for ${data.ticker} yet. Pick another company.` }),
      h("div", { class: "grid" }, checksCard(data)));
    return;
  }
  root.replaceChildren(summary, cards(data),
    h("div", { class: "grid" }, dailyCard(data)),
    h("div", { class: "grid" }, checksCard(data)),
    h("div", { class: "grid" }, headlinesCard(data)));
}
