// Market Impact (spec 8.3): positive-minus-negative return gaps and CAR paths around headline days.
import { getJSON } from "../api.js";
import { C, barsWithCI, lineChart, withTable } from "../charts.js";
import { card, fmt, h, note, segmented } from "../ui.js";
import { setParams } from "../state.js";

const VARIANTS = [["excluding", "Excluding price-move headlines"], ["all", "All relevant headlines"]];
let charts = [];

export function destroy() {
  charts.forEach((c) => c.dispose());
  charts = [];
}

const dayLabel = (d) => (d === 0 ? "Day 0" : `Day ${d > 0 ? "+" : "−"}${Math.abs(d)}`);

function spreadCard(data) {
  const returns = data.spread.filter((s) => s.unit === "pct");
  const volume = data.spread.find((s) => s.unit === "pts");
  const rows = returns.map((s) => ({ category: s.label, value: s.value, low: s.ci_low, high: s.ci_high,
    significant: s.significant, color: C.accent }));
  const chartEl = h("div", { class: "chart", role: "img",
    "aria-label": `Return gap, positive minus negative headline days: ${rows.map((r) => `${r.category} ${fmt.signed(r.value, 2)}%`).join(", ")}` });
  requestAnimationFrame(() => charts.push(barsWithCI(chartEl, { rows,
    categoryLabel: (r) => `${r.category}\n${r.significant ? "significant" : "not significant"}` })));
  const vol = h("div", { class: "volume-block" },
    h("div", { class: "stat-value" }, `${fmt.signed(volume.value, 1)} pts`),
    h("div", { class: "stat-label" }, "Abnormal trading volume, day 0"),
    h("div", { class: "stat-sub" }, `Positive minus negative days, 95% CI ${fmt.signed(volume.ci_low, 1)} to `
      + `${fmt.signed(volume.ci_high, 1)} pts of log volume. Negative-news days carry about ${data.volume_ratio}× the `
      + "abnormal volume of positive ones."));
  return card({ title: "Return Gap: Positive minus Negative Days", scope: data.scope, className: "span-12",
    info: "Mean market-adjusted return (stock minus QQQ) on positive-headline days minus negative-headline days, "
      + "per window. Whiskers are 95% bootstrap CIs clustered by date; a window is significant when its CI excludes 0. "
      + "Source: event_study_buckets.csv.",
    body: h("div", { class: "spread-split" },
      withTable(chartEl, { caption: "Return gap by window", rows: data.spread, columns: [
        { key: "label", label: "Window" },
        { key: "value", label: "Positive − negative", num: true,
          format: (v, r) => `${fmt.signed(v, 2)}${r.unit === "pts" ? " pts" : "%"}` },
        { key: "ci_low", label: "95% CI", num: true,
          format: (v, r) => `${fmt.signed(r.ci_low, 2)} to ${fmt.signed(r.ci_high, 2)}` },
        { key: "significant", label: "Significant?", format: (v) => (v ? "yes" : "no") },
      ] }),
      vol) });
}

function pathsCard(data, period) {
  const title = "Cumulative Abnormal Return Around Headline Days";
  if (!data.car_paths_available) {
    return card({ title, className: "span-12",
      body: note({ kind: "info", text: "Event-window paths are not saved on this server." }) });
  }
  const series = ["positive", "negative"].map((g) => ({
    name: g === "positive" ? "Positive days" : "Negative days",
    color: g === "positive" ? C.positive : C.negative,
    data: data.car_paths.filter((p) => p.group === g).map((p) => [p.day, p.mean_car]),
  }));
  const chartEl = h("div", { class: "chart", role: "img",
    "aria-label": "Mean cumulative abnormal return from day −5 to day +10 for positive and negative headline days" });
  requestAnimationFrame(() => charts.push(lineChart(chartEl, { series, xType: "value", xFormat: dayLabel,
    yFormat: (v) => `${fmt.signed(v, 2)}%`, area: false, endLabels: true, zoom: false,
    xName: "Trading days relative to the headline day" })));
  const rows = data.car_paths.filter((p) => p.group === "positive").map((p) => ({ day: p.day, positive: p.mean_car,
    negative: data.car_paths.find((q) => q.group === "negative" && q.day === p.day)?.mean_car }));
  return card({ title, className: "span-12",
    info: "Mean cumulative market-adjusted return from five trading days before to ten after the headline day, "
      + "for days with clearly positive (> +0.3) or negative (< −0.3) headline sentiment. Source: event_study_car_paths.csv.",
    body: [withTable(chartEl, { caption: title, rows, columns: [
      { key: "day", label: "Day", format: dayLabel },
      { key: "positive", label: "Positive days", num: true, format: (v) => `${fmt.signed(v, 2)}%` },
      { key: "negative", label: "Negative days", num: true, format: (v) => `${fmt.signed(v, 2)}%` },
    ] }), period ? null
      : h("p", { class: "caption" }, "The lines split before day 0: much of the move is already under way when the headline appears.")] });
}

function findingsCard(period) {
  return card({ title: period ? "What the Full 2019–2023 Study Shows" : "What This Shows", className: "span-12",
    body: h("ul", { class: "findings-list" },
    h("li", {}, h("strong", {}, "Mostly after the fact. "),
      "Much of the gap already exists the day before the headline date, and there is no significant drift over the following week. Headlines largely report moves."),
    h("li", {}, h("strong", {}, "Bad news draws more trading. "),
      "Negative-sentiment days carry clearly more abnormal volume than positive ones."),
    h("li", {}, h("strong", {}, "Dates, not times. "),
      "FNSPID has dates but no times, so a day +1 effect may simply be after-hours news.")) });
}

export async function render(root, params, ctx) {
  const variant = params.headlines === "all" ? "all" : "excluding";
  const data = await getJSON("/api/market-impact", { model: params.model, headlines: variant, period: params.period });
  if (!ctx.isCurrent()) return;
  destroy();
  const toggle = segmented({ label: "Which headlines", options: VARIANTS, value: variant,
    onChange: (key) => setParams({ headlines: key === "excluding" ? null : key }) });
  root.replaceChildren(
    h("div", { class: "page-tools" }, toggle,
      h("span", { class: "scope" }, `${data.scope} · ${fmt.int(data.events)} company-days · ${data.tickers} companies`)),
    h("div", { class: "grid" }, spreadCard(data)),
    h("div", { class: "grid" }, pathsCard(data, params.period)),
    h("div", { class: "grid" }, findingsCard(params.period)),
  );
}
