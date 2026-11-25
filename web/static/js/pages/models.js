// Model Evaluation (spec 8.5): accuracy on PhraseBank sentences and on real headlines, per company and pooled.
import { getJSON } from "../api.js";
import { C, barsWithCI, heatmap, withTable } from "../charts.js";
import { card, fmt, h, note } from "../ui.js";

// Names in the metrics files -> the names the rest of the site uses.
const DISPLAY = {
  "DistilBERT fine-tuned": "PhraseBank DistilBERT",
  "DistilBERT headline-adapted": "Headline-adapted DistilBERT",
  "DistilBERT class-weighted": "Class-weighted DistilBERT",
  "finbert (off-the-shelf)": "FinBERT (off the shelf)",
  "Zero-shot BART-MNLI": "Zero-shot BART-MNLI",
  "Majority class (neutral)": "Always neutral (baseline)",
  "Majority class (positive)": "Always positive (baseline)",
};
const SELECTED = { adapted: "DistilBERT headline-adapted", base: "DistilBERT fine-tuned" };
const BASELINE_COLOR = "#d5d9e0";
const OTHER_COLOR = "#9aa3b2";
let charts = [];

export function destroy() {
  charts.forEach((c) => c.dispose());
  charts = [];
}

const name = (m) => DISPLAY[m] || m;

function f1Card({ title, rows, selected, info, captions = [] }) {
  const sorted = [...rows].sort((a, b) => b.macro_f1 - a.macro_f1);
  const bars = sorted.map((r) => ({
    category: name(r.model), value: r.macro_f1, low: r.ci_low, high: r.ci_high,
    color: r.model === selected ? C.accent : r.model.startsWith("Majority") ? BASELINE_COLOR : OTHER_COLOR,
  }));
  const chartEl = h("div", { class: "chart", style: `height:${Math.max(220, 44 * bars.length + 40)}px`, role: "img",
    "aria-label": `${title}: ${bars.map((b) => `${b.category} ${fmt.num(b.value)}`).join(", ")}` });
  requestAnimationFrame(() => charts.push(barsWithCI(chartEl, { rows: bars, horizontal: true,
    yFormat: (v) => fmt.num(v, 2) })));
  return card({ title, info, className: "span-6", body: [
    withTable(chartEl, { caption: title, rows: sorted, columns: [
      { key: "model", label: "Model", format: name },
      { key: "macro_f1", label: "Macro F1", num: true, format: (v) => fmt.num(v, 3) },
      { key: "ci_low", label: "95% CI", num: true,
        format: (v, r) => (r.ci_low === null ? "–" : `${fmt.num(r.ci_low, 3)}–${fmt.num(r.ci_high, 3)}`) },
      { key: "accuracy", label: "Accuracy", num: true, format: (v) => fmt.pct(v, 1) },
      ...(rows[0].mcnemar_p !== undefined
        ? [{ key: "mcnemar_p", label: "McNemar p vs PhraseBank", num: true, format: (v) => (v === null ? "–" : fmt.num(v, 3)) }]
        : [{ key: "ece", label: "ECE", num: true, format: (v) => (v === null ? "–" : fmt.num(v, 3)) }]),
    ] }),
    ...captions.map((c) => h("p", { class: "caption" }, c)),
  ] });
}

function heatmapCard(data, params) {
  const models = [...new Set(data.per_company.map((r) => r.model))];
  const tickers = [...new Set(data.per_company.map((r) => r.ticker))];
  const cells = data.per_company.map((r) => [models.indexOf(r.model), tickers.indexOf(r.ticker), r.macro_f1]);
  const values = cells.map((c) => c[2]);
  const chartEl = h("div", { class: "chart", style: `height:${60 * tickers.length + 60}px`, role: "img",
    "aria-label": "Macro F1 for each company and model" });
  const selected = name(SELECTED[params.model || "adapted"]);
  requestAnimationFrame(() => charts.push(heatmap(chartEl, {
    xCats: models.map(name), yCats: tickers, cells, min: Math.min(...values), max: Math.max(...values),
    highlightRow: tickers.includes(params.ticker) ? params.ticker : null, highlightCol: selected })));
  return card({ title: "Macro F1 by Company", className: "span-12",
    info: "Macro F1 on LLM-annotated headlines that pass the relevance filter, for each annotated company. "
      + "Source: multi_company_per_company_accuracy.csv.",
    body: [withTable(chartEl, { caption: "Macro F1 by company", rows: data.per_company, columns: [
      { key: "ticker", label: "Company" },
      { key: "model", label: "Model", format: name },
      { key: "n", label: "Headlines", num: true, format: fmt.int },
      { key: "macro_f1", label: "Macro F1", num: true, format: (v) => fmt.num(v, 3) },
      { key: "accuracy", label: "Accuracy", num: true, format: (v) => fmt.pct(v, 1) },
    ] }), h("p", { class: "caption" }, `The selected model (${selected}) is in bold`
      + `${tickers.includes(params.ticker) ? `, and ${params.ticker} is outlined` : ""}.`)] });
}

function relevanceCard(data) {
  const pct = (v) => fmt.pct(v, 0);
  const table = h("table", { class: "data-table" },
    h("thead", {}, h("tr", {}, ["Company", "Precision", "Recall (estimate)", "On-topic share (estimate)"]
      .map((t, i) => h("th", { class: i ? "num" : null, scope: "col" }, t)))),
    h("tbody", {}, data.relevance.map((r) => h("tr", {}, h("th", { scope: "row" }, r.ticker),
      h("td", { class: "num" }, pct(r.precision)), h("td", { class: "num" }, pct(r.recall_estimate)),
      h("td", { class: "num" }, pct(r.on_topic_share_estimate))))));
  return card({ title: "Relevance Filter", className: "span-6",
    info: "Is the headline really about the company? Checked against LLM-annotated samples. Source: "
      + "multi_company_relevance_filter.csv.",
    body: [h("div", { class: "table-responsive" }, table), h("p", { class: "caption" },
      "Near-perfect precision. Recall is lower, but most on-topic headlines it drops are neutral listicles, "
      + "so it keeps almost every headline that carries sentiment.")] });
}

function stabilityCard(data) {
  return card({ title: "Stability and Calibration", className: "span-6", body: [
    h("p", { class: "m-0" }, `Seed variation (${data.seeds.n} seeds): macro F1 ${fmt.num(data.seeds.mean, 3)} ± `
      + `${fmt.num(data.seeds.std, 3)} on PhraseBank.`),
    h("p", { class: "mt-2 mb-0" }, `Calibration: expected calibration error ${fmt.num(data.calibration_ece, 3)}, `
      + "so the model's probabilities are used as is."),
  ] });
}

export async function render(root, params, ctx) {
  const data = await getJSON("/api/model-evaluation");
  if (!ctx.isCurrent()) return;
  destroy();
  const selected = SELECTED[params.model || "adapted"];
  root.replaceChildren(
    note({ kind: "caveat", text: "Domain shift: macro F1 falls from 0.84 on PhraseBank sentences to about "
      + "0.55–0.70 on real headlines. Read the timelines as a rough indicator of tone, not a precise measurement." }),
    h("div", { class: "grid" },
      f1Card({ title: "Financial PhraseBank Test (N = 726)", rows: data.phrasebank, selected,
        info: "Macro F1 on the held-out PhraseBank test sentences. Source: test_set_model_comparison.csv and "
          + "adapted_test_metrics.json.",
        captions: ["FinBERT was trained on PhraseBank, so its score there is likely inflated.",
          "No bootstrap CI is saved for the headline-adapted model's PhraseBank score, so its bar has no whiskers."] }),
      f1Card({ title: "Real Headlines, Six Companies (N = 554)", rows: data.headlines, selected,
        info: "Macro F1 on relevance-filtered headlines for TSLA, AAPL, NVDA, AMZN, GILD and SBUX. "
          + "Source: multi_company_pooled_accuracy.csv.",
        captions: ["Headline labels come from an LLM annotator and are not human-verified."] })),
    h("div", { class: "grid" }, heatmapCard(data, params)),
    h("div", { class: "grid" }, relevanceCard(data), stabilityCard(data)),
  );
}
