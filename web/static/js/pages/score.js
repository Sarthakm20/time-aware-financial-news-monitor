// Score a Headline (spec 8.6): run the deployed model on the visitor's own headlines.
import { ApiError, postJSON } from "../api.js";
import { card, fmt, h, note, sentimentBadge } from "../ui.js";

const MAX_LINES = 50;
const EXAMPLES = "Tesla Deliveries Top Estimates as Demand Surges\n"
  + "Intel Cuts Full-Year Guidance as PC Demand Slumps\n"
  + "Apple To Report Q3 Earnings On Thursday";
const PROB_ORDER = [["negative", "Negative"], ["neutral", "Neutral"], ["positive", "Positive"]];

// Kept across filter changes so a re-render doesn't wipe what the visitor typed or scored.
const state = { text: EXAMPLES, ticker: null, result: null, error: null };

function probabilityBar(probs) {
  return h("div", { class: "prob-bar", role: "img",
    "aria-label": PROB_ORDER.map(([k, label]) => `${label} ${fmt.pct(probs[k], 0)}`).join(", ") },
  PROB_ORDER.map(([k, label]) => {
    const share = probs[k];
    return h("span", { class: `prob prob-${k}`, style: `flex-grow:${Math.max(share, 0.0001)}`, title: `${label} ${fmt.pct(share, 1)}` },
      share >= 0.12 ? `${label} ${fmt.pct(share, 0)}` : "");
  }));
}

function resultsView() {
  if (state.error) return note({ kind: "error", text: state.error });
  if (!state.result) return null;
  const { predictions, model, ticker, truncated } = state.result;
  return h("div", {},
    truncated ? note({ kind: "info", text: `Only the first ${MAX_LINES} headlines were scored.` }) : null,
    h("ul", { class: "rows" }, predictions.map((p) => h("li", {},
      h("div", { class: "row-meta" }, sentimentBadge(p.label, p.sentiment_score),
        h("span", {}, `confidence ${fmt.pct(p.confidence)}`),
        p.relevant === null || p.relevant === undefined ? null
          : h("span", { class: `relevance-tag ${p.relevant ? "is-on" : "is-off"}` },
            p.relevant ? `About ${ticker}` : `Not about ${ticker}`)),
      h("p", { class: "row-title" }, p.headline),
      probabilityBar(p.probabilities)))),
    h("p", { class: "caption" }, `Model: ${model}. Known weakness: product recalls are often scored neutral rather than negative.`));
}

export async function render(root, params) {
  if (state.ticker === null) state.ticker = params.ticker || "";
  const textarea = h("textarea", { id: "s-text", class: "form-control", rows: 6, spellcheck: "true" });
  textarea.value = state.text;
  textarea.addEventListener("input", () => { state.text = textarea.value; });
  const ticker = h("input", { id: "s-ticker", class: "form-control", type: "text", maxlength: 10, value: state.ticker,
    placeholder: "e.g. TSLA", autocomplete: "off", spellcheck: "false" });
  ticker.addEventListener("input", () => { state.ticker = ticker.value; });
  const button = h("button", { type: "submit", class: "btn btn-dark" }, "Score headlines");
  const out = h("div", { class: "score-results", "aria-live": "polite" }, resultsView());

  const form = h("form", { class: "score-form" },
    h("div", { class: "field" }, h("label", { for: "s-text" }, "Headlines, one per line"), textarea),
    h("div", { class: "score-row" },
      h("div", { class: "field" }, h("label", { for: "s-ticker" }, "Company ticker (optional, adds the relevance check)"), ticker),
      button));
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    const lines = textarea.value.split("\n").map((l) => l.trim()).filter(Boolean);
    if (!lines.length) {
      state.result = null;
      state.error = "Add at least one headline, one per line.";
      out.replaceChildren(resultsView());
      return;
    }
    const symbol = ticker.value.trim().toUpperCase() || null;
    button.disabled = true;
    button.textContent = "Scoring…";
    try {
      const body = await postJSON("/predict", { headlines: lines.slice(0, MAX_LINES), ticker: symbol });
      state.result = { predictions: body.predictions, model: body.model, ticker: symbol, truncated: lines.length > MAX_LINES };
      state.error = null;
    } catch (err) {
      state.result = null;
      state.error = err instanceof ApiError && err.status === 503
        ? "The sentiment model isn't available on this server. Set SENTIMENT_MODEL to a Hugging Face Hub id."
        : err instanceof ApiError ? err.detail : "Scoring failed. Try again.";
    } finally {
      button.disabled = false;
      button.textContent = "Score headlines";
    }
    out.replaceChildren(resultsView());
  });

  root.replaceChildren(card({ title: "Your Headlines", className: "span-12",
    info: "Scores come from the deployed model (P(positive) − P(negative), from −1 to +1). The relevance check "
      + "says whether a headline is about the company you name.", body: [form, out] }));
}
