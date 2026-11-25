// News Explorer (spec 8.4): search and filter every scored headline. All state lives in the URL.
import { getJSON } from "../api.js";
import { card, debounce, emptyState, fmt, h, sentimentBadge } from "../ui.js";
import { setParams } from "../state.js";

const PAGE_SIZE = 25;
const SORTS = [["newest", "Newest first"], ["oldest", "Oldest first"], ["most_positive", "Most positive"],
  ["most_negative", "Most negative"]];
const LABELS = [["", "All sentiment"], ["positive", "Positive"], ["neutral", "Neutral"], ["negative", "Negative"]];

function field(id, label, control) {
  return h("div", { class: "field" }, h("label", { for: id }, label), control);
}

function select(id, options, value, onChange) {
  const el = h("select", { id, class: "form-select" },
    options.map(([v, t]) => h("option", { value: v, selected: v === value }, t)));
  el.addEventListener("change", () => onChange(el.value));
  return el;
}

function filterBar(params) {
  const search = h("input", { id: "n-q", class: "form-control", type: "search", value: params.q || "",
    placeholder: "Words in the headline", autocomplete: "off", maxlength: 200 });
  const commit = debounce(() => setParams({ q: search.value.trim() || null, page: null }, { replace: true }), 300);
  search.addEventListener("input", commit);
  const date = (id, key, label) => {
    const el = h("input", { id, class: "form-control", type: "date", value: params[key] || "", min: "2019-01-01",
      max: "2023-12-31" });
    el.addEventListener("change", () => setParams({ [key]: el.value || null, page: null }));
    return field(id, label, el);
  };
  return h("form", { class: "news-filters", role: "search", onsubmit: (e) => e.preventDefault() },
    field("n-q", "Search headlines", search),
    field("n-label", "Sentiment", select("n-label", LABELS, params.label || "",
      (v) => setParams({ label: v || null, page: null }))),
    date("n-start", "start", "From"),
    date("n-end", "end", "To"),
    field("n-sort", "Sort", select("n-sort", SORTS, params.sort || "newest",
      (v) => setParams({ sort: v === "newest" ? null : v, page: null }))));
}

function pageList(page, pages) {
  const set = new Set([1, pages, page - 1, page, page + 1].filter((p) => p >= 1 && p <= pages));
  const sorted = [...set].sort((a, b) => a - b);
  const out = [];
  sorted.forEach((p, i) => {
    if (i && p - sorted[i - 1] > 1) out.push("…");
    out.push(p);
  });
  return out;
}

function pagination(page, pages) {
  if (pages <= 1) return null;
  const go = (p) => setParams({ page: p === 1 ? null : p });
  const button = (label, target, attrs = {}) => h("button", { type: "button", class: "page-btn",
    disabled: target < 1 || target > pages, onclick: () => go(target), ...attrs }, label);
  return h("nav", { class: "pager", "aria-label": "Result pages" },
    button("Previous", page - 1),
    pageList(page, pages).map((p) => (p === "…" ? h("span", { class: "page-gap", "aria-hidden": "true" }, "…")
      : button(String(p), p, p === page ? { "aria-current": "page", class: "page-btn is-current" } : {}))),
    button("Next", page + 1));
}

function results(data, page) {
  const count = h("p", { class: "result-count" }, h("strong", {}, `${fmt.int(data.total)} headlines`),
    h("span", {}, "Every headline here already passed the relevance filter."));
  if (!data.total) {
    return [count, emptyState("No headlines match. Clear the search or widen the date range.")];
  }
  if (!data.rows.length) {
    return [count, emptyState(`There are only ${data.pages} pages of results.`,
      h("button", { type: "button", class: "btn btn-outline-secondary btn-sm", onclick: () => setParams({ page: null }) },
        "Go to page 1"))];
  }
  const list = h("ul", { class: "rows" }, data.rows.map((r) => h("li", {},
    h("div", { class: "row-meta" }, h("span", { class: "ticker" }, r.ticker), sentimentBadge(r.label, r.sentiment_score),
      h("span", {}, `confidence ${fmt.pct(r.confidence)}`), h("span", {}, fmt.date(r.date))),
    h("p", { class: "row-title" }, r.headline))));
  return [count, list, pagination(page, data.pages)];
}

export async function render(root, params, ctx) {
  const page = Math.max(1, Number(params.page) || 1);
  const start = params.start || (params.period ? `${params.period}-01-01` : null);
  const end = params.end || (params.period ? `${params.period}-12-31` : null);
  const data = await getJSON("/api/headlines", {
    q: params.q, ticker: params.ticker, label: params.label, start, end, sort: params.sort, page,
    page_size: PAGE_SIZE, model: params.model,
  });
  if (!ctx.isCurrent()) return;

  // Keep typing uninterrupted: the page re-renders from the URL, so restore focus and caret.
  const active = document.activeElement;
  const focusId = active && root.contains(active) ? active.id : null;
  const caret = focusId === "n-q" ? active.selectionStart : null;

  root.replaceChildren(card({ className: "news-panel", body: [filterBar(params), ...results(data, page)] }));

  if (focusId) {
    const el = document.getElementById(focusId);
    el?.focus();
    if (caret !== null && el) el.setSelectionRange(caret, caret);
  }
}
