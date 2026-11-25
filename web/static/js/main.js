// App shell: routing, the global filter row, and the page lifecycle.
import { ApiError, getJSON } from "./api.js";
import { card, fmt, loading, note } from "./ui.js";
import { onRoute, readRoute, setParams } from "./state.js";

// Which global filters each page honours (the rest are shown dimmed; see spec section 8.0).
const PAGES = {
  overview: { title: "Overview", desc: "Historical financial-news sentiment across the Nasdaq-100, 2019–2023",
    filters: ["company", "period", "model"], load: () => import("./pages/overview.js") },
  company: { title: "Company Analysis", desc: "Sentiment over time, lookback windows and headlines for one company",
    filters: ["company", "period", "model"], load: () => import("./pages/company.js") },
  market: { title: "Market Impact", desc: "How headline tone lines up with market-adjusted returns and trading volume",
    filters: ["model"], load: () => import("./pages/market.js") },
  news: { title: "News Explorer", desc: "Search, filter and read every scored headline",
    filters: ["company", "period", "model"], load: () => import("./pages/news.js") },
  models: { title: "Model Evaluation", desc: "How accurate the sentiment classifier is, and where it falls short",
    filters: ["company", "model"], load: () => import("./pages/models.js") },
  score: { title: "Score a Headline", desc: "Score your own headlines with the deployed model",
    filters: ["company"], load: () => import("./pages/score.js") },
  live: { title: "Live Feed", desc: "Headlines collected since the live feed started, scored as they arrive",
    filters: ["company"], load: () => import("./pages/live.js") },
};

const $ = (id) => document.getElementById(id);
const els = {
  title: $("page-title"), desc: $("page-desc"), body: $("page-body"), alert: $("page-alert"),
  company: $("f-company"), period: $("f-period"), model: $("f-model"), tickers: $("ticker-list"),
  status: $("data-status"),
};

let meta = null;
let tickerSet = new Set();
let carryAlert = null;
const current = { page: null, module: null, token: 0 };

function setStatus(text, live = false) {
  els.status.classList.toggle("is-live", live);
  els.status.lastElementChild.textContent = text;
}

function fillFilters() {
  tickerSet = new Set(meta.tickers.map((t) => t.ticker));
  els.tickers.replaceChildren(...meta.tickers.map((t) => {
    const option = document.createElement("option");
    option.value = t.ticker;
    option.label = `${fmt.int(t.headlines)} headlines`;
    return option;
  }));
  for (const year of meta.years) els.period.append(new Option(String(year), String(year)));
  for (const m of meta.models) els.model.append(new Option(m.label, m.id));
  const c = meta.corpus;
  $("ds-companies").textContent = `${meta.tickers.length} companies`;
  if (c) {
    $("ds-years").textContent = `${c.first_date.slice(0, 4)} – ${c.last_date.slice(0, 4)}`;
    $("ds-headlines").textContent = `${fmt.int(c.collected_headlines)} collected headlines`;
  }
}

function syncFilters({ page, params }) {
  els.company.value = params.ticker || "";
  els.period.value = params.period || "";
  els.model.value = params.model || "adapted";
  const applies = PAGES[page].filters;
  for (const [name, input] of [["company", els.company], ["period", els.period], ["model", els.model]]) {
    const idle = !applies.includes(name);
    input.closest(".filter").classList.toggle("is-idle", idle);
    input.title = idle ? "Doesn't apply on this page" : "";
  }
}

function bindFilters() {
  els.company.addEventListener("change", () => {
    const value = els.company.value.trim().toUpperCase();
    if (!value) return setParams({ ticker: null });
    if (tickerSet.has(value)) return setParams({ ticker: value });
    els.alert.replaceChildren(note({ kind: "error", text: `No company called ${value} in the dataset.` }));
    els.company.value = readRoute().params.ticker || "";
  });
  els.period.addEventListener("change", () => setParams({ period: els.period.value || null }));
  els.model.addEventListener("change", () =>
    setParams({ model: els.model.value === "adapted" ? null : els.model.value }));
}

function markNav(page) {
  for (const link of document.querySelectorAll("#nav .nav-link")) {
    if (link.dataset.page === page) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  document.body.classList.remove("sidebar-open"); // close the mobile drawer after navigating
}

async function render(route) {
  const token = ++current.token;
  const entry = PAGES[route.page];
  const pageChanged = route.page !== current.page;
  current.module?.destroy?.();
  current.module = null;
  current.page = route.page;

  els.title.textContent = entry.title;
  els.desc.textContent = entry.desc;
  document.title = `${entry.title} · Sentiment Monitor`;
  markNav(route.page);
  syncFilters(route);
  setStatus("Historical data · not live");
  // A carried alert survives redirects (e.g. Company Analysis defaulting to TSLA) until a render completes.
  els.alert.replaceChildren(...(carryAlert ? [carryAlert] : []));
  if (pageChanged) window.scrollTo(0, 0);
  els.body.replaceChildren(loading());

  try {
    const module = await entry.load();
    if (token !== current.token) return;
    current.module = module;
    await module.render(els.body, route.params, {
      meta, setStatus, isCurrent: () => token === current.token,
    });
    if (token === current.token) carryAlert = null;
  } catch (err) {
    if (token !== current.token) return;
    if (err instanceof ApiError && err.status === 404 && route.params.ticker) {
      carryAlert = note({ kind: "error", text: err.detail });
      setParams({ ticker: null }, { replace: true });
      return;
    }
    console.error(err);
    const text = err instanceof ApiError ? err.detail : "This page failed to load. Reload to try again.";
    els.body.replaceChildren(card({ body: note({ kind: "error", text }) }));
  }
}

async function start() {
  bindFilters();
  try {
    meta = await getJSON("/api/meta");
    fillFilters();
  } catch (err) {
    els.body.replaceChildren(card({ body: note({ kind: "error",
      text: err instanceof ApiError ? err.detail : "Couldn't load the dataset list. Reload to try again." }) }));
    return;
  }
  onRoute(render);
  render(readRoute());
}

start();
