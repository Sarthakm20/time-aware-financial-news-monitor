// URL-held view state: "#/<page>?ticker=NVDA&period=2022&model=adapted&...".
// The hash is the single source of truth, so back/forward and copied links restore the same view.

export const PAGES = ["overview", "company", "market", "news", "models", "score", "live"];

export function readRoute() {
  const raw = window.location.hash.replace(/^#\/?/, "");
  const [path, query = ""] = raw.split("?");
  const page = PAGES.includes(path) ? path : "overview";
  const params = Object.fromEntries(new URLSearchParams(query));
  if (params.ticker) params.ticker = params.ticker.trim().toUpperCase();
  for (const key of Object.keys(params)) if (params[key] === "") delete params[key];
  return { page, params };
}

function build(page, params) {
  const clean = Object.entries(params).filter(([, v]) => v !== null && v !== undefined && v !== "");
  const query = new URLSearchParams(clean).toString();
  return `#/${page}${query ? `?${query}` : ""}`;
}

export function go(page, params = {}) {
  window.location.hash = build(page, params);
}

// Merge into the current params; a null value removes the key. replace=true avoids a history entry
// (used for slider-type controls so Back still means "previous view").
export function setParams(patch, { replace = false } = {}) {
  const { page, params } = readRoute();
  const next = { ...params, ...patch };
  for (const [k, v] of Object.entries(patch)) if (v === null) delete next[k];
  const hash = build(page, next);
  if (hash === window.location.hash) return;
  if (replace) {
    history.replaceState(null, "", hash);
    window.dispatchEvent(new HashChangeEvent("hashchange"));
  } else {
    window.location.hash = hash;
  }
}

export function onRoute(callback) {
  window.addEventListener("hashchange", () => callback(readRoute()));
}
