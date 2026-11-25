// Small DOM helpers. Text always goes in through textContent, never innerHTML:
// headlines come from outside sources and must never be interpreted as markup.

export function h(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "text") node.textContent = value;
    else if (key.startsWith("on") && typeof value === "function") node.addEventListener(key.slice(2), value);
    else if (key === "dataset") Object.assign(node.dataset, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

export const icon = (name) => h("i", { class: `bi bi-${name}`, "aria-hidden": "true" });

export const fmt = {
  int: (n) => (n === null || n === undefined ? "–" : Number(n).toLocaleString("en-US")),
  pct: (x, digits = 0) => (x === null || x === undefined ? "–" : `${(x * 100).toFixed(digits)}%`),
  signed: (x, digits = 2) => {
    if (x === null || x === undefined) return "–";
    const s = Number(x).toFixed(digits);
    return Number(s) > 0 ? `+${s}` : Number(s) === 0 ? s.replace("-", "") : s.replace("-", "−");
  },
  num: (x, digits = 2) => (x === null || x === undefined ? "–" : Number(x).toFixed(digits)),
  date: (iso) => {
    if (!iso) return "–";
    const d = new Date(iso.length === 10 ? `${iso}T00:00:00Z` : iso);
    return d.toLocaleDateString("en-US", { year: "numeric", month: "short", day: "numeric", timeZone: "UTC" });
  },
};

// A content panel. `info` is the source/definition text shown in a popover from the title's (i) button.
export function card({ title, info, scope, tools, body, className = "" }) {
  const head = h("div", { class: "panel-head" });
  if (title) {
    const heading = h("h2", {}, title);
    if (info) {
      // <a tabindex=0>, not <button>: Safari doesn't focus a clicked button, so a focus-triggered popover
      // on a button never opens there (Bootstrap's documented pattern for dismissible popovers).
      const btn = h("a", { class: "info-btn", role: "button", tabindex: "0", "aria-label": `About ${title}` },
        icon("info-circle"));
      heading.append(btn);
      queueMicrotask(() => new window.bootstrap.Popover(btn, {
        content: info, trigger: "focus", placement: "top", customClass: "info-popover" }));
    }
    head.append(heading);
  }
  const toolBox = h("div", { class: "panel-tools" });
  if (scope) toolBox.append(h("span", { class: "scope" }, scope));
  if (tools) toolBox.append(...[tools].flat());
  if (toolBox.childNodes.length) head.append(toolBox);
  const panel = h("section", { class: `panel ${className}` });
  if (title || toolBox.childNodes.length) panel.append(head);
  if (body) panel.append(...[body].flat().filter((x) => x !== null && x !== undefined && x !== false));
  return panel;
}

export function statCard({ icon: iconName, tone = "blue", value, label, sub, scope, info }) {
  const text = h("div", {},
    h("div", { class: "stat-value" }, value),
    h("div", { class: "stat-label" }, label),
    sub ? h("div", { class: "stat-sub" }, sub) : null,
    scope ? h("div", { class: "stat-scope" }, scope) : null);
  const panel = h("section", { class: `panel stat tone-${tone}`, title: info || null },
    h("div", { class: "stat-icon" }, icon(iconName)), text);
  return panel;
}

const NOTE_ICONS = { caveat: "exclamation-triangle", info: "info-circle", error: "x-octagon" };

export function note({ kind = "info", text }) {
  return h("p", { class: `note note-${kind}`, role: kind === "error" ? "alert" : null },
    icon(NOTE_ICONS[kind]), h("span", {}, text));
}

export function emptyState(text, action = null) {
  return h("div", { class: "empty" }, icon("inbox"), h("p", { class: "m-0" }, text), action);
}

export const loading = (text = "Loading…") => h("div", { class: "loading", role: "status" }, text);

const LABELS = { positive: "Positive", neutral: "Neutral", negative: "Negative" };

// Sentiment is never colour-only: the word and the signed score always travel with the colour.
export function sentimentBadge(label, score) {
  return h("span", { class: `badge-sent sent-${label}` },
    `${LABELS[label] || label}${score === undefined || score === null ? "" : ` ${fmt.signed(score)}`}`);
}

export function segmented({ options, value, onChange, label }) {
  const group = h("div", { class: "seg", role: "group", "aria-label": label });
  let selected = value;
  for (const [key, text] of options) {
    const button = h("button", { type: "button", "aria-pressed": String(key === value) }, text);
    button.addEventListener("click", () => {
      if (key === selected) return;
      selected = key;
      for (const b of group.children) b.setAttribute("aria-pressed", String(b === button));
      onChange(key);
    });
    group.append(button);
  }
  return group;
}

export function debounce(fn, ms = 300) {
  let timer;
  return (...args) => {
    clearTimeout(timer);
    timer = setTimeout(() => fn(...args), ms);
  };
}
