// Shared ECharts builders. Every line chart has a crosshair with live labels on both axes;
// every bar/slice/cell has a hover tooltip; every chart can be shown as a table.
import { fmt, h } from "./ui.js";

const echarts = window.echarts;

export const C = {
  accent: "#2a78d6", positive: "#087a5a", neutral: "#b9bec8", negative: "#ec6a50",
  ink: "#141a26", ink2: "#4b5565", muted: "#7a8394", grid: "#e6e9ef", axis: "#c5cbd5", card: "#ffffff",
};
export const TONE_COLORS = { positive: C.positive, neutral: C.neutral, negative: C.negative };

const FONT = "Public Sans, system-ui, -apple-system, Segoe UI, sans-serif";
const reducedMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

const BASE = () => ({
  textStyle: { fontFamily: FONT, color: C.ink2 },
  animation: !reducedMotion(),
  animationDuration: 200,
  animationDurationUpdate: 200,
});

const AXIS_LABEL = { color: C.muted, fontSize: 12, fontFamily: FONT };
const TOOLTIP = {
  backgroundColor: "#ffffff", borderColor: C.grid, borderWidth: 1, padding: [8, 12],
  textStyle: { color: C.ink, fontSize: 13, fontFamily: FONT },
  extraCssText: "box-shadow: 0 2px 8px rgba(20, 26, 38, 0.12); border-radius: 8px;",
};
const POINTER_LABEL = { backgroundColor: C.ink, color: "#ffffff", fontFamily: FONT, fontSize: 12, padding: [4, 6] };

function mount(el, option) {
  const chart = echarts.init(el, null, { renderer: "svg" });
  chart.setOption({ ...BASE(), ...option });
  const observer = new ResizeObserver(() => chart.resize());
  observer.observe(el);
  return {
    chart,
    dispose() {
      observer.disconnect();
      chart.dispose();
    },
  };
}

// "Show as table" under a chart: the accessible twin of every plot.
export function withTable(chartEl, { columns, rows, caption }) {
  const wrap = h("div", { class: "table-wrap", hidden: true });
  const table = h("table", { class: "data-table" },
    caption ? h("caption", { class: "visually-hidden" }, caption) : null,
    h("thead", {}, h("tr", {}, columns.map((c) => h("th", { class: c.num ? "num" : null, scope: "col" }, c.label)))),
    h("tbody", {}, rows.map((r) => h("tr", {}, columns.map((c) => {
      const v = r[c.key];
      return h("td", { class: c.num ? "num" : null }, c.format ? c.format(v, r) : v ?? "–");
    })))));
  wrap.append(table);
  const button = h("button", { type: "button", class: "table-toggle", "aria-expanded": "false" }, "Show as table");
  button.addEventListener("click", () => {
    const open = wrap.hidden;
    wrap.hidden = !open;
    button.setAttribute("aria-expanded", String(open));
    button.textContent = open ? "Hide table" : "Show as table";
  });
  return h("div", {}, chartEl, button, wrap);
}

const dateLabel = (ms) => fmt.date(new Date(ms).toISOString());

// Break a time series where consecutive points are further apart than maxGapDays, so a stretch with no
// headlines shows as a gap (missing data), never as a straight line implying values that don't exist.
function breakGaps(data, maxGapDays) {
  const out = [];
  for (let i = 0; i < data.length; i += 1) {
    if (i && data[i][0] - data[i - 1][0] > maxGapDays * 864e5) out.push([(data[i][0] + data[i - 1][0]) / 2, null]);
    out.push(data[i]);
  }
  return out;
}

// Line chart over time (xType "time") or over a numeric axis (xType "value", e.g. event days).
export function lineChart(el, {
  series, xType = "time", xFormat = dateLabel, yFormat = (v) => fmt.signed(v, 2), yMin, yMax,
  zeroLine = true, area = true, endLabels = false, zoom = xType === "time", slider = zoom, xName, yName, maxGapDays = xType === "time" ? 14 : null,
}) {
  const multi = series.length > 1;
  const option = {
    grid: { left: 8, right: endLabels ? 104 : 16, top: multi ? 36 : 16, bottom: slider ? 56 : 28, containLabel: true },
    legend: multi ? { top: 0, left: 0, icon: "roundRect", itemWidth: 14, itemHeight: 4,
      textStyle: { color: C.ink2, fontSize: 12 } } : undefined,
    tooltip: {
      ...TOOLTIP, trigger: "axis",
      axisPointer: { type: "cross", lineStyle: { color: C.muted, type: "dashed" },
        crossStyle: { color: C.muted, type: "dashed" }, label: POINTER_LABEL },
      formatter: (items) => {
        const at = items[0]?.axisValue;
        items = items.filter((it) => it.value && it.value[1] !== null && it.value[1] !== undefined);
        if (!items.length) return at === undefined ? "" : `${xFormat(at)}<br>No headlines in this stretch`;
        const head = xFormat(items[0].axisValue);
        const lines = items.map((it) => `${it.marker}${it.seriesName}: <b>${yFormat(it.value[1])}</b>`
          + (it.value[2] !== undefined ? ` <span style="color:${C.muted}">(${fmt.int(it.value[2])} headline${it.value[2] === 1 ? "" : "s"})</span>` : ""));
        return [`<div style="margin-bottom:4px;color:${C.ink2}">${head}</div>`, ...lines].join("<br>");
      },
    },
    xAxis: {
      type: xType, name: xName, nameLocation: "middle", nameGap: 28, nameTextStyle: AXIS_LABEL,
      axisLine: { lineStyle: { color: C.axis } }, axisTick: { show: false },
      axisLabel: { ...AXIS_LABEL, hideOverlap: true, formatter: xType === "time" ? undefined : (v) => xFormat(v) },
      axisPointer: { label: { formatter: ({ value }) => xFormat(value) } },
      splitLine: { show: false },
      min: xType === "value" ? "dataMin" : undefined, max: xType === "value" ? "dataMax" : undefined,
    },
    yAxis: {
      type: "value", min: yMin, max: yMax, name: yName, nameTextStyle: AXIS_LABEL,
      axisLabel: { ...AXIS_LABEL, formatter: (v) => yFormat(v) },
      axisPointer: { label: { formatter: ({ value }) => yFormat(value) } },
      splitLine: { lineStyle: { color: C.grid } },
    },
    dataZoom: zoom ? [
      { type: "inside", filterMode: "none" },
      slider && { type: "slider", height: 18, bottom: 8, borderColor: C.grid, fillerColor: "rgba(42,120,214,0.10)",
        handleStyle: { color: "#ffffff", borderColor: C.axis }, textStyle: { color: C.muted, fontSize: 11 },
        dataBackground: { lineStyle: { color: C.axis }, areaStyle: { color: C.grid } }, labelFormatter: (v) => dateLabel(v) },
    ].filter(Boolean) : undefined,
    series: series.map((s, i) => ({
      type: "line", name: s.name, data: maxGapDays ? breakGaps(s.data, maxGapDays) : s.data, connectNulls: false,
      showSymbol: false, symbolSize: 8,
      lineStyle: { width: 2, color: s.color || C.accent }, itemStyle: { color: s.color || C.accent },
      areaStyle: area && !multi ? { color: s.color || C.accent, opacity: 0.1 } : undefined,
      emphasis: { disabled: true },
      endLabel: endLabels ? { show: true, formatter: s.name, color: C.ink2, fontSize: 12, fontFamily: FONT } : undefined,
      markLine: zeroLine && i === 0 ? { silent: true, symbol: "none", label: { show: false },
        lineStyle: { color: C.axis, type: "dashed", width: 1 }, data: [{ yAxis: 0 }] } : undefined,
    })),
  };
  const handle = mount(el, option);
  handle.zoomToYears = (years) => {
    const all = series[0].data;
    if (!all.length) return;
    const end = all[all.length - 1][0];
    const start = years ? new Date(end) : null;
    if (start) start.setFullYear(start.getFullYear() - years);
    handle.chart.dispatchAction({ type: "dataZoom", startValue: start ? start.getTime() : all[0][0], endValue: end });
  };
  return handle;
}

// Bars with 95% CI whiskers. rows: [{category, value, low, high, color}]
export function barsWithCI(el, { rows, yFormat = (v) => `${fmt.signed(v, 2)}%`, horizontal = false, categoryLabel }) {
  const cats = rows.map((r) => r.category);
  const valueAxis = {
    type: "value", axisLabel: { ...AXIS_LABEL, formatter: (v) => yFormat(v) },
    splitLine: { lineStyle: { color: C.grid } },
  };
  const catAxis = {
    type: "category", data: cats, axisTick: { show: false }, axisLine: { lineStyle: { color: C.axis } },
    axisLabel: { ...AXIS_LABEL, color: C.ink2, interval: 0,
      formatter: (v) => (categoryLabel ? categoryLabel(rows[cats.indexOf(v)]) : v) },
    inverse: horizontal,
  };
  const radius = (v) => (horizontal ? (v >= 0 ? [0, 4, 4, 0] : [4, 0, 0, 4]) : (v >= 0 ? [4, 4, 0, 0] : [0, 0, 4, 4]));
  const whisker = (params, api) => {
    const i = api.value(0);
    const lo = api.value(1);
    const hi = api.value(2);
    if (lo === null || hi === null || Number.isNaN(lo) || Number.isNaN(hi)) return null;
    const a = horizontal ? api.coord([lo, i]) : api.coord([i, lo]);
    const b = horizontal ? api.coord([hi, i]) : api.coord([i, hi]);
    const cap = 6;
    const style = { stroke: C.ink, lineWidth: 1.5 };
    const line = (x1, y1, x2, y2) => ({ type: "line", shape: { x1, y1, x2, y2 }, style });
    return {
      type: "group",
      children: horizontal
        ? [line(a[0], a[1], b[0], b[1]), line(a[0], a[1] - cap, a[0], a[1] + cap), line(b[0], b[1] - cap, b[0], b[1] + cap)]
        : [line(a[0], a[1], b[0], b[1]), line(a[0] - cap, a[1], a[0] + cap, a[1]), line(b[0] - cap, b[1], b[0] + cap, b[1])],
    };
  };
  const option = {
    grid: { left: 8, right: 24, top: 16, bottom: 8, containLabel: true },
    tooltip: {
      // trigger on the whole category band, so a near-zero bar is still easy to hover
      ...TOOLTIP, trigger: "axis",
      axisPointer: { type: "shadow", shadowStyle: { color: "rgba(20, 26, 38, 0.05)" } },
      formatter: (items) => {
        const p = items.find((it) => it.seriesType === "bar") || items[0];
        const r = rows[p.dataIndex];
        const ci = r.low === null || r.low === undefined ? "no confidence interval saved"
          : `95% CI ${yFormat(r.low)} to ${yFormat(r.high)}`;
        return `${r.category}<br><b>${yFormat(r.value)}</b><br><span style="color:${C.muted}">${ci}</span>`;
      },
    },
    xAxis: horizontal ? valueAxis : catAxis,
    yAxis: horizontal ? catAxis : valueAxis,
    series: [
      { type: "bar", barMaxWidth: 56, barCategoryGap: "35%",
        data: rows.map((r) => ({ value: r.value, itemStyle: { color: r.color || C.accent, borderRadius: radius(r.value) } })),
        markLine: { silent: true, symbol: "none", label: { show: false }, lineStyle: { color: C.axis, width: 1 },
          data: [horizontal ? { xAxis: 0 } : { yAxis: 0 }] } },
      { type: "custom", renderItem: whisker, silent: true, z: 10,
        encode: horizontal ? { x: [1, 2], y: 0 } : { x: 0, y: [1, 2] },
        data: rows.map((r, i) => [i, r.low ?? null, r.high ?? null]) },
    ],
  };
  return mount(el, option);
}

// Counts over time as bars (used under a line chart instead of a second y-axis). points: [[ms, count]]
export function timeBars(el, { points, name, color = C.muted, xMin, xMax }) {
  return mount(el, {
    grid: { left: 8, right: 16, top: 8, bottom: 24, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: "axis",
      axisPointer: { type: "line", lineStyle: { color: C.muted, type: "dashed" }, label: POINTER_LABEL },
      formatter: (items) => `${dateLabel(items[0].value[0])}<br>${name}: <b>${fmt.int(items[0].value[1])}</b>` },
    xAxis: { type: "time", min: xMin, max: xMax, axisLine: { lineStyle: { color: C.axis } }, axisTick: { show: false },
      axisLabel: { ...AXIS_LABEL, hideOverlap: true }, splitLine: { show: false } },
    yAxis: { type: "value", minInterval: 1, axisLabel: AXIS_LABEL, splitLine: { lineStyle: { color: C.grid } } },
    series: [{ type: "bar", name, data: points, barMaxWidth: 28,
      itemStyle: { color, borderRadius: [4, 4, 0, 0] } }],
  });
}

export function donut(el, { slices, center }) {
  const total = slices.reduce((s, x) => s + x.value, 0) || 1;
  return mount(el, {
    tooltip: { ...TOOLTIP, trigger: "item",
      formatter: (p) => `${p.name}<br><b>${fmt.pct(p.value / total, 1)}</b>` },
    legend: { orient: "vertical", right: 0, top: "middle", icon: "circle", itemWidth: 10, itemHeight: 10,
      textStyle: { color: C.ink2, fontSize: 13, lineHeight: 18 },
      formatter: (name) => `${name}\n${fmt.pct(slices.find((s) => s.name === name).value / total, 0)}` },
    graphic: center ? [{ type: "group", left: "29%", top: "middle", bounding: "raw", children: [
      { type: "text", style: { text: center.value, x: 0, y: -14, textAlign: "center", fill: C.ink,
        font: `700 24px ${FONT}` } },
      { type: "text", style: { text: center.label, x: 0, y: 14, textAlign: "center", fill: C.ink2,
        font: `400 13px ${FONT}` } },
    ] }] : undefined,
    series: [{ type: "pie", radius: ["56%", "80%"], center: ["30%", "50%"], startAngle: 90,
      avoidLabelOverlap: false, label: { show: false }, labelLine: { show: false },
      itemStyle: { borderColor: C.card, borderWidth: 2 },
      emphasis: { scale: true, scaleSize: 4, label: { show: false } },
      data: slices.map((s) => ({ name: s.name, value: s.value, itemStyle: { color: s.color } })) }],
  });
}

export function ring(el, { value, color = C.positive, label }) {
  return mount(el, {
    tooltip: { show: false },
    graphic: [{ type: "text", left: "center", top: "middle", silent: true,
      style: { text: fmt.pct(value, 0), fill: C.ink, font: `700 24px ${FONT}`, textAlign: "center" } }],
    series: [{ type: "pie", radius: ["66%", "82%"], startAngle: 90, silent: true, label: { show: false },
      itemStyle: { borderColor: C.card, borderWidth: 2 }, name: label,
      data: [{ value, itemStyle: { color } }, { value: 1 - value, itemStyle: { color: C.grid } }] }],
  });
}

// One-hue blue heatmap with the value printed in every cell (identity never by colour alone).
export function heatmap(el, { xCats, yCats, cells, min, max, format = (v) => fmt.num(v, 2), highlightRow, highlightCol }) {
  const ramp = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"];
  const dark = (v) => (v - min) / (max - min) > 0.55;
  return mount(el, {
    grid: { left: 8, right: 8, top: 8, bottom: 8, containLabel: true },
    tooltip: { ...TOOLTIP, trigger: "item",
      formatter: (p) => `${yCats[p.value[1]]} · ${xCats[p.value[0]]}<br><b>${format(p.value[2])}</b>` },
    xAxis: { type: "category", data: xCats, position: "top", axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { ...AXIS_LABEL, color: C.ink2, interval: 0, width: 110, overflow: "break",
        formatter: (v) => (v === highlightCol ? `{hl|${v}}` : v), rich: { hl: { fontWeight: 700, color: C.ink } } } },
    yAxis: { type: "category", data: yCats, inverse: true, axisTick: { show: false }, axisLine: { show: false },
      axisLabel: { ...AXIS_LABEL, color: C.ink2,
        formatter: (v) => (v === highlightRow ? `{hl|${v}}` : v), rich: { hl: { fontWeight: 700, color: C.ink } } } },
    visualMap: { show: false, min, max, inRange: { color: ramp } },
    series: [{ type: "heatmap", itemStyle: { borderColor: C.card, borderWidth: 2, borderRadius: 4 },
      data: cells.map(([x, y, v]) => ({ value: [x, y, v],
        label: { show: true, formatter: () => format(v), color: dark(v) ? "#ffffff" : C.ink, fontSize: 12 },
        itemStyle: yCats[y] === highlightRow ? { borderColor: C.ink, borderWidth: 2 } : undefined })) }],
  });
}

// Sync crosshairs, and make the first chart's zoom window drive the others (same x range assumed).
export function connect(...handles) {
  echarts.connect(handles.map((hd) => hd.chart));
  const [lead, ...rest] = handles;
  lead.chart.on("datazoom", () => {
    const dz = lead.chart.getOption().dataZoom?.[0];
    if (!dz) return;
    for (const other of rest) other.chart.dispatchAction({ type: "dataZoom", start: dz.start, end: dz.end });
  });
}

// Even y-range that always includes zero, rounded out to the nearest 0.1.
export function sentimentRange(values) {
  const lo = Math.min(0, ...values);
  const hi = Math.max(0, ...values);
  return [Math.max(-1, Math.floor((lo - 0.05) * 10) / 10), Math.min(1, Math.ceil((hi + 0.05) * 10) / 10)];
}
