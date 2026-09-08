/* Shell: language, routing, shared formatting and the Plotly theme.
   Views live in census.js / player.js / cohort.js and each exports render(el). */

import { renderCensus } from "/static/census.js";
import { renderPlayer } from "/static/player.js";
import { renderCohort } from "/static/cohort.js";

export const state = { lang: "ja", strings: {} };

export function t(key) {
  const entry = state.strings[key];
  if (!entry) return key;
  return entry[state.lang] || entry.en || key;
}

/** Published tokens (reason codes, statuses, dispositions) get a gloss where the
    UI has one and stay verbatim otherwise — an unglossed token is data, not a
    label, and inventing a translation for it would hide a schema change. */
export function token(value) {
  const key = `token.${value}`;
  return state.strings[key] ? t(key) : String(value);
}

/** A published token longer than its column has no space to break on, so CSS
    splits it mid-word.  Offering `<wbr>` after each separator wraps it where the
    schema already puts a boundary. */
export function wrappable(value) {
  const node = el("span", {});
  String(value)
    .split(/(?<=[_\-/])/)
    .forEach((part, index) => {
      if (index) node.append(document.createElement("wbr"));
      node.append(document.createTextNode(part));
    });
  return node;
}

export function num(value, digits = 0) {
  if (value === null || value === undefined || value === "") return "—";
  if (typeof value !== "number") return String(value);
  return value.toLocaleString(state.lang === "ja" ? "ja-JP" : "en-US", {
    maximumFractionDigits: digits,
    minimumFractionDigits: digits > 0 && Math.abs(value) < 1000 ? Math.min(digits, 2) : 0,
  });
}

export function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key === "html") node.innerHTML = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "style" && typeof value === "object") Object.assign(node.style, value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

export function panel(title, ...children) {
  return el("section", { class: "panel" }, el("h2", {}, title), ...children);
}

export function table(headers, rows, aligns = []) {
  const head = el(
    "tr",
    {},
    headers.map((label, i) => el("th", { class: aligns[i] === "num" ? "num" : null }, label)),
  );
  const body = rows.map((row) =>
    el(
      "tr",
      {},
      row.map((cell, i) =>
        el(
          "td",
          { class: aligns[i] === "num" ? "num" : null },
          cell && cell.nodeType ? cell : cell === null || cell === undefined ? "—" : String(cell),
        ),
      ),
    ),
  );
  return el("table", {}, el("thead", {}, head), el("tbody", {}, body));
}

/** One Plotly layout for every chart, so the figures read as one document. */
export function chartLayout(overrides = {}) {
  const css = getComputedStyle(document.documentElement);
  const line = css.getPropertyValue("--line").trim();
  const muted = css.getPropertyValue("--muted").trim();
  return Object.assign(
    {
      paper_bgcolor: "rgba(0,0,0,0)",
      plot_bgcolor: "rgba(0,0,0,0)",
      font: { family: "Plex Sans, Plex Sans JP, sans-serif", size: 11, color: muted },
      margin: { l: 56, r: 12, t: 8, b: 40 },
      xaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, automargin: true },
      yaxis: { gridcolor: line, zerolinecolor: line, linecolor: line, automargin: true },
      legend: { orientation: "h", y: -0.22, font: { size: 11 } },
      hoverlabel: { font: { family: "Plex Sans, Plex Sans JP, sans-serif", size: 11 } },
      barmode: "group",
      bargap: 0.28,
    },
    overrides,
  );
}

export const PLOT_CONFIG = { displayModeBar: false, responsive: true };

export function chart(node, traces, layout) {
  Plotly.newPlot(node, traces, chartLayout(layout), PLOT_CONFIG);
  return node;
}

export async function json(path) {
  const response = await fetch(path);
  if (!response.ok) throw new Error(`${path} -> ${response.status}`);
  return response.json();
}

const VIEWS = {
  census: { render: renderCensus, node: () => document.getElementById("view-census") },
  player: { render: renderPlayer, node: () => document.getElementById("view-player") },
  cohort: { render: renderCohort, node: () => document.getElementById("view-cohort") },
};

let current = "census";
const rendered = new Set();

async function show(name, force = false) {
  current = name;
  for (const tab of document.querySelectorAll(".tab")) {
    tab.classList.toggle("active", tab.dataset.view === name);
  }
  for (const [key, view] of Object.entries(VIEWS)) {
    view.node().classList.toggle("active", key === name);
  }
  if (force || !rendered.has(name)) {
    rendered.add(name);
    const node = VIEWS[name].node();
    node.replaceChildren(el("p", { class: "empty" }, t("common.loading")));
    await VIEWS[name].render(node);
  }
  location.hash = name;
}

function applyStaticText() {
  document.documentElement.lang = state.lang;
  for (const node of document.querySelectorAll("[data-i18n]")) {
    node.textContent = t(node.dataset.i18n);
  }
  document.getElementById("lang-toggle").textContent = state.lang === "ja" ? "EN" : "日本語";
}

async function boot() {
  // `?lang=en#cohort` makes a view deep-linkable in either language, which is
  // what a shared review link needs; the toggle then persists the choice.
  const requested = new URLSearchParams(location.search).get("lang");
  state.lang = ["ja", "en"].includes(requested)
    ? requested
    : localStorage.getItem("review-ui-lang") || "ja";
  state.strings = await json("/static/strings.json");
  applyStaticText();

  document.getElementById("tabs").addEventListener("click", (event) => {
    const tab = event.target.closest(".tab");
    if (tab) show(tab.dataset.view);
  });
  document.getElementById("lang-toggle").addEventListener("click", () => {
    state.lang = state.lang === "ja" ? "en" : "ja";
    localStorage.setItem("review-ui-lang", state.lang);
    applyStaticText();
    rendered.clear();
    show(current, true);
  });

  const initial = location.hash.replace("#", "");
  await show(VIEWS[initial] ? initial : "census");
}

boot();
