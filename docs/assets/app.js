import {buildInteractions, DAY, isContestMode, problemFeatures, ratingFeatures, replay,
  WINDOW_DAYS} from "./features.js?v=19";
import {calibrationFor, predictProbability} from "./model.js?v=19";
import {loadHistory, loadLiveCatalog} from "./codeforces.js?v=19";

const $ = id => document.getElementById(id);
const pct = (v, d = 0) => (100 * v).toFixed(d) + "%";
const num = v => new Intl.NumberFormat("en").format(v);
const fixed = (v, d = 3) => Number(v).toFixed(d);
const SVG = "http://www.w3.org/2000/svg";
const signed = v => (v > 0 ? "+" : v < 0 ? "−" : "±") + Math.abs(v);

function el(tag, text, cls) {
  const node = document.createElement(tag);
  if (text != null) node.textContent = text;
  if (cls) node.className = cls;
  return node;
}
function svg(tag, attrs = {}, text) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text != null) node.textContent = text;
  return node;
}
function problemLink(key, name) {
  const [contest, index] = key.split(":");
  const a = el("a", name || key);
  a.href = `https://codeforces.com/problemset/problem/${contest}/${index}`;
  a.target = "_blank";
  a.rel = "noopener noreferrer";
  return a;
}
const getJSON = path => fetch(new URL("../data/" + path, import.meta.url), {cache: "no-cache"}).then(r => {
  if (!r.ok) throw new Error("A data file could not load. Refresh the page to retry.");
  return r.json();
});

let assets;
const loadAssets = () => assets ??= Promise.all([getJSON("model.json"), getJSON("catalog.json")])
  .then(([model, catalog]) => ({model, catalog}))
  .catch(error => { assets = undefined; throw error; });

// ------------------------------------------------------------------ analysis tool

const LIKELY = 0.85;   // in the test period, estimates of 80–90% were solved 85% of the time
let session = null;   // {model, catalog, interactions, state, rating, now, attempted, topics, weak}
let active = null;

function busy(on, message) {
  $("analyse").disabled = on;
  $("example").disabled = on;
  $("cancel").hidden = !on;
  if (message !== undefined) $("status").textContent = message;
}

async function analyse(mode) {
  if (active) return;
  const controller = new AbortController();
  active = controller;
  $("error").hidden = true;
  busy(true, mode === "example" ? "Loading the example…" : "Loading the model…");
  try {
    const {model, catalog} = await loadAssets();
    let account, now;
    if (mode === "example") {
      const example = await getJSON("example.json");
      account = {handle: example.handle, history: example.history, ratings: example.ratings};
      now = example.now;
    } else {
      const handle = $("handle").value.trim();
      if (!/^[A-Za-z0-9_.-]{1,64}$/.test(handle)) throw new Error("Enter a valid Codeforces handle.");
      account = await loadHistory(handle, controller.signal, message => { $("status").textContent = message; });
      now = Math.floor(Date.now() / 1000);
    }
    controller.signal.throwIfAborted();
    const interactions = buildInteractions(account.history);
    if (interactions.length < 10) throw new Error("This account has fewer than 10 solo problems. The estimates need more history.");
    const state = replay(interactions, now);
    const rating = ratingFeatures(account.ratings, now);
    const attempted = new Set(account.history.map(s => s.key).filter(Boolean));
    const topics = topicRecord(interactions, now, model.tags);
    session = {model, catalog, account, interactions, state, rating, now, attempted, ...topics, example: mode === "example"};
    render();
    busy(false, mode === "example"
      ? "Showing a real anonymised account as it stood at the start of our test period."
      : `Analysed ${num(account.history.length)} submissions.`);
    $("analysis").hidden = false;
    $("analysis").scrollIntoView({behavior: "smooth", block: "start"});
  } catch (error) {
    busy(false, "");
    if (!controller.signal.aborted) {
      $("error").textContent = error.message || "The analysis could not finish. Please retry.";
      $("error").hidden = false;
    } else {
      $("status").textContent = "Cancelled.";
    }
  } finally {
    active = null;
  }
}

function render() {
  const {account, interactions, rating, now, state} = session;
  const date = new Date(now * 1000).toLocaleDateString("en-GB", {day: "numeric", month: "long", year: "numeric"});
  $("analysis-title").textContent = session.example ? "Example account" : account.handle;
  $("analysis-meta").textContent = "As of " + date;
  const closed = interactions.filter(i => i.attempt < now - WINDOW_DAYS * DAY);
  const solved = closed.filter(i => i.solve !== null && i.solve - i.attempt <= WINDOW_DAYS * DAY).length;
  fillStats($("profile-stats"), [
    ["Rating", rating.is_rated ? num(rating.user_rating) : "Unrated"],
    ["Rated contests", num(Math.round(Math.expm1(rating.log_contests)))],
    ["Problems attempted", num(state.attempted)],
    ["Solved within 30 days", closed.length ? pct(solved / closed.length) : "–"],
  ]);
  ratingChart();
  activityChart();
  topicChart();
  setupFilters();
  recommend();
  $("problem-result").replaceChildren();
}

function fillStats(target, pairs) {
  target.replaceChildren(...pairs.map(([k, v]) => {
    const div = el("div");
    div.append(el("dt", k), el("dd", v));
    return div;
  }));
}

function ratingChart() {
  const points = session.account.ratings.filter(r => r.time < session.now).sort((a, b) => a.time - b.time);
  const box = $("rating-chart");
  if (points.length < 2) { box.replaceChildren(el("p", "Not enough rated contests to draw a trend.", "muted small")); return; }
  const W = 340, H = 190, L = 34, R = 8, T = 8, B = 22;
  const t0 = points[0].time, t1 = points.at(-1).time;
  const lo = Math.floor(Math.min(...points.map(p => p.new)) / 200) * 200;
  const hi = Math.ceil(Math.max(...points.map(p => p.new)) / 200) * 200;
  const x = t => L + (W - L - R) * (t - t0) / Math.max(1, t1 - t0);
  const y = v => T + (H - T - B) * (1 - (v - lo) / Math.max(1, hi - lo));
  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Rating after each rated contest"});
  for (let v = lo; v <= hi; v += (hi - lo) > 1200 ? 400 : 200) {
    root.append(svg("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid"}),
      svg("text", {x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "axis"}, v));
  }
  const years = new Set(points.map(p => new Date(p.time * 1000).getFullYear()));
  const step = Math.ceil(years.size / 6);
  [...years].filter((_, i) => i % step === 0).forEach(year => {
    const t = Date.UTC(year, 0, 1) / 1000;
    if (t >= t0 && t <= t1) root.append(svg("text", {x: x(t), y: H - 6, "text-anchor": "middle", class: "axis"}, year));
  });
  root.append(svg("polyline", {points: points.map(p => `${x(p.time)},${y(p.new)}`).join(" "), class: "line"}));
  box.replaceChildren(root);
}

function activityChart() {
  const {interactions, now} = session;
  const quarter = t => { const d = new Date(t * 1000); return d.getUTCFullYear() * 4 + Math.floor(d.getUTCMonth() / 3); };
  const MONTHS = ["Jan–Mar", "Apr–Jun", "Jul–Sep", "Oct–Dec"];
  const name = q => MONTHS[q % 4] + " " + Math.floor(q / 4);
  const last = quarter(now - 1), first = last - 7;
  const counts = new Map();
  for (const i of interactions) {
    const q = quarter(i.attempt);
    if (q >= first && q <= last && i.attempt < now) counts.set(q, (counts.get(q) ?? 0) + 1);
  }
  const W = 340, H = 200, L = 4, R = 4, T = 18, B = 34;
  const max = Math.max(1, ...counts.values());
  const bw = (W - L - R) / 8;
  const y = v => T + (H - T - B) * (1 - v / max);
  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Problems started in each of the last eight quarters"});
  root.append(svg("line", {x1: L, x2: W - R, y1: y(0), y2: y(0), class: "grid"}));
  for (let q = first; q <= last; q++) {
    const n = counts.get(q) ?? 0;
    const x0 = L + (q - first) * bw + 5, w = bw - 10;
    root.append(svg("rect", {x: x0, y: y(n), width: w, height: y(0) - y(n), class: "bar", rx: 2}),
      svg("text", {x: x0 + w / 2, y: y(n) - 5, "text-anchor": "middle", class: "bar-label"}, n),
      svg("text", {x: x0 + w / 2, y: H - 19, "text-anchor": "middle", class: "axis tight"}, MONTHS[q % 4]),
      svg("text", {x: x0 + w / 2, y: H - 6, "text-anchor": "middle", class: "axis tight"}, Math.floor(q / 4)));
  }
  $("activity-chart").replaceChildren(root);
  const total = [...counts.values()].reduce((s, n) => s + n, 0);
  const busiest = [...counts.entries()].sort((a, b) => b[1] - a[1])[0];
  const sum = (from, to) => { let s = 0; for (let q = from; q <= to; q++) s += counts.get(q) ?? 0; return s; };
  const recent = sum(last - 3, last), earlier = sum(first, first + 3);
  const trend = recent > earlier * 1.2 ? "more active in the past year than the year before"
    : recent < earlier * 0.8 ? "less active in the past year than the year before" : "about as active as the year before";
  $("activity-note").textContent = total
    ? `${num(total)} new problems in the last two years, most in ${name(busiest[0])} (${busiest[1]}). You were ${trend}. The last bar (${name(last)}) is not finished yet.`
    : "No problems started in the last two years.";
}

// Plain record per topic: how often the first submission to a problem was accepted.
// Solved-within-30-days is 95-99% in almost every topic, so it cannot tell topics apart;
// first-try success varies much more and is easy to read.
const MIN_TOPIC = 10;   // fewest problems in a topic before we compare it with the average
function topicRecord(interactions, now, tags) {
  const past = interactions.filter(i => i.attempt < now);
  const firstTry = i => i.solve === i.attempt;
  const overall = past.length ? past.filter(firstTry).length / past.length : 0;
  const rows = tags.map(tag => {
    const items = past.filter(i => i.tags.includes(tag));
    const first = items.filter(firstTry).length;
    return {tag, n: items.length, first, rate: items.length ? first / items.length : 0};
  }).filter(r => r.n > 0);
  const weak = rows.filter(r => r.n >= MIN_TOPIC && r.rate < overall - 0.05)
    .sort((a, b) => a.rate - b.rate).slice(0, 3).map(r => r.tag);
  return {topics: rows, overallFirstTry: overall, weak};
}

function tagRecord(tag) {
  return session.topics.find(r => r.tag === tag) ?? {tag, n: 0, first: 0, rate: 0};
}

function topicChart() {
  const {topics, overallFirstTry, weak} = session;
  const enough = topics.filter(r => r.n >= MIN_TOPIC);
  const average = pct(overallFirstTry);
  const strong = enough.filter(r => r.rate > overallFirstTry + 0.05)
    .sort((a, b) => b.rate - a.rate).slice(0, 3);
  const sentence = r => `Right first time on ${num(r.first)} of ${num(r.n)} problems (${pct(r.rate)}).`;
  const column = (title, items, emptyText, cls) => {
    const box = el("div", null, "topic-col " + cls);
    box.append(el("h4", title));
    if (!items.length) { box.append(el("p", emptyText, "muted small")); return box; }
    const ul = el("ul", null, "topic-items");
    for (const r of items) {
      const li = el("li");
      li.append(el("strong", r.tag), el("span", sentence(r)));
      ul.append(li);
    }
    box.append(ul);
    return box;
  };
  const intro = el("p", `On average, your first submission to a problem is accepted ${average} of the time. `
    + "These topics are clearly below or above that.", "topic-summary");
  const grid = el("div", null, "topic-cols");
  grid.append(
    column("Topics to practise", weak.map(tagRecord), "No topic is clearly below your average.", "weak"),
    column("Your strongest topics", strong, "No topic is clearly above your average.", "strong"));

  const details = el("details", null, "topic-all");
  details.append(el("summary", "See all topics"));
  const table = el("table", null, "compact");
  const head = el("tr");
  ["Topic", "Problems tried", "Right first time"].forEach(h => { const th = el("th", h); th.scope = "col"; head.append(th); });
  const thead = el("thead");
  thead.append(head);
  const body = el("tbody");
  for (const r of [...topics].sort((a, b) => b.n - a.n)) {
    const tr = el("tr");
    const th = el("th", r.tag);
    th.scope = "row";
    tr.append(th, el("td", num(r.n)), el("td", pct(r.rate)));
    body.append(tr);
  }
  table.append(thead, body);
  const wrap = el("div", null, "table-wrap flush");
  wrap.append(table);
  details.append(wrap);
  $("gaps").replaceChildren(intro, grid, details);
}

function setupFilters() {
  const r = Math.round(session.rating.user_rating / 100) * 100;
  $("min-rating").value = Math.max(800, r);
  $("max-rating").value = Math.min(3500, r + 300);
  const select = $("topic");
  select.replaceChildren();
  const add = (value, text) => { const o = el("option", text); o.value = value; select.append(o); };
  add("", "All topics");
  if (session.weak.length) add("@weak", "My weaker topics (" + session.weak.join(", ") + ")");
  for (const tag of session.model.tags) add(tag, tag);
}

function explanation(key, p, features) {
  const {catalog, rating, model, weak} = session;
  const [, problemRating, tags] = catalog[key];
  const parts = [];
  const diff = problemRating - rating.user_rating;
  parts.push(`Rated ${problemRating}, ${diff === 0 ? "at" : Math.abs(diff) + (diff > 0 ? " above" : " below")} your rating.`);
  const nearA = Math.round(Math.expm1(features.log_near_attempted));
  const nearS = nearA ? Math.round((features.near_solve_rate * (nearA + 2)) - 1) : 0;
  if (nearA) parts.push(`You solved ${nearS} of ${nearA} problems within 200 points of this rating.`);
  const weakHit = tags.filter(t => weak.includes(t));
  if (weakHit.length) {
    const r = tagRecord(weakHit[0]);
    parts.push(`Practises ${weakHit[0]}, one of your weaker topics (${pct(r.rate)} accepted on the first try, against ${pct(session.overallFirstTry)} overall).`);
  } else {
    const fresh = tags.filter(t => tagRecord(t).n < 3);
    if (fresh.length) parts.push(`Little history in ${fresh[0]} so far.`);
  }
  const bin = calibrationFor(model, p);
  const note = `Of test-period problems estimated ${pct(bin.low)}–${pct(bin.high)}, ${pct(bin.observed)} were solved (${num(bin.n)} cases).`;
  return {text: parts.join(" "), note};
}

function recommend() {
  const {model, catalog, state, rating, attempted, now, weak} = session;
  let low = Number($("min-rating").value), high = Number($("max-rating").value);
  if (!(low <= high)) [low, high] = [high, low];
  const topic = $("topic").value;
  const want = topic === "@weak" ? weak : topic ? [topic] : null;
  const scored = [];
  for (const [key, [name, problemRating, tags]] of Object.entries(catalog)) {
    if (attempted.has(key) || problemRating < low || problemRating > high) continue;
    if (want && !tags.some(t => want.includes(t))) continue;
    const f = problemFeatures(state, rating, problemRating, tags, 0, now, model.tags);
    const p = predictProbability(model, model.columns.map(c => f[c]));
    scored.push({key, name, problemRating, tags, p, f});
  }
  const stretch = $("order").value === "stretch";
  let shown = scored;
  if (stretch) {
    shown = scored.filter(s => s.p >= LIKELY);
    if (shown.length < 10) shown = scored;
    shown.sort((a, b) => b.problemRating - a.problemRating || b.p - a.p);
  } else {
    shown.sort((a, b) => b.p - a.p || b.problemRating - a.problemRating);
  }
  $("rec-note").textContent = !scored.length
    ? "No unattempted problems match these filters. Widen the rating range or choose another topic."
    : stretch
      ? `${num(scored.length)} unattempted problems match. Showing the hardest ones with an estimated chance of at least ${pct(LIKELY)} of being solved within 30 days.`
      : `${num(scored.length)} unattempted problems match. Showing the ten with the highest estimated chance of being solved within 30 days.`;
  $("recs").replaceChildren(...shown.slice(0, 10).map(item => {
    const li = el("li", null, "rec");
    const head = el("div", null, "rec-head");
    const title = el("div", null, "rec-title");
    title.append(problemLink(item.key, item.name), el("span", " " + item.key, "muted small"));
    head.append(title, el("span", pct(item.p), "rec-p"));
    const {text, note} = explanation(item.key, item.p, item.f);
    li.append(head, el("p", item.tags.join(" · ") || "No tags", "rec-tags"), el("p", text, "rec-why"), el("p", note, "rec-note"));
    return li;
  }));
}

function parseProblem(value) {
  const text = value.trim();
  const url = text.match(/codeforces\.com\/(?:contest|problemset\/problem|gym)\/(\d+)\/(?:problem\/)?([A-Za-z]\d?)/i);
  if (url) return url[1] + ":" + url[2].toUpperCase();
  const id = text.match(/^(\d+)\s*[:\s-]?\s*([A-Za-z]\d?)$/);
  if (id) return id[1] + ":" + id[2].toUpperCase();
  throw new Error("Enter a problem as 1850:E, 1850E or a Codeforces problem link.");
}

async function checkProblem() {
  const out = $("problem-result");
  try {
    const key = parseProblem($("problem").value);
    const {model, rating, state, now, account} = session;
    let entry = session.catalog[key];
    if (!entry && !session.example) {
      out.replaceChildren(el("p", "Checking the current Codeforces catalog…", "muted small"));
      entry = (await loadLiveCatalog(new AbortController().signal))[key];
    }
    if (!entry) throw new Error("That problem is not in the catalog, or it has no difficulty rating yet.");
    if (entry[1] === null) throw new Error("That problem has no difficulty rating yet, so it cannot be estimated.");
    const previous = account.history.filter(s => s.key === key && s.time < now);
    const f = problemFeatures(state, rating, entry[1], entry[2], 0, now, model.tags);
    const p = predictProbability(model, model.columns.map(c => f[c]));
    const {text, note} = explanation(key, p, f);
    const title = el("p", null, "rec-title");
    title.append(problemLink(key, entry[0]), el("span", " " + key, "muted small"));
    const nodes = [title, el("p", pct(p), "score"), el("p", "Estimated chance of solving within 30 days", "muted small"), el("p", text), el("p", note, "rec-note")];
    if (previous.length) {
      const solvedBefore = previous.some(s => s.verdict === "OK");
      nodes.splice(1, 0, el("p", solvedBefore ? "You have already solved this problem." : `You have already submitted to this problem ${previous.length} time(s); the estimate treats it as new.`, "form-error"));
    }
    out.replaceChildren(...nodes);
  } catch (error) {
    out.replaceChildren(el("p", error.message, "form-error"));
  }
}

$("handle-form").addEventListener("submit", e => { e.preventDefault(); analyse("live"); });
$("example").addEventListener("click", () => analyse("example"));
$("cancel").addEventListener("click", () => active?.abort());
$("filters").addEventListener("submit", e => { e.preventDefault(); if (session) recommend(); });
$("topic").addEventListener("change", () => session && recommend());
$("order").addEventListener("change", () => session && recommend());
$("problem-form").addEventListener("submit", e => { e.preventDefault(); if (session) checkProblem(); });

// ------------------------------------------------------------------ report

function barList(rows, {rate = true, min = 0.8, max = 1} = {}) {
  const list = el("div", null, "bar-list");
  const top = Math.max(...rows.map(r => r[1]));
  for (const [name, n, value] of rows) {
    const line = el("div", null, "bar-line");
    const track = el("div", null, "bar-track");
    const fill = el("div", null, "bar-fill");
    fill.style.width = (rate ? 100 * Math.max(0, (value - min) / (max - min)) : 100 * n / top) + "%";
    track.append(fill);
    line.append(el("span", name, "bar-name"), track, el("span", rate ? pct(value, 1) : fixed(value), "bar-value"),
      el("span", n != null ? num(n) + " rows" : "", "bar-count"));
    list.append(line);
  }
  return list;
}

const edaViews = {
  gap: ["Solved within 30 days by difficulty gap", "Problem rating relative to the user’s rating just before the attempt. Bars start at 80%."],
  tags: ["Solved within 30 days by topic", "The twelve most common tags. A problem can carry several. Bars start at 80%."],
  bands: ["Solved within 30 days by user rating", "Rating just before the attempt. Bars start at 80%."],
  modes: ["Solved within 30 days by participation mode", "How the first submission was made. Bars start at 80%."],
};

function drawEDA(results, key) {
  const [title, text] = edaViews[key];
  $("eda").replaceChildren(el("h3", title), el("p", text, "chart-copy"), barList(results.eda[key]));
  document.querySelectorAll("[data-eda]").forEach(b => {
    b.classList.toggle("active", b.dataset.eda === key);
    b.setAttribute("aria-pressed", String(b.dataset.eda === key));
  });
}

function modelTable(results) {
  const table = $("model-table");
  const caption = table.querySelector("caption");
  const cols = [["validation ROC-AUC", "Validation ROC-AUC", 3], ["ROC-AUC", "Test ROC-AUC", 3],
    ["Precision (not solved)", "Precision (not solved)", "%"], ["Recall (not solved)", "Recall (not solved)", "%"],
    ["F1 (not solved)", "F1 (not solved)", 3], ["Accuracy", "Accuracy", "%"], ["Cross-entropy", "Cross-entropy", 3]];
  const head = el("thead");
  const hr = el("tr");
  hr.append(el("th", "Model"), ...cols.map(([, label]) => el("th", label)));
  hr.querySelectorAll("th").forEach(th => th.scope = "col");
  head.append(hr);
  const body = el("tbody");
  const best = results.evaluation.best;
  for (const row of results.evaluation.table) {
    const tr = el("tr", null, row.model === best ? "selected" : "");
    const th = el("th", row.model + (row.model === best ? " (selected)" : ""));
    th.scope = "row";
    tr.append(th, ...cols.map(([key, , format]) => el("td", format === "%" ? pct(row[key], 1) : fixed(row[key], format))));
    body.append(tr);
  }
  table.replaceChildren(caption, head, body);
}

function confusionTable(results) {
  const ev = results.evaluation;
  const c = ev.confusion[ev.best];
  const table = el("table", null, "compact matrix");
  const head = el("tr");
  ["Actual", "Flagged unsolved", "Not flagged"].forEach(h => { const th = el("th", h); th.scope = "col"; head.append(th); });
  const thead = el("thead");
  thead.append(head);
  const body = el("tbody");
  for (const [label, a, b] of [["Unsolved", c.tp, c.fn], ["Solved", c.fp, c.tn]]) {
    const tr = el("tr");
    const th = el("th", label);
    th.scope = "row";
    tr.append(th, el("td", num(a)), el("td", num(b)));
    body.append(tr);
  }
  table.append(thead, body);
  const wrap = el("div", null, "table-wrap flush");
  wrap.append(table);
  $("confusion").replaceChildren(wrap);
}

function calibrationChart(bins) {
  const W = 340, H = 240, L = 36, R = 10, T = 8, B = 32, lo = 0.4;
  const x = v => L + (W - L - R) * (v - lo) / (1 - lo);
  const y = v => T + (H - T - B) * (1 - (v - lo) / (1 - lo));
  const root = svg("svg", {viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "Observed against predicted solve rate on the test period"});
  for (let v = lo; v <= 1.001; v += 0.2) {
    root.append(svg("line", {x1: L, x2: W - R, y1: y(v), y2: y(v), class: "grid"}),
      svg("text", {x: L - 6, y: y(v) + 4, "text-anchor": "end", class: "axis"}, pct(v)),
      svg("text", {x: x(v), y: H - 16, "text-anchor": "middle", class: "axis"}, pct(v)));
  }
  root.append(svg("text", {x: (L + W - R) / 2, y: H - 2, "text-anchor": "middle", class: "axis"}, "Predicted"),
    svg("line", {x1: x(lo), y1: y(lo), x2: x(1), y2: y(1), class: "diag"}),
    svg("polyline", {points: bins.map(b => `${x(b.predicted)},${y(b.observed)}`).join(" "), class: "line"}));
  for (const b of bins) {
    const c = svg("circle", {cx: x(b.predicted), cy: y(b.observed), r: 2.5 + Math.sqrt(b.n) / 40, class: "dot"});
    c.append(svg("title", {}, `Predicted ${pct(b.predicted, 1)}, observed ${pct(b.observed, 1)}, ${num(b.n)} interactions`));
    root.append(c);
  }
  $("calibration").replaceChildren(root);
}

function drawSegments(results) {
  const rows = results.evaluation.segments[$("segment-select").value];
  const table = el("table", null, "compact");
  const head = el("tr");
  ["Segment", "Rows", "Solved", "Model AUC", "Baseline AUC"].forEach(h => { const th = el("th", h); th.scope = "col"; head.append(th); });
  const thead = el("thead");
  thead.append(head);
  table.append(thead);
  const body = el("tbody");
  for (const r of rows) {
    const tr = el("tr");
    const th = el("th", r.segment.replaceAll("_", " ").toLowerCase().replace(/^\w/, c => c.toUpperCase()));
    th.scope = "row";
    tr.append(th, el("td", num(r.rows)), el("td", pct(r.solved_rate, 1)), el("td", fixed(r["ROC-AUC"])), el("td", fixed(r["baseline ROC-AUC"])));
    body.append(tr);
  }
  table.append(body);
  const wrap = el("div", null, "table-wrap flush");
  wrap.append(table);
  $("segments").replaceChildren(wrap);
}

function exampleList(target, items) {
  target.replaceChildren(...items.map(e => {
    const li = el("li");
    li.append(problemLink(e.key, e.problem), ` (${e.problem_rating}): user rated ${e.user_rating}, ${e.mode}, estimated ${pct(e.p)}.`);
    return li;
  }));
}

function bullets(target, items) {
  target.replaceChildren(...items.map(([lead, rest]) => {
    const li = el("li");
    li.append(el("strong", lead), rest);
    return li;
  }));
}

function report(results) {
  const ev = results.evaluation, data = results.data;
  const row = name => ev.table.find(r => r.model === name);
  const best = row(ev.best), heuristic = row("Baseline (rating gap)");
  const lr = row("Logistic Regression"), runner = row(ev.runner_up);
  const month = s => new Date(s + "T00:00:00Z").toLocaleDateString("en-GB", {month: "short", year: "numeric", timeZone: "UTC"});
  const parts0 = data.partitions;
  const facts = {"rows-train": num(parts0.train.rows), "rows-val": num(parts0.val.rows), "rows-test": num(parts0.test.rows), auc: fixed(best["ROC-AUC"]), heuristic: fixed(heuristic["ROC-AUC"]), rows: num(data.interactions),
    users: num(data.users_sampled), cut1: month(data.cuts[0]), cut2: month(data.cuts[1]), embargo: num(data.embargo_dropped),
    "rec-model": fixed(results.recommendations.model_auc), "rec-rating": fixed(results.recommendations.rating_auc),
    "gap-diff": (100 * results.skill_gaps.mean_resid_diff).toFixed(1)};
  document.querySelectorAll("[data-fact]").forEach(n => { n.textContent = facts[n.dataset.fact]; });
  $("solved-rate").textContent = pct(data.solved_rate);
  const parts = data.partitions;
  fillStats($("data-stats"), [["Users sampled", num(data.users_sampled)], ["Submissions", num(data.submissions)],
    ["Interactions", num(data.interactions)], ["Train / val / test rows", `${num(parts.train.rows)} / ${num(parts.val.rows)} / ${num(parts.test.rows)}`]]);

  modelTable(results);
  const t = best["threshold P(not solved)"];
  const lift = (best["Precision (not solved)"] / ev.positive_rate_not_solved).toFixed(1);
  bullets($("results-text"), [
    [`${ev.best} is best:`, ` test ROC-AUC ${fixed(best["ROC-AUC"])} (95% interval ${fixed(ev.auc_ci[0])}–${fixed(ev.auc_ci[1])}), against ${fixed(lr["ROC-AUC"])} for Logistic Regression and ${fixed(heuristic["ROC-AUC"])} for the baseline.`],
    [`${ev.runner_up} is effectively tied`, ` (${fixed(runner["ROC-AUC"])}). The interval for the difference includes zero, so we do not claim either is better.`],
    ["Warnings are useful but not certain:", ` with the threshold chosen on validation (flag when P(not solved) ≥ ${best["threshold P(not solved)"].toFixed(3)}), flagged problems really went unsolved ${pct(best["Precision (not solved)"], 1)} of the time, ${lift}× the ${pct(ev.positive_rate_not_solved, 1)} base rate, and ${pct(best["Recall (not solved)"], 1)} of all unsolved problems were caught. Many failures come from things public data cannot see, such as a small bug or losing interest.`],
    ["Accuracy misleads:", ` predicting “solved” for everything is already ${pct(1 - ev.positive_rate_not_solved, 1)} accurate.`],
    ["The baseline’s probabilities are off", ` (cross-entropy ${fixed(heuristic["Cross-entropy"])} vs ${fixed(best["Cross-entropy"])}): it ranks reasonably, but the Elo formula predicts winning a contest, not finishing a problem within a month. Ranking well and giving honest probabilities are different skills.`],
    ["What this means:", ` personal history adds real signal beyond the rating gap (${fixed(heuristic["ROC-AUC"])} → ${fixed(best["ROC-AUC"])}), and trees beat the linear model, so interactions between features matter.`],
  ]);
  confusionTable(results);
  calibrationChart(ev.calibration);
  drawSegments(results);
  $("segment-select").addEventListener("change", () => drawSegments(results));

  const err = ev.errors;
  $("miss-text").textContent = `${num(err.confident_miss_count)} of ${num(err.confident_count)} test interactions estimated at 98% or more were not solved. `
    + `${pct(err.confident_miss_modes.PRACTICE ?? 0)} of them were practice attempts, where nothing forces a user to finish what they open. The examples are experienced users on problems a little above their level.`;
  exampleList($("miss-examples"), err.confident_miss_examples);
  $("surprise-text").textContent = `${num(err.surprise_count)} of ${num(err.low_count)} interactions estimated below 60% were solved. The examples are problems well above the user’s rating; whatever made them succeed, such as a strong day or an editorial, is invisible in the data.`;
  exampleList($("surprise-examples"), err.surprise_examples);
  const top = ev.importance[0].drop;
  $("importance").replaceChildren(barList(ev.importance.map(r => [r.feature.replaceAll("_", " "), null, r.drop]), {rate: false}));
  $("importance").querySelectorAll(".bar-fill").forEach((f, i) => { f.style.width = (100 * ev.importance[i].drop / top) + "%"; });

  const g = results.skill_gaps, r = results.recommendations;
  bullets($("gap-text"), [
    ["Test:", ` at the start of the test period we flagged each user’s three weakest and three strongest topics (${g.users} users).`],
    ["Raw result:", ` they then solved ${pct(g.mean_weak_rate, 1)} of problems in weak topics and ${pct(g.mean_strong_rate, 1)} in strong ones.`],
    ["After adjusting for difficulty and skill:", ` the gap is only ${(100 * g.mean_resid_diff).toFixed(1)} points (95% interval ${(100 * g.resid_diff_ci[0]).toFixed(1)} to +${(100 * g.resid_diff_ci[1]).toFixed(1)}), and dropping topic features leaves ROC-AUC unchanged (${fixed(g.ablation_auc)}).`],
    ["Conclusion:", " model-based skill gaps add no reliable predictive power, so the tool shows each user’s plain record by topic. A negative result, reported as one."],
  ]);
  bullets($("rec-text"), [
    ["Test:", ` for ${r.users} users, the problems they actually attempted at or above their rating in the test period (${num(r.stretch_rows)} attempts).`],
    ["Question:", " which ordering better separates the ones they solved: the model’s estimates, or “closest rating first”?"],
    ["Result:", ` the model wins, with mean per-user AUC ${fixed(r.model_auc)} against ${fixed(r.rating_auc)} (95% interval for the difference +${fixed(r.auc_diff_ci[0])} to +${fixed(r.auc_diff_ci[1])}).`],
    ["Harder, yet still solved:", ` the model’s top five picks per user were about ${Math.round(r.model_top5_gap)} rating points above the user, against ${Math.round(r.rating_top5_gap)} for the rating rule, and were solved slightly more often (${pct(r.model_top5, 1)} vs ${pct(r.rating_top5, 1)}). That is exactly the “appropriate difficulty” the proposal aimed for.`],
    ["Caveat:", " an offline check on problems users chose themselves, not a controlled experiment."],
  ]);
  fillStats($("rec-stats"), [["Model top-5 solved", pct(r.model_top5, 1)], ["Rating rule top-5 solved", pct(r.rating_top5, 1)],
    ["Model top-5 difficulty", "+" + Math.round(r.model_top5_gap)], ["Rating rule top-5 difficulty", "+" + Math.round(r.rating_top5_gap)]]);

  $("conclusion-text").textContent = `Yes, within limits. A ${ev.best.toLowerCase()} model trained on a user’s rating, contest record and history ranks which problems they will solve `
    + `much better than the rating gap alone (test ROC-AUC ${fixed(best["ROC-AUC"])} against ${fixed(heuristic["ROC-AUC"])}) and gives well-calibrated probabilities on a later year. `
    + `The strongest signal is personal: how often the user has solved problems at a similar level. Model-based recommendations pick harder problems that users still finish, `
    + `beating simple rating-based selection. Topic-wise skill gaps, the other half of our proposal, did not hold up as predictors once difficulty and general skill were accounted for, `
    + `so they are best read as a description of a user’s record.`;
}

document.querySelectorAll("[data-eda]").forEach(b => b.addEventListener("click", () => resultsPromise.then(r => drawEDA(r, b.dataset.eda))));
const resultsPromise = getJSON("results.json");
resultsPromise.then(results => { report(results); drawEDA(results, "gap"); })
  .catch(error => { $("eda").textContent = error.message; });

// Highlight the section being read in the header and the contents sidebar.
const sectionLinks = [...document.querySelectorAll('.toc a, .site-header nav a')];
const observer = new IntersectionObserver(entries => {
  for (const entry of entries) {
    if (!entry.isIntersecting) continue;
    const id = entry.target.id;
    sectionLinks.forEach(a => a.classList.toggle("active", a.getAttribute("href") === "#" + id));
  }
}, {rootMargin: "-20% 0px -70% 0px"});
document.querySelectorAll("main section[id]").forEach(s => observer.observe(s));
