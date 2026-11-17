// Browser port of src/features.py and src/skills.py. Keep the two in step:
// tests/check_parity.mjs compares this file's output with Python on fixed fixtures.

export const DAY = 86400;
export const WINDOW_DAYS = 30;
const DEFAULT_RATING = 1400;
const NEAR = 200;
const CONTEST_MODES = new Set(["CONTESTANT", "OUT_OF_COMPETITION"]);

export const eloProbability = (user, problem) => 1 / (1 + 10 ** ((problem - user) / 400));
const smoothed = (solved, attempted) => (solved + 1) / (attempted + 2);
const bucket = rating => Math.floor(rating / 100);

/** Normalise one submission from the Codeforces user.status API. */
export function normalizeSubmission(s) {
  const p = s.problem || {};
  const contest = p.contestId ?? null;
  return {
    key: contest == null || !p.index ? null : contest + ":" + p.index,
    id: s.id, time: s.creationTimeSeconds, verdict: s.verdict ?? null,
    rating: Number.isFinite(p.rating) ? p.rating : null,
    tags: [...new Set((p.tags || []).map(t => t.trim().toLowerCase()).filter(Boolean))].sort(),
    mode: s.author?.participantType ?? "PRACTICE",
    solo: (s.author?.members?.length ?? 1) <= 1,
    name: p.name ?? "",
  };
}

/** Normalise user.rating API rows. */
export const normalizeRatings = rows => rows.map(r => ({time: r.ratingUpdateTimeSeconds, old: r.oldRating, new: r.newRating}));

/** One interaction per problem: first solo attempt and first accepted submission. */
export function buildInteractions(submissions) {
  const items = new Map();
  const sorted = submissions.filter(s => s.solo && s.time > 0 && s.key != null)
    .sort((a, b) => a.time - b.time || a.id - b.id);
  for (const s of sorted) {
    let item = items.get(s.key);
    if (!item) {
      item = {key: s.key, rating: s.rating, tags: [...new Set(s.tags)].sort(), mode: s.mode,
        attempt: s.time, solve: null, name: s.name};
      items.set(s.key, item);
    }
    if (s.verdict === "OK" && item.solve === null) item.solve = s.time;
  }
  return [...items.values()];
}

function newState() {
  return {attempted: 0, solved: 0, firstTime: null, recent: [], tags: new Map(), buckets: new Map()};
}
function bump(map, key, slot) {
  if (!map.has(key)) map.set(key, [0, 0]);
  map.get(key)[slot] += 1;
}
function addAttempt(state, item) {
  state.attempted += 1;
  state.recent.push(item.attempt);
  if (state.firstTime === null) state.firstTime = item.attempt;
  for (const tag of item.tags) bump(state.tags, tag, 0);
  if (item.rating !== null) bump(state.buckets, bucket(item.rating), 0);
}
function addSolve(state, item) {
  state.solved += 1;
  for (const tag of item.tags) bump(state.tags, tag, 1);
  if (item.rating !== null) bump(state.buckets, bucket(item.rating), 1);
}

/** Replay events in time order, calling onAttempt(item, state) before each attempt updates the state. */
export function replay(interactions, until = Infinity, onAttempt = () => {}) {
  const events = [];
  interactions.forEach((item, n) => {
    events.push([item.attempt, 0, n]);
    if (item.solve !== null) events.push([item.solve, 1, n]);
  });
  events.sort((a, b) => a[0] - b[0] || a[1] - b[1] || a[2] - b[2]);
  const state = newState();
  let pos = 0;
  while (pos < events.length) {
    const now = events[pos][0];
    if (now >= until) break;
    const group = [];
    while (pos < events.length && events[pos][0] === now) group.push(events[pos++]);
    for (const [, kind, n] of group) if (kind === 0) onAttempt(interactions[n], state);
    for (const [, kind, n] of group) (kind === 0 ? addAttempt : addSolve)(state, interactions[n]);
  }
  return state;
}

export function ratingFeatures(ratings, now) {
  const past = ratings.filter(r => r.time < now).sort((a, b) => a.time - b.time);
  if (!past.length) {
    return {user_rating: DEFAULT_RATING, is_rated: 0, max_rating: DEFAULT_RATING, log_contests: 0,
      last_delta: 0, mean_delta_5: 0, log_days_since_contest: Math.log1p(3650)};
  }
  const deltas = past.map(r => r.new - r.old);
  const last5 = deltas.slice(-5);
  return {
    user_rating: past.at(-1).new, is_rated: 1, max_rating: Math.max(...past.map(r => r.new)),
    log_contests: Math.log1p(past.length), last_delta: deltas.at(-1),
    mean_delta_5: last5.reduce((a, b) => a + b, 0) / last5.length,
    log_days_since_contest: Math.log1p(Math.max(0, now - past.at(-1).time) / DAY),
  };
}

export function problemFeatures(state, rating, problemRating, tags, contestMode, now, vocab) {
  const overall = smoothed(state.solved, state.attempted);
  const low = bucket(problemRating - NEAR), high = bucket(problemRating + NEAR);
  let nearA = 0, nearS = 0;
  for (let b = low; b <= high; b++) {
    const c = state.buckets.get(b);
    if (c) { nearA += c[0]; nearS += c[1]; }
  }
  const rates = [], counts = [];
  for (const tag of tags) {
    const [a, s] = state.tags.get(tag) ?? [0, 0];
    rates.push(smoothed(s, a));
    counts.push(a);
  }
  const tagMean = rates.length ? rates.reduce((a, b) => a + b, 0) / rates.length : overall;
  const recent = state.recent.filter(t => t >= now - 30 * DAY).length;
  const row = {
    problem_rating: problemRating, ...rating,
    rating_gap: problemRating - rating.user_rating,
    elo_prob: eloProbability(rating.user_rating, problemRating),
    log_prior_attempted: Math.log1p(state.attempted),
    prior_solve_rate: overall,
    log_recent_attempted: Math.log1p(recent),
    log_days_active: state.firstTime !== null ? Math.log1p(Math.max(0, now - state.firstTime) / DAY) : 0,
    near_solve_rate: smoothed(nearS, nearA),
    log_near_attempted: Math.log1p(nearA),
    tag_skill_mean: tagMean,
    tag_skill_min: rates.length ? Math.min(...rates) : overall,
    tag_skill_rel: tagMean - overall,
    log_tag_attempted_min: counts.length ? Math.log1p(Math.min(...counts)) : 0,
    tag_count: tags.length,
    contest_mode: contestMode,
  };
  for (const tag of vocab) row["tag_" + tag] = tags.includes(tag) ? 1 : 0;
  return row;
}

export const isContestMode = mode => CONTEST_MODES.has(mode) ? 1 : 0;

/** Feature vectors for unattempted catalog problems in [rating+low, rating+high]. */
export function candidateMatrix(state, rating, catalog, attempted, now, vocab, columns, low = -100, high = 300) {
  const keys = [], rows = [];
  const user = rating.user_rating;
  for (const [key, [, problemRating, tags]] of Object.entries(catalog)) {
    if (attempted.has(key) || problemRating === null || problemRating < user + low || problemRating > user + high) continue;
    const f = problemFeatures(state, rating, problemRating, tags, 0, now, vocab);
    keys.push(key);
    rows.push(columns.map(c => f[c]));
  }
  return {keys, rows};
}

/** Copy of a feature row with topic-skill features replaced by the overall record. */
export function neutraliseTopics(row, columns) {
  const out = row.slice();
  const overall = row[columns.indexOf("prior_solve_rate")];
  out[columns.indexOf("tag_skill_mean")] = overall;
  out[columns.indexOf("tag_skill_min")] = overall;
  out[columns.indexOf("tag_skill_rel")] = 0;
  return out;
}

export function skillGaps(predict, keys, rows, catalog, columns, vocab, minCandidates = 5) {
  const p = rows.map(predict);
  const p0 = rows.map(r => predict(neutraliseTopics(r, columns)));
  const gaps = {};
  for (const tag of vocab) {
    let sum = 0, n = 0;
    keys.forEach((key, i) => {
      if (catalog[key][2].includes(tag)) { sum += p[i] - p0[i]; n++; }
    });
    if (n >= minCandidates) gaps[tag] = sum / n;
  }
  const values = Object.values(gaps);
  if (values.length) {
    const centre = values.reduce((a, b) => a + b, 0) / values.length;
    for (const tag of Object.keys(gaps)) gaps[tag] -= centre;
  }
  return {gaps, probabilities: p};
}
