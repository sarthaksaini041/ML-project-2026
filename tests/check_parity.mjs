// Checks the browser feature engine and model against Python reference values.
//   node tests/check_parity.mjs        (run `python -m src.pipeline` first to refresh fixtures)
import assert from "node:assert/strict";
import {readFileSync} from "node:fs";
import {buildInteractions, candidateMatrix, isContestMode, problemFeatures, ratingFeatures, replay, skillGaps}
  from "../docs/assets/features.js";
import {predictProbability} from "../docs/assets/model.js";

const load = path => JSON.parse(readFileSync(new URL(path, import.meta.url)));
const model = load("../docs/data/model.json");
const catalog = load("../docs/data/catalog.json");
const example = load("../docs/data/example.json");
const modelFixtures = load("fixtures/model_parity.json");
const featureFixtures = load("fixtures/feature_parity.json");
const cols = model.columns;

let worst = 0;
for (const {x, p} of modelFixtures) worst = Math.max(worst, Math.abs(predictProbability(model, x) - p));
assert.ok(worst < 1e-9, "model mismatch " + worst);
console.log(`model: ${modelFixtures.length} probabilities match (max diff ${worst.toExponential(1)})`);

const close = (a, b, label) => a.forEach((v, i) => assert.ok(Math.abs(v - b[i]) < 1e-9, `${label}: ${cols[i]} ${v} vs ${b[i]}`));
const interactions = buildInteractions(example.history);
const replayed = [];
replay(interactions, Infinity, (item, state) => {
  if (item.rating === null) return;
  const f = problemFeatures(state, ratingFeatures(example.ratings, item.attempt), item.rating, item.tags,
    isContestMode(item.mode), item.attempt, model.tags);
  replayed.push({key: item.key, x: cols.map(c => f[c])});
});
const tail = replayed.slice(-featureFixtures.interactions.length);
featureFixtures.interactions.forEach((ref, i) => {
  assert.equal(tail[i].key, ref.key);
  close(tail[i].x, ref.x, "history " + ref.key);
});
console.log(`features: ${tail.length} historical interactions match`);

const state = replay(interactions, example.now);
const rating = ratingFeatures(example.ratings, example.now);
const {keys, rows} = candidateMatrix(state, rating, catalog, new Set(interactions.map(i => i.key)), example.now, model.tags, cols);
assert.equal(keys.length, featureFixtures.candidate_count, "candidate count");
featureFixtures.candidates.forEach((ref, i) => {
  const j = keys.indexOf(ref.key);
  close(rows[j], ref.x, "candidate " + ref.key);
  assert.ok(Math.abs(predictProbability(model, rows[j]) - featureFixtures.probabilities[i]) < 1e-9);
});
const {gaps} = skillGaps(x => predictProbability(model, x), keys, rows, catalog, cols, model.tags);
for (const [tag, gap] of Object.entries(featureFixtures.gaps)) assert.ok(Math.abs(gaps[tag] - gap) < 1e-9, "gap " + tag);
console.log(`recommendations: ${keys.length} candidates and ${Object.keys(gaps).length} skill gaps match`);
