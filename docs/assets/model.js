// Evaluates the exported scikit-learn HistGradientBoostingClassifier (docs/data/model.json).

export function predictMargin(model, x) {
  let margin = model.baseline;
  for (const [feature, threshold, left, right, leaf, value, missingLeft] of model.trees) {
    let node = 0;
    while (!leaf[node]) {
      const v = x[feature[node]];
      node = Number.isNaN(v) ? (missingLeft[node] ? left[node] : right[node])
        : v <= threshold[node] ? left[node] : right[node];
    }
    margin += value[node];
  }
  return margin;
}

export const predictProbability = (model, x) => 1 / (1 + Math.exp(-predictMargin(model, x)));

/** The held-out calibration bin an estimate falls in, to show how similar estimates turned out. */
export function calibrationFor(model, p) {
  return model.calibration.find(bin => p >= bin.low && p < bin.high) ?? model.calibration.at(-1);
}
