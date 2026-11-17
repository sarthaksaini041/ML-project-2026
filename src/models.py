"""Baselines, candidate models, validation tuning and test evaluation."""
from __future__ import annotations

import time

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, log_loss, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 335


class RatingGapHeuristic(BaseEstimator, ClassifierMixin):
    """The baseline from the proposal: Elo expected score from the user-problem rating difference.

    Nothing is learned. The probability is 1 / (1 + 10^((problem - user) / 400)), read from the
    `elo_prob` column, so the heuristic can rank but is not calibrated to this target.
    """

    def __init__(self, column: int):
        self.column = column

    def fit(self, X, y):
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        p = np.asarray(X)[:, self.column]
        return np.column_stack([1 - p, p])


def candidates(feature_names: list[str]) -> dict[str, list]:
    """Hyperparameter grids. Each entry is a list of unfitted estimators to try on validation."""
    elo = feature_names.index("elo_prob")
    return {
        "Baseline (rating gap)": [RatingGapHeuristic(elo)],
        "Logistic Regression": [
            make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=2000))
            for c in (0.01, 0.1, 1.0, 10.0)],
        "Random Forest": [
            RandomForestClassifier(n_estimators=200, max_depth=d, min_samples_leaf=leaf,
                                   max_features="sqrt", n_jobs=-1, random_state=SEED)
            for d, leaf in ((10, 50), (16, 50), (None, 100))],
        "Gradient Boosting": [
            HistGradientBoostingClassifier(learning_rate=lr, max_leaf_nodes=leaves, max_iter=400,
                                           l2_regularization=1.0, min_samples_leaf=100,
                                           early_stopping=False, random_state=SEED)
            for lr, leaves in ((0.05, 15), (0.05, 31), (0.1, 15), (0.03, 63))],
    }


def scores(y, p) -> dict[str, float]:
    """Threshold-free metrics: ranking (ROC-AUC) and probability quality (binary cross-entropy)."""
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"ROC-AUC": roc_auc_score(y, p), "Cross-entropy": log_loss(y, p)}


def choose_threshold(y_val, p_val) -> float:
    """Cut-off on P(not solved) that maximises validation F1 for the rare "not solved" class.

    Chosen on validation only, then frozen before the test period is scored.
    """
    risk = 1 - np.asarray(p_val)
    grid = np.unique(np.quantile(risk, np.linspace(0.5, 0.999, 300)))
    f1 = [f1_score(1 - y_val, risk >= t, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(f1))])


def threshold_metrics(y, p, threshold) -> dict[str, float]:
    """Precision, recall and F1 for flagging "not solved" when P(not solved) >= threshold."""
    flag = (1 - np.asarray(p)) >= threshold
    unsolved = 1 - np.asarray(y)
    tn, fp, fn, tp = confusion_matrix(unsolved, flag, labels=[0, 1]).ravel()
    return {"Precision (not solved)": precision_score(unsolved, flag, zero_division=0),
            "Recall (not solved)": recall_score(unsolved, flag, zero_division=0),
            "F1 (not solved)": f1_score(unsolved, flag, zero_division=0),
            "Accuracy": float((tp + tn) / len(flag)),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}}


def tune(X_train, y_train, X_val, y_val, feature_names):
    """Fit every candidate on training rows and keep the best per family by validation ROC-AUC."""
    chosen, log = {}, []
    for family, options in candidates(feature_names).items():
        best = None
        for model in options:
            start = time.perf_counter()
            model.fit(X_train, y_train)
            auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            params = {k: v for k, v in model.get_params().items()
                      if k in ("C", "logisticregression__C", "max_depth", "min_samples_leaf",
                               "learning_rate", "max_leaf_nodes")}
            log.append({"family": family, "params": str(params), "validation ROC-AUC": auc,
                        "seconds": time.perf_counter() - start})
            if best is None or auc > best[0]:
                best = (auc, model, choose_threshold(y_val, model.predict_proba(X_val)[:, 1]))
        chosen[family] = best
    return chosen, log


def user_bootstrap(y, p, users, metric=roc_auc_score, draws=300, seed=SEED):
    """95% interval for a metric, resampling whole users so their rows move together."""
    codes = np.unique(users, return_inverse=True)[1]
    n = codes.max() + 1
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(draws):
        weights = rng.multinomial(n, np.full(n, 1 / n))[codes]
        if weights.sum() == 0:
            continue
        values.append(metric(y, p, sample_weight=weights))
    return np.percentile(values, [2.5, 97.5])
