"""Topic-wise skill gaps and problem recommendations built on the trained model.

Skill gap (per user, per tag)
    For candidate problems near the user's level that carry the tag, compare the model's solve
    probability using the user's real topic record with the probability after replacing that
    topic record by the user's overall record, averaged over those problems and then centred on
    the user's mean across topics. Positive means a relative strength, negative a relative
    weakness. Topics everyone finds hard do not show up as personal gaps.

Recommendations
    Unattempted rated problems in a difficulty range, ranked by solve probability among
    problems at or above the user's rating ("stretch" problems), optionally limited to tags.
`docs/assets/features.js` mirrors this logic in the browser.
"""
from __future__ import annotations

import numpy as np

from .features import problem_features

TOPIC_SKILL = ("tag_skill_mean", "tag_skill_min", "tag_skill_rel")


def neutralise_topics(X: np.ndarray, columns: list[str]) -> np.ndarray:
    """Copy of X with the topic-skill features set to the user's overall record."""
    X = X.copy()
    overall = X[:, columns.index("prior_solve_rate")]
    X[:, columns.index("tag_skill_mean")] = overall
    X[:, columns.index("tag_skill_min")] = overall
    X[:, columns.index("tag_skill_rel")] = 0.0
    return X


def candidate_matrix(state, rating_feats, catalog, attempted, now, tag_vocab, columns,
                     low=-100, high=300):
    """Feature matrix for unattempted catalog problems within [rating+low, rating+high]."""
    user = rating_feats["user_rating"]
    keys, rows = [], []
    for key, (name, rating, tags) in catalog.items():
        if key in attempted or rating is None or not (user + low <= rating <= user + high):
            continue
        feats = problem_features(state, rating_feats, rating, tuple(tags), 0, now, tag_vocab)
        keys.append(key)
        rows.append([feats[c] for c in columns])
    return keys, np.array(rows, dtype=float).reshape(len(rows), len(columns))


def skill_gaps(model, keys, X, catalog, columns, tag_vocab, min_candidates=5):
    """Mean (actual - neutral) solve probability per tag over candidate problems."""
    if not len(keys):
        return {}
    p = model.predict_proba(X)[:, 1]
    p0 = model.predict_proba(neutralise_topics(X, columns))[:, 1]
    gaps = {}
    for tag in tag_vocab:
        mask = np.array([tag in catalog[k][2] for k in keys])
        if mask.sum() >= min_candidates:
            gaps[tag] = float((p[mask] - p0[mask]).mean())
    # Small per-topic samples are shrunk toward 50%, which pushes every raw gap below zero.
    # Centre on the user's average topic so the gaps show relative strengths and weaknesses.
    if gaps:
        centre = sum(gaps.values()) / len(gaps)
        gaps = {tag: gap - centre for tag, gap in gaps.items()}
    return gaps
