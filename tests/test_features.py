"""Unit tests for the point-in-time feature engine and the chronological split.

    python -m pytest tests
"""
import pandas as pd

from src.dataset import chronological_split
from src.features import (DAY, WINDOW_DAYS, build_interactions, elo_probability, problem_features,
                          rating_features, replay, state_at)

VOCAB = ["dp", "greedy"]


def sub(key, time, verdict="WRONG_ANSWER", rating=1500, tags=("dp",), sid=None, solo=True, mode="PRACTICE"):
    return {"key": key, "time": time, "id": sid if sid is not None else time, "verdict": verdict,
            "rating": rating, "tags": tags, "mode": mode, "solo": solo}


def snapshots(subs):
    out = {}
    for item, state in replay(build_interactions(subs)):
        out[item.key] = problem_features(state, rating_features([], item.attempt), item.rating, item.tags,
                                         0, item.attempt, VOCAB)
    return out


def test_interaction_uses_first_attempt_and_first_accept():
    items = {i.key: i for i in build_interactions([
        sub("1:A", 100), sub("1:A", 200, "OK"), sub("1:A", 300, "OK"), sub("1:B", 150, "OK")])}
    assert items["1:A"].attempt == 100 and items["1:A"].solve == 200
    assert items["1:B"].attempt == 150 and items["1:B"].solve == 150


def test_team_and_unknown_submissions_are_ignored():
    items = build_interactions([sub("1:A", 100, solo=False), sub(None, 120), sub("1:B", 0)])
    assert items == []


def test_features_ignore_the_future():
    past = [sub("1:A", 100, "OK"), sub("1:B", 200)]
    future = past + [sub("1:B", 10_000, "OK"), sub("1:C", 20_000, "OK"), sub("1:D", 30_000)]
    assert snapshots(past)["1:B"] == snapshots(future)["1:B"]


def test_solve_counts_only_after_it_happens():
    feats = snapshots([sub("1:A", 100), sub("1:B", 200), sub("1:A", 300, "OK"), sub("1:C", 400)])
    assert feats["1:B"]["prior_solve_rate"] == (0 + 1) / (1 + 2)      # 1:A attempted, not yet solved
    assert feats["1:C"]["prior_solve_rate"] == (1 + 1) / (2 + 2)      # 1:A solved at 300 < 400


def test_same_second_events_do_not_see_each_other():
    feats = snapshots([sub("1:A", 100, "OK"), sub("1:B", 100)])
    assert feats["1:A"]["log_prior_attempted"] == 0 and feats["1:B"]["log_prior_attempted"] == 0


def test_state_at_matches_replay():
    subs = [sub("1:A", 100, "OK"), sub("1:B", 200), sub("1:C", 300, "OK")]
    state = state_at(build_interactions(subs), 250)
    assert (state.attempted, state.solved) == (2, 1)


def test_rating_features_use_only_earlier_contests():
    ratings = [{"time": 100, "old": 0, "new": 1200}, {"time": 500, "old": 1200, "new": 1500}]
    assert rating_features(ratings, 300)["user_rating"] == 1200
    assert rating_features(ratings, 500)["user_rating"] == 1200       # update at the same second is excluded
    assert rating_features([], 300)["is_rated"] == 0


def test_elo_probability():
    assert elo_probability(1500, 1500) == 0.5
    assert abs(elo_probability(1500, 1900) - 1 / 11) < 1e-12


def test_split_is_chronological_with_embargo():
    times = pd.Series(range(0, 400 * DAY, DAY // 2))
    table = pd.DataFrame({"attempt_time": times, "handle": "u", "solved": 1})
    train, val, test, (t_val, t_test) = chronological_split(table)
    window = WINDOW_DAYS * DAY
    assert (train["attempt_time"] + window < t_val).all()
    assert ((val["attempt_time"] >= t_val) & (val["attempt_time"] + window < t_test)).all()
    assert (test["attempt_time"] >= t_test).all()
