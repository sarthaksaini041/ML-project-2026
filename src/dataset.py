"""Build the user-problem interaction table and the chronological split."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .features import (DAY, NUMERIC_FEATURES, WINDOW_DAYS, build_interactions, problem_features,
                       rating_features, replay, CONTEST_MODES)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"


def parse_tags(value) -> tuple[str, ...]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ()
    return tuple(sorted({t.strip().lower() for t in json.loads(value) if t.strip()}))


def load_tables(data: Path = DATA) -> dict[str, pd.DataFrame]:
    return {name: pd.read_parquet(data / f"{name}.parquet")
            for name in ("submissions", "rating_history", "users", "problems")}


def tag_vocabulary(problems: pd.DataFrame, min_share: float = 0.02) -> list[str]:
    """Tags on at least `min_share` of catalog problems. Uses no outcome data."""
    counts = Counter(tag for value in problems["tags"] for tag in parse_tags(value))
    return sorted(tag for tag, n in counts.items()
                  if n >= min_share * len(problems) and not tag.startswith("*"))


def submissions_by_user(subs: pd.DataFrame) -> dict[str, list[dict]]:
    subs = subs.loc[subs["problem_contest_id"].notna()].drop_duplicates("submission_id")
    keys = subs["problem_contest_id"].astype(int).astype(str) + ":" + subs["problem_index"]
    frame = pd.DataFrame({
        "handle": subs["handle"].str.lower(), "key": keys, "time": subs["creation_time_seconds"],
        "id": subs["submission_id"], "verdict": subs["verdict"],
        "rating": subs["problem_rating"].astype(object).where(subs["problem_rating"].notna(), None),
        "tags": subs["problem_tags"].map(parse_tags), "mode": subs["participant_type"],
        "solo": subs["is_solo"],
    })
    return {h: g.drop(columns="handle").to_dict("records") for h, g in frame.groupby("handle")}


def ratings_by_user(history: pd.DataFrame) -> dict[str, list[dict]]:
    frame = pd.DataFrame({"handle": history["handle"].str.lower(),
                          "time": history["rating_update_time_seconds"],
                          "old": history["old_rating"], "new": history["new_rating"]})
    return {h: g.drop(columns="handle").to_dict("records") for h, g in frame.groupby("handle")}


def build_table(tables: dict[str, pd.DataFrame], tag_vocab: list[str]) -> tuple[pd.DataFrame, int]:
    """One row per (user, rated problem) interaction whose 30-day window is fully observed."""
    subs = tables["submissions"]
    data_end = int(subs["creation_time_seconds"].max())
    cutoff = data_end - WINDOW_DAYS * DAY
    complete = set(tables["users"].loc[tables["users"]["history_complete"], "handle"].str.lower())
    by_user = submissions_by_user(subs)
    ratings = ratings_by_user(tables["rating_history"])
    rows = []
    for handle, history in by_user.items():
        if handle not in complete:
            continue
        user_ratings = ratings.get(handle, [])
        for item, state in replay(build_interactions(history)):
            if item.rating is None or item.attempt > cutoff:
                continue
            feats = problem_features(state, rating_features(user_ratings, item.attempt), item.rating,
                                     item.tags, int(item.mode in CONTEST_MODES), item.attempt, tag_vocab)
            solved = item.solve is not None and item.solve - item.attempt <= WINDOW_DAYS * DAY
            rows.append({"handle": handle, "problem": item.key, "attempt_time": item.attempt,
                         "tags": item.tags, "mode": item.mode, **feats, "solved": int(solved)})
    table = pd.DataFrame(rows).sort_values(["attempt_time", "handle"]).reset_index(drop=True)
    return table, data_end


def feature_columns(tag_vocab: list[str]) -> list[str]:
    return NUMERIC_FEATURES + ["tag_" + t for t in tag_vocab]


def chronological_split(table: pd.DataFrame, val_share=0.15, test_share=0.15):
    """Split by attempt time into train / validation / test.

    Cut points are row quantiles of attempt time. A row is kept in an earlier partition only if
    its whole 30-day outcome window ends before the next partition starts (an embargo), so no
    training label depends on events inside the validation or test period.
    """
    times = table["attempt_time"]
    t_val = int(times.quantile(1 - val_share - test_share))
    t_test = int(times.quantile(1 - test_share))
    window = WINDOW_DAYS * DAY
    train = table.loc[times + window < t_val]
    val = table.loc[(times >= t_val) & (times + window < t_test)]
    test = table.loc[times >= t_test]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True), (t_val, t_test)
