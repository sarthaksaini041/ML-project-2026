"""Point-in-time features for one Codeforces user.

Everything here is computed from events strictly earlier than the prediction time, so the
same code builds the training table and answers live queries without leaking the future.
`docs/assets/features.js` is a line-by-line port; `tests/` checks the two agree.

Vocabulary
  interaction  one (user, problem) pair, created at the user's first solo submission to it
  attempt      the first submission time of an interaction (the prediction point)
  solve        the first accepted (OK) submission time of an interaction, if any
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

DAY = 86_400
WINDOW_DAYS = 30          # target: solved within this many days of the first attempt
DEFAULT_RATING = 1400     # Codeforces' internal starting rating, used for unrated users
NEAR = 200                # "similar difficulty" = within +-200 rating points
CONTEST_MODES = {"CONTESTANT", "OUT_OF_COMPETITION"}

NUMERIC_FEATURES = [
    "problem_rating", "user_rating", "is_rated", "rating_gap", "elo_prob", "max_rating",
    "log_contests", "last_delta", "mean_delta_5", "log_days_since_contest",
    "log_prior_attempted", "prior_solve_rate", "log_recent_attempted", "log_days_active",
    "near_solve_rate", "log_near_attempted", "tag_skill_mean", "tag_skill_min",
    "tag_skill_rel", "log_tag_attempted_min", "tag_count", "contest_mode",
]


def elo_probability(user_rating: float, problem_rating: float) -> float:
    """Codeforces/Elo expected score of a user against a problem of a given rating."""
    return 1.0 / (1.0 + 10.0 ** ((problem_rating - user_rating) / 400.0))


def smoothed(solved: int, attempted: int) -> float:
    """Beta(1, 1)-smoothed solve rate, so a 1/1 record is 0.67, not 1.0."""
    return (solved + 1.0) / (attempted + 2.0)


@dataclass
class Interaction:
    key: str
    rating: float | None
    tags: tuple[str, ...]
    mode: str
    attempt: int
    solve: int | None


@dataclass
class UserState:
    """Running counts of a user's past interactions, updated in time order."""
    attempted: int = 0
    solved: int = 0
    first_time: int | None = None
    recent: list[int] = field(default_factory=list)          # attempt times, for 30-day count
    tag_counts: dict[str, list[int]] = field(default_factory=dict)      # tag -> [attempted, solved]
    bucket_counts: dict[int, list[int]] = field(default_factory=dict)   # rating//100 -> [a, s]

    def add_attempt(self, item: Interaction) -> None:
        self.attempted += 1
        self.recent.append(item.attempt)
        if self.first_time is None:
            self.first_time = item.attempt
        for tag in item.tags:
            self.tag_counts.setdefault(tag, [0, 0])[0] += 1
        if item.rating is not None:
            self.bucket_counts.setdefault(int(item.rating) // 100, [0, 0])[0] += 1

    def add_solve(self, item: Interaction) -> None:
        self.solved += 1
        for tag in item.tags:
            self.tag_counts.setdefault(tag, [0, 0])[1] += 1
        if item.rating is not None:
            self.bucket_counts.setdefault(int(item.rating) // 100, [0, 0])[1] += 1


def rating_features(ratings: list[dict], now: int) -> dict:
    """Contest-performance features from rating changes published before `now`."""
    past = sorted((r for r in ratings if r["time"] < now), key=lambda r: r["time"])
    if not past:
        return {"user_rating": DEFAULT_RATING, "is_rated": 0, "max_rating": DEFAULT_RATING,
                "log_contests": 0.0, "last_delta": 0.0, "mean_delta_5": 0.0,
                "log_days_since_contest": math.log1p(3650)}
    deltas = [r["new"] - r["old"] for r in past]
    return {
        "user_rating": past[-1]["new"], "is_rated": 1, "max_rating": max(r["new"] for r in past),
        "log_contests": math.log1p(len(past)), "last_delta": float(deltas[-1]),
        "mean_delta_5": sum(deltas[-5:]) / len(deltas[-5:]),
        "log_days_since_contest": math.log1p(max(0, now - past[-1]["time"]) / DAY),
    }


def problem_features(state: UserState, rating_feats: dict, problem_rating: float,
                     tags: tuple[str, ...], contest_mode: int, now: int, tag_vocab: list[str]) -> dict:
    """Features for one candidate problem given a user's state just before `now`."""
    user_rating = rating_feats["user_rating"]
    overall = smoothed(state.solved, state.attempted)
    low, high = int(problem_rating - NEAR) // 100, int(problem_rating + NEAR) // 100
    near_a = sum(state.bucket_counts.get(b, [0, 0])[0] for b in range(low, high + 1))
    near_s = sum(state.bucket_counts.get(b, [0, 0])[1] for b in range(low, high + 1))
    tag_rates, tag_attempts = [], []
    for tag in tags:
        a, s = state.tag_counts.get(tag, [0, 0])
        tag_rates.append(smoothed(s, a))
        tag_attempts.append(a)
    tag_mean = sum(tag_rates) / len(tag_rates) if tag_rates else overall
    recent = sum(1 for t in state.recent if t >= now - 30 * DAY)
    row = {
        "problem_rating": float(problem_rating),
        **rating_feats,
        "rating_gap": float(problem_rating - user_rating),
        "elo_prob": elo_probability(user_rating, problem_rating),
        "log_prior_attempted": math.log1p(state.attempted),
        "prior_solve_rate": overall,
        "log_recent_attempted": math.log1p(recent),
        "log_days_active": math.log1p(max(0, now - state.first_time) / DAY) if state.first_time else 0.0,
        "near_solve_rate": smoothed(near_s, near_a),
        "log_near_attempted": math.log1p(near_a),
        "tag_skill_mean": tag_mean,
        "tag_skill_min": min(tag_rates) if tag_rates else overall,
        "tag_skill_rel": tag_mean - overall,
        "log_tag_attempted_min": math.log1p(min(tag_attempts)) if tag_attempts else 0.0,
        "tag_count": float(len(tags)),
        "contest_mode": float(contest_mode),
    }
    for tag in tag_vocab:
        row["tag_" + tag] = 1.0 if tag in tags else 0.0
    return row


def build_interactions(submissions: list[dict]) -> list[Interaction]:
    """Collapse solo submissions into one interaction per problem.

    Each submission dict has: key, time, id, verdict, rating, tags, mode, solo.
    """
    items: dict[str, Interaction] = {}
    for s in sorted(submissions, key=lambda s: (s["time"], s["id"])):
        if not s["solo"] or s["time"] <= 0 or s["key"] is None:
            continue
        item = items.get(s["key"])
        if item is None:
            item = items[s["key"]] = Interaction(s["key"], s["rating"], tuple(s["tags"]), s["mode"],
                                                 s["time"], None)
        if s["verdict"] == "OK" and item.solve is None:
            item.solve = s["time"]
    return list(items.values())


def replay(interactions: list[Interaction], until: int | None = None):
    """Yield (interaction, state-before-its-attempt) in time order.

    Events at the same second are all scored before any of them update the state.
    If `until` is given, stop before that time and return the final state via StopIteration.
    """
    events = [(i.attempt, 0, n) for n, i in enumerate(interactions)]
    events += [(i.solve, 1, n) for n, i in enumerate(interactions) if i.solve is not None]
    events.sort()
    state = UserState()
    pos = 0
    while pos < len(events):
        now = events[pos][0]
        if until is not None and now >= until:
            break
        group = []
        while pos < len(events) and events[pos][0] == now:
            group.append(events[pos])
            pos += 1
        for _, kind, n in group:
            if kind == 0:
                yield interactions[n], state
        for _, kind, n in group:
            (state.add_attempt if kind == 0 else state.add_solve)(interactions[n])
    return state


def state_at(interactions: list[Interaction], now: int) -> UserState:
    """The user's state from all events strictly before `now` (used for live queries)."""
    generator = replay(interactions, until=now)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value
