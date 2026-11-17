"""Resumable collection of a stratified 500-user Codeforces sample.

Primary source:
  - user.ratedList -> choose active users across rating bands
  - user.info -> current public profile snapshot
  - user.rating -> historical contest rating changes
  - user.status -> submission history (paginated)

Additional source:
  - contest.standings -> contest-level per-problem outcomes for contests in which
    sampled users appear. Standings rows are stored as a separate source and are
    NOT blindly merged into the primary labels.

Run from project root:
    python -m src.collect --config config.json
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

from src.cf_api import CodeforcesAPI, CodeforcesAPIError

ROOT = Path(".")
RAW = ROOT / "data" / "raw"
LOG_DIR = ROOT / "logs"
RAW.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "collection.log", encoding="utf-8"),
        logging.StreamHandler()
    ],
)
LOG = logging.getLogger("cf_collect")


def load_json(path: Path, default: Any):
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return default


def atomic_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    temp.replace(path)


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sample_users(api: CodeforcesAPI, cfg: dict) -> list[dict]:
    """Sample users from ratedList, stratified by current rating.

    The list is an eligibility pool, not a random population sample of all CF
    accounts. Rating bands are intentionally balanced for model development.
    """
    cache_file = RAW / "rated_list.json"
    if cache_file.exists():
        rated = load_json(cache_file, [])
        LOG.info("Loaded ratedList cache: %d users", len(rated))
    else:
        params = {
            "activeOnly": str(cfg.get("active_only", True)).lower(),
            "includeRetired": str(cfg.get("include_retired", False)).lower(),
        }
        rated = api.get("user.ratedList", params)
        atomic_json(cache_file, rated)
        LOG.info("Fetched ratedList: %d users", len(rated))

    rng = random.Random(cfg.get("random_seed", 42))
    selected = []
    seen = set()

    for band in cfg["rating_bands"]:
        candidates = [
            u for u in rated
            if band["min"] <= int(u.get("rating", -1)) <= band["max"]
        ]
        rng.shuffle(candidates)
        wanted = int(band["count"])
        chosen = []
        for user in candidates:
            handle = user.get("handle")
            if handle and handle.lower() not in seen:
                chosen.append(user)
                seen.add(handle.lower())
                if len(chosen) >= wanted:
                    break

        LOG.info(
            "Band %-26s eligible=%d selected=%d requested=%d",
            band["name"], len(candidates), len(chosen), wanted
        )
        selected.extend(chosen)

    target = int(cfg.get("sample_size", 500))
    if len(selected) < target:
        LOG.warning("Only %d users selected from configured bands (target=%d).",
                    len(selected), target)
    return selected[:target]


def collect_profile_and_rating(api: CodeforcesAPI, handles: list[str]):
    profiles = load_json(RAW / "profiles.json", {})
    ratings = load_json(RAW / "rating_histories.json", {})

    missing = [h for h in handles if h.lower() not in profiles]
    # user.info supports semicolon-separated handles; keep batches moderate.
    for start in range(0, len(missing), 100):
        batch = missing[start:start + 100]
        try:
            result = api.get("user.info", {"handles": ";".join(batch)})
            for item in result:
                profiles[item["handle"].lower()] = item
            atomic_json(RAW / "profiles.json", profiles)
        except Exception as exc:
            LOG.exception("user.info batch failed: %s", exc)

    for handle in tqdm(handles, desc="user.rating"):
        key = handle.lower()
        if key in ratings:
            continue
        try:
            ratings[key] = api.get("user.rating", {"handle": handle})
            atomic_json(RAW / "rating_histories.json", ratings)
        except Exception as exc:
            LOG.exception("user.rating failed for %s: %s", handle, exc)

    return profiles, ratings


def collect_all_submissions(api: CodeforcesAPI, handles: list[str], cfg: dict):
    """Fetch complete history. A per-user JSON file is the resume checkpoint."""
    submission_dir = RAW / "submissions"
    submission_dir.mkdir(parents=True, exist_ok=True)
    status_dir = RAW / "submission_status"
    status_dir.mkdir(parents=True, exist_ok=True)

    page_size = int(cfg.get("submissions_page_size", 1000))
    all_status = load_json(RAW / "history_completeness.json", {})

    for handle in tqdm(handles, desc="user.status histories"):
        key = handle.lower()
        user_file = submission_dir / f"{key}.json"
        status_file = status_dir / f"{key}.json"

        if user_file.exists() and status_file.exists():
            continue

        user_submissions = []
        start = 1
        page_num = 0
        complete = False
        error_message = None

        try:
            while True:
                page_num += 1
                batch = api.get(
                    "user.status",
                    {"handle": handle, "from": start, "count": page_size}
                )
                if not isinstance(batch, list):
                    raise RuntimeError(f"Unexpected user.status response for {handle}")

                user_submissions.extend(batch)
                start += len(batch)

                # An empty or short page proves the API history is exhausted.
                if len(batch) < page_size:
                    complete = True
                    break

            atomic_json(user_file, user_submissions)
            atomic_json(status_file, {
                "handle": handle,
                "complete": complete,
                "submission_count": len(user_submissions),
                "pages": page_num,
                "collected_at_utc": int(time.time()),
                "error": None
            })
            all_status[key] = {
                "complete": complete,
                "submission_count": len(user_submissions),
                "pages": page_num
            }
            atomic_json(RAW / "history_completeness.json", all_status)
            LOG.info("%s: %d submissions, complete=%s",
                     handle, len(user_submissions), complete)

        except Exception as exc:
            error_message = str(exc)
            # Save partial data separately; never label it as complete history.
            atomic_json(RAW / "partial_submissions" / f"{key}.json", user_submissions)
            atomic_json(status_file, {
                "handle": handle,
                "complete": False,
                "submission_count": len(user_submissions),
                "pages": page_num,
                "collected_at_utc": int(time.time()),
                "error": error_message
            })
            all_status[key] = {
                "complete": False,
                "submission_count": len(user_submissions),
                "pages": page_num,
                "error": error_message
            }
            atomic_json(RAW / "history_completeness.json", all_status)
            LOG.exception("History collection failed for %s", handle)
            if cfg.get("stop_on_api_error", False):
                raise


def collect_problemset(api: CodeforcesAPI):
    path = RAW / "problemset.json"
    if path.exists():
        payload = load_json(path, {})
        LOG.info("Loaded problemset cache.")
    else:
        payload = api.get("problemset.problems", {})
        atomic_json(path, payload)

    problems = payload.get("problems", [])
    stats = payload.get("problemStatistics", [])
    LOG.info("Problemset: %d problems, %d statistics rows",
             len(problems), len(stats))
    return payload



def collect_contest_standings(api: CodeforcesAPI, handles: list[str], cfg: dict):
    """Fetch public standings once per contest ID found in sample users' submissions.

    Anonymous access to regular public contests requires exactly one query
    parameter, contestId, and returns the complete official public standings.
    Some contest types and contests with hidden standings are not available.
    Standings are retained as an audit source, not converted directly into labels.
    """
    if not cfg.get("contest_standings_enabled", True):
        LOG.info("Contest standings disabled.")
        return

    submission_dir = RAW / "submissions"
    contest_to_sample_handles = defaultdict(set)

    for handle in handles:
        path = submission_dir / f"{handle.lower()}.json"
        if not path.exists():
            continue
        submissions = load_json(path, [])
        for sub in submissions:
            cid = sub.get("contestId")
            if cid is not None:
                contest_to_sample_handles[int(cid)].add(handle.lower())

    contest_ids = sorted(contest_to_sample_handles)
    max_contests = cfg.get("max_contests_to_fetch")
    if max_contests is not None:
        contest_ids = contest_ids[-int(max_contests):]

    standings_dir = RAW / "contest_standings"
    standings_dir.mkdir(parents=True, exist_ok=True)
    standings_status = load_json(RAW / "contest_standings_status.json", {})

    for cid in tqdm(contest_ids, desc="contest.standings"):
        out_file = standings_dir / f"{cid}.json"
        if out_file.exists():
            continue

        try:
            # Codeforces' anonymous public regular-contest mode accepts only
            # contestId; adding from/count/handles/showUnofficial is invalid.
            result = api.get("contest.standings", {"contestId": cid})
            if not isinstance(result, dict):
                raise RuntimeError(f"Unexpected standings response for contest {cid}")

            contest_info = result.get("contest", {})
            problems = result.get("problems", [])
            contest_rows = result.get("rows", [])

            payload = {
                "contestId": cid,
                "contest": contest_info,
                "problems": problems,
                "rows": contest_rows,
                "fetched_at_utc": int(time.time()),
                "sample_handles_in_contest": sorted(contest_to_sample_handles[cid]),
                "pages_fetched": 1,
                "possibly_truncated": False
            }
            atomic_json(out_file, payload)
            standings_status[str(cid)] = {
                "complete": True,
                "rows": len(contest_rows),
                "pages": 1
            }
            atomic_json(RAW / "contest_standings_status.json", standings_status)
            LOG.info("Contest %s: %d public standings rows", cid, len(contest_rows))

        except Exception as exc:
            standings_status[str(cid)] = {"complete": False, "error": str(exc)}
            atomic_json(RAW / "contest_standings_status.json", standings_status)
            LOG.exception("Standings fetch failed for contest %s", cid)
            if cfg.get("stop_on_api_error", False):
                raise


def export_flat_tables(handles: list[str]):
    """Export normalized CSV/Parquet tables for downstream EDA and modeling."""
    profiles = load_json(RAW / "profiles.json", {})
    ratings = load_json(RAW / "rating_histories.json", {})
    completeness = load_json(RAW / "history_completeness.json", {})

    user_rows = []
    for handle in handles:
        h = handle.lower()
        profile = profiles.get(h, {})
        user_rows.append({
            "handle": profile.get("handle", handle),
            "rating_current_snapshot": profile.get("rating"),
            "max_rating_current_snapshot": profile.get("maxRating"),
            "rank_current_snapshot": profile.get("rank"),
            "max_rank_current_snapshot": profile.get("maxRank"),
            "registration_time_seconds": profile.get("registrationTimeSeconds"),
            "last_online_time_seconds": profile.get("lastOnlineTimeSeconds"),
            "history_complete": completeness.get(h, {}).get("complete", False),
            "submission_count": completeness.get(h, {}).get("submission_count", 0),
        })

    rating_rows = []
    for handle in handles:
        h = handle.lower()
        for item in ratings.get(h, []):
            rating_rows.append({
                "handle": handle,
                "contest_id": item.get("contestId"),
                "contest_name": item.get("contestName"),
                "rank": item.get("rank"),
                "rating_update_time_seconds": item.get("ratingUpdateTimeSeconds"),
                "old_rating": item.get("oldRating"),
                "new_rating": item.get("newRating"),
            })

    submission_rows = []
    for handle in handles:
        path = RAW / "submissions" / f"{handle.lower()}.json"
        if not path.exists():
            continue
        for s in load_json(path, []):
            p = s.get("problem") or {}
            author = s.get("author") or {}
            members = author.get("members") or []
            submission_rows.append({
                "handle": handle,
                "submission_id": s.get("id"),
                "creation_time_seconds": s.get("creationTimeSeconds"),
                "relative_time_seconds": s.get("relativeTimeSeconds"),
                "contest_id": s.get("contestId"),
                "problem_contest_id": p.get("contestId", s.get("contestId")),
                "problem_index": p.get("index"),
                "problem_name": p.get("name"),
                "problem_rating": p.get("rating"),
                "problem_tags": json.dumps(p.get("tags") or []),
                "problem_type": p.get("type"),
                "verdict": s.get("verdict"),
                "programming_language": s.get("programmingLanguage"),
                "passed_test_count": s.get("passedTestCount"),
                "time_consumed_millis": s.get("timeConsumedMillis"),
                "memory_consumed_bytes": s.get("memoryConsumedBytes"),
                "participant_type": author.get("participantType"),
                "team_member_count": len(members),
                "is_solo": len(members) <= 1,
            })

    problem_payload = load_json(RAW / "problemset.json", {})
    problem_rows = []
    stats_lookup = {}
    for stat in problem_payload.get("problemStatistics", []):
        key = (stat.get("contestId"), stat.get("index"))
        stats_lookup[key] = stat.get("solvedCount")

    for p in problem_payload.get("problems", []):
        key = (p.get("contestId"), p.get("index"))
        problem_rows.append({
            "contest_id": p.get("contestId"),
            "problem_index": p.get("index"),
            "problem_key": f"{p.get('contestId')}:{p.get('index')}",
            "name": p.get("name"),
            "type": p.get("type"),
            "rating": p.get("rating"),
            "points": p.get("points"),
            "tags": json.dumps(p.get("tags") or []),
            "current_solved_count": stats_lookup.get(key),
        })

    tables = {
        "users": pd.DataFrame(user_rows),
        "rating_history": pd.DataFrame(rating_rows),
        "submissions": pd.DataFrame(submission_rows),
        "problems": pd.DataFrame(problem_rows),
    }

    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False)
        try:
            frame.to_parquet(out / f"{name}.parquet", index=False)
        except Exception as exc:
            LOG.warning("Parquet export skipped for %s: %s", name, exc)
        LOG.info("Exported %-16s rows=%d cols=%d", name, len(frame), len(frame.columns))

    return tables


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    cfg = load_config(args.config)

    api = CodeforcesAPI(
        min_interval_seconds=cfg.get("api_min_interval_seconds", 2.1),
        max_retries=cfg.get("max_retries", 6),
    )

    # Step 1: stratified rated-user sample.
    selected = sample_users(api, cfg)
    handles = [u["handle"] for u in selected]
    atomic_json(RAW / "selected_users.json", selected)
    LOG.info("Selected %d unique users.", len(handles))

    # Step 2: public profile + rating history.
    collect_profile_and_rating(api, handles)

    # Step 3: full paginated submission histories.
    collect_all_submissions(api, handles, cfg)

    # Step 4: problem metadata + global problem statistics.
    collect_problemset(api)

    # Step 5: public contest standings for contests represented in sample histories.
    collect_contest_standings(api, handles, cfg)

    # Step 6: normalized tables for EDA / labeling.
    export_flat_tables(handles)
    LOG.info("Collection pipeline finished.")


if __name__ == "__main__":
    main()
