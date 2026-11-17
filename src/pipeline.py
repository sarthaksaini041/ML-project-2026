"""End-to-end pipeline: dataset -> models -> evaluation -> skill gaps -> recommendations -> site.

    python -m src.pipeline            # from the repository root

Writes results/metrics.json (all numbers quoted on the website and in the README) and the
website data in docs/data/. The notebook calls the same functions step by step with plots.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score

from . import dataset, models, skills
from .features import CONTEST_MODES, DAY, build_interactions, problem_features, rating_features, replay, state_at

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SITE_DATA = ROOT / "docs" / "data"
TOPIC_HISTORY = ["tag_skill_mean", "tag_skill_min", "tag_skill_rel", "log_tag_attempted_min"]
RATING_BANDS = [(-np.inf, 1200, "Below 1200"), (1200, 1600, "1200–1599"), (1600, 2000, "1600–1999"),
                (2000, 2400, "2000–2399"), (2400, np.inf, "2400 and above")]


def band(values):
    out = np.empty(len(values), dtype=object)
    for low, high, name in RATING_BANDS:
        out[(values >= low) & (values < high)] = name
    return out


# --------------------------------------------------------------------------- data

def build():
    tables = dataset.load_tables()
    vocab = dataset.tag_vocabulary(tables["problems"])
    table, data_end = dataset.build_table(tables, vocab)
    train, val, test, cuts = dataset.chronological_split(table)
    return {"tables": tables, "vocab": vocab, "table": table, "data_end": data_end,
            "train": train, "val": val, "test": test, "cuts": cuts,
            "columns": dataset.feature_columns(vocab)}


def data_summary(d):
    t, subs = d["table"], d["tables"]["submissions"]
    parts = {name: d[name] for name in ("train", "val", "test")}
    fmt = lambda s: pd.to_datetime(s, unit="s").strftime("%Y-%m-%d")
    return {
        "users_sampled": int(len(d["tables"]["users"])),
        "users_complete": int(d["tables"]["users"]["history_complete"].sum()),
        "submissions": int(len(subs)),
        "solo_submissions": int(subs["is_solo"].sum()),
        "catalog_problems": int(len(d["tables"]["problems"])),
        "interactions": int(len(t)),
        "users_with_rows": int(t["handle"].nunique()),
        "solved_rate": float(t["solved"].mean()),
        "data_end": fmt(d["data_end"]),
        "first_attempt": fmt(t["attempt_time"].min()),
        "cuts": [fmt(c) for c in d["cuts"]],
        "embargo_dropped": int(len(t) - sum(len(p) for p in parts.values())),
        "partitions": {name: {"rows": int(len(p)), "users": int(p["handle"].nunique()),
                              "solved_rate": float(p["solved"].mean()),
                              "start": fmt(p["attempt_time"].min()), "end": fmt(p["attempt_time"].max())}
                       for name, p in parts.items()},
        "features": len(d["columns"]),
        "tags": d["vocab"],
    }


def eda(d):
    train = d["train"]
    gap_bins = [-np.inf, -600, -300, -100, 100, 300, 600, np.inf]
    gap_names = ["600+ below", "300–599 below", "100–299 below", "Within 100", "100–299 above",
                 "300–599 above", "600+ above"]
    gap = pd.cut(train["rating_gap"], gap_bins, right=False, labels=gap_names)
    rows = lambda g: [[str(k), int(v["size"]), float(v["mean"])] for k, v in g.iterrows()]
    by_gap = train.groupby(gap, observed=True)["solved"].agg(["size", "mean"])
    exploded = train[["tags", "solved"]].explode("tags").dropna()
    by_tag = exploded.groupby("tags")["solved"].agg(["size", "mean"]).nlargest(12, "size")
    by_band = train.assign(b=band(train["user_rating"].to_numpy())).groupby("b")["solved"].agg(["size", "mean"])
    by_band = by_band.reindex([n for *_, n in RATING_BANDS])
    by_mode = train.groupby("mode")["solved"].agg(["size", "mean"]).sort_values("size", ascending=False)
    month = pd.to_datetime(d["table"]["attempt_time"], unit="s").dt.to_period("Q").astype(str)
    timeline = d["table"].groupby(month)["solved"].agg(["size", "mean"]).tail(16)
    return {"gap": rows(by_gap), "tags": rows(by_tag.sort_values("mean")), "bands": rows(by_band),
            "modes": rows(by_mode), "timeline": rows(timeline)}


# --------------------------------------------------------------------------- models

def train_models(d):
    cols = d["columns"]
    Xtr, ytr = d["train"][cols].to_numpy(), d["train"]["solved"].to_numpy()
    Xva, yva = d["val"][cols].to_numpy(), d["val"]["solved"].to_numpy()
    chosen, log = models.tune(Xtr, ytr, Xva, yva, cols)
    # Refit each family's chosen configuration on train + validation, then score test once.
    both = pd.concat([d["train"], d["val"]])
    final = {}
    for family, (val_auc, model, _) in chosen.items():
        final[family] = clone(model).fit(both[cols].to_numpy(), both["solved"].to_numpy())
    return {"tuning": log, "validation": {f: float(a) for f, (a, _, _) in chosen.items()},
            "thresholds": {f: t for f, (_, _, t) in chosen.items()}, "final": final}


def evaluate(d, m):
    cols, test = d["columns"], d["test"]
    X, y, users = test[cols].to_numpy(), test["solved"].to_numpy(), test["handle"].to_numpy()
    probs = {f: model.predict_proba(X)[:, 1] for f, model in m["final"].items()}
    table, confusion = [], {}
    for family, p in probs.items():
        decision = models.threshold_metrics(y, p, m["thresholds"][family])
        confusion[family] = decision.pop("confusion")
        row = {"model": family, "validation ROC-AUC": m["validation"][family], **models.scores(y, p),
               **decision, "threshold P(not solved)": m["thresholds"][family]}
        table.append(row)
    best = max((f for f in probs if f != "Baseline (rating gap)"),
               key=lambda f: m["validation"][f])
    p = probs[best]
    others = sorted(((m["validation"][f], f) for f in probs if f != best), reverse=True)
    runner = others[0][1]
    auc_ci = models.user_bootstrap(y, p, users)
    diff_ci = models.user_bootstrap(
        y, p, users, metric=lambda yy, pp, sample_weight: roc_auc_score(yy, pp, sample_weight=sample_weight)
        - roc_auc_score(yy, probs[runner], sample_weight=sample_weight))
    # Calibration bins (the website quotes these to put each estimate in context).
    edges = [0, 0.6, 0.8, 0.9, 0.95, 0.98, 1.0001]
    idx = np.digitize(p, edges) - 1
    calibration = [{"low": edges[i], "high": min(edges[i + 1], 1.0), "n": int((idx == i).sum()),
                    "predicted": float(p[idx == i].mean()), "observed": float(y[idx == i].mean())}
                   for i in range(len(edges) - 1) if (idx == i).sum()]
    # Segments: by user rating band, problem rating band, mode, and common tags.
    def segment(labels):
        out = []
        for name in pd.unique(labels):
            mask = labels == name
            if mask.sum() >= 200 and 0 < y[mask].mean() < 1:
                out.append({"segment": str(name), "rows": int(mask.sum()), "solved_rate": float(y[mask].mean()),
                            "ROC-AUC": float(roc_auc_score(y[mask], p[mask])),
                            "baseline ROC-AUC": float(roc_auc_score(y[mask], probs["Baseline (rating gap)"][mask]))})
        return out
    user_band = segment(band(test["user_rating"].to_numpy()))
    order = [n for *_, n in RATING_BANDS]
    user_band.sort(key=lambda r: order.index(r["segment"]))
    problem_band = segment(band(test["problem_rating"].to_numpy()))
    problem_band.sort(key=lambda r: order.index(r["segment"]))
    mode = segment(test["mode"].to_numpy())
    tag_rows = []
    for tag in d["vocab"]:
        mask = test["tags"].map(lambda t: tag in t).to_numpy()
        if mask.sum() >= 1000:
            tag_rows.append({"segment": tag, "rows": int(mask.sum()), "solved_rate": float(y[mask].mean()),
                             "ROC-AUC": float(roc_auc_score(y[mask], p[mask])),
                             "baseline ROC-AUC": float(roc_auc_score(y[mask], probs["Baseline (rating gap)"][mask]))})
    tag_rows.sort(key=lambda r: -r["ROC-AUC"])
    # Representative errors from the selected model.
    frame = test.assign(p=p)
    confident_miss = frame.loc[(frame["p"] >= 0.98) & (frame["solved"] == 0)]
    surprise = frame.loc[(frame["p"] < 0.6) & (frame["solved"] == 1)]
    names = d["tables"]["problems"].set_index("problem_key")["name"]
    def examples(rows, n=4):
        rows = rows.sample(min(n, len(rows)), random_state=models.SEED)
        return [{"problem": names.get(r.problem, r.problem), "key": r.problem,
                 "problem_rating": int(r.problem_rating), "user_rating": int(r.user_rating),
                 "mode": r.mode.replace("_", " ").lower(), "p": float(r.p),
                 "prior_attempted": int(round(np.expm1(r.log_prior_attempted)))} for r in rows.itertuples()]
    errors = {
        "confident_miss_count": int(len(confident_miss)), "confident_count": int((p >= 0.98).sum()),
        "surprise_count": int(len(surprise)), "low_count": int((p < 0.6).sum()),
        "confident_miss_modes": confident_miss["mode"].value_counts(normalize=True).round(3).to_dict(),
        "confident_miss_examples": examples(confident_miss), "surprise_examples": examples(surprise),
    }
    sample = test.sample(min(20000, len(test)), random_state=models.SEED)
    imp = permutation_importance(m["final"][best], sample[cols].to_numpy(), sample["solved"].to_numpy(),
                                 scoring="roc_auc", n_repeats=3, random_state=models.SEED, n_jobs=-1)
    importance = sorted(({"feature": c, "drop": float(v)} for c, v in zip(cols, imp.importances_mean)),
                        key=lambda r: -r["drop"])[:12]
    return {"table": table, "confusion": confusion, "positive_rate_not_solved": float(1 - y.mean()),
            "best": best, "runner_up": runner, "auc_ci": auc_ci.tolist(),
            "diff_ci": diff_ci.tolist(), "calibration": calibration,
            "segments": {"user_rating": user_band, "problem_rating": problem_band, "mode": mode, "tags": tag_rows},
            "errors": errors, "importance": importance}, probs


# --------------------------------------------------------------------------- skills and recommendations

def catalog_of(tables):
    problems = tables["problems"]
    problems = problems.loc[problems["type"].eq("PROGRAMMING")]
    return {r.problem_key: [r.name, None if np.isnan(r.rating) else float(r.rating),
                            list(dataset.parse_tags(r.tags))] for r in problems.itertuples()}


def evaluate_skills_and_recommendations(d, m, best, probs):
    """Do the flagged gaps and the model-ranked recommendations hold up in the test period?"""
    cols, vocab, test = d["columns"], d["vocab"], d["test"]
    t_test = d["cuts"][1]
    catalog = catalog_of(d["tables"])
    model = m["final"][best]
    # Ablation: the same model without any topic-history features.
    keep = [c for c in cols if c not in TOPIC_HISTORY]
    both = pd.concat([d["train"], d["val"]])
    ablation = HistGradientBoostingClassifier(**model.get_params()).fit(both[keep].to_numpy(), both["solved"].to_numpy())
    p_no_topic = ablation.predict_proba(test[keep].to_numpy())[:, 1]
    ablation_auc = roc_auc_score(test["solved"], p_no_topic)

    by_user = dataset.submissions_by_user(d["tables"]["submissions"])
    ratings = dataset.ratings_by_user(d["tables"]["rating_history"])
    frame = test.assign(resid=test["solved"] - p_no_topic)
    per_user = []
    for handle, rows in frame.groupby("handle"):
        history = [s for s in by_user[handle] if s["time"] < t_test]
        interactions = build_interactions(history)
        if len(interactions) < 30 or len(rows) < 20:
            continue
        state = state_at(interactions, t_test)
        rfeats = rating_features(ratings.get(handle, []), t_test)
        keys, X = skills.candidate_matrix(state, rfeats, catalog, {i.key for i in interactions}, t_test, vocab, cols)
        gaps = skills.skill_gaps(model, keys, X, catalog, cols, vocab)
        if len(gaps) < 6:
            continue
        ranked = sorted(gaps, key=gaps.get)
        weak, strong = set(ranked[:3]), set(ranked[-3:])
        w = rows["tags"].map(lambda t: bool(weak & set(t)))
        s = rows["tags"].map(lambda t: bool(strong & set(t)))
        if w.sum() >= 3 and s.sum() >= 3:
            per_user.append({"handle": handle, "weak_resid": rows.loc[w, "resid"].mean(),
                             "strong_resid": rows.loc[s, "resid"].mean(),
                             "weak_rate": rows.loc[w, "solved"].mean(), "strong_rate": rows.loc[s, "solved"].mean(),
                             "weak_n": int(w.sum()), "strong_n": int(s.sum())})
    gap_users = pd.DataFrame(per_user)
    diff = (gap_users["weak_resid"] - gap_users["strong_resid"]).to_numpy()
    rng = np.random.default_rng(models.SEED)
    boot = [rng.choice(diff, len(diff)).mean() for _ in range(2000)]
    skill_eval = {
        "users": int(len(gap_users)), "ablation_auc": float(ablation_auc),
        "mean_weak_rate": float(gap_users["weak_rate"].mean()), "mean_strong_rate": float(gap_users["strong_rate"].mean()),
        "mean_resid_diff": float(diff.mean()), "resid_diff_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
        "share_users_weak_lower": float((diff < 0).mean()),
    }

    # Recommendation ranking: among each user's test attempts at or above their rating,
    # does the model order them by eventual success better than "closest rating first"?
    p = probs[best]
    stretch = test.assign(p=p).loc[test["rating_gap"] >= 0]
    rec_rows = []
    for handle, rows in stretch.groupby("handle"):
        if len(rows) >= 10 and 0 < rows["solved"].mean() < 1:
            top_model = rows.nlargest(5, "p")["solved"].mean()
            top_rating = rows.nsmallest(5, "rating_gap")["solved"].mean()
            rec_rows.append({"model_auc": roc_auc_score(rows["solved"], rows["p"]),
                             "rating_auc": roc_auc_score(rows["solved"], -rows["rating_gap"]),
                             "model_top5": top_model, "rating_top5": top_rating, "n": len(rows),
                             "mean_gap_model_top5": rows.nlargest(5, "p")["rating_gap"].mean(),
                             "mean_gap_rating_top5": rows.nsmallest(5, "rating_gap")["rating_gap"].mean()})
    rec = pd.DataFrame(rec_rows)
    d_auc = (rec["model_auc"] - rec["rating_auc"]).to_numpy()
    boot = [rng.choice(d_auc, len(d_auc)).mean() for _ in range(2000)]
    rec_eval = {
        "users": int(len(rec)), "stretch_rows": int(len(stretch)), "stretch_solved_rate": float(stretch["solved"].mean()),
        "model_auc": float(rec["model_auc"].mean()), "rating_auc": float(rec["rating_auc"].mean()),
        "auc_diff_ci": np.percentile(boot, [2.5, 97.5]).tolist(),
        "share_users_model_better": float((d_auc > 0).mean()),
        "model_top5": float(rec["model_top5"].mean()), "rating_top5": float(rec["rating_top5"].mean()),
        "model_top5_gap": float(rec["mean_gap_model_top5"].mean()),
        "rating_top5_gap": float(rec["mean_gap_rating_top5"].mean()),
    }
    return skill_eval, rec_eval, catalog


# --------------------------------------------------------------------------- export

def export_tree_model(model, columns, vocab, calibration):
    """Serialise a fitted HistGradientBoostingClassifier for docs/assets/model.js."""
    if not isinstance(model, HistGradientBoostingClassifier):
        raise TypeError("The browser evaluator supports the Gradient Boosting model only.")
    trees = []
    for (predictor,) in model._predictors:
        n = predictor.nodes
        trees.append([n["feature_idx"].tolist(), [float(x) for x in n["num_threshold"]],
                      n["left"].tolist(), n["right"].tolist(), n["is_leaf"].astype(int).tolist(),
                      [float(x) for x in n["value"]], n["missing_go_to_left"].astype(int).tolist()])
    return {"version": "gb-solve30-v1", "columns": columns, "tags": vocab,
            "baseline": float(np.ravel(model._baseline_prediction)[0]), "trees": trees,
            "calibration": calibration}


def demo_user(d, catalog):
    """A real public history from the test period, anonymised, for the offline example."""
    t_test = d["cuts"][1]
    test = d["test"]
    counts = test.groupby("handle").size()
    candidates = counts[(counts > 150) & (counts < 400)].index
    ratings = dataset.ratings_by_user(d["tables"]["rating_history"])
    def rating_then(handle):
        past = sorted((r for r in ratings.get(handle, []) if r["time"] < t_test), key=lambda r: r["time"])
        return past[-1]["new"] if past else 0
    pick = next(h for h in sorted(candidates) if 1500 <= rating_then(h) < 1900)
    subs = dataset.submissions_by_user(d["tables"]["submissions"])[pick]
    history = [{"key": s["key"], "id": i, "time": int(s["time"]), "verdict": s["verdict"],
                "rating": s["rating"], "tags": list(s["tags"]), "mode": s["mode"], "solo": bool(s["solo"])}
               for i, s in enumerate(sorted(subs, key=lambda s: (s["time"], s["id"]))) if s["time"] < t_test]
    return {"handle": "Example user", "now": int(t_test), "history": history,
            "ratings": [{"time": int(r["time"]), "old": int(r["old"]), "new": int(r["new"])}
                        for r in sorted(ratings.get(pick, []), key=lambda r: r["time"]) if r["time"] < t_test]}


def parity_fixtures(d, model, n=300):
    cols = d["columns"]
    rows = d["test"].sample(n, random_state=models.SEED)
    p = model.predict_proba(rows[cols].to_numpy())[:, 1]
    return [{"x": [float(v) for v in r], "p": float(q)} for r, q in zip(rows[cols].to_numpy(), p)]


def feature_fixtures(d, example, catalog, model, n_candidates=40):
    """Python reference features for the example history, for the JavaScript parity test."""
    cols, vocab = d["columns"], d["vocab"]
    interactions = build_interactions(example["history"])
    replayed = []
    for item, state in replay(interactions):
        if item.rating is None:
            continue
        feats = problem_features(state, rating_features(example["ratings"], item.attempt), item.rating,
                                 item.tags, int(item.mode in CONTEST_MODES), item.attempt, vocab)
        replayed.append({"key": item.key, "x": [feats[c] for c in cols]})
    now = example["now"]
    state = state_at(interactions, now)
    rfeats = rating_features(example["ratings"], now)
    keys, X = skills.candidate_matrix(state, rfeats, catalog, {i.key for i in interactions}, now, vocab, cols)
    pick = list(range(0, len(keys), max(1, len(keys) // n_candidates)))[:n_candidates]
    gaps = skills.skill_gaps(model, keys, X, catalog, cols, vocab)
    return {"interactions": replayed[-60:], "candidates": [{"key": keys[i], "x": X[i].tolist()} for i in pick],
            "candidate_count": len(keys), "gaps": gaps,
            "probabilities": model.predict_proba(X[pick])[:, 1].tolist()}


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False))


def export_site(d, m, results, catalog):
    """Write results/metrics.json, the website data files and the parity fixtures."""
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "metrics.json").write_text(json.dumps(results, indent=2, default=float))
    ev = results["evaluation"]
    model = m["final"][ev["best"]]
    write_json(SITE_DATA / "model.json", export_tree_model(model, d["columns"], d["vocab"], ev["calibration"]))
    write_json(SITE_DATA / "results.json", results)
    write_json(SITE_DATA / "catalog.json", {k: v for k, v in catalog.items() if v[1] is not None})
    example = demo_user(d, catalog)
    write_json(SITE_DATA / "example.json", example)
    write_json(ROOT / "tests" / "fixtures" / "model_parity.json", parity_fixtures(d, model))
    write_json(ROOT / "tests" / "fixtures" / "feature_parity.json", feature_fixtures(d, example, catalog, model))


def run():
    d = build()
    m = train_models(d)
    ev, probs = evaluate(d, m)
    skill_eval, rec_eval, catalog = evaluate_skills_and_recommendations(d, m, ev["best"], probs)
    results = {"data": data_summary(d), "eda": eda(d), "tuning": m["tuning"], "evaluation": ev,
               "skill_gaps": skill_eval, "recommendations": rec_eval}
    export_site(d, m, results, catalog)
    return d, m, results


if __name__ == "__main__":
    _, _, results = run()
    for row in results["evaluation"]["table"]:
        print({k: round(v, 4) if isinstance(v, float) else v for k, v in row.items()})
    print("skill gaps:", results["skill_gaps"])
    print("recommendations:", results["recommendations"])
