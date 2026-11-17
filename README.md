# Personalized Competitive Programming Difficulty and Skill-Gap Estimator

ES 335 Machine Learning course project, IIT Gandhinagar, 2026  
Sarthak Saini (26120007) and Aditya Anup Dhomne (26120001)

- **Website:** https://sarthaksaini041.github.io/ML-project-2026/
- **Repository:** https://github.com/sarthaksaini041/ML-project-2026
- **Tools:** Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Jupyter

## Question

Can classical machine-learning models estimate the probability that a Codeforces user will solve a programming problem from their rating, previous contest performance, submission history, problem difficulty and problem tags? Can these predictions identify a user's weak topics and recommend problems at an appropriate difficulty?

| | |
| --- | --- |
| **One row** | a user–problem interaction, created at the user's first solo submission to that problem |
| **Target** | 1 if the user gets an accepted submission on it within 30 days of that first attempt |
| **Prediction time** | just before the first attempt; every feature uses only earlier events |
| **Split** | chronological: train before 4 Jan 2025, validation until 25 Oct 2025, test after, with a 30-day embargo at each boundary |

## Results (test period, 37,817 interactions)

| Model | Validation ROC-AUC | Test ROC-AUC | Precision (not solved) | Recall (not solved) | F1 (not solved) | Accuracy | Cross-entropy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Baseline (rating gap, Elo) | 0.709 | 0.694 | 10.7% | 44.8% | 0.173 | 75.7% | 1.068 |
| Logistic Regression | 0.784 | 0.773 | 27.8% | 30.6% | 0.291 | 91.6% | 0.186 |
| Random Forest | 0.801 | 0.797 | 30.6% | 30.2% | 0.304 | 92.2% | 0.181 |
| **Gradient Boosting** (selected) | **0.810** | **0.800** | **32.8%** | 29.4% | **0.310** | 92.6% | **0.180** |

Precision, recall and F1 are computed for the rare "not solved" class (5.7% of test rows). The flagging threshold on P(not solved) is chosen on validation for each model. The metrics are the ones taught in the course: ROC/AUC, precision, recall, F1, confusion matrix, calibration and binary cross-entropy. Predicting "solved" for everything would already be 94.3% accurate while flagging nothing, which shows why accuracy misleads when one class is rare. The baseline is the rating-gap rule from the proposal: the Elo expected score from the user–problem rating difference, with nothing learned from data.

- The Gradient Boosting test ROC-AUC has a 95% user-bootstrap interval of 0.779 to 0.818. Its difference from Random Forest is within noise.
- **Recommendations:** on each user's real stretch attempts in the test period, ranking by the model beats choosing problems closest to the user's rating. Mean per-user AUC is 0.655 against 0.622, and the 95% interval for the difference is +0.001 to +0.065.
- **Skill gaps:** topics flagged as weak were solved less often afterwards (89.2% against 93.9%). After adjusting for difficulty and general skill, though, the difference was not significant, so the site presents skill gaps as a description of the record rather than a validated prediction.

All numbers are written to `results/metrics.json` by the pipeline and loaded by the website from `docs/data/results.json`.

## Repository structure

```text
.
├── src/
│   ├── cf_api.py         Rate-limited, cached Codeforces API client
│   ├── collect.py        Samples 80 rated users per rating band; downloads profiles, ratings, submissions, catalog
│   ├── smoke_test.py     API connectivity check
│   ├── features.py       Point-in-time feature engine (replays each user's history in time order)
│   ├── dataset.py        Interaction table, tag vocabulary, chronological split with embargo
│   ├── models.py         Baselines, candidate models, tuning grids, metrics, user bootstrap
│   ├── skills.py         Topic-wise skill gaps and recommendation candidates
│   └── pipeline.py       End-to-end run: data, models, evaluation, skill gaps, recommendations, export
├── notebooks/
│   └── skill_gap_estimator.ipynb   The analysis, step by step, with plots (executed)
├── tests/
│   ├── test_features.py  Leakage, tie handling, target definition, split embargo
│   ├── check_parity.mjs  Browser features, model and skill gaps reproduce Python exactly
│   └── fixtures/
├── docs/                 GitHub Pages site
│   ├── index.html
│   ├── notebook.html     Rendered notebook
│   ├── assets/           app.js, features.js (port of src/features.py), model.js, codeforces.js, style.css
│   └── data/             model.json, results.json, catalog.json, example.json (written by the pipeline)
├── data/
│   ├── processed/        submissions, rating_history, users, problems (Parquet, about 22 MB, committed)
│   └── raw/              API cache (about 900 MB, not committed)
├── results/metrics.json
├── config.json           Sample design, seed, API spacing
└── requirements.txt
```

## Data access

All data comes from the public [Codeforces API](https://codeforces.com/apiHelp) endpoints `user.ratedList`, `user.info`, `user.rating`, `user.status` and `problemset.problems`. No account, password or API key is used. `.env.example` is only a placeholder.

The sample has 400 rated users, 80 from each of five rating bands, with complete submission histories (668,355 submissions, 11,511 catalog problems). After keeping solo submissions on rated problems with a fully observed 30-day window, there are 252,109 interactions.

`data/processed/` holds the four Parquet tables the analysis reads. They contain only public handles and submission metadata, with no source code. The raw API cache is too large to commit. `python -m src.collect` rebuilds it, but because Codeforces data changes over time, a new collection will not exactly match ours.

## Setup and reproduction

Requires Python 3.10+. Node 18+ is needed only for the parity check.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

1. **Collect the data.** This step is optional because the processed tables are included, and it takes several hours.
   ```bash
   python -m src.smoke_test --handle tourist
   python -m src.collect --config config.json
   ```
2. **Run the full pipeline** (about 5 minutes):
   ```bash
   python -m src.pipeline
   ```
   This builds the dataset, tunes and tests every model, evaluates skill gaps and recommendations, and writes `results/metrics.json`, `docs/data/` and the test fixtures. To run the same steps with plots, use `notebooks/skill_gap_estimator.ipynb`.
3. **Run the tests:**
   ```bash
   python -m pytest tests
   node tests/check_parity.mjs
   ```

## Website

```bash
python -m http.server 8000 --directory docs
```

Open http://localhost:8000/. Enter a Codeforces handle, or click **Use an example**, to see:
- rating and activity trends
- topic-wise skill gaps
- recommended problems, filterable by rating range and topic, each with an explanation and a confidence note
- an estimate for any specific problem

Live lookups call the public Codeforces API from the browser at most once every two seconds. The report sections below the tool follow the course guidelines: question, data, preprocessing, methods, evaluation, results, errors, skill gaps and recommendations, limitations, conclusion and reproduction.

**Publish:** push to GitHub. Then open **Settings → Pages**, choose **Deploy from a branch**, and select the default branch with the **/docs** folder.

## Changes from the proposal

- **Gradient Boosting** is scikit-learn's `HistGradientBoostingClassifier`.
- **"Solves"** is defined as accepted within 30 days of the first attempt. Problems a user never attempted are not treated as failures.
- **Only rated problems** are modelled.
- **Skill gaps** did not prove predictive, and we report that result rather than hiding it.

## AI-use disclosure

We used AI assistants for brainstorming, coding help, debugging, and editing text and the website. We ran and checked every analysis ourselves and are responsible for all results.

## References

- Codeforces API documentation: https://codeforces.com/apiHelp
- A. Elo. *The Rating of Chessplayers, Past and Present.* 1978.
- L. Breiman. *Random Forests.* Machine Learning 45, 2001.
- J. Friedman. *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics, 2001.
- F. Pedregosa et al. *Scikit-learn: Machine Learning in Python.* JMLR 12, 2011.
- pandas, NumPy, Matplotlib, Seaborn.
