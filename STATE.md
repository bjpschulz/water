# STATE.md

Snapshot of *where the project currently stands*. Changes often — every
time a stage finishes or an open decision resolves. Two things this file
deliberately does **not** contain:

- **Exact measured numbers** — those live in `output/`,
  produced by reproducible scripts (like eda, baseline or training scripts).
- **Settled judgment calls** (e.g. "this comparison is confounded, don't
  use it") — once something here resolves, its one-line conclusion moves
  to AGENTS.md's "Settled facts & scope decisions" and gets deleted from
  this file. If you're looking for something that isn't here anymore,
  check there.

Rules, protocol, and anything that shouldn't change session-to-session
live in AGENTS.md, not here.

Last updated: all pipeline stages (1–6) are done and the test split is spent.
Remaining work is the report (discussion, literature, open `\todo`s) and the
repository clean-up for grading. Solo project.

## Where we are

All stages run end to end via `run_all.py` (the prediction plotter
`src/plot_predictions.py` is run by hand):

- EDA and cleaning (`src/eda.py` → `output/eda/`).
- Feature table: lags 1–15, 30, 45, 60, 1440, 10080 and calendar features in
  raw and cyclic encodings, local time, warm-up NaN rows kept
  (`src/build_features.py`).
- Stage 1, splits (`src/core/data.py`): Monday-00:00-local boundaries (~70/15/15),
  valid/test at full weekly cycles, warm-up rows excluded from train.
  Downstream stages must obtain splits via `split_series()`.
- Stage 2, naive baselines (`src/baselines.py`).
- Stage 3, diagnostic models (`src/compare_models.py`): linear regression and
  LightGBM, each L2 and L1, full feature set.
- Stage 4, feature ablation (`src/feature_ablation.py`); final feature set in
  AGENTS.md.
- Stage 5, grouped TreeSHAP importance and residual ACF
  (`src/feature_importance.py`, `src/residual_acf.py`), on validation.
- Stage 6, single test evaluation (`src/test_evaluation.py`). The test split is
  spent; no further modelling decision may be based on it.

`tex/main.tex` is written up through Stage 6.

## Next steps

1. Report: Ch.5 drafted. Built with pdflatex (`latexmk -pdf`), Computer Modern;
   Arial lines are commented out in the preamble pending the teacher's answer.
   Open: the remaining `\todo`s in `tex/main.tex` (zero/non-zero breakdown,
   negative predictions, efficiency), front matter, figures into `tex/images/`.
2. Repository clean-up for grading (README, code structure, what is committed).
3. Finally, remove AGENTS.md and STATE.md.

## Report

The final report is `tex/main.tex` (single file). Update it only after asking
and checking in with the user (AGENTS.md → "Living documentation"). Open
`\todo{}` items mark what is still unwritten.

## Modelling target

`y_t` is `avg_rate` (L/min) at native 1-minute resolution, predicted from
known calendar features and earlier lags only.

## Active feature set

`hour` + `lag_1`–`lag_15` for all final-stage models (AGENTS.md, settled
facts). The feature table keeps all columns; the Stage 3 diagnostic models
still use the full set. Rolling statistics are excluded (AGENTS.md); exogenous
data such as weather are out of scope.

## Evaluation metric

Report MAE and RMSE against all explicit naive baselines. See
`AGENTS.md` → "Evaluation principles" for the settled rationale.
