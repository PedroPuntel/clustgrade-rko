# Experiment 01 — RKO parameter study (2×2 ablation)

Crosses two binary factors that control how candidate grids are scored
inside the ClustGrade-RKO loop:

- `quadrat_filter ∈ {True, False}` — reject grids that fail Pearson's
  quadrat test for complete spatial randomness before evaluating density.
- `msi_space ∈ {"feature", "mds"}` — compute the inner silhouette objective
  on the original feature matrix or on the scaled 2D MDS projection.

The study runs on a stratified 20-dataset subset (12 CLUST + 8 CLASSF), 5
seeds per configuration, 60 s time budget per run. The winning configuration
is serialized to `artifacts/best_config.json` and inherited by Experiments
02 and 03.

## Run

```bash
poetry run python -m experiments.01-param-study.run         # ~ 5–7 h on a laptop
poetry run python -m experiments.01-param-study.aggregate   # < 1 min
poetry run python -m experiments.01-param-study.plot        # < 1 min
```

Optional sensitivity scatter plots:

```bash
poetry run python -m experiments.01-param-study.plot_largest_msi_sensitivity
poetry run python -m experiments.01-param-study.plot_largest_ari_sensitivity
```

## Outputs (`artifacts/`)

- `raw_runs.csv` — one row per `(dataset, quadrat_filter, msi_space, seed)`.
- `summary_per_config.csv`, `win_loss_tie.csv`, `statistical_tests.csv` —
  win/loss/tie counts, Friedman + Wilcoxon/Holm-Bonferroni post-hoc.
- `best_config.json` — selected `(quadrat_filter, msi_space)` pair.
- `raincloud_msi.png`, `raincloud_ari_classf.png`, sensitivity scatters.
