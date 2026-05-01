# Experiment 02 — Q-learning impact + metaheuristic comparison

Two complementary studies on the ClustGrade-RKO solver pool:

1. **Q-learning factor study** (`run.py` + `aggregate.py`). Compares
   `q_learning=False` vs `q_learning=True` under matched 120 s budgets on
   the same 20-dataset subset used in Experiment 01. The base
   `(quadrat_filter, msi_space)` configuration is inherited from Exp 01's
   `best_config.json`, so Q-learning is the sole varying factor.

2. **Per-metaheuristic comparison** (`run_mh_comparison.py` +
   `aggregate_mh_comparison.py`). Runs each of the three RKO
   metaheuristics — BRKGA, VNS, ILS — in isolation across all 50 datasets
   (20 seeds, 60 s each, Q-learning enabled). Identifies the strongest
   single metaheuristic for downstream scatter comparisons in Exp 03.

## Run

Q-learning study:

```bash
poetry run python -m experiments.02-qlearning-mh-comparison.run
poetry run python -m experiments.02-qlearning-mh-comparison.aggregate
poetry run python -m experiments.02-qlearning-mh-comparison.plot
```

Metaheuristic comparison (long: ≈ 50 h sequential, ≈ 6–7 h with `--jobs 8`):

```bash
poetry run python -m experiments.02-qlearning-mh-comparison.run_mh_comparison --jobs 8
poetry run python -m experiments.02-qlearning-mh-comparison.aggregate_mh_comparison
poetry run python -m experiments.02-qlearning-mh-comparison.plot_mh_comparison
```

If a per-shard subprocess crashes mid-run, the failed shards can be
re-driven with `recover_failed_run_results.py` instead of restarting the
whole comparison.

## Outputs (`artifacts/`)

Q-learning ablation:

- `raw_runs.csv` — one row per `(dataset, q_learning, seed)`.
- `summary_qlearning.csv`, `statistical_tests.csv`, delta plots.

Metaheuristic comparison (under `artifacts/mh_comparison/`):

- `raw_runs.csv` — 3 000 rows (50 × 3 × 20).
- `per_dataset_medians.csv`, `holm_adjusted.csv`,
  `mh_comparison_msi.png`, `mh_comparison_ari_classf.png`.
