# ClustGrade-RKO

Hybrid grid + density clustering wrapped as a Random-Key Optimizer.

ClustGrade-RKO discretizes a 2D MDS projection of the input data into an
adaptive grid, computes per-cell density and spatial-statistics features,
and runs DBSCAN-style cell agglomeration. Grid construction is searched by
a pool of cooperating metaheuristics — BRKGA, VNS, and ILS — with optional
Q-learning adaptive parameter selection. The published evaluation shows
statistical parity with DBSCAN-DistK across 50 datasets while removing the
per-dataset `epsilon` hyperparameter from the user's hands.

## Quick start

ClustGrade-RKO uses the upstream **Random-Key Optimizer** framework. That
framework is distributed separately and is not vendored in this
repository. Clone it alongside this repo and point the
`RKO_FRAMEWORK_PATH` environment variable at the directory that contains
its `RKO.py` and `Environment.py` source files.

```bash
# 1. Get the RKO framework (one-time setup)
git clone <RKO-framework-public-url> /path/to/rko-framework
export RKO_FRAMEWORK_PATH=/path/to/rko-framework      # bash / zsh
# Windows PowerShell: $env:RKO_FRAMEWORK_PATH = "C:\path\to\rko-framework"

# 2. Install this project
poetry install

# 3. Smoke test
poetry run pytest -q

# 4. Reproduce experiments (in order)
poetry run python -m experiments.01-param-study.run
poetry run python -m experiments.01-param-study.aggregate

poetry run python -m experiments.02-qlearning-mh-comparison.run
poetry run python -m experiments.02-qlearning-mh-comparison.aggregate

poetry run python -m experiments.03-vs-dbscan-distk.run_dbscan_distk
poetry run python -m experiments.03-vs-dbscan-distk.run
poetry run python -m experiments.03-vs-dbscan-distk.aggregate
```

## Repository layout

```bash
clustgrade-rko/
├── rko/             # ClustGrade-RKO algorithm: environment, pipeline, runner
├── src/             # Supporting modules (preprocessing, grid, density, metrics)
├── data/            # 50 datasets (32 clustering + 18 classification) and MDS projections
├── tests/           # pytest suite
├── configs/         # Per-experiment YAML configs
├── experiments/
│   ├── 01-param-study/
│   ├── 02-qlearning-mh-comparison/
│   └── 03-vs-dbscan-distk/
└── pyproject.toml
```

## Algorithm at a glance

1. Standardize features → Classical MDS to 2D → MinMax scale to `[0,1]^2`.
2. Encode candidate grids as random-key vectors and decode adaptively (ASG).
3. For each candidate grid: compute per-cell Hopkins statistic, agglomerate
   density-similar cells (DBSCAN over cell features), score the resulting
   clustering by silhouette on the scaled MDS.
4. Drive the search with a pool of BRKGA + VNS + ILS, optionally with
   Q-learning over the parameter grid. The pool shares solutions via a
   common elite set.

## Reproducing the paper results

The artifacts/ subdirectories shipped under each experiment are the canonical
outputs used in the SBPO article. Re-running the scripts above will recompute
those artifacts with identical numerical seeds and therefore (modulo
floating-point determinism) identical figures and tables.
