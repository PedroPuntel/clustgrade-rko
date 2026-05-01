# Experiment 03 — ClustGrade-RKO vs DBSCAN-DistK

Head-to-head comparison between ClustGrade-RKO and the DBSCAN-DistK
baseline (Semaan, 2012) on 50 datasets. Both algorithms operate on the
same MinMax-scaled `[0,1]^2` MDS coordinates and are scored by
silhouette-on-MDS (MSI) plus ARI-against-truth on the 18 classification
datasets.

## Run

DBSCAN-DistK epsilon selection (per-dataset `k* × rule` sweep, MinPts = 5,
selection by max silhouette):

```bash
poetry run python -m experiments.03-vs-dbscan-distk.run_dbscan_distk
```

ClustGrade-RKO (Q-learning ON, base config inherited from Exp 01,
20 seeds × 50 datasets × 60 s):

```bash
poetry run python -m experiments.03-vs-dbscan-distk.run --jobs 4
```

Aggregation (Wilcoxon paired tests, win/loss/tie, gap statistics) and
plots:

```bash
poetry run python -m experiments.03-vs-dbscan-distk.aggregate
poetry run python -m experiments.03-vs-dbscan-distk.plot
poetry run python -m experiments.03-vs-dbscan-distk.plot_mh_vs_dbscan
```

`run_mh_vs_dbscan.py` is a companion analysis that benchmarks each
individual metaheuristic from Exp 02 against DBSCAN-DistK on the same
50 datasets.

## Outputs (`artifacts/`)

- `dbscan_distk_runs.csv` — 50 rows (one DBSCAN-DistK run per dataset).
- `rko_runs.csv` — 1 000 rows (50 datasets × 20 seeds, ClustGrade-RKO).
- `comparison.csv` — wide per-dataset join: best-of-20 RKO vs DBSCAN-DistK,
  MSI/ARI deltas. Drives Tables 4 and following in the article.
- `wilcoxon_tests.csv`, `win_loss_tie.csv`, `gap_summary.csv`.
- `plot-01..06-*.png` — paired scatter panels on top-MSI / top-ARI deltas.
- `mh_vs_dbscan/` — best-MH vs DBSCAN-DistK panels and gap stats.
- `distk_scaled/<dataset_id>_distk_scaled.png` — DistK k-NN distance plot
  per dataset.

## Configuration

Per-dataset DBSCAN epsilons live in
`configs/exp03_epsilons_distk.yaml`. The file is regenerated from scratch
by `run_dbscan_distk.py`.
