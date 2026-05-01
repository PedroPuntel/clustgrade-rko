# Datasets

50 datasets used in the ClustGrade-RKO evaluation, split into two groups:

- **`raw/clust/`** — 32 clustering datasets (no ground truth). Includes the
  7-dataset stability subset under `raw/clust/subset/`.
- **`raw/classf/`** — 18 classification datasets. Last column of each file is
  the class label.

## Pre-computed projections

Every dataset has a 2D classical-MDS projection cached to avoid re-running
the O(n³) MDS at every experiment. These live under `processed/`:

- `processed/clustering/<ID>_mds.npy`
- `processed/classification/<ID>_mds.npy`
- `processed/classification/clusters/<ID>_labels.npy`

## Manifest

The single source of truth is `experiments/_shared/datasets_manifest.csv`
(shipped with the experiments tree). It lists every dataset with its
group, file path, expected dimensionality, and the cluster-count hint when
applicable.

## Provenance

These datasets are well-known benchmarks from the clustering and machine
learning literature. Original references are listed in the SBPO article.
This repository ships a working copy in CSV/TXT form for reproducibility;
no preprocessing other than feature/label separation has been applied to
the raw versions.
