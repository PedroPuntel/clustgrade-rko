"""
Shared utilities for ClustGrade experiments.

All helpers enforce the noise-exclusion policy (label == -1 is noise),
the seeding strategy, and consistent metric computation across experiments.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score as skl_sil
from sklearn.preprocessing import LabelEncoder

# ---------------------------------------------------------------------------
# Path bootstrap — all experiment scripts import this module first.
# ---------------------------------------------------------------------------
_SRC = Path(__file__).parents[2] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

_ROOT = Path(__file__).parents[2]


def _as_bool(value: Any) -> bool:
    """Parse flexible boolean values stored in manifest CSV fields."""
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def _load_raw_df(row: pd.Series) -> pd.DataFrame:
    """Load raw dataset using manifest-controlled header behavior."""
    from utils.data_loader import load_dataset

    raw_path = _ROOT / row["raw_path"]
    has_header = _as_bool(row.get("has_header", "false"))
    return load_dataset(raw_path, has_header=has_header)


def _resolve_label_col(row: pd.Series, df: pd.DataFrame) -> int | str:
    """Resolve manifest label_col token into a concrete DataFrame column."""
    token = str(row.get("label_col", "last")).strip().lower()
    if token in {"", "last"}:
        return df.columns[-1]
    if token == "first":
        return df.columns[0]
    idx = int(token)
    if idx < 0 or idx >= len(df.columns):
        raise ValueError(f"Invalid label_col index {idx} for dataset {row.get('dataset_id')}")
    return df.columns[idx]


def _extract_features_and_labels(
    row: pd.Series,
) -> tuple[npt.NDArray[np.float64], npt.NDArray[Any] | None]:
    """Extract numeric feature matrix and optional labels from a manifest row."""
    from utils.data_loader import split_features_labels

    df = _load_raw_df(row)
    has_label = _as_bool(row.get("has_label", "false"))

    if has_label:
        label_col = _resolve_label_col(row, df)
        feat_df, labels = split_features_labels(df, label_col=label_col)
    else:
        feat_df = df
        labels = None

    numeric_features = feat_df.apply(pd.to_numeric, errors="coerce")
    keep_cols = numeric_features.notna().any(axis=0)
    numeric_features = numeric_features.loc[:, keep_cols]

    if numeric_features.shape[1] == 0:
        raise ValueError(f"No numeric feature columns found for dataset {row.get('dataset_id')}")
    if numeric_features.isna().any().any():
        bad_cols = numeric_features.columns[numeric_features.isna().any(axis=0)].tolist()
        raise ValueError(
            f"Non-numeric values remain in feature columns {bad_cols} for dataset {row.get('dataset_id')}"
        )

    features = numeric_features.to_numpy(dtype=float)
    label_arr = labels.to_numpy(dtype=object) if labels is not None else None
    return features, label_arr


# ---------------------------------------------------------------------------
# Manifest loading
# ---------------------------------------------------------------------------


def load_manifest() -> pd.DataFrame:
    """Load the canonical dataset manifest."""
    path = Path(__file__).parent / "datasets_manifest.csv"
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _dataset_row(dataset_id: str) -> pd.Series:
    """Fetch a single dataset row from the manifest by ID."""
    manifest = load_manifest()
    match = manifest[manifest["dataset_id"] == dataset_id]
    if match.empty:
        raise ValueError(f"Dataset {dataset_id} not found in manifest")
    return match.iloc[0]


def manifest_clust(df: pd.DataFrame) -> pd.DataFrame:
    """Return clustering-base rows."""
    return df[df["group"] == "clust"].reset_index(drop=True)


def manifest_classf(df: pd.DataFrame) -> pd.DataFrame:
    """Return classification-base rows."""
    return df[df["group"] == "classf"].reset_index(drop=True)


def manifest_subset7(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 7-base stability-analysis subset."""
    return df[df["subset_7"] == "true"].reset_index(drop=True)


# 20-dataset stratified subset for Exp 01 (param study).
# 12 CLUST (size-stratified) + 8 CLASSF (size-stratified).
_SUBSET_20_IDS: set[str] = {
    # CLUST — small
    "RUSPINI",
    "OUTLIERS",
    # CLUST — medium
    "200DATA",
    "FACE",
    "400P3C",
    # CLUST — large
    "CHART",
    "BROKEN-RING",
    "GAUSS9",
    "TRIPADVISOR",
    # CLUST — very large
    "CONCRETEDATA",
    "WAVEFORM21",
    # CLUST — extra (to reach 12)
    "SONAR",
    # CLASSF — small
    "IRIS",
    "WINE",
    # CLASSF — medium
    "ECOLI",
    "JAIN",
    # CLASSF — large
    "WDBC",
    "AGGREGATION",
    "PIMA-INDIANS",
    # CLASSF — very large
    "YEAST",
}


def manifest_subset20(df: pd.DataFrame) -> pd.DataFrame:
    """Return the 20-base stratified subset for param study (Exp 01)."""
    return df[df["dataset_id"].isin(_SUBSET_20_IDS)].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_features(row: pd.Series) -> npt.NDArray[np.float64]:
    """
    Load feature matrix from a manifest row.

    For classf bases the label column is stripped automatically.
    Returns a float64 numpy array of shape (n_samples, n_features).
    """
    features, _ = _extract_features_and_labels(row)
    return features


def load_labels(row: pd.Series) -> npt.NDArray[Any]:
    """
    Load raw (string or numeric) labels for a classf manifest row.

    Returns the label array as-is (preserving string labels for LabelEncoder).
    """
    _, labels = _extract_features_and_labels(row)
    if labels is None:
        raise ValueError(f"Dataset {row.get('dataset_id')} has no labels in manifest")
    return labels


def load_mds(dataset_id: str, group: str) -> npt.NDArray[np.float64]:
    """Load the pre-computed MDS projection for a dataset."""
    if group == "classf":
        path = _ROOT / "data" / "processed" / "classf" / f"{dataset_id}_mds.npy"
    else:
        row = _dataset_row(dataset_id)
        if _as_bool(row.get("subset_7", "false")):
            path = _ROOT / "data" / "processed" / "clust" / "subset" / f"{dataset_id}_mds.npy"
        else:
            path = _ROOT / "data" / "processed" / "clust" / f"{dataset_id}_mds.npy"
    return np.load(path)


def load_stored_labels(dataset_id: str) -> npt.NDArray[Any]:
    """Load the stored ground-truth labels for a classf dataset."""
    path = _ROOT / "data" / "processed" / "classf" / f"{dataset_id}_labels.npy"
    return np.load(path, allow_pickle=True)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def seed_run(seed: int) -> None:
    """
    Seed both Python random and numpy for reproducible stochastic runs.

    ESG BRKGA uses Python random; ASG BRKGA uses both Python random and
    numpy.random. Call this before every stochastic model.fit().
    ESG may route to deterministic BFESGA or stochastic ESGBRKGA depending on
    the runtime gateway threshold.
    """
    random.seed(seed)
    np.random.seed(seed)


# ---------------------------------------------------------------------------
# Metric helpers (noise-exclusion policy enforced here)
# ---------------------------------------------------------------------------


def noise_mask(labels: list[int] | npt.NDArray[np.int64]) -> npt.NDArray[np.bool_]:
    """Return boolean mask of non-noise points (label != -1)."""
    return np.array(labels) != -1


def compute_msi(
    coords: npt.NDArray[np.float64],
    labels: list[int] | npt.NDArray[np.int64],
) -> float | None:
    """
    Compute Mean Silhouette Index on 2D MDS coordinates, noise excluded.

    Returns None when fewer than 2 clusters remain after noise removal.
    """
    arr = np.array(labels)
    mask = arr != -1
    if mask.sum() < 2:
        return None
    valid_labels = arr[mask]
    if len(set(valid_labels)) <= 1:
        return None
    return float(skl_sil(coords[mask], valid_labels, metric="euclidean"))


def compute_ari(
    true_labels: npt.NDArray[Any],
    pred_labels: list[int] | npt.NDArray[np.int64],
) -> float | None:
    """
    Compute Adjusted Rand Index, noise excluded from pred_labels.

    Handles string true_labels via LabelEncoder.
    true_labels and pred_labels must have the same original length (before masking).
    """
    pred_arr = np.array(pred_labels)
    mask = pred_arr != -1
    if mask.sum() == 0:
        return None
    le = LabelEncoder()
    true_int = le.fit_transform(true_labels)
    return float(adjusted_rand_score(true_int[mask], pred_arr[mask]))


def cv(values: list[float] | npt.NDArray[np.float64]) -> float:
    """
    Coefficient of Variation: (std / mean) * 100, using ddof=1 (R-compatible).

    Returns 0.0 when the mean is zero or the input has fewer than 2 elements.
    """
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return 0.0
    mean = float(arr.mean())
    if mean == 0.0:
        return 0.0
    return float(np.std(arr, ddof=1) / mean * 100)


def summary_stats(values: list[float] | npt.NDArray[np.float64]) -> dict[str, float]:
    """Return min, median, max, and CV for a list of values."""
    arr = np.array(values, dtype=float)
    valid = arr[~np.isnan(arr)]
    if len(valid) == 0:
        return {"min": float("nan"), "med": float("nan"), "max": float("nan"), "cv": float("nan")}
    return {
        "min": float(np.min(valid)),
        "med": float(np.median(valid)),
        "max": float(np.max(valid)),
        "cv": cv(valid),
    }
