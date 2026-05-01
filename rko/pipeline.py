"""
ClustGrade-RKO pipeline.

Preprocessing is computed once per dataset; evaluation runs per candidate
solution decoded from the random-key vector. Imports from src/.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Path bootstrap — add src/ to sys.path for sibling imports.
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from density.cell_mapping import get_cell_mapping
from density.clustering import assess_grid_density, get_neighbor_cells
from grid.validation import quadrat_test
from metrics.clustering import silhouette_score
from prep.mds import compute_mds
from prep.standardization import standardize
from spatial.point_pattern import PointPattern

# Fixed Hopkins merge threshold. Previously exposed as a decoder key; studies
# showed no meaningful variation across values, so it is now a constant.
ALPHA_MERGE: float = 0.05

MsiSpace = Literal["feature", "mds"]


# ---------------------------------------------------------------------------
# Preprocessing (called once per dataset)
# ---------------------------------------------------------------------------


def preprocess(
    data: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], PointPattern]:
    """Standardize -> MDS -> MinMax scale to [0,1]^2.

    Returns:
        scaled_mds: MDS projection scaled to [0,1] per column.
        ppp: PointPattern wrapping the scaled MDS coordinates.
    """
    std_data = standardize(data.astype(float))
    mds_proj = compute_mds(std_data, n_components=2)
    scaler = MinMaxScaler()
    scaled_mds = scaler.fit_transform(mds_proj)
    ppp = PointPattern(scaled_mds)
    return scaled_mds, ppp


def compute_max_cells(n: int) -> int:
    """Return M = floor(sqrt(n)), the max total cells constraint."""
    return int(math.floor(math.sqrt(n)))


def compute_max_axis(M: int) -> int:
    """Return A = floor(M/2), the max divisions per single axis."""
    return max(2, M // 2)


# ---------------------------------------------------------------------------
# Solution evaluation (called per RKO candidate)
# ---------------------------------------------------------------------------


def _score_solution(
    data: npt.NDArray[np.float64],
    scaled_mds: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int64],
    msi_space: MsiSpace,
) -> float | None:
    """Silhouette on either original features or preprocessed [0,1]^2 MDS.

    Noise (label == -1) is excluded before scoring, matching the
    `experiments/_shared.compute_msi` contract.
    """
    mask = labels != -1
    if mask.sum() < 2:
        return None
    valid_labels = labels[mask]
    if len(set(valid_labels.tolist())) <= 1:
        return None
    if msi_space == "mds":
        # silhouette_score re-standardizes; pass raw MDS coords (already in
        # [0,1]^2) so the re-standardization mirrors compute_msi semantics
        # up to a monotonic transform. The ranking is what RKO cares about.
        return float(silhouette_score(scaled_mds[mask], valid_labels))
    return float(silhouette_score(data[mask], valid_labels))


def evaluate_solution(
    data: npt.NDArray[np.float64],
    ppp: PointPattern,
    scaled_mds: npt.NDArray[np.float64],
    xbreaks: npt.NDArray[np.float64],
    ybreaks: npt.NDArray[np.float64],
    *,
    msi_space: MsiSpace = "feature",
    quadrat_filter: bool = True,
    grid_alpha: float = 0.05,
) -> dict[str, Any]:
    """Evaluate a single decoded grid + density solution.

    Args:
        data: Original feature matrix (n, d).
        ppp: PointPattern on scaled MDS coords (for quadrat + density tests).
        scaled_mds: 2D MDS coords scaled to [0,1]^2 — used when msi_space="mds".
        xbreaks, ybreaks: Grid break arrays in [0,1].
        msi_space: "feature" (score on data) or "mds" (score on scaled_mds).
        quadrat_filter: When True, reject grids that fail the quadrat test.
        grid_alpha: Significance level for the quadrat test.

    Returns:
        Dictionary with keys: cost, silhouette, k, labels, quadrat_pval.
    """
    # 1. Validate grid via quadrat test. Passing grids reject the null of CSR.
    pval = quadrat_test(ppp, xbreaks=xbreaks, ybreaks=ybreaks)
    if quadrat_filter and pval > grid_alpha:
        n = data.shape[0]
        return {
            "cost": 1.0,
            "silhouette": -1.0,
            "k": 0,
            "labels": [-1] * n,
            "quadrat_pval": pval,
            "quadrat_pass": False,
        }

    # 2. Density stage: cell mapping -> neighbors -> Hopkins merge -> labels.
    cell_map = get_cell_mapping(ppp, "ASGBRKGA", [xbreaks, ybreaks])
    neighbors = get_neighbor_cells(cell_map["cells_index"], cell_map["which_empty"])
    cluster_vec = assess_grid_density(cell_map, neighbors, "hopkins", ALPHA_MERGE)
    pcm = cell_map["points_cell_mapping"].merge(cluster_vec, on="Index", how="left")
    labels = pcm["Cluster"].fillna(-1).astype(int).to_list()
    labels_arr = np.asarray(labels, dtype=np.int64)

    # 3. All noise -> penalty.
    if all(label == -1 for label in labels):
        return {
            "cost": 1.0,
            "silhouette": -1.0,
            "k": 0,
            "labels": labels,
            "quadrat_pval": pval,
            "quadrat_pass": True,
        }

    # 4. Compute number of non-noise clusters.
    unique_clusters = set(labels)
    k = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    # 5. Score with silhouette in the requested space.
    score = _score_solution(data, scaled_mds, labels_arr, msi_space)
    if score is None:
        return {
            "cost": 1.0,
            "silhouette": -1.0,
            "k": k,
            "labels": labels,
            "quadrat_pval": pval,
            "quadrat_pass": True,
        }

    return {
        "cost": -score,
        "silhouette": score,
        "k": k,
        "labels": labels,
        "quadrat_pval": pval,
        "quadrat_pass": True,
    }
