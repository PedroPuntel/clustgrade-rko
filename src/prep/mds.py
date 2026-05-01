from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt
from scipy.spatial.distance import pdist, squareform

from utils.logging import get_logger

LOGGER = get_logger(__name__)


def compute_mds(data: npt.NDArray[np.float64], n_components: int = 2) -> npt.NDArray[np.float64]:
    """
    Classical MDS using double-centering of the distance matrix.

    Notes:
        O(n^3) due to eigendecomposition of the full distance matrix.
    """

    # Start timing for profiling.
    start = time.perf_counter()
    # Compute pairwise Euclidean distances.
    dist = squareform(pdist(data, metric="euclidean"))
    # Build the double-centering matrix.
    n = dist.shape[0]
    j = np.eye(n) - np.ones((n, n)) / n
    # Apply classical MDS transformation.
    b = -0.5 * j @ (dist**2) @ j
    # Solve for the leading eigenpairs.
    eigvals, eigvecs = np.linalg.eigh(b)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    # Keep non-negative components only.
    positive = np.maximum(eigvals[:n_components], 0.0)
    coords = eigvecs[:, :n_components] * np.sqrt(positive)
    # Log elapsed time for profiling.
    elapsed = time.perf_counter() - start
    LOGGER.info("MDS computed in %.4fs for n=%d", elapsed, n)
    result: npt.NDArray[np.float64] = coords
    return result
