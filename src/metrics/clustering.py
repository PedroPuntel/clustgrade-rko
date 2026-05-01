from __future__ import annotations

import numpy as np
import numpy.typing as npt
from sklearn.metrics import calinski_harabasz_score
from sklearn.metrics import silhouette_score as skl_silhouette

from prep.standardization import standardize


def silhouette_score(data: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float | None:
    """
    Compute mean silhouette score using standardized data.
    """

    # Guard against single-cluster scoring.
    if len(set(labels)) <= 1:
        return None
    # Standardize features before computing distances.
    std_data = standardize(data.astype(float))
    return float(skl_silhouette(std_data, labels, metric="euclidean"))


def calinski_score(data: npt.NDArray[np.float64], labels: npt.NDArray[np.int64]) -> float | None:
    """
    Compute Calinski-Harabasz score using standardized data.
    """

    # Guard against single-cluster scoring.
    if len(set(labels)) <= 1:
        return None
    # Standardize features before computing the index.
    std_data = standardize(data.astype(float))
    return float(calinski_harabasz_score(std_data, labels))
