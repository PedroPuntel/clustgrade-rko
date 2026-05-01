from __future__ import annotations

import numpy as np
import numpy.typing as npt


def standardize(data: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Standardize columns using z-score normalization.

    Uses sample standard deviation (ddof=1) to match the R implementation.

    Notes:
        O(n x d).
    """

    # Compute per-feature means for centering.
    means = data.mean(axis=0)
    # Compute per-feature standard deviations for scaling.
    stds = data.std(axis=0, ddof=1)  #! Uses ddof=1 to match R implementation
    # Protect against zero-variance features.
    stds[stds == 0.0] = 1.0
    # Return standardized matrix.
    result: npt.NDArray[np.float64] = (data - means) / stds
    return result
