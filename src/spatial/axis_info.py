from __future__ import annotations

import numpy as np
import numpy.typing as npt


def axis_info(
    mds_proj: npt.NDArray[np.float64], axis: int, tol: float = 0.1
) -> npt.NDArray[np.float64]:
    """
    Compute axis statistics used by grid generation.
    """

    # Select axis coordinates.
    coords = mds_proj[:, axis]
    # Compute axis bounds.
    min_coord = float(coords.min())
    max_coord = float(coords.max())
    max_delta = max_coord - min_coord
    # Match R AxisInfo: minimum absolute adjacent gap in original order.
    diffs = np.abs(np.diff(coords))
    # R's min(abs(diff(...))) on empty input yields +Inf (with warning).
    min_delta = float(diffs.min()) if diffs.size > 0 else float("inf")
    # Guard against extremely small gaps.
    if min_delta < tol:
        min_delta = tol
    # Return summary statistics.
    return np.array([min_coord, max_coord, max_delta, min_delta], dtype=float)
