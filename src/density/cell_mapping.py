from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from spatial.point_pattern import PointPattern


def _expand_breaks(breaks: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    # Preserve empty breaks as-is.
    if breaks.size == 0:
        return breaks
    # BUG FIX: The previous implementation used the sign of breaks[0] / breaks[-1]
    # to decide expansion direction, which shifted the minimum boundary *inward*
    # (wrong direction) for positive coordinate ranges, causing boundary points to
    # fall outside the grid and be silently excluded. The unused `coords` parameter
    # was also removed. The minimum boundary must always move left (subtract) and
    # the maximum boundary must always move right (add), regardless of sign.
    noise = np.random.uniform(0.0, 1e-3, size=2)
    breaks = breaks.copy()
    breaks[0] -= noise[0]  # always shrink minimum to capture left-boundary points
    breaks[-1] += noise[1]  # always grow maximum to capture right-boundary points
    return breaks


def get_cell_mapping(
    ppp: PointPattern,
    grid_method: str,
    grid: Any,
) -> dict[str, Any]:
    # Extract MDS coordinates.
    coords = ppp.coords
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]

    # Build grid breaks for ESG or ASG.
    if grid_method == "ASGBRKGA":
        xbreaks = np.asarray(grid[0], dtype=float)
        ybreaks = np.asarray(grid[1], dtype=float)
    else:
        nx, ny = int(grid[0]), int(grid[1])
        x_min, x_max = float(x_coords.min()), float(x_coords.max())
        y_min, y_max = float(y_coords.min()), float(y_coords.max())
        xbreaks = np.linspace(x_min, x_max, nx + 1)
        ybreaks = np.linspace(y_min, y_max, ny + 1)

    # Expand breaks to avoid boundary exclusions.
    xbreaks = _expand_breaks(xbreaks)
    ybreaks = _expand_breaks(ybreaks)

    # Map coordinates to grid indices.
    x_cells = np.digitize(x_coords, xbreaks, right=False)
    y_cells = np.digitize(y_coords, ybreaks, right=False)

    # Compute point counts per grid cell.
    # BUG FIX: np.histogram2d (called inside quadrat_counts) returns shape (nx, ny)
    # — x varies on axis 0, y on axis 1. Transpose so that the matrix is oriented
    # (ny, nx) = (rows=y, cols=x), matching the grid_index_mat convention used by
    # the index lookup `grid_index_mat[y_cells-1, x_cells-1]` on line 74.
    grid_qcount = ppp.quadrat_counts(xbreaks=xbreaks, ybreaks=ybreaks).T
    n_y, n_x = grid_qcount.shape

    # Assign deterministic indices bottom-to-top, left-to-right.
    grid_index_mat = np.zeros((n_y, n_x), dtype=int)
    counter = 1
    for col in range(n_x):
        for row in range(n_y - 1, -1, -1):
            grid_index_mat[row, col] = counter
            counter += 1

    # Track empty cell indices.
    empty_cells = grid_index_mat[grid_qcount == 0].tolist()

    # Build point-level mapping for later density operations.
    index = grid_index_mat[y_cells - 1, x_cells - 1]
    mapping = pd.DataFrame(
        {
            "Xcoord": x_coords,
            "Ycoord": y_coords,
            "Xcell": x_cells,
            "Ycell": y_cells,
            "Index": index,
        }
    )

    # Return both point mapping and grid metadata.
    return {
        "points_cell_mapping": mapping,
        "cells_index": grid_index_mat,
        "which_empty": empty_cells,
    }
