from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from density.cell_mapping import get_cell_mapping
from metrics.clustering import calinski_score, silhouette_score
from spatial.point_pattern import PointPattern
from spatial.statistics import clark_evans_test, hopkins_pvalue, hopkins_statistic


def get_neighbor_cells(
    grid_index_mat: npt.NDArray[np.int64], empty_cells: list[int]
) -> dict[int, list[int]]:
    """Build 8-neighborhood adjacency lists for each non-empty grid cell.

    Each cell is connected to all valid positions in its Moore neighborhood
    (3x3 window centered at the cell, including itself). Empty cells are not
    used as dictionary keys, but they may still appear inside neighbor lists
    when they are adjacent to a non-empty cell, preserving the legacy behavior.

    Args:
        grid_index_mat: Matrix of cell ids, shape ``(n_rows, n_cols)``.
        empty_cells: Cell ids that should not be keys in the output mapping.

    Returns:
        Dictionary ``cell_id -> sorted neighbor cell ids``.

    Notes:
        Time complexity is ``O(n_rows * n_cols)`` with a constant-size
        3x3 neighborhood scan for each visited matrix position.
    """
    # Cache grid dimensions for bounds checks.
    n_rows, n_cols = grid_index_mat.shape
    # Use a set for O(1) membership checks when skipping empty cells.
    empty_cell_ids = set(empty_cells)
    # Output map: key = source cell id, value = sorted adjacent ids.
    neighbor_map: dict[int, list[int]] = {}

    # Scan every grid position once.
    for row_idx in range(n_rows):
        # Move left-to-right for the current row.
        for col_idx in range(n_cols):
            # Resolve the cell id stored at the current matrix location.
            source_cell_id = int(grid_index_mat[row_idx, col_idx])
            # Skip cells marked as empty: they do not become adjacency keys.
            if source_cell_id in empty_cell_ids:
                continue

            # Collect neighbors from the local 3x3 window (including self).
            neighbor_ids: list[int] = []
            # Row offset in {-1, 0, +1} around current row.
            for row_offset in (-1, 0, 1):
                # Column offset in {-1, 0, +1} around current column.
                for col_offset in (-1, 0, 1):
                    # Compute candidate neighbor coordinates.
                    neighbor_row = row_idx + row_offset
                    neighbor_col = col_idx + col_offset
                    # Keep only in-bounds coordinates.
                    if 0 <= neighbor_row < n_rows and 0 <= neighbor_col < n_cols:
                        # Append neighbor cell id from the valid matrix position.
                        neighbor_ids.append(int(grid_index_mat[neighbor_row, neighbor_col]))

            # Keep sorted neighbors (including duplicates), matching R behavior.
            neighbor_map[source_cell_id] = sorted(neighbor_ids)

    # Return adjacency list for all non-empty source cells.
    return neighbor_map


def assess_cluster_tendency(
    mapped: pd.DataFrame,
    cells_to_merge: list[int],
    density_test: str,
    alpha: float,
) -> bool | int:
    # Filter points within the candidate cells.
    aux = mapped[mapped["Index"].isin(cells_to_merge)][["Xcoord", "Ycoord"]].to_numpy()
    # This is coherent with DBSCAN's MinPts=5 default parameter
    if aux.shape[0] <= 5:
        return -1
    if density_test == "hopkins":
        # Use squared distances (d=2 for 2D data) so H ~ Beta(m, m) under CSR,
        # which is required for hopkins_pvalue() to return a valid p-value.
        m = aux.shape[0] - 1
        h_stat = hopkins_statistic(aux, m=m, squared=True)
        pval = hopkins_pvalue(h_stat, m)
        return bool(pval > alpha)

    # Clark-Evans regularity or two-sided CSR tests.
    p_regular, p_two_sided = clark_evans_test(aux)
    return bool((p_regular <= alpha) or (p_two_sided > alpha))


def assess_grid_density(
    grid_mapping: dict[str, Any],
    cell_neighborhood: dict[int, list[int]],
    density_test: str,
    alpha: float,
) -> pd.DataFrame:
    # Start from non-empty cells only.
    to_visit = sorted(set(grid_mapping["cells_index"].ravel()) - set(grid_mapping["which_empty"]))

    # Union-find structure for merged cells.
    parent = {cell: cell for cell in to_visit}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for cell in to_visit:
        # Collect valid neighbors for this cell.
        neighbors = [
            n for n in cell_neighborhood.get(cell, []) if n not in grid_mapping["which_empty"]
        ]
        if not neighbors:
            # Isolated cells are evaluated in the second pass below.
            continue
        # Attempt to merge with neighbors when homogeneity is supported.
        for neighbor in neighbors:
            flag_merge = assess_cluster_tendency(
                grid_mapping["points_cell_mapping"], [cell, neighbor], density_test, alpha
            )
            if flag_merge is True:
                union(cell, neighbor)

    # Assign dense cluster labels from union-find roots.
    cluster_map: dict[int, int] = {}
    next_cluster = 1
    for cell in to_visit:
        root = find(cell)
        if root not in cluster_map:
            cluster_map[root] = next_cluster
            next_cluster += 1

    # Build cluster table aligned to cell indices.
    cluster_vec = pd.DataFrame(
        {"Index": to_visit, "Cluster": [cluster_map[find(cell)] for cell in to_visit]}
    )

    # Re-evaluate each cell as clustered or noise.
    clustered_flags = []
    for cell in cluster_vec["Index"].to_list():
        eval_flag = assess_cluster_tendency(
            grid_mapping["points_cell_mapping"], [cell], density_test, alpha
        )
        clustered_flags.append(-1 if eval_flag == -1 else (not eval_flag))

    # Apply noise labels.
    cluster_vec["Clustered"] = clustered_flags
    cluster_vec["Cluster"] = np.where(cluster_vec["Clustered"] == -1, -1, cluster_vec["Cluster"])
    cluster_vec = cluster_vec.drop(columns=["Clustered"])
    return cluster_vec


def assess_cluster_quality(
    data: npt.NDArray[np.float64],
    mapping: pd.DataFrame,
    clust_fobj: str,
) -> float | None:
    # Remove noise points before scoring.
    if (mapping["Cluster"] == -1).any():
        keep = mapping["Cluster"] != -1
        mapping = mapping.loc[keep, ["Xcoord", "Ycoord", "Cluster"]]
        data = data[keep.to_numpy()]
    else:
        mapping = mapping[["Xcoord", "Ycoord", "Cluster"]]

    if mapping.empty:
        return None

    # Compute the chosen cluster quality metric.
    labels = mapping["Cluster"].to_numpy()
    if clust_fobj == "silhouette":
        return float(silhouette_score(data, labels))
    return float(calinski_score(data, labels))


def dgbclust(
    data: npt.NDArray[np.float64],
    ppp: PointPattern,
    elite_grids: list[Any],
    elite_scores: list[float],
    grid_method: str,
    *,
    alpha: float = 0.05,
    density_test: str = "clarkevans",
    clust_fobj: str = "silhouette",
) -> dict[str, Any]:
    """
    Run density-stage clustering over elite grid candidates.

    Notes:
        When `only_ics=True` is used in the grid-search stage, elite grids may be
        selected purely by ICS and later produce all-noise labels under the
        density test (especially with `density_test="hopkins"` on sparse/noisy
        data). This is treated as a valid output (`k=0`, `score=None`) rather
        than an exception.
    """
    # BFESGA yields a single grid candidate.
    if grid_method == "BFESGA":
        elite_grids = [elite_grids[0]]

    # Map points to cells and compute neighborhood structure.
    grids_cell_mapping = [get_cell_mapping(ppp, grid_method, grid) for grid in elite_grids]
    grids_neighbor_cells = [
        get_neighbor_cells(g["cells_index"], g["which_empty"]) for g in grids_cell_mapping
    ]
    # Merge cells based on density tests.
    grids_cluster_mapping = [
        assess_grid_density(grids_cell_mapping[i], grids_neighbor_cells[i], density_test, alpha)
        for i in range(len(grids_cell_mapping))
    ]

    # Map points to their cluster assignments.
    points_cluster_mapping = [
        grids_cell_mapping[i]["points_cell_mapping"].merge(
            grids_cluster_mapping[i], on="Index", how="left"
        )
        for i in range(len(grids_cell_mapping))
    ]

    # Score each grid candidate and select the best.
    scores = [assess_cluster_quality(data, pcm, clust_fobj) for pcm in points_cluster_mapping]
    score_array = np.array([s if s is not None else np.nan for s in scores], dtype=float)
    valid_scores = [s for s in scores if s is not None]
    best_idx = int(np.nanargmax(score_array)) if valid_scores else 0

    best_mapping = points_cluster_mapping[best_idx]
    labels = best_mapping["Cluster"].fillna(-1).astype(int).to_list()
    if all(label == -1 for label in labels):
        return {
            "cluster": labels,
            "score": None,
            "k": 0,
            "dist": dict(pd.Series(labels).value_counts()),
            "best_grid": elite_grids[best_idx],
            "best_grid_ics": elite_scores[best_idx] if elite_scores else None,
            "grid_method": grid_method,
            "spatial_ppp_obj": ppp,
        }

    # Compute number of non-noise clusters.
    unique_clusters = set(labels)
    k = len(unique_clusters) - (1 if -1 in unique_clusters else 0)

    return {
        "cluster": labels,
        "score": scores[best_idx] if valid_scores else None,
        "k": k,
        "dist": dict(pd.Series(labels).value_counts()),
        "best_grid": elite_grids[best_idx],
        "best_grid_ics": elite_scores[best_idx] if elite_scores else None,
        "grid_method": grid_method,
        "spatial_ppp_obj": ppp,
    }
