from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt
from scipy.spatial import KDTree
from scipy.stats import beta, norm


def hopkins_statistic(
    points: npt.NDArray[np.float64],
    m: int | None = None,
    n_runs: int = 1,
    squared: bool = False,
) -> float:
    """Compute the Hopkins statistic for spatial randomness.

    Args:
        points: Coordinate array, shape (n_points, n_dims).
        m: Number of points sampled per run. Defaults to n-1.
            To match R's convention (00_Simulations.R lines 25-53), pass
            ``m=int(n * 0.5)`` with ``n_runs=10``.
        n_runs: Number of independent estimates to average. Defaults to 1.
            R uses 10 repetitions for whole-dataset tendency analysis;
            the density clustering phase uses 1 (per-cell candidate).
        squared: If False (default), sums raw Euclidean distances — matches
            R's reference implementation and is used for whole-dataset
            tendency analysis (e.g. MDS validation).
            If True, sums squared Euclidean distances (d-th power for d=2).
            **Must be True when the result will be passed to**
            ``hopkins_pvalue()``, because the Beta(m, m) null distribution
            is only valid when d-th power distances are used (see Notes).

    Returns:
        Hopkins H statistic in [0, 1]. Values near 1 indicate clustering,
        near 0.5 indicate CSR, near 0 indicate regularity.

    Notes:
        **Distance formula choice and null distribution:**

        The theoretically exact null distribution $H \\sim \\text{Beta}(m, m)$
        under CSR requires d-th power distances. In d dimensions, each
        $u_i^d \\sim \\text{Exp}(\\lambda V_d)$, so $\\sum u_i^d \\sim
        \\text{Gamma}(m, \\lambda V_d)$ and the ratio follows Beta(m, m) exactly.
        For 2D data this means **squared** distances must be used.

        With raw Euclidean distances (``squared=False``), distances follow a
        Rayleigh distribution in 2D — not Exponential — so neither
        $\\text{Beta}(m, m)$ nor $U(0, 1)$ holds for $m > 1$. ($U(0, 1)$ is
        only valid for $m = 1$ with d-th power distances, because
        $\\text{Beta}(1, 1) = U(0, 1)$.)

        Raw distances (``squared=False``) match R's reference
        (00_Simulations.R, calcula_U / calcula_W), which confirmed the
        Pearson correlation between h_orig and h_mds of ~0.637 (vs ~0.453
        with squared distances). Use ``squared=True`` only when a valid
        formal p-value is required via ``hopkins_pvalue()``.
    """
    # Handle degenerate inputs.
    n = points.shape[0]
    if n < 2:
        return 0.5
    # Determine sample size for Hopkins statistic.
    m_val = n - 1 if m is None else max(1, min(m, n - 1))
    # Build KD-tree for nearest-neighbor queries.
    tree = KDTree(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    dims = int(points.shape[1])

    h_vals: list[float] = []
    for _ in range(max(1, n_runs)):
        # Sample random points within the bounding box.
        random_points = np.random.uniform(mins, maxs, size=(int(m_val), dims))
        # Distances from random points to nearest data points.
        u_dist, _ = tree.query(random_points, k=1)
        # Distances from sampled data points to nearest neighbors.
        sample_idx = np.random.choice(n, size=int(m_val), replace=False)
        w_dist, _ = tree.query(points[sample_idx], k=2)
        w_dist = w_dist[:, 1]
        # Apply d-th power (squared for 2D) when a valid Beta p-value is needed.
        if squared:
            u_vals = u_dist**2
            w_vals = w_dist**2
        else:
            u_vals = u_dist  # raw Euclidean — matches R reference
            w_vals = w_dist
        u_sum = float(np.sum(u_vals))
        w_sum = float(np.sum(w_vals))
        denom = u_sum + w_sum
        h_vals.append(u_sum / denom if denom > 0.0 else 0.5)

    return float(np.mean(h_vals))


def hopkins_pvalue(h_stat: float, m: int) -> float:
    """Exact right-tail p-value for Hopkins statistic under CSR.

    Args:
        h_stat: Hopkins H value computed with **squared** (d-th power)
            distances, i.e. ``hopkins_statistic(..., squared=True)``.
            Passing H from raw-distance computation produces an invalid
            p-value because the Beta(m, m) distribution no longer holds
            (see ``hopkins_statistic`` Notes).
        m: Sample size used in the Hopkins computation.

    Returns:
        P(H >= h_stat | CSR) under the Beta(m, m) null distribution.
        Small values indicate significant clustering tendency.

    Notes:
        Under CSR, $H = \\sum u_i^d / (\\sum u_i^d + \\sum w_i^d) \\sim
        \\text{Beta}(m, m)$ when d-th power distances are used (exact result,
        not an approximation). The right tail $P(H \\geq h) = 1 -
        F_{\\text{Beta}(m,m)}(h)$ is the p-value for rejecting CSR in favour
        of clustering.
    """
    # Guard invalid m values.
    if m <= 0:
        return 1.0
    return float(1.0 - beta.cdf(h_stat, m, m))


def clark_evans_test(points: npt.NDArray[np.float64]) -> tuple[float, float]:
    """Clark-Evans test for spatial randomness (Clark & Evans, 1954).

    Tests whether the nearest-neighbour distance distribution departs
    from Complete Spatial Randomness (CSR).

    Under CSR the mean nearest-neighbour distance is asymptotically normal:

        r̄ ~ N( 1/(2√λ),  (4−π)/(4πλn) )

    where λ = n/area is the point density and area is the bounding-box area.
    The SE constant 0.26136 = √(4−π) / √(4π) follows directly.

    Args:
        points: 2-D coordinate array, shape (n_points, 2).

    Returns:
        (p_regular, p_two_sided) where:
        - p_regular   = P(Z ≥ z | CSR) — one-sided right-tail p-value.
                        Small (≤ α) when mean_nn > E(R): points are
                        significantly MORE DISPERSED than CSR → regularity.
        - p_two_sided = P(|Z| ≥ |z| | CSR) — two-sided p-value.
                        Small (≤ α) when the pattern departs from CSR in
                        either direction (clustering OR regularity).

    Notes:
        To test for clustering (mean_nn < E(R), Z < 0), use the left-tail
        p-value: ``p_cluster = 1 - p_regular`` (i.e. norm.cdf(z)).
    """
    # Handle degenerate inputs.
    n = points.shape[0]
    if n < 2:
        return 1.0, 1.0
    # Compute nearest-neighbor distances.
    tree = KDTree(points)
    dists, _ = tree.query(points, k=2)
    nn_dist = dists[:, 1]
    mean_nn = float(nn_dist.mean())
    # Estimate window area from bounding box.
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    area = float(np.prod(maxs - mins))
    if area == 0.0:
        return 1.0, 1.0
    # Expected distance and SE under CSR (formula 3.10 from ENCE monograph).
    density = n / area
    expected = 0.5 / math.sqrt(density)  # E(R) = 1 / (2√λ)
    se = 0.26136 / math.sqrt(n * density)  # SE  = √((4−π)/(4πλn))
    if se == 0.0:
        return 1.0, 1.0
    # Z-score and p-values.
    z = (mean_nn - expected) / se
    p_regular = float(1.0 - norm.cdf(z))  # right-tail: dispersion / regularity
    p_two_sided = float(2.0 * min(norm.cdf(z), 1.0 - norm.cdf(z)))
    return p_regular, p_two_sided
