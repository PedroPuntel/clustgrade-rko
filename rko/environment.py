"""
RKO environment for ClustGrade-RKO.

Maps a fixed-size random key vector in [0,1) to a grid + density solution,
then evaluates via silhouette score. Memoized to guarantee same-keys-same-cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

# Resolve and inject the upstream RKO framework before importing from it.
from rko._framework import bootstrap_framework

bootstrap_framework()

from Environment import RKOEnvAbstract

from rko.pipeline import (
    MsiSpace,
    compute_max_axis,
    compute_max_cells,
    evaluate_solution,
)

if TYPE_CHECKING:
    from spatial.point_pattern import PointPattern


class ClustGradeEnv(RKOEnvAbstract):
    """RKO environment for ClustGrade-RKO clustering.

    Random key vector layout (tam_solution = 2*A):
        [0]          -> nx (x-axis divisions)
        [1]          -> ny (y-axis divisions, constrained by nx)
        [2 .. A]     -> x-axis interior break positions (A-1 keys)
        [A+1 .. 2A-1]-> y-axis interior break positions (A-1 keys)

    Where:
        M = floor(sqrt(n)) = max total cells (nx * ny <= M)
        A = floor(M / 2)   = max divisions per single axis

    The Hopkins merge threshold is fixed at ALPHA_MERGE = 0.05 (see pipeline).
    """

    def __init__(
        self,
        data: npt.NDArray[np.float64],
        ppp: PointPattern,
        scaled_mds: npt.NDArray[np.float64],
        *,
        instance_name: str = "default",
        max_time: int = 60,
        msi_space: MsiSpace = "feature",
        quadrat_filter: bool = True,
        q_learning: bool = False,
    ) -> None:
        super().__init__()

        # Store precomputed objects for cost evaluation.
        self._data = data
        self._ppp = ppp
        self._scaled_mds = scaled_mds
        self._msi_space: MsiSpace = msi_space
        self._quadrat_filter: bool = quadrat_filter

        # Compute vector sizing constraints.
        n = data.shape[0]
        self._M = compute_max_cells(n)
        self._A = compute_max_axis(self._M)

        # RKO interface attributes.
        self.tam_solution: int = 2 * self._A
        self.max_time: int = max_time
        self.LS_type: str = "First"  # faster for expensive cost functions
        self.instance_name: str = instance_name
        self.dict_best: dict = {}
        # Q-learning report is written by RKO only when online tuning is active
        # (i.e. when any parameter dict has multi-valued lists).
        self.save_q_learning_report: bool = q_learning

        # Metaheuristic parameters. Only BRKGA/VNS/ILS are used in the default
        # solver mix; the others are left at base-class defaults (required by
        # RKO's check_env) but are never invoked.
        #
        # When q_learning=True, the three active dicts are expanded to
        # multi-value lists, which triggers RKO's online Q-learning adaptive
        # tuning (see misc/rko/RKO.py::_setup_parameters).
        if q_learning:
            self.BRKGA_parameters = {
                "p": [50, 100, 150, 200, 300],
                "pe": [0.10, 0.15, 0.20, 0.25, 0.30],
                "pm": [0.05, 0.10, 0.15, 0.20, 0.25],
                "rhoe": [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
            }
            self.VNS_parameters = {
                "kMax": [2, 3, 5, 7, 10],
                "betaMin": [0.01, 0.03, 0.05, 0.10, 0.15],
            }
            self.ILS_parameters = {
                "betaMin": [0.01, 0.03, 0.05, 0.10, 0.15],
                "betaMax": [0.10, 0.15, 0.20, 0.25, 0.30],
            }
        else:
            self.BRKGA_parameters = {
                "p": [100],
                "pe": [0.25],
                "pm": [0.10],
                "rhoe": [0.70],
            }
            self.VNS_parameters = {
                "kMax": [5],
                "betaMin": [0.05],
            }
            self.ILS_parameters = {
                "betaMin": [0.10],
                "betaMax": [0.20],
            }

        # Memoization cache: decoded solution tuple -> cost.
        self._cache: dict[tuple, float] = {}
        self._cache_hits: int = 0  # repeated decoded grids that avouided recomputation
        self._eval_count: int = 0  # total calls to cost()
        self._cap_hits: int = (
            0  # How often nx*ny would exceed M (i.e. decoder clamps ny feasibility)
        )

    # ------------------------------------------------------------------
    # Decoder: random keys -> (xbreaks, ybreaks)
    # ------------------------------------------------------------------

    def decoder(self, keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Decode a random key vector into a grid solution.

        Returns:
            (xbreaks, ybreaks) tuple.
        """
        M = self._M
        A = self._A

        # --- Grid dimensions ---
        # nx in [2, A]: maps key[0] from [0,1) to integer range.
        nx = 2 + int(keys[0] * (A - 1))
        nx = min(nx, A)

        # ny in [2, max_ny] where max_ny = min(A, floor(M/nx)).
        max_ny = min(A, M // nx)
        if max_ny < 2:
            max_ny = 2
            self._cap_hits += 1
        ny = 2 + int(keys[1] * (max_ny - 1))
        ny = min(ny, max_ny)

        # --- X-axis breaks ---
        # Interior breaks: first (nx-1) keys from indices [2..A], sorted.
        x_interior_count = nx - 1
        x_keys = keys[2 : 2 + x_interior_count]
        x_interior = np.sort(x_keys)
        xbreaks = np.concatenate([[0.0], x_interior, [1.0]])

        # --- Y-axis breaks ---
        # Interior breaks: first (ny-1) keys from indices [A+1..2A-1], sorted.
        y_start = A + 1
        y_interior_count = ny - 1
        y_keys = keys[y_start : y_start + y_interior_count]
        y_interior = np.sort(y_keys)
        ybreaks = np.concatenate([[0.0], y_interior, [1.0]])

        return xbreaks, ybreaks

    # ------------------------------------------------------------------
    # Cost: evaluate decoded solution (memoized)
    # ------------------------------------------------------------------

    def cost(self, solution: tuple, final_solution: bool = False) -> float:
        """Compute clustering cost for a decoded solution.

        Returns negative silhouette (minimized by RKO) or 1.0 for invalid.
        """
        xbreaks, ybreaks = solution
        self._eval_count += 1

        # Build cache key from decoded arrays.
        cache_key = (tuple(xbreaks), tuple(ybreaks))
        if cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[cache_key]

        # Evaluate the full pipeline.
        result = evaluate_solution(
            self._data,
            self._ppp,
            self._scaled_mds,
            xbreaks,
            ybreaks,
            msi_space=self._msi_space,
            quadrat_filter=self._quadrat_filter,
        )
        cost_val = float(result["cost"])
        self._cache[cache_key] = cost_val
        return cost_val

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def diagnostics(self) -> dict:
        """Return instrumentation counters."""
        return {
            "eval_count": self._eval_count,
            "cache_hits": self._cache_hits,
            "cache_size": len(self._cache),
            "cap_hits": self._cap_hits,
            "M": self._M,
            "A": self._A,
            "tam_solution": self.tam_solution,
        }

    def decode_and_evaluate(self, keys: np.ndarray) -> dict:
        """Full decode + evaluate for final result inspection."""
        xbreaks, ybreaks = self.decoder(keys)
        result = evaluate_solution(
            self._data,
            self._ppp,
            self._scaled_mds,
            xbreaks,
            ybreaks,
            msi_space=self._msi_space,
            quadrat_filter=self._quadrat_filter,
        )
        result["xbreaks"] = xbreaks.tolist()
        result["ybreaks"] = ybreaks.tolist()
        result["nx"] = len(xbreaks) - 1
        result["ny"] = len(ybreaks) - 1
        result["total_cells"] = (len(xbreaks) - 1) * (len(ybreaks) - 1)
        return result
