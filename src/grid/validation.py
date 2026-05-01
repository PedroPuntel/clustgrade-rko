from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.stats import chisquare

from spatial.point_pattern import PointPattern


def check_chisqr(
    ppp: PointPattern,
    *,
    nx: int | None = None,
    ny: int | None = None,
    xbreaks: npt.NDArray[np.float64] | None = None,
    ybreaks: npt.NDArray[np.float64] | None = None,
) -> bool:
    """
    Check chi-square constraints for quadrat tests.
    """

    # Count points per cell and check minimum requirements.
    counts = ppp.quadrat_counts(nx=nx, ny=ny, xbreaks=xbreaks, ybreaks=ybreaks).ravel()
    m = counts.size
    if m <= 6:
        return False
    # Enforce mean points per cell threshold.
    mean = counts.mean() if m > 0 else 0.0
    return mean > 1.0


def quadrat_test(
    ppp: PointPattern,
    *,
    nx: int | None = None,
    ny: int | None = None,
    xbreaks: npt.NDArray[np.float64] | None = None,
    ybreaks: npt.NDArray[np.float64] | None = None,
) -> float:
    """
    Compute p-value for quadrat chi-square test.
    """

    # Compute quadrat counts for the chi-square test.
    counts = ppp.quadrat_counts(nx=nx, ny=ny, xbreaks=xbreaks, ybreaks=ybreaks).ravel()
    if counts.size == 0:
        return 1.0
    # Build expected frequencies using the observed mean.
    expected = np.full_like(counts, counts.mean(), dtype=float)
    stat = chisquare(counts, f_exp=expected)
    return float(stat.pvalue)
