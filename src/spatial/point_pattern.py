from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class PointPattern:
    """
    Lightweight point-pattern object for 2D coordinates.
    """

    coords: npt.NDArray[np.float64]

    @property
    def window(self) -> tuple[tuple[float, float], tuple[float, float]]:
        # Compute axis-aligned bounds for the point pattern.
        x_min = float(self.coords[:, 0].min())
        x_max = float(self.coords[:, 0].max())
        y_min = float(self.coords[:, 1].min())
        y_max = float(self.coords[:, 1].max())
        return (x_min, x_max), (y_min, y_max)

    def quadrat_counts(
        self,
        *,
        nx: int | None = None,
        ny: int | None = None,
        xbreaks: npt.NDArray[np.float64] | None = None,
        ybreaks: npt.NDArray[np.float64] | None = None,
    ) -> npt.NDArray[np.int64]:
        """
        Count points per quadrat using either counts or explicit breaks.
        """

        # Build breaks from grid counts when not provided.
        if xbreaks is None or ybreaks is None:
            if nx is None or ny is None:
                raise ValueError("Provide nx/ny or xbreaks/ybreaks.")
            (x_min, x_max), (y_min, y_max) = self.window
            xbreaks = np.linspace(x_min, x_max, nx + 1)
            ybreaks = np.linspace(y_min, y_max, ny + 1)
        # Compute 2D histogram counts per quadrat.
        hist, _, _ = np.histogram2d(self.coords[:, 0], self.coords[:, 1], bins=[xbreaks, ybreaks])
        return hist.astype(int)
