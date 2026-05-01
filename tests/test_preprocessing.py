from __future__ import annotations

import numpy as np

from src.prep.mds import compute_mds
from src.prep.standardization import standardize


def test_standardize_mean_std() -> None:
    """Z-score standardization: column mean ~0, sample std (ddof=1) ~1."""
    data = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    std = standardize(data)
    assert np.allclose(std.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(std.std(axis=0, ddof=1), 1.0, atol=1e-8)


def test_mds_shape() -> None:
    """MDS returns a stable 2D embedding shape."""
    rng = np.random.default_rng(0)
    data = rng.normal(size=(10, 5))
    proj = compute_mds(data, n_components=2)
    assert proj.shape == (10, 2)
