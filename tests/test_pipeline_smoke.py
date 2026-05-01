"""Pipeline-level smoke test for ClustGrade-RKO.

Exercises the public surface of ``rko.pipeline`` without pulling in the
upstream RKO framework. A two-cluster Gaussian dataset is preprocessed and
evaluated against a hand-picked 2x2 grid; the result must be a valid
solution (finite silhouette, non-empty labels, expected dictionary shape).
"""

from __future__ import annotations

import numpy as np

from rko.pipeline import evaluate_solution, preprocess


def _two_cluster_data(seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(loc=-2.0, scale=0.3, size=(40, 4))
    b = rng.normal(loc=2.0, scale=0.3, size=(40, 4))
    return np.vstack([a, b])


def test_preprocess_outputs_unit_square() -> None:
    data = _two_cluster_data(seed=0)
    scaled_mds, ppp = preprocess(data)
    assert scaled_mds.shape == (80, 2)
    assert scaled_mds.min() >= 0.0 and scaled_mds.max() <= 1.0
    assert ppp.coords.shape == (80, 2)


def test_evaluate_solution_smoke() -> None:
    data = _two_cluster_data(seed=1)
    scaled_mds, ppp = preprocess(data)

    xbreaks = np.array([0.0, 0.5, 1.0])
    ybreaks = np.array([0.0, 0.5, 1.0])

    result = evaluate_solution(
        data=data,
        ppp=ppp,
        scaled_mds=scaled_mds,
        xbreaks=xbreaks,
        ybreaks=ybreaks,
        msi_space="mds",
        quadrat_filter=False,
    )

    assert set(result.keys()) >= {"cost", "silhouette", "k", "labels"}
    assert isinstance(result["k"], int) and result["k"] >= 0
    assert len(result["labels"]) == data.shape[0]
    assert np.isfinite(result["silhouette"])
