"""
Experiment 03 — DBSCAN-DistK on MinMax-scaled MDS space.

Semaan (2012) DistK epsilon estimation. For each dataset, sweep a small set
of k* values and, for each k*, extract 4 candidate epsilons via distinct
rules applied to the sorted k*-NN distance vector (Vdist). Each (k*, rule,
epsilon) triple is evaluated with DBSCAN; the triple with the maximum
silhouette (on non-noise points, subject to K >= 2) wins.

MinPts is fixed at 5 across all DBSCAN configurations in this experiment
(KDist, GridSearch, DistK) for cross-method comparability. This departs
from Semaan's original formulation where qtdeObjetos = k*.

Candidate rules (per k*):
    median  : np.median(Vdist)
    max     : np.max(Vdist)
    pico10  : split Vdist into 10 equal-count parts via np.array_split;
              epsilon = average of the two boundary values at the split
              with the largest gap between consecutive parts.
    pico20  : identical to pico10 with 20 parts.

Outputs:
    configs/exp03_epsilons_distk.yaml
    artifacts/dbscan_distk_runs.csv
    artifacts/distk_scaled/<dataset_id>_distk_scaled.png

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.run_dbscan_distk
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from experiments._shared.utils import (
    compute_ari,
    compute_msi,
    load_manifest,
    load_mds,
    load_stored_labels,
)
from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

DISTK_PLOT_DIR = ARTIFACTS / "distk_scaled"
DISTK_PLOT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = _ROOT / "configs"
EPSILON_CONFIG = CONFIGS / "exp03_epsilons_distk.yaml"

MIN_PTS = 5
ALGORITHM = "DBSCAN-DistK"
K_STAR_SET: list[int] = [3, 4, 5, 10, 15, 20, 50]
RULES: list[str] = ["median", "max", "pico10", "pico20"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def scale_mds(mds: np.ndarray) -> np.ndarray:
    scaler = MinMaxScaler()
    return scaler.fit_transform(mds)


def kstar_distances(scaled_mds: np.ndarray, k_star: int) -> np.ndarray:
    """Sorted k*-NN distances on scaled MDS (Vdist in Semaan's notation)."""
    nbrs = NearestNeighbors(n_neighbors=k_star + 1).fit(scaled_mds)
    dists, _ = nbrs.kneighbors(scaled_mds)
    return np.sort(dists[:, -1])


def pico_epsilon(vdist: np.ndarray, n_parts: int) -> float | None:
    """Pico rule: split Vdist into n_parts equal-count chunks, find the pair
    of consecutive parts with the largest gap between their boundary values
    (last value of part i vs first value of part i+1), and return the average
    of those two boundary values.
    """
    if vdist.size < n_parts + 1:
        return None
    parts = np.array_split(vdist, n_parts)
    # Drop any empty splits defensively (shouldn't happen given size check).
    parts = [p for p in parts if p.size > 0]
    if len(parts) < 2:
        return None
    best_gap = -np.inf
    best_avg: float | None = None
    for i in range(len(parts) - 1):
        left_boundary = float(parts[i][-1])
        right_boundary = float(parts[i + 1][0])
        gap = right_boundary - left_boundary
        if gap > best_gap:
            best_gap = gap
            best_avg = 0.5 * (left_boundary + right_boundary)
    return best_avg


def rule_epsilon(vdist: np.ndarray, rule: str) -> float | None:
    if vdist.size == 0:
        return None
    if rule == "median":
        return float(np.median(vdist))
    if rule == "max":
        return float(np.max(vdist))
    if rule == "pico10":
        return pico_epsilon(vdist, 10)
    if rule == "pico20":
        return pico_epsilon(vdist, 20)
    raise ValueError(f"Unknown rule: {rule}")


def evaluate_epsilon(
    scaled_mds: np.ndarray, eps: float,
) -> tuple[np.ndarray | None, int, float | None]:
    """Run DBSCAN and compute silhouette on non-noise points."""
    labels = DBSCAN(eps=eps, min_samples=MIN_PTS).fit_predict(scaled_mds)
    unique = set(labels.tolist())
    k = len(unique) - (1 if -1 in unique else 0)
    if k < 2:
        return labels, k, None
    mask = labels != -1
    if int(np.sum(mask)) < 2 or len(set(labels[mask].tolist())) < 2:
        return labels, k, None
    try:
        sil = float(silhouette_score(scaled_mds[mask], labels[mask]))
    except ValueError:
        sil = None
    return labels, k, sil


def select_best_triple(
    scaled_mds: np.ndarray,
) -> tuple[int | None, str | None, float | None, np.ndarray | None, int, float | None, np.ndarray | None]:
    """Sweep all (k*, rule) triples, pick the one with max silhouette.

    Returns (k_star, rule, eps, labels, k, silhouette, winning_vdist). If no
    candidate yields K>=2 with a valid silhouette, returns
    (None, None, None, None, 0, None, None).
    """
    best_k_star: int | None = None
    best_rule: str | None = None
    best_eps: float | None = None
    best_labels: np.ndarray | None = None
    best_k = 0
    best_sil: float | None = None
    best_vdist: np.ndarray | None = None

    vdist_cache: dict[int, np.ndarray] = {}

    for k_star in K_STAR_SET:
        if scaled_mds.shape[0] <= k_star:
            logger.info("    skip k*=%d (n_points=%d)", k_star, scaled_mds.shape[0])
            continue
        vdist = kstar_distances(scaled_mds, k_star)
        vdist_cache[k_star] = vdist
        for rule in RULES:
            eps = rule_epsilon(vdist, rule)
            if eps is None or eps <= 0:
                continue
            labels, k, sil = evaluate_epsilon(scaled_mds, eps)
            if sil is None:
                continue
            if best_sil is None or sil > best_sil:
                best_sil = sil
                best_k_star = k_star
                best_rule = rule
                best_eps = eps
                best_labels = labels
                best_k = k
                best_vdist = vdist

    return best_k_star, best_rule, best_eps, best_labels, best_k, best_sil, best_vdist


# ---------------------------------------------------------------------------
# YAML writer
# ---------------------------------------------------------------------------

def write_epsilons_yaml(
    path: Path,
    selections: dict[str, dict[str, float | int | str | None]],
) -> None:
    """Write epsilon config with header, one block per dataset."""
    lines = [
        "# Experiment 03 — DBSCAN-DistK (Semaan 2012) on MinMax-scaled MDS [0,1]^2 space",
        f"# k* sweep: {K_STAR_SET}",
        f"# Rules: {RULES}",
        f"# MinPts fixed at {MIN_PTS} for cross-method comparability (departs from Semaan's qtdeObjetos=k*).",
        "# Selection: max silhouette on scaled MDS with labels != -1, subject to K >= 2.",
    ]
    for key in sorted(selections.keys()):
        sel = selections[key]
        eps = sel["epsilon"]
        if eps is None:
            lines.append(f"{key}:")
            lines.append("  epsilon: null")
            lines.append("  k_star: null")
            lines.append("  rule: null")
        else:
            lines.append(f"{key}:")
            lines.append(f"  epsilon: {float(eps):.6f}")
            lines.append(f"  k_star: {sel['k_star']}")
            lines.append(f"  rule: {sel['rule']}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def plot_distk(
    dataset_id: str,
    vdist: np.ndarray | None,
    k_star: int | None,
    rule: str | None,
    eps: float | None,
) -> None:
    """Per-dataset sorted k*-NN curve (winning k*) with selected epsilon overlaid."""
    fig, ax = plt.subplots(figsize=(6, 3))
    if vdist is None or k_star is None:
        ax.text(0.5, 0.5, "No valid candidate (K<2 for all triples)",
                ha="center", va="center", transform=ax.transAxes, fontsize=10)
        ax.set_title(f"{dataset_id} — DBSCAN-DistK: no selection")
    else:
        ax.plot(vdist, color="tab:blue", linewidth=1.2)
        if eps is not None:
            ax.axhline(eps, color="tab:green", linestyle="-", linewidth=1.2,
                       label=f"eps={eps:.4f} (rule={rule})")
            ax.legend(loc="best", fontsize=8)
        ax.set_title(f"{dataset_id} — k*={k_star}-NN curve [scaled MDS, DistK]")
        ax.set_xlabel("Points (sorted)")
        ax.set_ylabel(f"{k_star}-NN distance")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = DISTK_PLOT_DIR / f"{dataset_id}_distk_scaled.png"
    fig.savefig(out, dpi=100)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    manifest = load_manifest()
    records: list[dict] = []
    selections: dict[str, dict[str, float | int | str | None]] = {}

    for idx, row in manifest.iterrows():
        dataset_id: str = row["dataset_id"]
        group: str = row["group"]

        logger.info("=== [%d/%d] %s [%s] ===", idx + 1, len(manifest), dataset_id, group)

        try:
            mds = load_mds(dataset_id, group)
            scaled = scale_mds(mds)
            true_labels = load_stored_labels(dataset_id) if group == "classf" else None

            (best_k_star, best_rule, best_eps, best_labels, best_k,
             best_sil, best_vdist) = select_best_triple(scaled)

            selections[dataset_id] = {
                "epsilon": best_eps,
                "k_star": best_k_star,
                "rule": best_rule,
            }

            plot_distk(dataset_id, best_vdist, best_k_star, best_rule, best_eps)

            if best_eps is None or best_labels is None:
                logger.warning("  No (k*, rule) triple yielded K>=2 — recording as null")
                records.append({
                    "dataset_id": dataset_id, "group": group,
                    "algorithm": ALGORITHM, "seed": 0,
                    "k": None, "msi": None, "ari": None,
                    "k_star": None, "rule": None, "epsilon": None,
                })
                continue

            msi = compute_msi(scaled, best_labels)
            ari = compute_ari(true_labels, best_labels) if true_labels is not None else None
            records.append({
                "dataset_id": dataset_id, "group": group,
                "algorithm": ALGORITHM, "seed": 0,
                "k": best_k,
                "msi": msi,
                "ari": ari,
                "k_star": best_k_star,
                "rule": best_rule,
                "epsilon": round(float(best_eps), 6),
            })
            logger.info(
                "  DBSCAN-DistK(k*=%d, rule=%s, eps=%.4f)  K=%d  sil=%s  MSI=%s  ARI=%s",
                best_k_star, best_rule, best_eps, best_k,
                f"{best_sil:.4f}" if best_sil is not None else "N/A",
                f"{msi:.4f}" if msi is not None else "N/A",
                f"{ari:.4f}" if ari is not None else "N/A",
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  FAILED %s: %s", dataset_id, exc)
            selections[dataset_id] = {"epsilon": None, "k_star": None, "rule": None}
            records.append({
                "dataset_id": dataset_id, "group": group,
                "algorithm": ALGORITHM, "seed": 0,
                "k": None, "msi": None, "ari": None,
                "k_star": None, "rule": None, "epsilon": None,
            })

    write_epsilons_yaml(EPSILON_CONFIG, selections)
    out_path = ARTIFACTS / "dbscan_distk_runs.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)

    logger.info("Saved %d rows -> %s", len(records), out_path)
    logger.info("Saved epsilon config -> %s", EPSILON_CONFIG)
    logger.info("Saved %d plots -> %s", len(manifest), DISTK_PLOT_DIR)


if __name__ == "__main__":
    main()
