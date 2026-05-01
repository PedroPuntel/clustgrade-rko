"""
Experiment 03 — Visual comparison of top-delta clustering solutions (RKO vs DBSCAN-DistK).

DBSCAN-DistK (Semaan 2012) is the primary DBSCAN baseline (per-dataset epsilon chosen
from a k* x rule sweep by silhouette-on-scaled-MDS), so all side-by-side plots benchmark
ClustGrade-RKO against it specifically.

Identifies the 5 datasets with the highest absolute MSI delta and the 5 with
the highest absolute ARI delta between ClustGrade-RKO and DBSCAN-DistK. For each,
reruns both algorithms (using the best-MSI seed for RKO, deterministic for DBSCAN)
and produces paired scatter plots on MinMax-scaled [0,1]^2 MDS coordinates.

Both algorithms operate on the same [0,1]^2 space. DBSCAN-DistK uses epsilon
values from configs/exp03_epsilons_distk.yaml.

Produces:
    plot-01-rko-top5-msi.png     — RKO scatter on top-5 MSI-delta datasets
    plot-02-dbscan-top5-msi.png  — DBSCAN-DistK scatter on same datasets
    plot-03-rko-top5-ari.png     — RKO scatter on top-5 ARI-delta datasets (CLASSF)
    plot-04-dbscan-top5-ari.png  — DBSCAN-DistK scatter on same datasets
    plot-05-dbscan-top5-msi.png  — DBSCAN-DistK scatter on top-5 where DBSCAN > RKO (MSI)
    plot-06-rko-top5-msi-loss.png — RKO scatter on same datasets

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.plot
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"
_RKO_FW = _ROOT / "misc" / "rko"

for p in [str(_SRC), str(_RKO_FW), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RKO import RKO

from rko.environment import ClustGradeEnv
from rko.pipeline import preprocess
from utils.logging import get_logger

from experiments._shared.utils import (
    compute_ari,
    compute_msi,
    load_features,
    load_manifest,
    load_mds,
    load_stored_labels,
    seed_run,
)

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
CONFIGS = _ROOT / "configs"

PANEL_COUNT = 5
MIN_PTS = 5
RKO_TIME_BUDGET = 60
RKO_MIX = {"brkga": 1, "vns": 1, "ils": 1}

# Inherit RKO base config from Exp 01 (kept consistent with run.py).
_BEST_CONFIG_PATH = Path(__file__).parents[1] / "01-param-study" / "artifacts" / "best_config.json"
with open(_BEST_CONFIG_PATH) as f:
    _BEST_CONFIG = json.load(f)
QUADRAT_FILTER: bool = _BEST_CONFIG["quadrat_filter"]
MSI_SPACE: str = _BEST_CONFIG["msi_space"]

# Cluster coloring (same palette as Exp 03/04).
VIBRANT_CLUSTER_COLORS = [
    "#00A6FB", "#F7B801", "#06D6A0", "#8338EC", "#FF9F1C",
    "#118AB2", "#3A86FF", "#FB5607", "#2EC4B6", "#9B5DE5",
    "#80ED99", "#E9C46A",
]
OUTLIER_COLOR = "#FF1E1E"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{float(v):.2f}"


def _k_from_labels(labels: np.ndarray) -> int:
    return len(set(labels.tolist()) - {-1})


def _scatter_labels(ax: Axes, mds: np.ndarray, labels: np.ndarray, title: str) -> None:
    """Scatter plot with cluster-colour mapping."""
    unique = sorted(set(labels.tolist()))
    non_noise = [lbl for lbl in unique if lbl != -1]
    color_map = {
        lbl: VIBRANT_CLUSTER_COLORS[i % len(VIBRANT_CLUSTER_COLORS)]
        for i, lbl in enumerate(non_noise)
    }
    for lbl in unique:
        mask = labels == lbl
        color = OUTLIER_COLOR if lbl == -1 else color_map[lbl]
        label_text = "noise" if lbl == -1 else f"C{lbl}"
        ax.scatter(
            mds[mask, 0], mds[mask, 1],
            c=[color], s=15, alpha=0.7, edgecolors="none", label=label_text,
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _title_block(record: pd.Series, k: int, algo: str) -> str:
    dataset_id = str(record["dataset_id"])
    msi_rko = _fmt(record.get("msi_rko"))
    msi_db = _fmt(record.get("msi_dbscan_distk"))

    if str(record["group"]) == "classf":
        ari_rko = _fmt(record.get("ari_rko"))
        ari_db = _fmt(record.get("ari_dbscan_distk"))
        return (
            f"{dataset_id}\n"
            f"MSI R={msi_rko} | DDK={msi_db}\n"
            f"ARI R={ari_rko} | DDK={ari_db}\n"
            f"{algo} K={k}"
        )

    return f"{dataset_id}\nMSI R={msi_rko} | DDK={msi_db}\n{algo} K={k}"


def load_epsilons() -> dict[str, float | None]:
    """Load epsilon values used by DBSCAN-DistK on MinMax-scaled MDS [0,1]^2.

    The DistK config is nested per dataset (`epsilon`, `k_star`, `rule`); only
    the `epsilon` value is needed here to rerun DBSCAN.
    """
    path = CONFIGS / "exp03_epsilons_distk.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    out: dict[str, float | None] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            eps = v.get("epsilon")
            out[k] = float(eps) if eps is not None else None
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def _run_rko(
    features: np.ndarray, seed: int, dataset_id: str,
) -> dict[str, Any]:
    """Run clustgrade-rko and return labels."""
    scaled_mds, ppp = preprocess(features)

    seed_run(seed)

    env = ClustGradeEnv(
        features, ppp, scaled_mds,
        instance_name=dataset_id,
        max_time=RKO_TIME_BUDGET,
        msi_space=MSI_SPACE,
        quadrat_filter=QUADRAT_FILTER,
        q_learning=False,
    )

    solver = RKO(env, print_best=False)
    best_cost, best_keys, _ = solver.solve(
        time_total=RKO_TIME_BUDGET,
        brkga=RKO_MIX["brkga"],
        vns=RKO_MIX["vns"],
        ils=RKO_MIX["ils"],
        runs=1,
    )

    if best_keys is not None:
        result = env.decode_and_evaluate(np.array(best_keys))
        labels = np.array(result["labels"], dtype=np.int64)
    else:
        labels = np.full(features.shape[0], -1, dtype=np.int64)

    return {"labels": labels}


def _run_dbscan(mds: np.ndarray, eps: float) -> dict[str, Any]:
    """Run DBSCAN and return labels."""
    labels = DBSCAN(eps=eps, min_samples=MIN_PTS).fit_predict(mds)
    return {"labels": labels}


# ---------------------------------------------------------------------------
# Context / caching
# ---------------------------------------------------------------------------

def _build_context() -> dict[str, Any]:
    manifest = load_manifest()
    epsilons = load_epsilons()
    manifest_map = {str(row["dataset_id"]): row for _, row in manifest.iterrows()}
    return {
        "manifest_map": manifest_map,
        "epsilons": epsilons,
        "feature_cache": {},
        "mds_cache": {},
        "labels_cache": {},
    }


def _get_features(dataset_id: str, context: dict[str, Any]) -> np.ndarray:
    cache: dict[str, np.ndarray] = context["feature_cache"]
    if dataset_id not in cache:
        row = context["manifest_map"][dataset_id]
        cache[dataset_id] = load_features(row)
    return cache[dataset_id]


def _get_scaled_mds(dataset_id: str, group: str, context: dict[str, Any]) -> np.ndarray:
    """Return MinMax-scaled [0,1]^2 MDS, matching the RKO operating space."""
    cache: dict[tuple[str, str], np.ndarray] = context["mds_cache"]
    key = (dataset_id, group)
    if key not in cache:
        from sklearn.preprocessing import MinMaxScaler
        raw = load_mds(dataset_id, group)
        cache[key] = MinMaxScaler().fit_transform(raw)
    return cache[key]


def _get_labels(
    dataset_id: str, group: str, algo: str, seed: int, context: dict[str, Any],
) -> np.ndarray | None:
    cache: dict[tuple[str, str, str, int], np.ndarray | None] = context["labels_cache"]
    key = (dataset_id, group, algo, seed)
    if key in cache:
        return cache[key]

    features = _get_features(dataset_id, context)
    scaled_mds = _get_scaled_mds(dataset_id, group, context)

    if algo == "ClustGrade-RKO":
        result = _run_rko(features, seed, dataset_id)
        labels = result["labels"]
    else:
        eps = context["epsilons"].get(dataset_id)
        if eps is None:
            labels = None
        else:
            result = _run_dbscan(scaled_mds, eps)
            labels = result["labels"]

    cache[key] = labels
    return labels


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _scatter_ranked_instances(
    records: pd.DataFrame,
    algo: str,
    fig_title: str,
    out_name: str,
    context: dict[str, Any],
) -> None:
    n = min(PANEL_COUNT, len(records))
    fig, axes = plt.subplots(1, PANEL_COUNT, figsize=(24, 4.8))

    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue

        rec = records.iloc[idx]
        dataset_id = str(rec["dataset_id"])
        group = str(rec["group"])

        # Use best_seed for RKO, 0 for DBSCAN.
        seed = int(rec.get("best_seed", 0)) if algo == "ClustGrade-RKO" else 0
        labels = _get_labels(dataset_id, group, algo, seed, context)

        if labels is None:
            ax.set_title(f"{dataset_id}\n(no epsilon for DBSCAN-DistK)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        scaled_mds = _get_scaled_mds(dataset_id, group, context)
        k = _k_from_labels(labels)
        _scatter_labels(ax, scaled_mds, labels, _title_block(rec, k, algo))

    fig.suptitle(fig_title, y=1.05)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating Experiment 03 comparison plots (RKO vs DBSCAN-DistK) ...")

    comp = pd.read_csv(ARTIFACTS / "comparison.csv")
    context = _build_context()

    # --- Top-5 datasets where RKO outperforms DBSCAN-DistK (MSI) ---
    msi_valid = comp.dropna(subset=["msi_delta_distk"]).copy()
    msi_rko_wins = msi_valid.sort_values("msi_delta_distk", ascending=False).head(PANEL_COUNT)
    msi_rko_wins = msi_rko_wins.reset_index(drop=True)

    _scatter_ranked_instances(
        msi_rko_wins, "ClustGrade-RKO",
        "Experiment 03: Top 5 margins where RKO outperforms DBSCAN-DistK (MSI)",
        "plot-01-rko-top5-msi.png", context,
    )
    _scatter_ranked_instances(
        msi_rko_wins, "DBSCAN-DistK",
        "Experiment 03: DBSCAN-DistK on same datasets (comparison)",
        "plot-02-dbscan-top5-msi.png", context,
    )

    # --- Top-5 datasets where DBSCAN-DistK outperforms RKO (MSI) ---
    msi_db_wins = msi_valid.sort_values("msi_delta_distk", ascending=True).head(PANEL_COUNT)
    msi_db_wins = msi_db_wins.reset_index(drop=True)

    _scatter_ranked_instances(
        msi_db_wins, "DBSCAN-DistK",
        "Experiment 03: Top 5 margins where DBSCAN-DistK outperforms RKO (MSI)",
        "plot-05-dbscan-top5-msi.png", context,
    )
    _scatter_ranked_instances(
        msi_db_wins, "ClustGrade-RKO",
        "Experiment 03: RKO on same datasets (comparison)",
        "plot-06-rko-top5-msi-loss.png", context,
    )

    # --- Top-5 datasets where RKO outperforms DBSCAN-DistK (ARI, CLASSF) ---
    classf = comp[comp["group"] == "classf"].copy()
    ari_valid = classf.dropna(subset=["ari_delta_distk"])
    if len(ari_valid) > 0:
        ari_rko_wins = ari_valid.sort_values("ari_delta_distk", ascending=False).head(PANEL_COUNT)
        ari_rko_wins = ari_rko_wins.reset_index(drop=True)

        _scatter_ranked_instances(
            ari_rko_wins, "ClustGrade-RKO",
            "Experiment 03: Top 5 margins where RKO outperforms DBSCAN-DistK (ARI, CLASSF)",
            "plot-03-rko-top5-ari.png", context,
        )
        _scatter_ranked_instances(
            ari_rko_wins, "DBSCAN-DistK",
            "Experiment 03: DBSCAN-DistK on same datasets (comparison)",
            "plot-04-dbscan-top5-ari.png", context,
        )

    print("Done.")


if __name__ == "__main__":
    main()
