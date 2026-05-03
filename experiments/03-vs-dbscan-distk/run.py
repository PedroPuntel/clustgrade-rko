"""
Experiment 03 — ClustGrade-RKO (RKO-only runner).

Runs the best RKO config from Exp 01 on all 50 datasets with 20 seeds each,
with Q-learning enabled. DBSCAN variants are handled by separate standalone
scripts (run_dbscan_scaled.py for DBSCAN-KDist, run_dbscan_distk.py for
DBSCAN-DistK — the primary baseline), each producing its own CSV.
Aggregation across all three runners happens in aggregate.py.

RKO config (from Exp 01's best_config.json):
    quadrat_filter, msi_space — inherited
    q_learning = True
    time budget = 60s
    seeds = 0..19

Headline metrics: MSI on scaled [0,1]^2 MDS, ARI on CLASSF.

Output:
    artifacts/rko_runs.csv — 1000 rows (50 datasets x 20 seeds), all
                             algorithm="ClustGrade-RKO"

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.run
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"
_RKO_FW = _ROOT / "misc" / "rko"

for p in [str(_SRC), str(_RKO_FW), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments._shared.utils import (
    compute_ari,
    compute_msi,
    load_features,
    load_manifest,
    load_stored_labels,
    seed_run,
)
from RKO import RKO
from rko.environment import ClustGradeEnv
from rko.pipeline import preprocess
from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

N_SEEDS = 20
RKO_TIME_BUDGET = 60
Q_LEARNING = True
RKO_MIX = {"brkga": 1, "vns": 1, "ils": 1}

# Inherit RKO base config from Exp 01.
_BEST_CONFIG_PATH = Path(__file__).parents[1] / "01-param-study" / "artifacts" / "best_config.json"
with open(_BEST_CONFIG_PATH) as f:
    _BEST_CONFIG = json.load(f)
QUADRAT_FILTER: bool = _BEST_CONFIG["quadrat_filter"]
MSI_SPACE: str = _BEST_CONFIG["msi_space"]


# ---------------------------------------------------------------------------
# RKO runner
# ---------------------------------------------------------------------------


def run_rko(
    dataset_id: str,
    group: str,
    features: np.ndarray,
    scaled_mds: np.ndarray,
    ppp: "PointPattern",  # noqa: F821
    true_labels: np.ndarray | None,
    seed: int,
) -> dict:
    seed_run(seed)

    env = ClustGradeEnv(
        features,
        ppp,
        scaled_mds,
        instance_name=dataset_id,
        max_time=RKO_TIME_BUDGET,
        msi_space=MSI_SPACE,
        quadrat_filter=QUADRAT_FILTER,
        q_learning=Q_LEARNING,
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
        best_keys_arr = np.array(best_keys)
        result = env.decode_and_evaluate(best_keys_arr)
        labels_arr = np.array(result["labels"], dtype=np.int64)
        k = result["k"]
    else:
        labels_arr = np.full(features.shape[0], -1, dtype=np.int64)
        k = 0

    # Headline metrics on scaled [0,1]^2 MDS (same space RKO optimizes in).
    headline_msi = compute_msi(scaled_mds, labels_arr)
    ari = compute_ari(true_labels, labels_arr) if true_labels is not None else None

    return {
        "dataset_id": dataset_id,
        "group": group,
        "algorithm": "ClustGrade-RKO",
        "seed": seed,
        "k": k,
        "msi": headline_msi,
        "ari": ari,
        "internal_cost": round(best_cost, 4) if best_cost is not None else None,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Load existing partial results and return (dataframe, set of completed dataset_ids)."""
    out_path = ARTIFACTS / "rko_runs.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        expected_per_dataset = N_SEEDS
        counts = df.groupby("dataset_id").size()
        done = set(counts[counts >= expected_per_dataset].index)
        logger.info("Checkpoint: loaded %d rows, %d datasets complete", len(df), len(done))
        return df, done
    return pd.DataFrame(), set()


def main() -> None:
    manifest = load_manifest()

    logger.info("Experiment 03: ClustGrade-RKO on %d datasets", len(manifest))
    logger.info(
        "  RKO config: quadrat_filter=%s, msi_space=%s, q_learning=%s, %ds, seeds=%d",
        QUADRAT_FILTER,
        MSI_SPACE,
        Q_LEARNING,
        RKO_TIME_BUDGET,
        N_SEEDS,
    )

    existing_df, done_datasets = _load_checkpoint()
    records: list[dict] = list(existing_df.to_dict("records")) if len(existing_df) > 0 else []

    out_path = ARTIFACTS / "rko_runs.csv"

    for idx, row in manifest.iterrows():
        dataset_id: str = row["dataset_id"]
        group: str = row["group"]

        if dataset_id in done_datasets:
            logger.info(
                "=== [%d/%d] %s [%s] === SKIPPED (checkpoint)",
                idx + 1,
                len(manifest),
                dataset_id,
                group,
            )
            continue

        logger.info("=== [%d/%d] %s [%s] ===", idx + 1, len(manifest), dataset_id, group)

        # Remove any partial rows for this dataset from a prior interrupted run.
        records = [r for r in records if r["dataset_id"] != dataset_id]

        features = load_features(row)
        true_labels = load_stored_labels(dataset_id) if group == "classf" else None

        # Preprocess once — both algorithms use the [0,1]^2 scaled MDS space.
        scaled_mds, ppp = preprocess(features)

        for seed in range(N_SEEDS):
            try:
                rec = run_rko(
                    dataset_id,
                    group,
                    features,
                    scaled_mds,
                    ppp,
                    true_labels,
                    seed,
                )
                records.append(rec)
                logger.info(
                    "  RKO seed=%d  K=%d  MSI=%s  ARI=%s",
                    seed,
                    rec["k"],
                    f"{rec['msi']:.4f}" if rec["msi"] is not None else "N/A",
                    f"{rec['ari']:.4f}" if rec["ari"] is not None else "N/A",
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("  RKO seed=%d FAILED: %s", seed, exc)
                records.append(
                    {
                        "dataset_id": dataset_id,
                        "group": group,
                        "algorithm": "ClustGrade-RKO",
                        "seed": seed,
                        "k": None,
                        "msi": None,
                        "ari": None,
                        "internal_cost": None,
                    }
                )

        # Checkpoint: save after each dataset completes.
        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)
        logger.info("  Checkpoint saved (%d rows)", len(df))

    logger.info("All datasets complete. Final: %d rows -> %s", len(records), out_path)


if __name__ == "__main__":
    main()
