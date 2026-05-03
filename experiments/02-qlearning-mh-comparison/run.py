"""
Experiment 02 — Q-learning impact assessment.

Compare q_learning=False vs q_learning=True under matched 120s time budgets
on the 20-dataset stratified subset (same as Exp 01).

Base RKO configuration (quadrat_filter, msi_space) is inherited from
Experiment 01's best_config.json to isolate Q-learning as the sole factor.

Datasets: subset_20 (12 CLUST + 8 CLASSF, stratified by size).
Seeds: 5 per configuration.
Time budget: 120s (doubled to give Q-learning more episodes).

Output:
    artifacts/raw_runs.csv — one row per (dataset, q_learning, seed)

Usage:
    poetry run python -m experiments.02-qlearning-mh-comparison.run
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
    load_mds,
    load_stored_labels,
    manifest_subset20,
    seed_run,
)
from RKO import RKO
from rko.environment import ClustGradeEnv
from rko.pipeline import preprocess
from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

_BEST_CONFIG_PATH = Path(__file__).parents[1] / "01-param-study" / "artifacts" / "best_config.json"
with open(_BEST_CONFIG_PATH) as f:
    _BEST_CONFIG = json.load(f)
QUADRAT_FILTER: bool = _BEST_CONFIG["quadrat_filter"]
MSI_SPACE: str = _BEST_CONFIG["msi_space"]

N_SEEDS = 5
TIME_BUDGET = 120
RKO_MIX = {"brkga": 1, "vns": 1, "ils": 1}


def run_rko_ql(
    dataset_id: str,
    group: str,
    features: np.ndarray,
    scaled_mds: np.ndarray,
    ppp: "PointPattern",  # noqa: F821
    mds: np.ndarray,
    true_labels: np.ndarray | None,
    seed: int,
    q_learning: bool,
) -> dict:
    """Run RKO with or without Q-learning."""
    seed_run(seed)

    env = ClustGradeEnv(
        features,
        ppp,
        scaled_mds,
        instance_name=dataset_id,
        max_time=TIME_BUDGET,
        msi_space=MSI_SPACE,
        quadrat_filter=QUADRAT_FILTER,
        q_learning=q_learning,
    )

    solver = RKO(env, print_best=False)
    best_cost, best_keys, _ = solver.solve(
        time_total=TIME_BUDGET,
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

    diag = env.diagnostics()
    headline_msi = compute_msi(mds, labels_arr)
    ari = compute_ari(true_labels, labels_arr) if true_labels is not None else None

    return {
        "dataset_id": dataset_id,
        "group": group,
        "q_learning": q_learning,
        "seed": seed,
        "k": k,
        "msi": headline_msi,
        "ari": ari,
        "internal_cost": round(best_cost, 4) if best_cost is not None else None,
        "eval_count": diag["eval_count"],
        "cache_hits": diag["cache_hits"],
    }


def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Load existing partial results and return (dataframe, set of completed dataset_ids)."""
    out_path = ARTIFACTS / "raw_runs.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        # A dataset is complete when it has both QL settings × all seeds.
        expected_per_dataset = 2 * N_SEEDS  # {False, True} × N_SEEDS
        counts = df.groupby("dataset_id").size()
        done = set(counts[counts >= expected_per_dataset].index)
        logger.info("Checkpoint: loaded %d rows, %d datasets complete", len(df), len(done))
        return df, done
    return pd.DataFrame(), set()


def main() -> None:
    manifest = load_manifest()
    subset = manifest_subset20(manifest)

    logger.info(
        "Experiment 02: Q-learning impact on %d datasets, %d seeds, %ds budget "
        "(base config: quadrat_filter=%s, msi_space=%s)",
        len(subset),
        N_SEEDS,
        TIME_BUDGET,
        QUADRAT_FILTER,
        MSI_SPACE,
    )

    existing_df, done_datasets = _load_checkpoint()
    records: list[dict] = list(existing_df.to_dict("records")) if len(existing_df) > 0 else []

    out_path = ARTIFACTS / "raw_runs.csv"

    for idx, row in subset.iterrows():
        dataset_id: str = row["dataset_id"]
        group: str = row["group"]

        if dataset_id in done_datasets:
            logger.info(
                "=== [%d/%d] %s [%s] === SKIPPED (checkpoint)",
                idx + 1,
                len(subset),
                dataset_id,
                group,
            )
            continue

        logger.info("=== [%d/%d] %s [%s] ===", idx + 1, len(subset), dataset_id, group)

        # Remove any partial rows for this dataset from a prior interrupted run.
        records = [r for r in records if r["dataset_id"] != dataset_id]

        features = load_features(row)
        mds = load_mds(dataset_id, group)
        true_labels = load_stored_labels(dataset_id) if group == "classf" else None
        scaled_mds, ppp = preprocess(features)

        for ql in [False, True]:
            ql_label = "QL=ON" if ql else "QL=OFF"
            for seed in range(N_SEEDS):
                try:
                    rec = run_rko_ql(
                        dataset_id,
                        group,
                        features,
                        scaled_mds,
                        ppp,
                        mds,
                        true_labels,
                        seed,
                        ql,
                    )
                    records.append(rec)
                    logger.info(
                        "  %s seed=%d K=%d MSI=%s evals=%d",
                        ql_label,
                        seed,
                        rec["k"],
                        f"{rec['msi']:.4f}" if rec["msi"] is not None else "N/A",
                        rec["eval_count"],
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.error("  %s seed=%d FAILED: %s", ql_label, seed, exc)
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "group": group,
                            "q_learning": ql,
                            "seed": seed,
                            "k": None,
                            "msi": None,
                            "ari": None,
                            "internal_cost": None,
                            "eval_count": None,
                            "cache_hits": None,
                        }
                    )

        # Checkpoint: save after each dataset completes.
        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)
        logger.info("  Checkpoint saved (%d rows)", len(df))

    logger.info("All datasets complete. Final: %d rows -> %s", len(records), out_path)


if __name__ == "__main__":
    main()
