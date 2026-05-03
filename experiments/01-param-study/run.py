"""
Experiment 01 — RKO parameter study: 2x2 ablation.

Factors:
    quadrat_filter ∈ {True, False}
    msi_space      ∈ {"feature", "mds"}

20-dataset stratified subset (12 CLUST + 8 CLASSF), 5 seeds per configuration, 60s time budget.

CLUST (12 from 32) — stratified by size:

    Small (n<200): RUSPINI (75), BREASTB (49), OUTLIERS (131)
    → pick RUSPINI, OUTLIERS

    Medium (200-500): 200DATA (required), FACE (296), 400P3C (400), BUPA (345)
    → pick 200DATA, FACE, 400P3C

    Large (500-1000): CHART (600), BROKEN-RING (800), GAUSS9 (900), TRIPADVISOR (980)
    → pick CHART, BROKEN-RING, GAUSS9, TRIPADVISOR

    Very large (1000+): BANKNOTE (1372), CONCRETEDATA (1030), A1 (3000), WAVEFORM21 (5000)
    → pick CONCRETEDATA, WAVEFORM21

CLASSF (8 from 18) — stratified by size:

    Small (n<250): IRIS (150), WINE (178), GLASS (214)
    → pick IRIS, WINE
    Medium (250-400): ECOLI (336), COMPOUND (399), JAIN (373)
    → pick ECOLI, JAIN
    Large (400-800): WDBC (569), R15 (600), AGGREGATION (788), PIMA-INDIANS (769)
    → pick WDBC, AGGREGATION, PIMA-INDIANS
    Very large (1000+): YEAST (1484)
    → pick YEAST

The RKO internal optimization objective uses the specified msi_space, but the
**reported MSI** is always computed via `_shared/utils.py::compute_msi` on
MDS 2D coordinates (headline metric consistency with Exp 01-03).

Output:
    artifacts/raw_runs.csv — one row per (dataset, quadrat_filter, msi_space, seed)

Usage:
    poetry run python -m experiments.01-param-study.run
"""

from __future__ import annotations

import itertools
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

N_SEEDS = 5
TIME_BUDGET = 60
RKO_MIX = {"brkga": 1, "vns": 1, "ils": 1}

# 2x2 ablation grid.
QUADRAT_VALUES = [True, False]
MSI_SPACE_VALUES = ["feature", "mds"]


def run_rko_config(
    dataset_id: str,
    group: str,
    features: np.ndarray,
    scaled_mds: np.ndarray,
    ppp: "PointPattern",  # noqa: F821
    mds: np.ndarray,
    true_labels: np.ndarray | None,
    seed: int,
    quadrat_filter: bool,
    msi_space: str,
) -> dict:
    """Run one RKO configuration on one dataset."""
    # NOTE: Seed both random and np.random BEFORE env construction so any future
    # ClustGradeEnv.__init__ RNG use would be covered. seed_run handles both.
    seed_run(seed)

    env = ClustGradeEnv(
        features,
        ppp,
        scaled_mds,
        instance_name=dataset_id,
        max_time=TIME_BUDGET,
        msi_space=msi_space,  # type: ignore[arg-type]
        quadrat_filter=quadrat_filter,
        q_learning=False,
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

    # Headline metrics always on MDS 2D.
    headline_msi = compute_msi(mds, labels_arr)
    ari = compute_ari(true_labels, labels_arr) if true_labels is not None else None

    return {
        "dataset_id": dataset_id,
        "group": group,
        "quadrat_filter": quadrat_filter,
        "msi_space": msi_space,
        "seed": seed,
        "k": k,
        "msi": headline_msi,
        "ari": ari,
        # NOTE: Full precision — do not round. aggregate.py formats at display time.
        "internal_cost": best_cost,
    }


def _load_checkpoint() -> tuple[pd.DataFrame, set[str]]:
    """Load existing partial results and return (dataframe, set of completed dataset_ids)."""
    out_path = ARTIFACTS / "raw_runs.csv"
    if out_path.exists():
        df = pd.read_csv(out_path)
        # NOTE: De-dup on (dataset_id, quadrat_filter, msi_space, seed) BEFORE counting.
        # A prior interrupted run could have written duplicate rows (e.g. crash mid-save);
        # without dedup, a dataset with duplicates could falsely pass the "complete" check.
        df = df.drop_duplicates(
            subset=["dataset_id", "quadrat_filter", "msi_space", "seed"],
            keep="last",
        ).reset_index(drop=True)
        # NOTE: Strict equality (==) not >=. A dataset is complete ONLY when it has
        # EXACTLY the expected row count. >= would skip datasets with stale duplicates.
        expected_per_dataset = len(QUADRAT_VALUES) * len(MSI_SPACE_VALUES) * N_SEEDS
        counts = df.groupby("dataset_id").size()
        done = set(counts[counts == expected_per_dataset].index)
        logger.info("Checkpoint: loaded %d rows, %d datasets complete", len(df), len(done))
        return df, done
    return pd.DataFrame(), set()


def main() -> None:
    manifest = manifest_subset20(load_manifest())
    configs = list(itertools.product(QUADRAT_VALUES, MSI_SPACE_VALUES))
    total_configs = len(configs)

    logger.info(
        "Experiment 01: 2x2 ablation on %d datasets, %d seeds, %d configs",
        len(manifest),
        N_SEEDS,
        total_configs,
    )

    existing_df, done_datasets = _load_checkpoint()
    records: list[dict] = list(existing_df.to_dict("records")) if len(existing_df) > 0 else []

    out_path = ARTIFACTS / "raw_runs.csv"

    # NOTE: enumerate + iterrows — pandas >= 2.x iterrows yields Hashable index,
    # not guaranteed int. Using enumerate gives us a stable int for progress logging.
    for idx, (_, row) in enumerate(manifest.iterrows()):
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
        mds = load_mds(dataset_id, group)
        true_labels = load_stored_labels(dataset_id) if group == "classf" else None

        # Preprocess once per dataset (shared across configs).
        scaled_mds, ppp = preprocess(features)

        for qf, ms in configs:
            config_label = f"qf={qf},ms={ms}"
            for seed in range(N_SEEDS):
                try:
                    rec = run_rko_config(
                        dataset_id,
                        group,
                        features,
                        scaled_mds,
                        ppp,
                        mds,
                        true_labels,
                        seed,
                        qf,
                        ms,
                    )
                    records.append(rec)
                    logger.info(
                        "  %s seed=%d K=%d MSI=%s",
                        config_label,
                        seed,
                        rec["k"],
                        f"{rec['msi']:.4f}" if rec["msi"] is not None else "N/A",
                    )
                except Exception as exc:  # noqa: BLE001
                    # NOTE: Broad catch is intentional — one flaky RKO seed must not
                    # abort the remaining 399 runs. exc_info=True ensures the stack
                    # trace reaches the log for post-hoc diagnosis.
                    logger.error(
                        "  %s seed=%d FAILED: %s",
                        config_label,
                        seed,
                        exc,
                        exc_info=True,
                    )
                    records.append(
                        {
                            "dataset_id": dataset_id,
                            "group": group,
                            "quadrat_filter": qf,
                            "msi_space": ms,
                            "seed": seed,
                            "k": None,
                            "msi": None,
                            "ari": None,
                            "internal_cost": None,
                        }
                    )

        # Checkpoint: save after each dataset completes.
        # NOTE: Full precision required — Exp 01 ranking gap is ~0.055 and
        # aggregate.pairwise_win_loss uses tol=0.01. Truncating to 2 decimals
        # here (as a previous version did) collapses the ranking into rounding
        # noise. Display formatting belongs in aggregate.py, not the raw CSV.
        df = pd.DataFrame(records)
        df.to_csv(out_path, index=False)
        logger.info("  Checkpoint saved (%d rows)", len(df))

    logger.info("All datasets complete. Final: %d rows -> %s", len(records), out_path)


if __name__ == "__main__":
    main()
