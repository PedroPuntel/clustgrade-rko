"""
Experiment 02 — Metaheuristic comparison (BRKGA vs VNS vs ILS).

Runs each of the three RKO metaheuristics in isolation on all 50 datasets to
identify which metaheuristic performs best on the ClustGrade-RKO grid-decoding
problem. Base config (quadrat_filter, msi_space) is inherited from Exp 01's
best_config.json; q_learning is fixed to True, even if Exp 02's verdict differs,
as this was a direct request from other article contributors.

Each metaheuristic runs with its own native parameter set as defined in
rko/environment.py::ClustGradeEnv.

Scope:
    Datasets:        50 (full manifest)
    Metaheuristics:  brkga, vns, ils (each run in isolation)
    Seeds:           20 per (dataset, metaheuristic)
    Time budget:     60s per run
    Q-Learning:      enabled

Total work: 50 * 3 * 20 = 3,000 RKO runs (~50h sequential, ~6-7h at --jobs 8).

Execution modes:
    Coordinator (default):  dispatches per-(dataset, metaheuristic) subprocess
                            shards through a ThreadPoolExecutor. The main thread
                            is the sole writer to the consolidated raw_runs.csv.
    Shard (--shard D MH):   worker mode. Runs 20 seeds for one (dataset, MH)
                            and writes its 20 rows to
                            artifacts/mh_comparison/_shards/run_D_MH.csv
    Sequential (--sequential or --jobs 1): runs all shards in-process (useful
                            for debugging; avoids subprocess plumbing).

Concurrency safety:
    Each subprocess owns exactly one shard CSV (unique path per (dataset, MH));
    no two processes ever write to the same file. The coordinator merges the
    shards into raw_runs.csv on a single thread after each shard finishes. RKO
    itself spawns multiprocessing.Process workers inside each subprocess, which
    is safe because each shard runs in a fresh Python interpreter with its own
    process tree (the same pattern rko/run.py uses).

Output:
    artifacts/mh_comparison/_shards/run_<dataset_id>_<mh>.csv — per-shard rows
    artifacts/mh_comparison/_shards/run_<dataset_id>_<mh>.log — per-shard logs
    artifacts/mh_comparison/raw_runs.csv — consolidated, columns:
        dataset_id, group, metaheuristic, seed, k, msi, ari, internal_cost,
        time_to_best, eval_count, cache_hits

Usage:
    # Parallel (default, 8 workers)
    poetry run python -m experiments.02-qlearning-mh-comparison.run_mh_comparison

    # Custom worker count
    poetry run python -m experiments.02-qlearning-mh-comparison.run_mh_comparison --jobs 4

    # Sequential (for debugging)
    poetry run python -m experiments.02-qlearning-mh-comparison.run_mh_comparison --sequential
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"
_RKO_FW = _ROOT / "misc" / "rko"

for p in [str(_SRC), str(_RKO_FW), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RKO import RKO  # noqa: E402

from rko.environment import ClustGradeEnv  # noqa: E402
from rko.pipeline import preprocess  # noqa: E402
from utils.logging import get_logger  # noqa: E402

from experiments._shared.utils import (  # noqa: E402
    compute_ari,
    compute_msi,
    load_features,
    load_manifest,
    load_stored_labels,
    seed_run,
)

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison"
SHARDS_DIR = ARTIFACTS / "_shards"
RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"

METAHEURISTICS: list[str] = ["brkga", "vns", "ils"]
N_SEEDS: int = 20
TIME_BUDGET: int = 60
DEFAULT_JOBS: int = 8
N_DATASETS_LIMIT: int | None = None  # None = full manifest; set by --smoke

# Smoke-test overrides — kept small enough to finish in a couple of minutes
# with --jobs 2 on any laptop. Writes to a separate artifacts folder so a
# smoke run never pollutes the real experiment's checkpoints.
_SMOKE_N_SEEDS: int = 2
_SMOKE_TIME_BUDGET: int = 10
_SMOKE_N_DATASETS: int = 2
_SMOKE_ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison_smoke"

SHARD_COLUMNS: list[str] = [
    "dataset_id", "group", "metaheuristic", "seed",
    "k", "msi", "ari", "internal_cost",
    "time_to_best", "eval_count", "cache_hits",
]

def _apply_smoke_overrides() -> None:
    """Swap runtime constants to the smoke-test profile.

    Redirects artifacts to a `_smoke` subfolder so the real checkpoints are
    never touched. Safe to call multiple times (idempotent).
    """
    global N_SEEDS, TIME_BUDGET, N_DATASETS_LIMIT
    global ARTIFACTS, SHARDS_DIR, RAW_RUNS_PATH
    N_SEEDS = _SMOKE_N_SEEDS
    TIME_BUDGET = _SMOKE_TIME_BUDGET
    N_DATASETS_LIMIT = _SMOKE_N_DATASETS
    ARTIFACTS = _SMOKE_ARTIFACTS
    SHARDS_DIR = ARTIFACTS / "_shards"
    RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"


# Inherit RKO base config from Exp 01.
_BEST_CONFIG_PATH = (
    Path(__file__).parents[1] / "01-param-study" / "artifacts" / "best_config.json"
)
with open(_BEST_CONFIG_PATH) as f:
    _BEST_CONFIG = json.load(f)
QUADRAT_FILTER: bool = _BEST_CONFIG["quadrat_filter"]
MSI_SPACE: str = _BEST_CONFIG["msi_space"]

Q_LEARNING: bool = True  # fixed to True per article co-author request, even if Exp 02's verdict differs

# ---------------------------------------------------------------------------
# Core RKO runner (single seed, single metaheuristic)
# ---------------------------------------------------------------------------

def run_rko_single_mh(
    dataset_id: str,
    group: str,
    features: np.ndarray,
    scaled_mds: np.ndarray,
    ppp: "PointPattern",  # noqa: F821
    true_labels: np.ndarray | None,
    seed: int,
    metaheuristic: str,
) -> dict:
    """Run a single RKO seed with exactly one metaheuristic enabled.

    The metaheuristic's native (single-valued) parameter set is applied via
    ClustGradeEnv(q_learning=False). Headline MSI is computed on the scaled
    [0,1]^2 MDS space (same space RKO optimizes in when msi_space='mds').
    """
    if metaheuristic not in METAHEURISTICS:
        raise ValueError(f"Unknown metaheuristic: {metaheuristic!r}")

    seed_run(seed)

    env = ClustGradeEnv(
        features, ppp, scaled_mds,
        instance_name=dataset_id,
        max_time=TIME_BUDGET,
        msi_space=MSI_SPACE,
        quadrat_filter=QUADRAT_FILTER,
        q_learning=Q_LEARNING,
    )

    solver = RKO(env, print_best=False)
    solve_kwargs = {mh: (1 if mh == metaheuristic else 0) for mh in METAHEURISTICS}
    best_cost, best_keys, time_to_best = solver.solve(
        time_total=TIME_BUDGET,
        **solve_kwargs,
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

    headline_msi = compute_msi(scaled_mds, labels_arr)
    ari = compute_ari(true_labels, labels_arr) if true_labels is not None else None
    diag = env.diagnostics()

    return {
        "dataset_id": dataset_id,
        "group": group,
        "metaheuristic": metaheuristic,
        "seed": seed,
        "k": k,
        "msi": headline_msi,
        "ari": ari,
        "internal_cost": round(float(best_cost), 6) if best_cost is not None else None,
        "time_to_best": round(float(abs(time_to_best)) if time_to_best else 0.0, 2),
        "eval_count": diag["eval_count"],
        "cache_hits": diag["cache_hits"],
    }


# ---------------------------------------------------------------------------
# Shard mode: run all N_SEEDS for one (dataset, metaheuristic) and save CSV
# ---------------------------------------------------------------------------

def _shard_path(dataset_id: str, metaheuristic: str) -> Path:
    return SHARDS_DIR / f"run_{dataset_id}_{metaheuristic}.csv"


def _shard_is_complete(dataset_id: str, metaheuristic: str) -> bool:
    """A shard is complete when its CSV exists and has exactly N_SEEDS rows."""
    p = _shard_path(dataset_id, metaheuristic)
    if not p.exists():
        return False
    try:
        df = pd.read_csv(p)
    except Exception:
        return False
    return len(df) == N_SEEDS


def _load_existing_shard(out_path: Path) -> tuple[pd.DataFrame, set[int]]:
    """Return (existing_complete_rows, completed_seeds) for seed-level resume.

    A row is considered "completed" iff its `k` is non-null. This covers both
    successful rows (fully populated) and log-recovered rows (NaN in
    internal_cost/time_to_best/cache_hits but populated k/msi/ari/eval_count).
    Old error-branch rows have k=NaN and are excluded, so they get re-run.
    """
    if not out_path.exists():
        return pd.DataFrame(columns=SHARD_COLUMNS), set()
    try:
        df = pd.read_csv(out_path)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=SHARD_COLUMNS), set()
    if "seed" not in df.columns or "k" not in df.columns or df.empty:
        return pd.DataFrame(columns=SHARD_COLUMNS), set()
    df = df.drop_duplicates(subset=["seed"], keep="first")
    completed_mask = df["k"].notna()
    complete_rows = df.loc[completed_mask].copy()
    completed_seeds: set[int] = set(complete_rows["seed"].astype(int).tolist())
    
    # Enforce the canonical schema; missing columns (e.g., when resuming from
    # a log-recovered CSV) come out as NaN, which downstream aggregators ignore.
    for col in SHARD_COLUMNS:
        if col not in complete_rows.columns:
            complete_rows[col] = None
    complete_rows = complete_rows[SHARD_COLUMNS]
    return complete_rows, completed_seeds


def run_shard(dataset_id: str, metaheuristic: str) -> pd.DataFrame:
    """Run the missing RKO fits for one (dataset, metaheuristic) and persist the shard.

    Seed-level resume: if `out_path` already contains rows from a prior partial
    run (or from `recover_failed_run_results.py`), only the missing seeds are
    executed. Completed rows are merged into the final CSV as-is.

    Returns the shard DataFrame.
    """
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _shard_path(dataset_id, metaheuristic)

    existing_rows, completed_seeds = _load_existing_shard(out_path)
    missing: list[int] = [s for s in range(N_SEEDS) if s not in completed_seeds]

    logger.info(
        "[%s/%s] resume: %d/%d seeds already present; running %d missing: %s",
        dataset_id, metaheuristic, len(completed_seeds), N_SEEDS, len(missing),
        missing if len(missing) <= 20 else f"{missing[:5]}...",
    )

    if not missing:
        # Already complete — write back the existing rows in canonical order
        # to normalize the CSV and return.
        df = existing_rows.sort_values("seed").reset_index(drop=True)[SHARD_COLUMNS]
        df.to_csv(out_path, index=False)
        return df

    manifest = load_manifest()
    row = manifest[manifest["dataset_id"] == dataset_id].iloc[0]
    group: str = row["group"]

    features = load_features(row)
    true_labels = load_stored_labels(dataset_id) if group == "classf" else None
    scaled_mds, ppp = preprocess(features)

    new_records: list[dict] = []
    for seed in missing:
        try:
            rec = run_rko_single_mh(
                dataset_id, group, features, scaled_mds, ppp,
                true_labels, seed, metaheuristic,
            )
            new_records.append(rec)
            logger.info(
                "  [%s/%s] seed=%d K=%s MSI=%s ARI=%s evals=%s",
                dataset_id, metaheuristic, seed, rec["k"],
                f"{rec['msi']:.4f}" if rec["msi"] is not None else "N/A",
                f"{rec['ari']:.4f}" if rec["ari"] is not None else "N/A",
                rec["eval_count"],
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("  [%s/%s] seed=%d FAILED: %s", dataset_id, metaheuristic, seed, exc)
            new_records.append({
                "dataset_id": dataset_id, "group": group,
                "metaheuristic": metaheuristic, "seed": seed,
                "k": None, "msi": None, "ari": None,
                "internal_cost": None, "time_to_best": None,
                "eval_count": None, "cache_hits": None,
            })

    new_df = pd.DataFrame(new_records, columns=SHARD_COLUMNS)
    merged = pd.concat([existing_rows, new_df], ignore_index=True)
    merged = merged.sort_values("seed").reset_index(drop=True)[SHARD_COLUMNS]
    merged.to_csv(out_path, index=False)
    return merged


# ---------------------------------------------------------------------------
# Subprocess dispatch (coordinator-side)
# ---------------------------------------------------------------------------

def _run_shard_subprocess(dataset_id: str, metaheuristic: str) -> pd.DataFrame | None:
    """Dispatch a single shard to a fresh Python subprocess; return its shard DataFrame.

    The subprocess inherits the current interpreter and invokes this module with
    --shard. Log output is redirected to a per-shard .log file to avoid
    stdout/stderr interleaving across concurrent workers.
    """
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)
    log_path = SHARDS_DIR / f"run_{dataset_id}_{metaheuristic}.log"
    out_path = _shard_path(dataset_id, metaheuristic)

    cmd = [
        sys.executable, "-m",
        "experiments.02-qlearning-mh-comparison.run_mh_comparison",
        "--shard", dataset_id, metaheuristic,
    ]
    # Propagate smoke mode so the worker applies the same overrides and writes
    # to the smoke artifacts folder.
    if ARTIFACTS == _SMOKE_ARTIFACTS:
        cmd.append("--smoke")

    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    # Generous timeout: 20 seeds * 60s + 120s headroom for preprocessing & framework overhead.
    timeout = N_SEEDS * TIME_BUDGET + 120

    try:
        with open(log_path, "w", encoding="utf-8") as log_file:
            proc = subprocess.run(
                cmd,
                stdout=log_file, stderr=subprocess.STDOUT,
                env=env, timeout=timeout,
                cwd=str(_ROOT),
            )
        if proc.returncode != 0:
            logger.error("[%s/%s] subprocess FAILED (rc=%d); see %s",
                         dataset_id, metaheuristic, proc.returncode, log_path.name)
            return None
    except subprocess.TimeoutExpired:
        logger.error("[%s/%s] subprocess TIMED OUT after %ds", dataset_id, metaheuristic, timeout)
        return None

    if not out_path.exists():
        logger.error("[%s/%s] shard CSV missing after subprocess exit", dataset_id, metaheuristic)
        return None

    df = pd.read_csv(out_path)
    if len(df) != N_SEEDS:
        logger.error("[%s/%s] shard CSV has %d rows, expected %d",
                     dataset_id, metaheuristic, len(df), N_SEEDS)
        return None
    return df


# ---------------------------------------------------------------------------
# Consolidation (coordinator-only writer to raw_runs.csv)
# ---------------------------------------------------------------------------

def _consolidate_raw_runs() -> pd.DataFrame:
    """Read every complete shard CSV and write the consolidated raw_runs.csv."""
    frames: list[pd.DataFrame] = []
    if not SHARDS_DIR.exists():
        return pd.DataFrame(columns=SHARD_COLUMNS)

    for shard_csv in sorted(SHARDS_DIR.glob("run_*.csv")):
        try:
            df = pd.read_csv(shard_csv)
        except Exception as exc:  # noqa: BLE001
            logger.warning("  Skipping unreadable shard %s: %s", shard_csv.name, exc)
            continue
        if len(df) != N_SEEDS:
            continue  # partial shard — will be re-run
        frames.append(df)

    if not frames:
        consolidated = pd.DataFrame(columns=SHARD_COLUMNS)
    else:
        consolidated = pd.concat(frames, ignore_index=True)
        consolidated = consolidated[SHARD_COLUMNS]

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    consolidated.to_csv(RAW_RUNS_PATH, index=False)
    return consolidated


# ---------------------------------------------------------------------------
# Coordinator
# ---------------------------------------------------------------------------

def _build_job_list() -> list[tuple[str, str]]:
    manifest = load_manifest()
    if N_DATASETS_LIMIT is not None:
        manifest = manifest.head(N_DATASETS_LIMIT)
    jobs: list[tuple[str, str]] = []
    for _, row in manifest.iterrows():
        dataset_id = row["dataset_id"]
        for mh in METAHEURISTICS:
            if _shard_is_complete(dataset_id, mh):
                continue
            jobs.append((dataset_id, mh))
    return jobs


def _run_coordinator(jobs: list[tuple[str, str]], n_workers: int) -> None:
    total_shards = 50 * len(METAHEURISTICS)
    already_done = total_shards - len(jobs)
    logger.info(
        "Coordinator: %d/%d shards already complete; dispatching %d with %d workers.",
        already_done, total_shards, len(jobs), n_workers,
    )

    completed = 0
    failed: list[tuple[str, str]] = []

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_run_shard_subprocess, did, mh): (did, mh)
            for did, mh in jobs
        }
        for fut in as_completed(futures):
            did, mh = futures[fut]
            try:
                df = fut.result()
            except Exception as exc:  # noqa: BLE001
                logger.error("[%s/%s] dispatcher exception: %s", did, mh, exc)
                df = None

            if df is None:
                failed.append((did, mh))
            else:
                completed += 1
                # Coordinator is the sole writer to raw_runs.csv.
                _consolidate_raw_runs()
                logger.info(
                    "  [%d/%d] %s/%s DONE (median MSI=%s)",
                    completed, len(jobs), did, mh,
                    f"{df['msi'].median():.4f}" if df["msi"].notna().any() else "N/A",
                )

    # Final consolidation pass ensures raw_runs.csv reflects the complete run.
    final_df = _consolidate_raw_runs()
    logger.info("Coordinator done: %d rows in %s", len(final_df), RAW_RUNS_PATH)
    if failed:
        logger.warning("FAILED shards (%d): %s", len(failed), failed)


def _run_sequential(jobs: list[tuple[str, str]]) -> None:
    logger.info("Sequential mode: running %d shards in-process (no subprocesses).", len(jobs))
    for idx, (did, mh) in enumerate(jobs, 1):
        logger.info("=== [%d/%d] %s / %s ===", idx, len(jobs), did, mh)
        try:
            run_shard(did, mh)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s/%s] shard FAILED in-process: %s", did, mh, exc)
            continue
        _consolidate_raw_runs()

    final_df = _consolidate_raw_runs()
    logger.info("Sequential done: %d rows in %s", len(final_df), RAW_RUNS_PATH)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--jobs", type=int, default=DEFAULT_JOBS,
        help=f"Number of parallel subprocess workers (default: {DEFAULT_JOBS}).",
    )
    parser.add_argument(
        "--sequential", action="store_true",
        help="Run all shards in-process (alias for --jobs 1, bypasses subprocess layer).",
    )
    parser.add_argument(
        "--shard", nargs=2, metavar=("DATASET_ID", "METAHEURISTIC"), default=None,
        help="Worker mode: run one shard and exit (invoked by coordinator subprocesses).",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help=(
            f"Smoke mode: overrides N_SEEDS={_SMOKE_N_SEEDS}, "
            f"TIME_BUDGET={_SMOKE_TIME_BUDGET}s, first {_SMOKE_N_DATASETS} datasets only. "
            f"Artifacts go to artifacts/mh_comparison_smoke/ so real checkpoints are untouched."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.smoke:
        _apply_smoke_overrides()
        logger.info(
            "SMOKE MODE: N_SEEDS=%d, TIME_BUDGET=%ds, %d datasets, artifacts -> %s",
            N_SEEDS, TIME_BUDGET, N_DATASETS_LIMIT, ARTIFACTS,
        )

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    SHARDS_DIR.mkdir(parents=True, exist_ok=True)

    if args.shard is not None:
        dataset_id, metaheuristic = args.shard
        logger.info("Shard worker starting: dataset=%s metaheuristic=%s", dataset_id, metaheuristic)
        run_shard(dataset_id, metaheuristic)
        return

    logger.info(
        "Experiment 02 (MH comparison): 50 datasets x %d MHs x %d seeds x %ds "
        "(quadrat_filter=%s, msi_space=%s, q_learning=False)",
        len(METAHEURISTICS), N_SEEDS, TIME_BUDGET, QUADRAT_FILTER, MSI_SPACE,
    )

    jobs = _build_job_list()
    if not jobs:
        logger.info("All shards already complete; consolidating and exiting.")
        _consolidate_raw_runs()
        return

    if args.sequential or args.jobs <= 1:
        _run_sequential(jobs)
    else:
        _run_coordinator(jobs, n_workers=args.jobs)


if __name__ == "__main__":
    main()
