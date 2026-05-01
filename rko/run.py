"""
ClustGrade-RKO runner: end-to-end execution with a single default solver mix.

Supports parallel dataset runs via subprocess (avoids nested multiprocessing
conflicts with RKO's internal Process workers on Windows).

Usage:
    poetry run python -m rko.run                                 # Single dataset (200DATA)
    poetry run python -m rko.run --dataset 200DATA A1 2-FACE     # Multiple datasets
    poetry run python -m rko.run --group clust                   # All clustering datasets
    poetry run python -m rko.run --group classf                  # All classification datasets
    poetry run python -m rko.run --group all                     # All 50 datasets
    poetry run python -m rko.run --time 120 --jobs 4             # 120s budget, 4 parallel jobs
    poetry run python -m rko.run --q-learning                    # Enable adaptive tuning
    poetry run python -m rko.run --msi-space mds                 # Score on MDS coords
"""

from __future__ import annotations

import argparse
import os
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path bootstrap. The repo's own src/ goes on sys.path here; the upstream RKO
# framework is added by `rko._framework.bootstrap_framework` (resolved via the
# RKO_FRAMEWORK_PATH env var — see README).
# ---------------------------------------------------------------------------
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"

for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from rko._framework import bootstrap_framework

bootstrap_framework()

from RKO import RKO
from rko.environment import ClustGradeEnv
from rko.pipeline import preprocess

# ---------------------------------------------------------------------------
# Dataset loading — imports _shared/utils.py via importlib to avoid shadowing
# src/utils/ package.
# ---------------------------------------------------------------------------


def _load_shared_utils():
    """Load experiments/_shared/utils.py without polluting sys.path."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_shared_utils", _ROOT / "experiments" / "_shared" / "utils.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_shared = _load_shared_utils()


def load_dataset(dataset_id: str) -> np.ndarray:
    """Load feature matrix from manifest."""
    manifest = _shared.load_manifest()
    row = manifest[manifest["dataset_id"] == dataset_id].iloc[0]
    return _shared.load_features(row)


def resolve_dataset_ids(args: argparse.Namespace) -> list[str]:
    """Resolve --dataset and --group flags into a list of dataset IDs."""
    manifest = _shared.load_manifest()

    if args.group:
        if args.group == "all":
            return manifest["dataset_id"].tolist()
        elif args.group == "clust":
            return _shared.manifest_clust(manifest)["dataset_id"].tolist()
        elif args.group == "classf":
            return _shared.manifest_classf(manifest)["dataset_id"].tolist()
        else:
            raise ValueError(f"Unknown group: {args.group}. Use 'all', 'clust', or 'classf'.")

    return args.dataset


# ---------------------------------------------------------------------------
# Default solver mix: BRKGA + VNS + ILS (collapsed from the v1 comparison loop;
# see git history pre-Phase 3 refinements for the multi-mix experimentation).
# ---------------------------------------------------------------------------

DEFAULT_MIX = {"brkga": 1, "vns": 1, "ils": 1}

RESULT_COLS = [
    "dataset",
    "n",
    "d",
    "q_learning",
    "msi_space",
    "quadrat_filter",
    "best_cost",
    "silhouette",
    "k",
    "nx",
    "ny",
    "total_cells",
    "time_to_best",
    "total_time",
    "eval_count",
    "cache_hits",
    "cap_hits",
]


def run_one(
    env: ClustGradeEnv,
    time_budget: int,
    seed: int,
) -> dict:
    """Run the default BRKGA+VNS+ILS mix once on the given env."""
    random.seed(seed)
    np.random.seed(seed)

    solver = RKO(env, print_best=False)
    start = time.perf_counter()
    best_cost, best_keys, time_to_best = solver.solve(
        time_total=time_budget,
        brkga=DEFAULT_MIX["brkga"],
        vns=DEFAULT_MIX["vns"],
        ils=DEFAULT_MIX["ils"],
        runs=1,
    )
    elapsed = time.perf_counter() - start

    # Decode best solution for detailed inspection.
    if best_keys is not None:
        best_keys_arr = np.array(best_keys)
        result = env.decode_and_evaluate(best_keys_arr)
    else:
        result = {
            "silhouette": -1.0,
            "k": 0,
            "nx": 0,
            "ny": 0,
            "total_cells": 0,
        }

    diag = env.diagnostics()

    return {
        "best_cost": round(best_cost, 4),
        "silhouette": round(result["silhouette"], 4),
        "k": result["k"],
        "nx": result.get("nx", 0),
        "ny": result.get("ny", 0),
        "total_cells": result.get("total_cells", 0),
        "time_to_best": round(abs(time_to_best) if time_to_best else 0.0, 2),
        "total_time": round(elapsed, 2),
        "eval_count": diag["eval_count"],
        "cache_hits": diag["cache_hits"],
        "cap_hits": diag["cap_hits"],
    }


# ---------------------------------------------------------------------------
# Per-dataset job
# ---------------------------------------------------------------------------


def run_dataset(
    dataset_id: str,
    time_budget: int,
    seed: int,
    msi_space: str,
    q_learning: bool,
    quadrat_filter: bool = True,
    verbose: bool = True,
) -> list[dict]:
    """Run the default solver mix once for a single dataset."""
    try:
        data = load_dataset(dataset_id)
    except Exception as e:
        print(f"[{dataset_id}] ERROR loading: {e}")
        return []

    n, d = data.shape
    if verbose:
        print(f"[{dataset_id}] n={n}, d={d} — preprocessing...")

    try:
        scaled_mds, ppp = preprocess(data)
    except Exception as e:
        print(f"[{dataset_id}] ERROR in preprocessing: {e}")
        return []

    env = ClustGradeEnv(
        data,
        ppp,
        scaled_mds,
        instance_name=dataset_id,
        max_time=time_budget,
        msi_space=msi_space,  # type: ignore[arg-type]
        quadrat_filter=quadrat_filter,
        q_learning=q_learning,
    )
    diag = env.diagnostics()
    if verbose:
        print(
            f"[{dataset_id}] M={diag['M']}, A={diag['A']}, tam={diag['tam_solution']}, "
            f"msi_space={msi_space}, quadrat_filter={quadrat_filter}, q_learning={q_learning}"
        )
        print(f"[{dataset_id}] Running BRKGA+VNS+ILS...")

    result = run_one(env, time_budget, seed)
    result["dataset"] = dataset_id
    result["n"] = n
    result["d"] = d
    result["q_learning"] = q_learning
    result["msi_space"] = msi_space
    result["quadrat_filter"] = quadrat_filter

    if verbose:
        print(
            f"[{dataset_id}] sil={result['silhouette']}, "
            f"k={result['k']}, grid={result['nx']}x{result['ny']}"
        )

    return [result]


# ---------------------------------------------------------------------------
# Parallel dataset execution via subprocesses
# ---------------------------------------------------------------------------


def _run_single_dataset_subprocess(
    dataset_id: str,
    time_budget: int,
    seed: int,
    msi_space: str,
    q_learning: bool,
    quadrat_filter: bool = True,
) -> pd.DataFrame | None:
    """Run a single dataset in a fresh subprocess. Returns DataFrame or None."""
    artifacts_dir = _ROOT / "rko" / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)
    out_path = artifacts_dir / f"results_{dataset_id}.csv"

    cmd = [
        sys.executable,
        str(_ROOT / "rko" / "run.py"),
        "--dataset",
        dataset_id,
        "--time",
        str(time_budget),
        "--seed",
        str(seed),
        "--msi-space",
        msi_space,
        "--quiet",
    ]
    if q_learning:
        cmd.append("--q-learning")
    if quadrat_filter:
        cmd.append("--quadrat-filter")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
            timeout=time_budget + 120,  # generous timeout
        )
        if proc.returncode != 0:
            print(f"[{dataset_id}] subprocess FAILED (rc={proc.returncode})")
            if proc.stderr:
                lines = proc.stderr.strip().splitlines()
                for line in lines[-5:]:
                    print(f"  {line}")
            return None
    except subprocess.TimeoutExpired:
        print(f"[{dataset_id}] subprocess TIMED OUT")
        return None

    if out_path.exists():
        return pd.read_csv(out_path)
    return None


def _run_datasets_parallel(
    dataset_ids: list[str],
    time_budget: int,
    seed: int,
    msi_space: str,
    q_learning: bool,
    quadrat_filter: bool,
    n_jobs: int,
    verbose: bool,
) -> list[dict]:
    """Run multiple datasets in parallel subprocesses."""
    results_dfs = []
    with ThreadPoolExecutor(max_workers=n_jobs) as executor:
        futures = {
            executor.submit(
                _run_single_dataset_subprocess,
                did,
                time_budget,
                seed,
                msi_space,
                q_learning,
                quadrat_filter,
            ): did
            for did in dataset_ids
        }
        for i, future in enumerate(as_completed(futures), 1):
            did = futures[future]
            try:
                df = future.result()
                if df is not None:
                    results_dfs.append(df)
                    if verbose:
                        best = df.iloc[0]
                        print(
                            f"  [{i}/{len(dataset_ids)}] {did}: "
                            f"sil={best['silhouette']}, k={int(best['k'])}"
                        )
                else:
                    print(f"  [{i}/{len(dataset_ids)}] {did}: FAILED")
            except Exception as e:
                print(f"  [{i}/{len(dataset_ids)}] {did}: ERROR {e}")

    if not results_dfs:
        return []
    combined = pd.concat(results_dfs, ignore_index=True)
    return combined.to_dict("records")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="ClustGrade-RKO runner")
    parser.add_argument(
        "--dataset",
        type=str,
        nargs="+",
        default=["200DATA"],
        help="One or more dataset IDs from manifest",
    )
    parser.add_argument(
        "--group",
        type=str,
        default=None,
        choices=["all", "clust", "classf"],
        help="Run on a dataset group (overrides --dataset)",
    )
    parser.add_argument("--time", type=int, default=60, help="Time budget (s)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of parallel dataset jobs (default: 1 = sequential)",
    )
    parser.add_argument(
        "--msi-space",
        type=str,
        default="feature",
        choices=["feature", "mds"],
        help="Space in which to compute the silhouette (MSI) objective",
    )
    parser.add_argument(
        "--quadrat-filter",
        action="store_true",
        help="Enable quadrat spatial randomness test as a grid validation gate",
    )
    parser.add_argument(
        "--q-learning",
        action="store_true",
        help="Enable RKO Q-learning adaptive tuning on BRKGA/VNS/ILS",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress per-dataset progress")
    args = parser.parse_args()

    dataset_ids = resolve_dataset_ids(args)
    n_datasets = len(dataset_ids)
    verbose = not args.quiet

    print("=== ClustGrade-RKO ===")
    print(
        f"Datasets: {n_datasets}, Time budget: {args.time}s, "
        f"Seed: {args.seed}, Jobs: {args.jobs}"
    )
    print(
        f"MSI space: {args.msi_space}, Quadrat filter: {args.quadrat_filter}, "
        f"Q-learning: {args.q_learning}"
    )
    print("Default mix: BRKGA+VNS+ILS")
    if n_datasets <= 10:
        print(f"IDs: {dataset_ids}")
    print()

    if args.jobs == 1:
        all_results = []
        for i, did in enumerate(dataset_ids, 1):
            print(f"--- [{i}/{n_datasets}] {did} ---")
            results = run_dataset(
                did,
                args.time,
                args.seed,
                args.msi_space,
                args.q_learning,
                args.quadrat_filter,
                verbose,
            )
            all_results.extend(results)
    else:
        print(f"Launching up to {args.jobs} parallel subprocess jobs...")
        all_results = _run_datasets_parallel(
            dataset_ids,
            args.time,
            args.seed,
            args.msi_space,
            args.q_learning,
            args.quadrat_filter,
            args.jobs,
            verbose,
        )

    if not all_results:
        print("No results produced.")
        return

    df = pd.DataFrame(all_results)[RESULT_COLS]

    print()
    print("=" * 100)
    print("RESULTS")
    print("=" * 100)
    print(df.to_string(index=False))
    print()

    # Save results.
    artifacts_dir = _ROOT / "rko" / "artifacts"
    artifacts_dir.mkdir(exist_ok=True)

    if n_datasets == 1:
        out_path = artifacts_dir / f"results_{dataset_ids[0]}.csv"
    else:
        out_path = artifacts_dir / "results_batch.csv"
    df.to_csv(out_path, index=False)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
