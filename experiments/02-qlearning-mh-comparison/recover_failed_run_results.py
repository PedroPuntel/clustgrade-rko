"""
Experiment 02 — Partial-shard recovery from per-shard log files.

When a shard crashes mid-run (most often from OOM under parallel load), its
CSV is never written — `run_shard()` only flushes at the very end of all
N_SEEDS iterations. However, every completed seed is logged in real time via
`logger.info(...)` at run_mh_comparison.py:261-267, so the per-shard `.log`
file preserves the seed results on disk.

This script parses every log whose sibling `.csv` is missing (or has fewer
than N_SEEDS rows) and reconstructs a partial shard CSV from the logged
seed lines. Combined with the seed-level resume logic in `run_shard()`, this
means a re-run only executes the **missing** seeds per shard — potentially
saving several hours of wall time.

Recovered rows are NaN in three columns that the logger does not emit:
    internal_cost, time_to_best, cache_hits
The downstream aggregators (`aggregate_mh_comparison.py`,
`plot_mh_comparison.py`, `run_mh_vs_dbscan.py`) consume only `msi`, `ari`,
and `k`, so this loss is inert for analysis.

Usage:
    # Dry-run (parse everything, print summary, write nothing)
    poetry run python -m experiments.02-qlearning-mh-comparison.recover_failed_run_results --dry-run

    # Real run (write reconstructed shard CSVs)
    poetry run python -m experiments.02-qlearning-mh-comparison.recover_failed_run_results
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"

for p in [str(_SRC), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from utils.logging import get_logger  # noqa: E402

from experiments._shared.utils import load_manifest  # noqa: E402

logger = get_logger(__name__)

# Re-declare the run_mh_comparison contract locally. We cannot `import` from
# the runner module directly because its parent package name
# (`02-qlearning-mh-comparison`) starts with a digit and contains
# hyphens, so it is not a valid Python identifier. Keep in sync with
# `run_mh_comparison.py::{N_SEEDS, SHARD_COLUMNS, SHARDS_DIR}`.
N_SEEDS: int = 20
SHARD_COLUMNS: list[str] = [
    "dataset_id", "group", "metaheuristic", "seed",
    "k", "msi", "ari", "internal_cost",
    "time_to_best", "eval_count", "cache_hits",
]
SHARDS_DIR: Path = Path(__file__).parent / "artifacts" / "mh_comparison" / "_shards"


def _shard_path(dataset_id: str, metaheuristic: str) -> Path:
    return SHARDS_DIR / f"run_{dataset_id}_{metaheuristic}.csv"

# ---------------------------------------------------------------------------
# Log parsing
# ---------------------------------------------------------------------------

# Matches the exact seed-log line emitted by run_mh_comparison.py:261-267, e.g.:
#   2026-04-23 01:12:42 | INFO | __main__ |   [CONCRETEDATA/brkga] seed=0 K=5 MSI=0.3443 ARI=N/A evals=20
# Captures: dataset_id, metaheuristic, seed, k, msi, ari, eval_count.
SEED_LINE_RE = re.compile(
    r"\[(?P<dataset>[^/\]]+)/(?P<mh>[a-z]+)\]\s+"
    r"seed=(?P<seed>\d+)\s+K=(?P<k>\S+)\s+MSI=(?P<msi>\S+)\s+ARI=(?P<ari>\S+)\s+evals=(?P<evals>\d+)"
)


def _parse_float_or_nan(tok: str) -> float | None:
    """Return float(tok) or None for the 'N/A' sentinel emitted by the logger."""
    if tok == "N/A":
        return None
    try:
        return float(tok)
    except ValueError:
        return None


def parse_log(log_path: Path) -> list[dict]:
    """Parse a shard log into a list of seed-result dicts (possibly empty).

    One dict per matched seed line. Duplicates (should not occur — shards do
    not retry seeds — but defensive) are deduped keeping the first occurrence.
    """
    records: list[dict] = []
    seen_seeds: set[int] = set()

    with log_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            m = SEED_LINE_RE.search(line)
            if m is None:
                continue
            seed = int(m.group("seed"))
            if seed in seen_seeds:
                continue
            seen_seeds.add(seed)
            records.append({
                "dataset_id": m.group("dataset"),
                "metaheuristic": m.group("mh"),
                "seed": seed,
                "k": int(m.group("k")),
                "msi": _parse_float_or_nan(m.group("msi")),
                "ari": _parse_float_or_nan(m.group("ari")),
                "eval_count": int(m.group("evals")),
            })

    return records


# ---------------------------------------------------------------------------
# Recovery
# ---------------------------------------------------------------------------

def _existing_row_count(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    try:
        return len(pd.read_csv(csv_path))
    except Exception:  # noqa: BLE001
        return 0


def recover_shard(
    log_path: Path,
    group_by_dataset: dict[str, str],
    dry_run: bool,
) -> dict:
    """Parse one shard log, build a DataFrame, and write it if it beats the existing CSV.

    Returns a summary dict with keys: dataset_id, metaheuristic, log_seeds,
    existing_rows, action ('write' | 'skip-empty' | 'skip-worse' | 'dry-run').
    """
    stem = log_path.stem  # e.g. "run_CONCRETEDATA_brkga"
    parts = stem.split("_", 2)  # ["run", "<dataset_id>", "<mh>"]
    if len(parts) != 3 or parts[0] != "run":
        return {"log": log_path.name, "action": "skip-malformed-name"}
    _, dataset_id, metaheuristic = parts

    records = parse_log(log_path)
    log_seeds = len(records)
    csv_path = _shard_path(dataset_id, metaheuristic)
    existing_rows = _existing_row_count(csv_path)

    summary = {
        "dataset_id": dataset_id,
        "metaheuristic": metaheuristic,
        "log_seeds": log_seeds,
        "existing_rows": existing_rows,
    }

    if log_seeds == 0:
        summary["action"] = "skip-empty"
        return summary

    # Never downgrade: if an existing CSV already has at least as many rows
    # as we could recover, leave it alone.
    if existing_rows >= log_seeds:
        summary["action"] = "skip-worse"
        return summary

    if dry_run:
        summary["action"] = "dry-run"
        return summary

    # Build the DataFrame with the full SHARD_COLUMNS schema. Columns not
    # emitted by the logger (internal_cost, time_to_best, cache_hits) are NaN
    # — downstream aggregators never read them.
    group = group_by_dataset.get(dataset_id)
    if group is None:
        logger.warning("  %s/%s: dataset not in manifest; skipping", dataset_id, metaheuristic)
        summary["action"] = "skip-unknown-dataset"
        return summary

    for rec in records:
        rec["group"] = group
        rec["internal_cost"] = None
        rec["time_to_best"] = None
        rec["cache_hits"] = None

    df = pd.DataFrame(records)[SHARD_COLUMNS].sort_values("seed").reset_index(drop=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    summary["action"] = "write"
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _print_summary_table(summaries: list[dict]) -> None:
    """Pretty-print a per-shard summary; mirrors the style of aggregate console output."""
    if not summaries:
        print("(no shard logs found)")
        return

    print(f"\n{'Dataset':<22} {'MH':<6} {'Log':>4} {'Existing':>8} {'Action':<18}")
    print("-" * 64)
    for s in sorted(summaries, key=lambda x: (x.get("dataset_id", ""), x.get("metaheuristic", ""))):
        print(
            f"{s.get('dataset_id', '-'):<22} "
            f"{s.get('metaheuristic', '-'):<6} "
            f"{s.get('log_seeds', 0):>4} "
            f"{s.get('existing_rows', 0):>8} "
            f"{s.get('action', '-'):<18}"
        )
    print()


def _print_totals(summaries: list[dict]) -> None:
    actions = {}
    recovered_seeds = 0
    for s in summaries:
        a = s.get("action", "-")
        actions[a] = actions.get(a, 0) + 1
        if a in ("write", "dry-run"):
            recovered_seeds += int(s.get("log_seeds", 0))

    print("Totals:")
    for a, n in sorted(actions.items()):
        print(f"  {a:<20} {n:>3}")
    print(f"  seeds recovered (write/dry-run): {recovered_seeds}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1] if __doc__ else "")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and summarise; do not write any CSVs.",
    )
    args = parser.parse_args()

    if not SHARDS_DIR.exists():
        logger.error("Shards directory does not exist: %s", SHARDS_DIR)
        sys.exit(1)

    manifest = load_manifest()
    group_by_dataset = dict(zip(manifest["dataset_id"], manifest["group"]))

    log_paths = sorted(SHARDS_DIR.glob("run_*.log"))
    logger.info(
        "Scanning %d shard logs under %s (dry_run=%s)",
        len(log_paths), SHARDS_DIR, args.dry_run,
    )

    summaries: list[dict] = []
    for log_path in log_paths:
        summary = recover_shard(log_path, group_by_dataset, dry_run=args.dry_run)
        summaries.append(summary)

    # Only show actionable entries in the detailed table.
    actionable = [s for s in summaries if s.get("action") in {"write", "dry-run", "skip-empty", "skip-worse"}]
    _print_summary_table(actionable)
    _print_totals(summaries)

    if args.dry_run:
        logger.info("Dry-run complete — no files written.")
    else:
        n_written = sum(1 for s in summaries if s.get("action") == "write")
        logger.info("Recovery complete: %d shard CSVs written to %s", n_written, SHARDS_DIR)


if __name__ == "__main__":
    main()
