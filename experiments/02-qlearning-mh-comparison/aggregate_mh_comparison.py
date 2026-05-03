"""
Experiment 02 — Aggregation: metaheuristic comparison (BRKGA vs VNS vs ILS).

Reads artifacts/mh_comparison/raw_runs.csv and produces:
  1. Per-dataset median MSI/ARI/K per metaheuristic (wide form).
  2. Overall rankings per metric.
  3. Pairwise win/loss/tie counts (3 pairs per metric, tol = 0.01).
  4. Friedman omnibus test (k=3 matched blocks) + pairwise Wilcoxon post-hoc,
     with Holm-Bonferroni correction across the 3-test family.
  5. best_mh.json with the winning metaheuristic and full provenance.

Statistical design:
    - MSI: Friedman across all 50 datasets, k=3 (brkga, vns, ils) matched blocks.
    - ARI: Friedman across 18 CLASSF datasets, same k=3 structure.
    - Post-hoc: Wilcoxon signed-rank for each of 3 MH pairs per metric, with
      Holm-Bonferroni correction over the per-metric family (m=3).

Usage:
    poetry run python -m experiments.02-qlearning-mh-comparison.aggregate_mh_comparison
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison"
_SMOKE_ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison_smoke"
RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"
N_SEEDS_COMPLETE: int = 20  # a (dataset, MH) shard is "complete" when it has this many seeds


def _apply_smoke_overrides() -> None:
    global ARTIFACTS, RAW_RUNS_PATH
    ARTIFACTS = _SMOKE_ARTIFACTS
    RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"


METAHEURISTICS: list[str] = ["brkga", "vns", "ils"]
MH_LABELS: dict[str, str] = {"brkga": "BRKGA", "vns": "VNS", "ils": "ILS"}
WLT_TOL: float = 0.01  # clinically-meaningful MSI delta (matches Exp 01 convention)


# ---------------------------------------------------------------------------
# Loading + per-dataset medians
# ---------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    if not RAW_RUNS_PATH.exists():
        raise FileNotFoundError(
            f"Expected raw runs at {RAW_RUNS_PATH}. Run run_mh_comparison.py first."
        )
    return pd.read_csv(RAW_RUNS_PATH)


def compute_per_dataset_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Long-form medians grouped by (dataset_id, group, metaheuristic)."""
    return (
        df.groupby(["dataset_id", "group", "metaheuristic"])
        .agg(
            msi_median=("msi", "median"),
            ari_median=("ari", "median"),
            k_median=("k", "median"),
            eval_median=("eval_count", "median"),
            time_to_best_median=("time_to_best", "median"),
        )
        .reset_index()
    )


def pivot_wide(medians: pd.DataFrame, value_col: str) -> pd.DataFrame:
    """Pivot long medians into wide form: one row per dataset, one column per MH."""
    wide = medians.pivot_table(
        index=["dataset_id", "group"],
        columns="metaheuristic",
        values=value_col,
    ).reset_index()
    wide.columns.name = None
    # Enforce deterministic column order.
    ordered_cols = ["dataset_id", "group"] + [mh for mh in METAHEURISTICS if mh in wide.columns]
    return wide[ordered_cols]


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------


def friedman_on_matrix(wide: pd.DataFrame, label: str) -> dict:
    """Friedman test across the 3 MH columns of a wide table. Drops rows with NaN."""
    cols = [mh for mh in METAHEURISTICS if mh in wide.columns]
    block = wide[cols].dropna()
    n = len(block)
    if n < 5:
        logger.warning("  Friedman [%s]: only %d complete blocks — skipping", label, n)
        return {
            "metric": label,
            "test": "friedman",
            "n_blocks": n,
            "statistic": None,
            "p_value": None,
            "significant": None,
        }
    stat, p = stats.friedmanchisquare(*[block[c].values for c in cols])
    return {
        "metric": label,
        "test": "friedman",
        "n_blocks": n,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant": bool(p < 0.05),
    }


def pairwise_wilcoxon(wide: pd.DataFrame, label: str) -> list[dict]:
    """Pairwise Wilcoxon signed-rank tests across the 3 MH columns."""
    rows: list[dict] = []
    cols = [mh for mh in METAHEURISTICS if mh in wide.columns]
    for a, b in combinations(cols, 2):
        paired = wide[[a, b]].dropna()
        n_pairs = len(paired)
        if n_pairs < 5:
            rows.append(
                {
                    "metric": label,
                    "test": "wilcoxon",
                    "pair": f"{a}_vs_{b}",
                    "n_pairs": n_pairs,
                    "statistic": None,
                    "p_value": None,
                    "significant": None,
                }
            )
            continue
        diffs = paired[a].values - paired[b].values
        if np.all(diffs == 0):
            rows.append(
                {
                    "metric": label,
                    "test": "wilcoxon",
                    "pair": f"{a}_vs_{b}",
                    "n_pairs": n_pairs,
                    "statistic": 0.0,
                    "p_value": 1.0,
                    "significant": False,
                }
            )
            continue
        res = stats.wilcoxon(paired[a].values, paired[b].values, alternative="two-sided")
        rows.append(
            {
                "metric": label,
                "test": "wilcoxon",
                "pair": f"{a}_vs_{b}",
                "n_pairs": n_pairs,
                "statistic": round(float(res.statistic), 4),
                "p_value": round(float(res.pvalue), 6),
                "significant": bool(res.pvalue < 0.05),
            }
        )
    return rows


def holm_bonferroni(
    p_values: list[float | None],
    alpha: float = 0.05,
) -> tuple[list[bool | None], list[float | None]]:
    """Step-down Holm-Bonferroni correction over a family of p-values.

    Returns:
        (decisions, adjusted_p_values)
            - decisions: True/False per input (None for None inputs).
            - adjusted_p_values: running-max step-down adjusted p, capped at 1.0.
    """
    n = len(p_values)
    m = len([p for p in p_values if p is not None])
    if m == 0:
        return [None] * n, [None] * n

    order = sorted(
        [(i, p) for i, p in enumerate(p_values) if p is not None],
        key=lambda x: x[1],
    )
    decisions: list[bool | None] = [None] * n
    p_adj: list[float | None] = [None] * n

    running_max = 0.0
    for rank, (orig_idx, p) in enumerate(order):
        raw_adj = min(1.0, float(p) * (m - rank))
        running_max = max(running_max, raw_adj)
        p_adj[orig_idx] = round(running_max, 6)

    stopped = False
    for rank, (orig_idx, p) in enumerate(order):
        if stopped:
            decisions[orig_idx] = False
            continue
        adj_alpha = alpha / (m - rank)
        if p <= adj_alpha:
            decisions[orig_idx] = True
        else:
            decisions[orig_idx] = False
            stopped = True
    return decisions, p_adj


def win_loss_tie_pairwise(wide: pd.DataFrame, label: str, tol: float = WLT_TOL) -> list[dict]:
    """Per-pair win/loss/tie counts across datasets."""
    rows: list[dict] = []
    cols = [mh for mh in METAHEURISTICS if mh in wide.columns]
    for a, b in combinations(cols, 2):
        paired = wide[[a, b]].dropna()
        diffs = (paired[a].values - paired[b].values).astype(float)
        wins = int(np.sum(diffs > tol))
        losses = int(np.sum(diffs < -tol))
        ties = int(len(diffs) - wins - losses)
        rows.append(
            {
                "metric": label,
                "pair": f"{a}_vs_{b}",
                f"{a}_wins": wins,
                f"{b}_wins": losses,
                "ties": ties,
                "total": int(len(diffs)),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Rankings
# ---------------------------------------------------------------------------


def ranking_table(wide: pd.DataFrame, label: str) -> pd.DataFrame:
    """Overall ranking of metaheuristics by median-of-medians on the given metric."""
    columns = [
        "rank",
        "metaheuristic",
        "label",
        "n_datasets",
        "median",
        "mean",
        "std",
        "min",
        "max",
        "metric",
    ]
    rows = []
    for mh in METAHEURISTICS:
        if mh not in wide.columns:
            continue
        vals = wide[mh].dropna().values.astype(float)
        rows.append(
            {
                "metaheuristic": mh,
                "label": MH_LABELS[mh],
                "n_datasets": int(len(vals)),
                "median": round(float(np.median(vals)), 6) if len(vals) else None,
                "mean": round(float(np.mean(vals)), 6) if len(vals) else None,
                "std": round(float(np.std(vals, ddof=1)), 6) if len(vals) > 1 else None,
                "min": round(float(np.min(vals)), 6) if len(vals) else None,
                "max": round(float(np.max(vals)), 6) if len(vals) else None,
            }
        )

    if not rows:
        logger.warning("Ranking [%s]: no data for any metaheuristic — returning empty table", label)
        return pd.DataFrame(columns=columns)

    rank_df = pd.DataFrame(rows)
    rank_df = rank_df.sort_values(
        "median",
        ascending=False,
        na_position="last",
    ).reset_index(drop=True)
    rank_df.insert(0, "rank", rank_df.index + 1)
    rank_df["metric"] = label
    return rank_df[columns]


# ---------------------------------------------------------------------------
# Reviewer-format CSV (Friedman + Wilcoxon + Holm-adj p)
# ---------------------------------------------------------------------------


def format_statistical_tests_table(tests: list[dict]) -> pd.DataFrame:
    """Reshape the tests list into a polished, reviewer-friendly frame.

    Columns: familia, teste, comparativo, p_valor, p_valor_holm.
    """
    family_labels = {
        "MSI": "ISM",
        "ARI_CLASSF": "IRA",
    }
    rows: list[dict] = []
    for t in tests:
        family = t.get("metric", "")
        test_kind = t.get("test", "")
        if test_kind == "friedman":
            n = t.get("n_blocks")
            comparativo = "---"
            p_value = t.get("p_value")
            p_holm = f"{float(p_value):.4e}" if p_value is not None else "—"
        else:  # wilcoxon
            n = t.get("n_pairs")
            pair = t.get("pair", "")
            parts = pair.split("_vs_")
            comparativo = " vs. ".join(MH_LABELS.get(p, p.upper()) for p in parts)
            p_holm_value = t.get("p_holm")
            p_holm = f"{float(p_holm_value):.4f}" if p_holm_value is not None else "—"

        rows.append(
            {
                "_family": family,
                "familia": family_labels.get(family, family),
                "teste": test_kind.title() if test_kind else "",
                "comparativo": comparativo,
                "_n": n,
                "p_valor": t.get("p_value"),
                "p_valor_holm": p_holm,
            }
        )
    family_order = {"MSI": 0, "ARI_CLASSF": 1}
    test_order = {"Friedman": 0, "Wilcoxon": 1}
    return (
        pd.DataFrame(rows)
        .assign(
            _fo=lambda d: d["_family"].map(family_order).fillna(99),
            _to=lambda d: d["teste"].map(test_order).fillna(99),
        )
        .sort_values(["_fo", "_to"], kind="stable")
        .drop(columns=["_family", "_fo", "_to"])
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Read from the smoke-mode artifacts folder (artifacts/mh_comparison_smoke/).",
    )
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_overrides()
        logger.info("SMOKE MODE: reading from %s", ARTIFACTS)

    df = load_raw()
    logger.info("Loaded %d raw runs from %s", len(df), RAW_RUNS_PATH)

    # --- Per-dataset medians (long form) ---
    medians = compute_per_dataset_medians(df)
    medians.to_csv(ARTIFACTS / "per_dataset_medians_long.csv", index=False)

    # --- Wide pivots (one column per MH) ---
    msi_wide = pivot_wide(medians, "msi_median")
    ari_wide = pivot_wide(medians, "ari_median")
    k_wide = pivot_wide(medians, "k_median")

    # Consolidated wide table for inspection.
    msi_wide_r = msi_wide.rename(columns={mh: f"msi_{mh}" for mh in METAHEURISTICS})
    ari_wide_r = ari_wide.rename(columns={mh: f"ari_{mh}" for mh in METAHEURISTICS})
    k_wide_r = k_wide.rename(columns={mh: f"k_{mh}" for mh in METAHEURISTICS})
    combined = msi_wide_r.merge(ari_wide_r, on=["dataset_id", "group"], how="outer")
    combined = combined.merge(k_wide_r, on=["dataset_id", "group"], how="outer")
    combined.to_csv(ARTIFACTS / "per_dataset_medians.csv", index=False)

    ari_classf_wide = ari_wide[ari_wide["group"] == "classf"].copy()

    # --- Rankings ---
    ranking_msi = ranking_table(msi_wide, "MSI")
    ranking_msi.to_csv(ARTIFACTS / "ranking_msi.csv", index=False)

    ranking_ari = ranking_table(ari_classf_wide, "ARI_CLASSF")
    ranking_ari.to_csv(ARTIFACTS / "ranking_ari_classf.csv", index=False)

    # --- Win/loss/tie (3 pairs per metric) ---
    wlt_msi = win_loss_tie_pairwise(msi_wide, "MSI")
    pd.DataFrame(wlt_msi).to_csv(ARTIFACTS / "win_loss_tie_msi.csv", index=False)

    wlt_ari = win_loss_tie_pairwise(ari_classf_wide, "ARI_CLASSF")
    pd.DataFrame(wlt_ari).to_csv(ARTIFACTS / "win_loss_tie_ari.csv", index=False)

    # --- Friedman + pairwise Wilcoxon + Holm-Bonferroni ---
    tests: list[dict] = []

    fr_msi = friedman_on_matrix(msi_wide, "MSI")
    tests.append(fr_msi)
    wilcoxon_msi = pairwise_wilcoxon(msi_wide, "MSI")
    msi_pvals = [r["p_value"] for r in wilcoxon_msi]
    msi_holm, msi_p_adj = holm_bonferroni(msi_pvals)
    for r, h, p_adj in zip(wilcoxon_msi, msi_holm, msi_p_adj):
        r["significant_holm"] = h
        r["p_holm"] = p_adj
    tests.extend(wilcoxon_msi)

    fr_ari = friedman_on_matrix(ari_classf_wide, "ARI_CLASSF")
    tests.append(fr_ari)
    wilcoxon_ari = pairwise_wilcoxon(ari_classf_wide, "ARI_CLASSF")
    ari_pvals = [r["p_value"] for r in wilcoxon_ari]
    ari_holm, ari_p_adj = holm_bonferroni(ari_pvals)
    for r, h, p_adj in zip(wilcoxon_ari, ari_holm, ari_p_adj):
        r["significant_holm"] = h
        r["p_holm"] = p_adj
    tests.extend(wilcoxon_ari)

    pd.DataFrame(tests).to_csv(ARTIFACTS / "statistical_tests.csv", index=False)

    # --- Reviewer-format CSV (Friedman + Wilcoxon + Holm-adj p) ---
    reviewer_df = format_statistical_tests_table(tests)
    reviewer_df.drop(columns=["_n"]).to_csv(ARTIFACTS / "reviewer_stat_tests.csv", index=False)

    # --- best_mh.json (MSI ranking primary; Friedman p as provenance) ---
    best_row = ranking_msi.iloc[0]
    best_mh = {
        "metaheuristic": str(best_row["metaheuristic"]),
        "label": str(best_row["label"]),
        "median_msi": float(best_row["median"]) if pd.notna(best_row["median"]) else None,
        "mean_msi": float(best_row["mean"]) if pd.notna(best_row["mean"]) else None,
        "n_datasets": int(best_row["n_datasets"]),
        "friedman_p_msi": fr_msi.get("p_value"),
        "friedman_significant_msi": fr_msi.get("significant"),
        "friedman_p_ari_classf": fr_ari.get("p_value"),
        "friedman_significant_ari_classf": fr_ari.get("significant"),
    }
    with open(ARTIFACTS / "best_mh.json", "w") as f:
        json.dump(best_mh, f, indent=2)

    # --- Console summary ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 02 — METAHEURISTIC COMPARISON (BRKGA vs VNS vs ILS)")
    print("=" * 80)

    print("\n--- MSI ranking (all 50 datasets) ---")
    print(ranking_msi.to_string(index=False))
    print(
        f"\nFriedman MSI: stat={fr_msi['statistic']}, p={fr_msi['p_value']}, "
        f"significant={fr_msi['significant']}"
    )
    print("Pairwise Wilcoxon MSI (Holm-Bonferroni m=3):")
    for r in wilcoxon_msi:
        print(f"  {r['pair']}: p={r['p_value']}, sig_holm={r['significant_holm']}")

    print("\n--- ARI ranking (18 CLASSF datasets) ---")
    print(ranking_ari.to_string(index=False))
    print(
        f"\nFriedman ARI: stat={fr_ari['statistic']}, p={fr_ari['p_value']}, "
        f"significant={fr_ari['significant']}"
    )
    print("Pairwise Wilcoxon ARI (Holm-Bonferroni m=3):")
    for r in wilcoxon_ari:
        print(f"  {r['pair']}: p={r['p_value']}, sig_holm={r['significant_holm']}")

    print(f"\nBest MH (by median MSI): {best_mh['label']}  -> {ARTIFACTS / 'best_mh.json'}")

    print("\n--- Reviewer stat-tests table ---")
    print(reviewer_df.drop(columns=["_n"]).to_string(index=False))

    logger.info("Aggregation complete. Artifacts in %s", ARTIFACTS)


if __name__ == "__main__":
    main()
