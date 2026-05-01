"""
Experiment 03 — Aggregation: ClustGrade-RKO vs DBSCAN-DistK.

Reads two standalone CSVs and produces:
    1. Per-dataset comparison table (RKO best-of-N vs DBSCAN-DistK)
    2. Wilcoxon signed-rank tests (RKO vs DBSCAN-DistK) for MSI on all datasets
         and ARI on classification datasets
    3. Win/loss/tie counts
    4. Delta distribution bar charts (MSI + ARI)

Baseline: DBSCAN-DistK (Semaan 2012, 7 k* values x 4 rules, MinPts=5 fixed).

`comparison.csv` schema (wide):
    dataset_id, group, best_seed,
    k_rko, msi_rko, ari_rko,
    k_dbscan_distk, msi_dbscan_distk, ari_dbscan_distk,
    msi_delta_distk, ari_delta_distk

Downstream scripts (plot.py) must be updated to read this schema.

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.aggregate
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"

DBSCAN_VARIANTS: list[tuple[str, str]] = [
    ("DBSCAN-DistK", "distk"),      # baseline
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_raw() -> pd.DataFrame:
    rko = pd.read_csv(ARTIFACTS / "rko_runs.csv")
    distk = pd.read_csv(ARTIFACTS / "dbscan_distk_runs.csv")
    # Restrict DBSCAN frame to the columns aggregate.py actually uses; the
    # DistK runner carries extra provenance columns (k_star, rule, epsilon)
    # that would otherwise leak into the concatenated frame.
    keep = ["dataset_id", "group", "algorithm", "seed", "k", "msi", "ari"]
    return pd.concat(
        [rko, distk[keep]],
        ignore_index=True,
    )


def compute_rko_best(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset best RKO run (highest MSI) across seeds."""
    # This aggregate intentionally reports the deployed Exp 03 algorithm: a
    # mixed-solver RKO run (BRKGA+VNS+ILS, q_learning=True) summarized as the
    # best seed per dataset. It is therefore not numerically comparable to the
    # reviewer-only MH breakdown, which uses isolated MH medians from Exp 02.
    rko = df[df["algorithm"] == "ClustGrade-RKO"].copy()
    idx = rko.groupby(["dataset_id", "group"])["msi"].idxmax()
    best = rko.loc[idx.dropna()][["dataset_id", "group", "seed", "k", "msi", "ari"]].copy()
    best = best.rename(columns={
        "seed": "best_seed", "k": "k_rko", "msi": "msi_rko", "ari": "ari_rko",
    })
    return best.reset_index(drop=True)


def build_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Wide-format comparison with columns for DBSCAN-DistK."""
    merged = compute_rko_best(df)

    for algo_name, suffix in DBSCAN_VARIANTS:
        sub = df[df["algorithm"] == algo_name][
            ["dataset_id", "group", "k", "msi", "ari"]
        ].copy()
        sub = sub.rename(columns={
            "k": f"k_dbscan_{suffix}",
            "msi": f"msi_dbscan_{suffix}",
            "ari": f"ari_dbscan_{suffix}",
        })
        merged = merged.merge(sub, on=["dataset_id", "group"], how="left")
        merged[f"msi_delta_{suffix}"] = merged["msi_rko"] - merged[f"msi_dbscan_{suffix}"]
        merged[f"ari_delta_{suffix}"] = merged["ari_rko"] - merged[f"ari_dbscan_{suffix}"]

    return merged


# ---------------------------------------------------------------------------
# Statistical helpers
# ---------------------------------------------------------------------------

def wilcoxon_test(
    vals_a: np.ndarray, vals_b: np.ndarray, label: str,
) -> dict:
    mask = ~(np.isnan(vals_a) | np.isnan(vals_b))
    a_clean = vals_a[mask]
    b_clean = vals_b[mask]
    n = len(a_clean)

    if n < 5:
        return {"metric": label, "n_pairs": n, "statistic": None,
                "p_value": None, "significant": None}

    wt = stats.wilcoxon(a_clean, b_clean, alternative="two-sided")
    return {
        "metric": label,
        "n_pairs": n,
        "statistic": round(float(wt.statistic), 4),
        "p_value": round(float(wt.pvalue), 6),
        "significant": bool(wt.pvalue < 0.05),
    }


def win_loss_tie(deltas: np.ndarray, label: str, tol: float = 0.01) -> dict:
    valid = deltas[~np.isnan(deltas)]
    wins = int(np.sum(valid > tol))
    losses = int(np.sum(valid < -tol))
    ties = int(len(valid) - wins - losses)
    return {"metric": label, "rko_wins": wins, "rko_losses": losses, "ties": ties,
            "total": len(valid)}


def holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    m = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [False] * m
    for rank, (orig_idx, p) in enumerate(indexed):
        adjusted_alpha = alpha / (m - rank)
        if p <= adjusted_alpha:
            results[orig_idx] = True
        else:
            break
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison_barplot(
    comp: pd.DataFrame, metric: str, suffix: str, group_label: str, out_path: Path,
) -> None:
    """Side-by-side bar chart for RKO vs a single DBSCAN variant."""
    sub = comp.sort_values(f"{metric}_delta_{suffix}")
    ids = sub["dataset_id"].tolist()
    rko_vals = sub[f"{metric}_rko"].values
    dbscan_vals = sub[f"{metric}_dbscan_{suffix}"].values

    fig, ax = plt.subplots(figsize=(12, max(5, len(ids) * 0.3)))
    y = np.arange(len(ids))
    h = 0.35
    ax.barh(y - h / 2, rko_vals, h, label="ClustGrade-RKO", color="#3498db", alpha=0.7)
    ax.barh(y + h / 2, dbscan_vals, h, label=f"DBSCAN-{suffix}", color="#e74c3c", alpha=0.7)
    ax.set_yticks(y)
    ax.set_yticklabels(ids, fontsize=7)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Experiment 03: {metric.upper()} — RKO vs DBSCAN-{suffix} ({group_label})")
    ax.legend(loc="best")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)


def plot_delta_distribution(
    deltas: np.ndarray,
    dataset_ids: list[str],
    metric_name: str,
    suffix: str,
    out_path: Path,
) -> None:
    """Bar chart of per-dataset deltas (RKO - DBSCAN variant)."""
    mask = ~np.isnan(deltas)
    valid_deltas = deltas[mask]
    valid_ids = [d for d, m in zip(dataset_ids, mask) if m]

    order = np.argsort(valid_deltas)
    sorted_deltas = valid_deltas[order]
    sorted_ids = [valid_ids[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, max(6, len(sorted_ids) * 0.25)))
    colors = ["#2ecc71" if d > 0 else "#e74c3c" for d in sorted_deltas]
    ax.barh(range(len(sorted_ids)), sorted_deltas, color=colors, edgecolor="none")
    ax.set_yticks(range(len(sorted_ids)))
    ax.set_yticklabels(sorted_ids, fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(f"{metric_name} delta (RKO - DBSCAN-{suffix})")
    ax.set_title(f"Experiment 03: Per-dataset {metric_name} delta (vs DBSCAN-{suffix})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    df = load_raw()
    logger.info("Loaded %d raw runs (RKO + DBSCAN-DistK)", len(df))

    comp = build_comparison(df)
    comp.to_csv(ARTIFACTS / "comparison.csv", index=False)

    classf = comp[comp["group"] == "classf"]

    print("\n" + "=" * 80)
    print("EXPERIMENT 03 — RKO vs DBSCAN-DistK")
    print("=" * 80)

    all_tests: list[dict] = []
    all_wlt: list[dict] = []

    print(f"\n--- MSI_ALL ({len(comp)} datasets) ---")
    msi_rko = comp["msi_rko"].values.astype(float)
    print(f"  Median MSI (RKO):              {np.nanmedian(msi_rko):.4f}")

    for algo_name, suffix in DBSCAN_VARIANTS:
        msi_db = comp[f"msi_dbscan_{suffix}"].values.astype(float)
        msi_delta = comp[f"msi_delta_{suffix}"].values.astype(float)

        print(f"  Median MSI ({algo_name}): {np.nanmedian(msi_db):.4f}")

        wt_msi = wilcoxon_test(msi_rko, msi_db, f"MSI_ALL_vs_{suffix}")
        wlt_msi = win_loss_tie(msi_delta, f"MSI_ALL_vs_{suffix}")
        all_tests.append(wt_msi)
        all_wlt.append(wlt_msi)

        print(
            f"    MSI W/L/T: RKO wins={wlt_msi['rko_wins']}, "
            f"losses={wlt_msi['rko_losses']}, ties={wlt_msi['ties']}"
        )
        print(f"    MSI Wilcoxon: stat={wt_msi['statistic']}, p={wt_msi['p_value']}")

    print(f"\n--- ARI_CLASSF ({len(classf)} datasets) ---")
    ari_rko = classf["ari_rko"].values.astype(float)
    print(f"  Median ARI (RKO):              {np.nanmedian(ari_rko):.4f}")

    for algo_name, suffix in DBSCAN_VARIANTS:
        ari_db = classf[f"ari_dbscan_{suffix}"].values.astype(float)
        ari_delta = classf[f"ari_delta_{suffix}"].values.astype(float)

        print(f"  Median ARI ({algo_name}): {np.nanmedian(ari_db):.4f}")

        wt_ari = wilcoxon_test(ari_rko, ari_db, f"ARI_CLASSF_vs_{suffix}")
        wlt_ari = win_loss_tie(ari_delta, f"ARI_CLASSF_vs_{suffix}")
        all_tests.append(wt_ari)
        all_wlt.append(wlt_ari)

        print(
            f"    ARI W/L/T: RKO wins={wlt_ari['rko_wins']}, "
            f"losses={wlt_ari['rko_losses']}, ties={wlt_ari['ties']}"
        )
        print(f"    ARI Wilcoxon: stat={wt_ari['statistic']}, p={wt_ari['p_value']}")

    # --- Holm-Bonferroni across the two reported tests only ---
    p_vals = [t["p_value"] if t["p_value"] is not None else 1.0 for t in all_tests]
    corrections = holm_bonferroni(p_vals)
    for test, sig in zip(all_tests, corrections):
        test["significant_holm"] = sig

    print("\n--- Holm-Bonferroni corrected significance ---")
    for t in all_tests:
        print(f"  {t['metric']}: p={t['p_value']}, sig_holm={t.get('significant_holm')}")

    # --- Save artifacts ---
    pd.DataFrame(all_tests).to_csv(ARTIFACTS / "statistical_tests.csv", index=False)
    pd.DataFrame(all_wlt).to_csv(ARTIFACTS / "win_loss_tie.csv", index=False)

    # --- Comparison bar plots (per DBSCAN variant) ---
    for algo_name, suffix in DBSCAN_VARIANTS:
        plot_comparison_barplot(
            comp, "msi", suffix, "ALL",
            ARTIFACTS / f"barplot_msi_all_{suffix}.png",
        )
        if len(classf) > 0:
            plot_comparison_barplot(
                classf, "ari", suffix, "CLASSF",
                ARTIFACTS / f"barplot_ari_classf_{suffix}.png",
            )

    # --- Delta bar charts (per DBSCAN variant) ---
    for _algo_name, suffix in DBSCAN_VARIANTS:
        all_msi_delta = comp[f"msi_delta_{suffix}"].values.astype(float)
        plot_delta_distribution(
            all_msi_delta, comp["dataset_id"].tolist(), "MSI", suffix,
            ARTIFACTS / f"msi_delta_barplot_{suffix}.png",
        )
        if len(classf) > 0:
            ari_delta = classf[f"ari_delta_{suffix}"].values.astype(float)
            plot_delta_distribution(
                ari_delta, classf["dataset_id"].tolist(), "ARI", suffix,
                ARTIFACTS / f"ari_delta_barplot_{suffix}.png",
            )

    logger.info("Aggregation complete.")


if __name__ == "__main__":
    main()
