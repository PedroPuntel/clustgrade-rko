"""
Experiment 02 — Aggregation: Q-learning impact.

Reads raw_runs.csv and produces:
  1. Per-dataset median MSI/ARI per Q-learning setting
  2. Wilcoxon signed-rank tests (MSI all datasets + ARI CLASSF)
  3. Win/loss/tie counts
  4. Boxplots and delta bar charts (MSI + ARI)

Usage:
    poetry run python -m experiments.02-qlearning-mh-comparison.aggregate
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


def load_raw() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "raw_runs.csv")


def compute_per_dataset_medians(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby(["dataset_id", "group", "q_learning"]).agg(
        msi_median=("msi", "median"),
        ari_median=("ari", "median"),
        k_median=("k", "median"),
        eval_median=("eval_count", "median"),
    ).reset_index()
    return grouped


def plot_delta_distribution(
    deltas: np.ndarray,
    dataset_ids: list[str],
    metric_name: str,
    out_path: Path,
) -> None:
    """Bar chart of per-dataset deltas (QL=ON - QL=OFF)."""
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
    ax.set_xlabel(f"{metric_name} delta (QL=ON - QL=OFF)")
    ax.set_title(f"Experiment 02: Per-dataset {metric_name} delta")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)


def _wilcoxon_test(
    values_off: np.ndarray,
    values_on: np.ndarray,
    label: str,
) -> dict:
    """Paired Wilcoxon signed-rank test. Returns dict with test results."""
    mask = ~(np.isnan(values_off) | np.isnan(values_on))
    off_clean = values_off[mask]
    on_clean = values_on[mask]
    n_pairs = len(off_clean)

    if n_pairs < 5:
        logger.warning("  %s: only %d valid pairs — skipping Wilcoxon", label, n_pairs)
        return {"metric": label, "n_pairs": n_pairs, "statistic": None,
                "p_value": None, "significant": None}

    stat_result = stats.wilcoxon(off_clean, on_clean, alternative="two-sided")
    return {
        "metric": label,
        "n_pairs": n_pairs,
        "statistic": round(float(stat_result.statistic), 4),
        "p_value": round(float(stat_result.pvalue), 6),
        "significant": bool(stat_result.pvalue < 0.05),
    }


def _win_loss_tie(deltas: np.ndarray, label: str, tol: float = 0.01) -> dict:
    """Count wins (QL=ON > QL=OFF), losses (QL=ON < QL=OFF), ties."""
    valid = deltas[~np.isnan(deltas)]
    wins = int(np.sum(valid > tol))
    losses = int(np.sum(valid < -tol))
    ties = int(len(valid) - wins - losses)
    return {"metric": label, "ql_wins": wins, "ql_losses": losses, "ties": ties,
            "total": len(valid)}


def _holm_bonferroni(p_values: list[float], alpha: float = 0.05) -> list[bool]:
    """Holm-Bonferroni correction. Returns list of booleans (significant or not)."""
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


def main() -> None:
    df = load_raw()
    logger.info("Loaded %d raw runs", len(df))

    medians = compute_per_dataset_medians(df)
    medians.to_csv(ARTIFACTS / "per_dataset_medians.csv", index=False)

    # --- MSI analysis (all datasets) ---
    msi_wide = medians.pivot_table(
        index=["dataset_id", "group"], columns="q_learning", values="msi_median",
    ).reset_index()
    msi_wide.columns.name = None
    msi_wide = msi_wide.rename(columns={False: "msi_off", True: "msi_on"})
    msi_wide["msi_delta"] = msi_wide["msi_on"] - msi_wide["msi_off"]
    msi_wide.to_csv(ARTIFACTS / "msi_comparison.csv", index=False)

    msi_off = msi_wide["msi_off"].values.astype(float)
    msi_on = msi_wide["msi_on"].values.astype(float)
    msi_deltas = msi_wide["msi_delta"].values.astype(float)

    wt_msi = _wilcoxon_test(msi_off, msi_on, "MSI")
    wlt_msi = _win_loss_tie(msi_deltas, "MSI")

    # --- ARI analysis (CLASSF datasets) ---
    ari_wide = medians.pivot_table(
        index=["dataset_id", "group"], columns="q_learning", values="ari_median",
    ).reset_index()
    ari_wide.columns.name = None
    ari_wide = ari_wide.rename(columns={False: "ari_off", True: "ari_on"})
    ari_wide["ari_delta"] = ari_wide["ari_on"] - ari_wide["ari_off"]
    ari_classf = ari_wide[ari_wide["group"] == "classf"].copy()
    ari_classf.to_csv(ARTIFACTS / "ari_comparison.csv", index=False)

    ari_off = ari_classf["ari_off"].values.astype(float)
    ari_on = ari_classf["ari_on"].values.astype(float)
    ari_deltas = ari_classf["ari_delta"].values.astype(float)

    wt_ari = _wilcoxon_test(ari_off, ari_on, "ARI")
    wlt_ari = _win_loss_tie(ari_deltas, "ARI")

    # --- Holm-Bonferroni correction ---
    p_values = []
    for wt in [wt_msi, wt_ari]:
        p_values.append(wt["p_value"] if wt["p_value"] is not None else 1.0)
    corrections = _holm_bonferroni(p_values)
    wt_msi["significant_holm"] = corrections[0]
    wt_ari["significant_holm"] = corrections[1]

    # --- Print summary ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 02 — Q-LEARNING IMPACT")
    print("=" * 80)

    print("\n--- MSI (all datasets) ---")
    print(f"  Median MSI (QL=OFF): {np.nanmedian(msi_off):.4f}")
    print(f"  Median MSI (QL=ON):  {np.nanmedian(msi_on):.4f}")
    print(f"  Win/Loss/Tie: QL wins={wlt_msi['ql_wins']}, "
          f"losses={wlt_msi['ql_losses']}, ties={wlt_msi['ties']}")
    print(f"  Wilcoxon: stat={wt_msi['statistic']}, p={wt_msi['p_value']}, "
          f"sig(Holm)={wt_msi['significant_holm']}")

    print("\n--- ARI (CLASSF datasets) ---")
    print(f"  Median ARI (QL=OFF): {np.nanmedian(ari_off):.4f}")
    print(f"  Median ARI (QL=ON):  {np.nanmedian(ari_on):.4f}")
    print(f"  Win/Loss/Tie: QL wins={wlt_ari['ql_wins']}, "
          f"losses={wlt_ari['ql_losses']}, ties={wlt_ari['ties']}")
    print(f"  Wilcoxon: stat={wt_ari['statistic']}, p={wt_ari['p_value']}, "
          f"sig(Holm)={wt_ari['significant_holm']}")

    # --- Save statistical test results ---
    stats_df = pd.DataFrame([wt_msi, wt_ari])
    stats_df.to_csv(ARTIFACTS / "statistical_tests.csv", index=False)

    wlt_df = pd.DataFrame([wlt_msi, wlt_ari])
    wlt_df.to_csv(ARTIFACTS / "win_loss_tie.csv", index=False)

    # --- Boxplot MSI ---
    fig, ax = plt.subplots(figsize=(6, 5))
    data_lists = [msi_off[~np.isnan(msi_off)], msi_on[~np.isnan(msi_on)]]
    bp = ax.boxplot(data_lists, labels=["QL=OFF", "QL=ON"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#e74c3c")
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax.set_ylabel("Per-dataset median MSI")
    ax.set_title("Experiment 02: Q-learning impact on MSI")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "boxplot_msi.png", dpi=150)
    plt.close(fig)

    # --- Boxplot ARI (CLASSF) ---
    fig, ax = plt.subplots(figsize=(6, 5))
    data_lists = [ari_off[~np.isnan(ari_off)], ari_on[~np.isnan(ari_on)]]
    bp = ax.boxplot(data_lists, labels=["QL=OFF", "QL=ON"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#3498db")
    bp["boxes"][1].set_facecolor("#e74c3c")
    for box in bp["boxes"]:
        box.set_alpha(0.6)
    ax.set_ylabel("Per-dataset median ARI")
    ax.set_title("Experiment 02: Q-learning impact on ARI (CLASSF)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "boxplot_ari_classf.png", dpi=150)
    plt.close(fig)

    # --- Delta bar charts ---
    plot_delta_distribution(
        msi_deltas, msi_wide["dataset_id"].tolist(), "MSI",
        ARTIFACTS / "msi_delta_barplot.png",
    )
    plot_delta_distribution(
        ari_deltas, ari_classf["dataset_id"].tolist(), "ARI",
        ARTIFACTS / "ari_delta_barplot.png",
    )

    logger.info("Aggregation complete.")


if __name__ == "__main__":
    main()
