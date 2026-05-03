"""
Experiment 01 — Aggregation: win/loss/tie counts, boxplots, descriptive stats,
Friedman test + Wilcoxon/Holm-Bonferroni post-hoc.

Reads raw_runs.csv and produces:
  1. Per-dataset median MSI/ARI per config -> summary table
  2. Win/loss/tie counts across all 4 configs
  3. Boxplots of MSI/ARI distributions
  4. Friedman test + (if significant) Wilcoxon pairwise post-hoc with Holm-Bonferroni
  5. Best config selection for downstream Exp 03 (serialized to best_config.json)

Usage:
    poetry run python -m experiments.01-param-study.aggregate
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"

ALPHA = 0.05


def load_raw() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "raw_runs.csv")


def config_label(row: pd.Series) -> str:
    qf = "QF" if row["quadrat_filter"] else "noQF"
    ms = row["msi_space"].upper()
    return f"{qf}+{ms}"


def compute_per_dataset_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Per-dataset median MSI/ARI/K grouped by config."""
    grouped = (
        df.groupby(["dataset_id", "group", "quadrat_filter", "msi_space"])
        .agg(
            msi_median=("msi", "median"),
            ari_median=("ari", "median"),
            k_median=("k", "median"),
        )
        .reset_index()
    )
    grouped["config"] = grouped.apply(config_label, axis=1)
    return grouped


def pairwise_win_loss(medians: pd.DataFrame, metric: str, tol: float = 0.01) -> pd.DataFrame:
    """Win/loss/tie between all config pairs on a given metric.

    NOTE: tol=0.01 is a deliberate *clinically-meaningful* threshold — a 1% MSI
    delta — NOT a rounding artifact. Requires full-precision input from
    raw_runs.csv; earlier versions truncated to 2 decimals, which collapsed the
    tie band onto the truncation step and made ties meaningless.
    """
    col = f"{metric}_median"
    configs = sorted(medians["config"].unique())
    rows = []

    for c1, c2 in itertools.combinations(configs, 2):
        d1 = medians[medians["config"] == c1][["dataset_id", col]].set_index("dataset_id")
        d2 = medians[medians["config"] == c2][["dataset_id", col]].set_index("dataset_id")
        joined = d1.join(d2, lsuffix="_1", rsuffix="_2", how="inner").dropna()
        delta = joined[f"{col}_2"] - joined[f"{col}_1"]
        rows.append(
            {
                "config_a": c1,
                "config_b": c2,
                "metric": metric,
                "a_wins": int((delta < -tol).sum()),
                "b_wins": int((delta > tol).sum()),
                "ties": int(((delta >= -tol) & (delta <= tol)).sum()),
                "total": len(delta),
            }
        )
    return pd.DataFrame(rows)


def overall_ranking(medians: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Rank configs by overall median of per-dataset medians."""
    col = f"{metric}_median"
    ranking = (
        medians.groupby("config")[col]
        .agg(["median", "mean", "std", "count"])
        .sort_values("median", ascending=False)
        .reset_index()
    )
    ranking.columns = ["config", "median", "mean", "std", "n_datasets"]
    return ranking


def plot_boxplots(medians: pd.DataFrame, metric: str, out_path: Path) -> None:
    """Boxplot of per-dataset median metric across configs."""
    col = f"{metric}_median"
    configs = sorted(medians["config"].unique())

    fig, ax = plt.subplots(figsize=(8, 5))
    data_lists = [medians[medians["config"] == c][col].dropna().values for c in configs]
    # NOTE: tick_labels= replaces the deprecated labels= kwarg (matplotlib >= 3.9).
    bp = ax.boxplot(data_lists, tick_labels=configs, patch_artist=True)
    colors = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    ax.set_ylabel(f"Per-dataset median {metric}")
    ax.set_title(f"Experiment 01: {metric} by RKO configuration")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    logger.info("  Saved %s", out_path.name)


def friedman_posthoc(medians: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Friedman test on per-dataset medians across 4 configs, with Wilcoxon
    post-hoc + Holm-Bonferroni correction if Friedman is significant.

    NOTE: Friedman is the canonical non-parametric ANOVA-equivalent for k>2
    matched samples (here: k=4 configs × N datasets block design). If p<0.05,
    we run all 6 pairwise Wilcoxon signed-rank tests and correct via
    Holm-Bonferroni (step-down Bonferroni) — same correction scheme as Exp 04,
    but applied to 6 comparisons here instead of 2.

    Returns a DataFrame with columns:
        test, config_a, config_b, statistic, p_value, p_holm, significant
    Row 0 is the omnibus Friedman row (config_a/config_b empty). Rows 1-6 are
    pairwise Wilcoxon post-hoc (only written if Friedman is significant).
    """
    col = f"{metric}_median"
    configs = sorted(medians["config"].unique())

    # Pivot to (dataset, config) wide form, drop datasets missing any config.
    wide = medians.pivot_table(
        index="dataset_id",
        columns="config",
        values=col,
        aggfunc="first",
    )[configs].dropna()

    if len(wide) < 2:
        logger.warning(
            "  Friedman (%s): only %d complete datasets — skipping test.",
            metric,
            len(wide),
        )
        return pd.DataFrame(
            columns=[
                "test",
                "config_a",
                "config_b",
                "statistic",
                "p_value",
                "p_holm",
                "significant",
                "n_datasets",
            ]
        )

    # --- Omnibus Friedman test ---
    friedman_stat, friedman_p = stats.friedmanchisquare(*[wide[c].values for c in configs])
    friedman_sig = friedman_p < ALPHA

    rows = [
        {
            "test": "friedman",
            "config_a": "",
            "config_b": "",
            "statistic": float(friedman_stat),
            "p_value": float(friedman_p),
            "p_holm": float(friedman_p),  # identity for the omnibus row
            "significant": bool(friedman_sig),
            "n_datasets": int(len(wide)),
        }
    ]

    if not friedman_sig:
        logger.info(
            "  Friedman (%s): chi2=%.4f, p=%.4f (N=%d) — NOT significant; " "skipping post-hoc.",
            metric,
            friedman_stat,
            friedman_p,
            len(wide),
        )
        return pd.DataFrame(rows)

    logger.info(
        "  Friedman (%s): chi2=%.4f, p=%.4f (N=%d) — significant; running "
        "Wilcoxon post-hoc with Holm-Bonferroni.",
        metric,
        friedman_stat,
        friedman_p,
        len(wide),
    )

    # --- Pairwise Wilcoxon ---
    pairs = list(itertools.combinations(configs, 2))
    pairwise = []
    for c1, c2 in pairs:
        diff = wide[c2].values - wide[c1].values
        # NOTE: scipy.stats.wilcoxon's default zero_method="wilcox" drops zero-
        # difference pairs. With Holm-Bonferroni we rely on p-values being
        # comparable under the null, which this zero-drop behavior preserves.
        try:
            w_stat, w_p = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
        except ValueError:
            # All differences are zero — degenerate, no signal to test.
            w_stat, w_p = float("nan"), 1.0
        pairwise.append(
            {
                "config_a": c1,
                "config_b": c2,
                "statistic": float(w_stat),
                "p_value": float(w_p),
            }
        )

    # --- Holm-Bonferroni step-down correction on m=6 ---
    m = len(pairwise)
    # NOTE: Hand-rolled Holm: sort ascending, multiply by (m - rank), enforce
    # monotonicity, cap at 1.0. Identical procedure to Exp 04's aggregate.py;
    # kept explicit here because m=6 is larger than Exp 04's m=2 and the
    # monotonicity step actually matters (ranks 2-5 can be pulled up by rank 1).
    sorted_pairs = sorted(pairwise, key=lambda r: r["p_value"])
    last_p = 0.0
    for rank, r in enumerate(sorted_pairs):
        raw = r["p_value"] * (m - rank)
        adjusted = min(max(raw, last_p), 1.0)
        r["p_holm"] = adjusted
        last_p = adjusted

    for r in sorted_pairs:
        rows.append(
            {
                "test": "wilcoxon_holm",
                "config_a": r["config_a"],
                "config_b": r["config_b"],
                "statistic": r["statistic"],
                "p_value": r["p_value"],
                "p_holm": r["p_holm"],
                "significant": bool(r["p_holm"] < ALPHA),
                "n_datasets": int(len(wide)),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    df = load_raw()
    logger.info("Loaded %d raw runs", len(df))

    medians = compute_per_dataset_medians(df)
    medians.to_csv(ARTIFACTS / "per_dataset_medians.csv", index=False)

    # --- Rankings ---
    print("\n" + "=" * 80)
    print("EXPERIMENT 01 — AGGREGATION RESULTS")
    print("=" * 80)

    for metric in ("msi", "ari"):
        ranking = overall_ranking(medians, metric)
        ranking.to_csv(ARTIFACTS / f"ranking_{metric}.csv", index=False)
        subset_label = "all datasets" if metric == "msi" else "CLASSF only"

        if metric == "ari":
            ranking_classf = overall_ranking(
                medians[medians["group"] == "classf"],
                metric,
            )
            ranking_classf.to_csv(ARTIFACTS / f"ranking_{metric}_classf.csv", index=False)
            ranking = ranking_classf

        print(f"\n--- {metric.upper()} ranking ({subset_label}) ---")
        print(ranking.to_string(index=False))

    # --- Win/loss/tie ---
    for metric in ("msi", "ari"):
        if metric == "ari":
            subset = medians[medians["group"] == "classf"]
        else:
            subset = medians
        wlt = pairwise_win_loss(subset, metric)
        wlt.to_csv(ARTIFACTS / f"win_loss_tie_{metric}.csv", index=False)
        print(f"\n--- {metric.upper()} win/loss/tie ---")
        print(wlt.to_string(index=False))

    # --- Boxplots ---
    plot_boxplots(medians, "msi", ARTIFACTS / "boxplot_msi.png")
    plot_boxplots(
        medians[medians["group"] == "classf"],
        "ari",
        ARTIFACTS / "boxplot_ari_classf.png",
    )

    # --- Statistical tests: Friedman + Wilcoxon/Holm post-hoc ---
    print("\n--- Friedman test + Wilcoxon/Holm-Bonferroni post-hoc ---")
    stats_frames: list[pd.DataFrame] = []

    msi_stats = friedman_posthoc(medians, "msi")
    if not msi_stats.empty:
        msi_stats.insert(0, "metric", "msi")
        stats_frames.append(msi_stats)
        print("\n  MSI (all datasets):")
        print(msi_stats.drop(columns=["metric"]).to_string(index=False))

    # NOTE: ARI test runs on CLASSF only (8 datasets). 8 is on the low end for
    # Friedman with k=4 — report but caveat in results.md.
    ari_stats = friedman_posthoc(medians[medians["group"] == "classf"], "ari")
    if not ari_stats.empty:
        ari_stats.insert(0, "metric", "ari")
        stats_frames.append(ari_stats)
        print("\n  ARI (CLASSF only):")
        print(ari_stats.drop(columns=["metric"]).to_string(index=False))

    if stats_frames:
        all_stats = pd.concat(stats_frames, ignore_index=True)
        all_stats.to_csv(ARTIFACTS / "statistical_tests.csv", index=False)
        logger.info("  Saved statistical_tests.csv")

    # --- Best config selection ---
    # NOTE: Selection is purely descriptive (highest median MSI over all 20
    # datasets). The Friedman/Holm tests above determine whether that winner
    # is *significantly* different from the runners-up; check statistical_tests.csv
    # before promoting this config to Exp 03.
    msi_ranking = overall_ranking(medians, "msi")
    best_row = msi_ranking.iloc[0]
    best_config = str(best_row["config"])
    best_median = float(best_row["median"])

    # Parse config label back into (quadrat_filter, msi_space).
    qf_part, ms_part = best_config.split("+")
    best_qf = qf_part == "QF"
    best_ms = ms_part.lower()

    # Look up Friedman p-value for provenance.
    friedman_p = None
    friedman_sig = False
    if stats_frames:
        all_stats = pd.concat(stats_frames, ignore_index=True)
        friedman_row = all_stats[(all_stats["metric"] == "msi") & (all_stats["test"] == "friedman")]
        if not friedman_row.empty:
            friedman_p = float(friedman_row.iloc[0]["p_value"])
            friedman_sig = bool(friedman_row.iloc[0]["significant"])

    best_config_payload = {
        "config": best_config,
        "quadrat_filter": best_qf,
        "msi_space": best_ms,
        "selected_by": "highest_median_msi_all_datasets",
        "median_msi": best_median,
        "friedman_p_msi": friedman_p,
        "friedman_significant_msi": friedman_sig,
    }
    with open(ARTIFACTS / "best_config.json", "w") as f:
        json.dump(best_config_payload, f, indent=2)
    logger.info("  Saved best_config.json")

    print(f"\n>>> Best config by median MSI: {best_config}  (median={best_median:.4f})")
    if friedman_p is not None:
        print(
            f"    Friedman p (MSI) = {friedman_p:.4f}  "
            f"{'[significant]' if friedman_sig else '[NOT significant]'}"
        )
    print("    Consult statistical_tests.csv before promoting to Exp 03.")

    logger.info("Aggregation complete.")


if __name__ == "__main__":
    main()
