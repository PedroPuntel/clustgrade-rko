"""
Experiment 03 — Per-metaheuristic comparison vs DBSCAN-DistK.

For each metaheuristic (BRKGA, VNS, ILS) independently, compare its per-dataset
median MSI/ARI against the DBSCAN-DistK baseline and report:

    - overall win proportion  = wins / total wins across the 3 metaheuristics
    - relative win proportion = wins / (wins + losses + ties)

Two scopes:
    - MSI across all 50 datasets (CLUST + CLASSF).
    - ARI across the 18 CLASSF datasets only.

Statistical tests: paired Wilcoxon (MH vs DBSCAN) + Holm-Bonferroni

Reads:
    - experiments/02-qlearning-mh-comparison/artifacts/mh_comparison/per_dataset_medians.csv
    - experiments/03-vs-dbscan-distk/artifacts/dbscan_distk_runs.csv

Writes (under experiments/03-vs-dbscan-distk/artifacts/mh_vs_dbscan/):
    mh_vs_dbscan_msi.csv
    mh_vs_dbscan_ari_classf.csv
    mh_vs_dbscan_gap_by_dataset.csv
    mh_vs_dbscan_gap_summary.csv
    mh_vs_dbscan_reviewer_gap.csv
    barplot_mh_vs_dbscan_msi_overall.png
    barplot_mh_vs_dbscan_msi_relative.png
    barplot_mh_vs_dbscan_ari_classf_overall.png
    barplot_mh_vs_dbscan_ari_classf_relative.png

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.run_mh_vs_dbscan
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_EXP06_MH_DIR = (
    _ROOT / "experiments" / "02-qlearning-mh-comparison"
    / "artifacts" / "mh_comparison"
)
EXP06_MEDIANS = _EXP06_MH_DIR / "per_dataset_medians.csv"
EXP07_DBSCAN = (
    _ROOT / "experiments" / "03-vs-dbscan-distk"
    / "artifacts" / "dbscan_distk_runs.csv"
)
EXP07_HYBRID_STATS = (
    _ROOT / "experiments" / "03-vs-dbscan-distk"
    / "artifacts" / "statistical_tests.csv"
)
EXP07_COMPARISON = (
    _ROOT / "experiments" / "03-vs-dbscan-distk"
    / "artifacts" / "comparison.csv"
)
OUT_DIR = (
    _ROOT / "experiments" / "03-vs-dbscan-distk"
    / "artifacts" / "mh_vs_dbscan"
)


# ---------------------------------------------------------------------------
# Constants (declared locally to avoid cross-experiment imports)
# ---------------------------------------------------------------------------

METAHEURISTICS: list[str] = ["brkga", "vns", "ils"]
MH_LABELS: dict[str, str] = {"brkga": "BRKGA", "vns": "VNS", "ils": "ILS"}
MH_COLORS: dict[str, str] = {"brkga": "#4C72B0", "vns": "#DD8452", "ils": "#55A868"}
DBSCAN_COLOR: str = "#8172B2"
HYBRID_METHOD_CODE: str = "hybrid"
HYBRID_LABEL: str = "ClustGrade-RKO Híbrido"
METHOD_ORDER: dict[str, int] = {"brkga": 0, "vns": 1, "ils": 2, HYBRID_METHOD_CODE: 3}

TOL: float = 0.01  # clinically-meaningful delta for win/loss; consistent with Exp 01/06.

ALGO_LABELS: dict[str, str] = {**MH_LABELS, "dbscan": "DBSCAN"}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})


# ---------------------------------------------------------------------------
# Statistical tests (paired Wilcoxon vs DBSCAN + Holm-Bonferroni)
# ---------------------------------------------------------------------------

def _pairwise_wilcoxon_vs_dbscan(
    wide: pd.DataFrame,
    label: str,
    metric_prefix: str,
) -> list[dict]:
    rows: list[dict] = []
    dbscan_col = f"{metric_prefix}_dbscan"
    if dbscan_col not in wide.columns:
        return rows
    for mh in METAHEURISTICS:
        mh_col = f"{metric_prefix}_{mh}"
        if mh_col not in wide.columns:
            continue
        paired = wide[[mh_col, dbscan_col]].dropna()
        n = len(paired)
        if n < 5:
            rows.append({"metric": label, "test": "wilcoxon", "pair": f"{mh_col}_vs_{dbscan_col}",
                         "n_pairs": n, "statistic": None, "p_value": None, "significant": None})
            continue
        diffs = paired[mh_col].values - paired[dbscan_col].values
        if np.all(diffs == 0):
            rows.append({"metric": label, "test": "wilcoxon", "pair": f"{mh_col}_vs_{dbscan_col}",
                         "n_pairs": n, "statistic": 0.0, "p_value": 1.0, "significant": False})
            continue
        res = stats.wilcoxon(
            paired[mh_col].values,
            paired[dbscan_col].values,
            alternative="two-sided",
        )
        rows.append({
            "metric": label, "test": "wilcoxon", "pair": f"{mh_col}_vs_{dbscan_col}",
            "n_pairs": n,
            "statistic": round(float(res.statistic), 4),
            "p_value": round(float(res.pvalue), 6),
            "significant": bool(res.pvalue < 0.05),
        })
    return rows


def _holm_bonferroni(
    p_values: list[float | None], alpha: float = 0.05,
) -> tuple[list[bool | None], list[float | None]]:
    n = len(p_values)
    m = len([p for p in p_values if p is not None])
    if m == 0:
        return [None] * n, [None] * n
    order = sorted([(i, p) for i, p in enumerate(p_values) if p is not None], key=lambda x: x[1])
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


def _format_reviewer_table(tests: list[dict]) -> pd.DataFrame:
    """Reshape tests list into a publication-ready reviewer table."""
    family_labels = {"MSI": "ISM", "ARI_CLASSF": "IRA"}
    rows: list[dict] = []
    for t in tests:
        family = t.get("metric", "")
        comparison = t.get("comparison_label")
        if comparison is None:
            pair = str(t.get("pair", ""))
            parts = pair.split("_vs_")
            comparison = " vs ".join(
                str(ALGO_LABELS.get(p.split("_", 1)[-1], p.upper()) or p.upper()) for p in parts
            )
        n = t.get("n_pairs")
        rows.append({
            "_family": family,
            "familia": family_labels.get(family, family),
            "n": n,
            "comparativo": comparison,
            "p_valor": t.get("p_value"),
        })
    family_order = {"MSI": 0, "ARI_CLASSF": 1}
    return (
        pd.DataFrame(rows)
        .assign(
            _fo=lambda d: d["_family"].map(family_order).fillna(99),
        )
        .sort_values(["_fo"], kind="stable")
        .drop(columns=["_family", "_fo"])
        .reset_index(drop=True)
    )


def run_statistical_tests(
    medians: pd.DataFrame, dbscan: pd.DataFrame, out_dir: Path,
) -> list[dict]:
    """Paired Wilcoxon (MH vs DBSCAN) + Holm-Bonferroni for each metric family.

    MSI uses all 50 datasets. ARI uses the 18 CLASSF datasets only.
    Holm-Bonferroni is applied within each metric family across the three
    MH-vs-DBSCAN comparisons.
    """
    merged = medians.merge(dbscan, on=["dataset_id", "group"], how="inner")

    # MSI: all datasets.
    msi_cols = [f"msi_{mh}" for mh in METAHEURISTICS if f"msi_{mh}" in merged.columns]
    msi_wide = merged[["dataset_id", "group"] + msi_cols + ["msi_dbscan"]].copy()

    # ARI: CLASSF only.
    ari_cols = [f"ari_{mh}" for mh in METAHEURISTICS if f"ari_{mh}" in merged.columns]
    ari_wide = merged.loc[
        merged["group"] == "classf",
        ["dataset_id", "group"] + ari_cols + ["ari_dbscan"],
    ].copy()

    all_tests: list[dict] = []

    wilcoxon_msi = _pairwise_wilcoxon_vs_dbscan(msi_wide, "MSI", "msi")
    msi_holm, msi_p_adj = _holm_bonferroni([r["p_value"] for r in wilcoxon_msi])
    for r, h, p_adj in zip(wilcoxon_msi, msi_holm, msi_p_adj):
        r["significant_holm"] = h
        r["p_holm"] = p_adj
    all_tests.extend(wilcoxon_msi)

    wilcoxon_ari = _pairwise_wilcoxon_vs_dbscan(ari_wide, "ARI_CLASSF", "ari")
    ari_holm, ari_p_adj = _holm_bonferroni([r["p_value"] for r in wilcoxon_ari])
    for r, h, p_adj in zip(wilcoxon_ari, ari_holm, ari_p_adj):
        r["significant_holm"] = h
        r["p_holm"] = p_adj
    all_tests.extend(wilcoxon_ari)

    all_tests.extend(_load_hybrid_tests())

    pd.DataFrame(all_tests).to_csv(out_dir / "mh_vs_dbscan_statistical_tests.csv", index=False)
    logger.info("Saved mh_vs_dbscan_statistical_tests.csv")

    reviewer_df = _format_reviewer_table(all_tests)
    reviewer_df.to_csv(out_dir / "mh_vs_dbscan_reviewer_stat_tests.csv", index=False)

    return all_tests


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_mh_medians() -> pd.DataFrame:
    if not EXP06_MEDIANS.exists():
        raise FileNotFoundError(
            f"Missing MH medians at {EXP06_MEDIANS}. "
            "Run aggregate_mh_comparison.py first."
        )
    # This reviewer analysis intentionally uses Exp 02 per-dataset medians for
    # each isolated metaheuristic. That makes it a different estimand from
    # aggregate.py, which evaluates the final Exp 03 mixed-solver algorithm via
    # the best seed per dataset; differing W/L/T and p-values are expected.
    df = pd.read_csv(EXP06_MEDIANS)
    required = {"dataset_id", "group"} | {f"msi_{mh}" for mh in METAHEURISTICS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"MH medians missing required columns: {missing}")
    return df


def _load_dbscan() -> pd.DataFrame:
    if not EXP07_DBSCAN.exists():
        raise FileNotFoundError(
            f"Missing DBSCAN-DistK runs at {EXP07_DBSCAN}. "
            "Run run_dbscan_distk.py first."
        )
    df = pd.read_csv(EXP07_DBSCAN)
    required = {"dataset_id", "group", "msi", "ari"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DBSCAN runs missing required columns: {missing}")
    keep = df[["dataset_id", "group", "msi", "ari"]].rename(
        columns={"msi": "msi_dbscan", "ari": "ari_dbscan"},
    )
    return keep


def _load_hybrid_tests() -> list[dict]:
    if not EXP07_HYBRID_STATS.exists():
        raise FileNotFoundError(
            f"Missing hybrid statistical tests at {EXP07_HYBRID_STATS}. "
            "Run aggregate.py first."
        )

    df = pd.read_csv(EXP07_HYBRID_STATS)
    required = {"metric", "n_pairs", "p_value", "statistic", "significant"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hybrid statistical tests missing required columns: {missing}")

    selected = df[df["metric"].isin(["MSI_ALL_vs_distk", "ARI_CLASSF_vs_distk"])].copy()
    if selected.empty:
        return []

    selected["p_value"] = selected["p_value"].astype(float)
    holm_decisions, holm_p_adj = _holm_bonferroni(selected["p_value"].tolist())
    selected["significant_holm"] = holm_decisions
    selected["p_holm"] = holm_p_adj

    family_map = {
        "MSI_ALL_vs_distk": "MSI",
        "ARI_CLASSF_vs_distk": "ARI_CLASSF",
    }

    rows: list[dict] = []
    for _, row in selected.iterrows():
        rows.append({
            "metric": family_map[str(row["metric"])],
            "test": "wilcoxon",
            "pair": str(row["metric"]),
            "comparison_label": "ClustGrade-RKO Híbrido vs DBSCAN",
            "n_pairs": int(row["n_pairs"]),
            "statistic": float(row["statistic"]),
            "p_value": float(row["p_value"]),
            "significant": bool(row["significant"]),
            "significant_holm": row["significant_holm"],
            "p_holm": row["p_holm"],
        })
    return rows


def _load_hybrid_comparison() -> pd.DataFrame:
    """Load Exp 03 hybrid-vs-DBSCAN wide comparison."""
    if not EXP07_COMPARISON.exists():
        raise FileNotFoundError(
            f"Missing hybrid comparison at {EXP07_COMPARISON}. "
            "Run aggregate.py first."
        )

    df = pd.read_csv(EXP07_COMPARISON)
    required = {
        "dataset_id",
        "group",
        "msi_rko",
        "ari_rko",
        "msi_dbscan_distk",
        "ari_dbscan_distk",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Hybrid comparison missing required columns: {missing}")
    return df[list(required)].copy()


# ---------------------------------------------------------------------------
# Win/loss/tie computation
# ---------------------------------------------------------------------------

def compute_wlt_vs_dbscan(
    medians: pd.DataFrame,
    dbscan: pd.DataFrame,
    metric_prefix: str,
    dbscan_col: str,
    group_filter: str | None = None,
    tol: float = TOL,
) -> pd.DataFrame:
    """Per-MH wins/losses/ties vs DBSCAN on the given metric.

    Args:
        medians: Exp 02 wide medians with `{metric_prefix}_{mh}` columns.
        dbscan: DBSCAN runs with `dbscan_col` holding the per-dataset metric value.
        metric_prefix: "msi" or "ari".
        dbscan_col: "msi_dbscan" or "ari_dbscan".
        group_filter: optional group ("clust" / "classf") to subset both frames.
        tol: symmetric tie tolerance.

    Returns:
        DataFrame with columns: metaheuristic, label, n_total, V, D, E,
        overall_prop, relative_prop. Letters V/D/E denote Vit\u00f3ria/Derrota/Empate
        (Win/Loss/Tie) from the MH's perspective vs DBSCAN.
        `overall_prop` is the share of total MH wins within the family, while
        `relative_prop` is the per-MH win rate over V + D + E.
    """
    mh_df = medians.copy()
    db_df = dbscan.copy()
    if group_filter is not None:
        mh_df = mh_df[mh_df["group"] == group_filter]
        db_df = db_df[db_df["group"] == group_filter]

    merged = mh_df.merge(db_df, on=["dataset_id", "group"], how="inner")
    merged = merged.dropna(subset=[dbscan_col])

    rows: list[dict] = []
    for mh in METAHEURISTICS:
        col = f"{metric_prefix}_{mh}"
        if col not in merged.columns:
            continue
        sub = merged.dropna(subset=[col])
        diffs = sub[col].values.astype(float) - sub[dbscan_col].values.astype(float)
        wins = int(np.sum(diffs > tol))
        losses = int(np.sum(diffs < -tol))
        ties = int(len(diffs) - wins - losses)
        n_total = int(len(diffs))
        rows.append({
            "metaheuristic": mh,
            "label": MH_LABELS[mh],
            "n_total": n_total,
            "V": wins,
            "D": losses,
            "E": ties,
            "overall_prop": None,
            "relative_prop": round(wins / n_total, 4) if n_total else None,
        })
    cols = ["metaheuristic", "label", "n_total", "V", "D", "E",
            "overall_prop", "relative_prop"]
    table = pd.DataFrame(rows, columns=cols)
    total_wins = int(table["V"].sum()) if not table.empty else 0
    if total_wins:
        table["overall_prop"] = (table["V"] / total_wins).round(4)
    return table


def _compute_gap_percent(method_values: np.ndarray, dbscan_values: np.ndarray) -> np.ndarray:
    """Compute percent GAP = 100 * (method - dbscan) / |dbscan|.

    For zero-denominator rows, returns 0.0 only when both values are zero,
    otherwise NaN (undefined percentage delta).
    """
    eps = 1e-12
    raw = method_values - dbscan_values
    denom = np.abs(dbscan_values)
    with np.errstate(divide="ignore", invalid="ignore"):
        gap_pct = np.where(denom > eps, (raw / denom) * 100.0, np.nan)
    both_zero = (np.abs(method_values) <= eps) & (np.abs(dbscan_values) <= eps)
    gap_pct = np.where(both_zero, 0.0, gap_pct)
    return gap_pct.astype(float)


def _build_gap_rows(
    frame: pd.DataFrame,
    metric: str,
    method_code: str,
    method_label: str,
    method_col: str,
    dbscan_col: str,
) -> list[dict]:
    required = {"dataset_id", "group", method_col, dbscan_col}
    if not required.issubset(frame.columns):
        return []

    sub = frame[["dataset_id", "group", method_col, dbscan_col]].dropna()
    if sub.empty:
        return []

    method_vals = sub[method_col].values.astype(float)
    dbscan_vals = sub[dbscan_col].values.astype(float)
    gap_raw = method_vals - dbscan_vals
    gap_pct = _compute_gap_percent(method_vals, dbscan_vals)

    rows: list[dict] = []
    for ds, grp, raw, pct in zip(sub["dataset_id"], sub["group"], gap_raw, gap_pct):
        rows.append({
            "dataset_id": str(ds),
            "group": str(grp),
            "metric": metric,
            "method_code": method_code,
            "method_label": method_label,
            "comparison_label": f"{method_label} vs DBSCAN",
            "gap_raw": float(raw),
            "gap_pct": float(pct),
        })
    return rows


def compute_gap_vs_dbscan(
    medians: pd.DataFrame,
    dbscan: pd.DataFrame,
    hybrid_comp: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute raw and percent GAP per dataset vs DBSCAN.

    Returns:
        - gap_by_dataset: one row per dataset/method/metric with raw + percent GAP.
        - gap_summary: mean raw GAP and mean percent GAP per method and metric.
    """
    rows: list[dict] = []

    merged = medians.merge(dbscan, on=["dataset_id", "group"], how="inner")
    merged_ari = merged[merged["group"] == "classf"].copy()

    for mh in METAHEURISTICS:
        rows.extend(
            _build_gap_rows(
                merged,
                metric="MSI",
                method_code=mh,
                method_label=MH_LABELS[mh],
                method_col=f"msi_{mh}",
                dbscan_col="msi_dbscan",
            )
        )
        rows.extend(
            _build_gap_rows(
                merged_ari,
                metric="ARI_CLASSF",
                method_code=mh,
                method_label=MH_LABELS[mh],
                method_col=f"ari_{mh}",
                dbscan_col="ari_dbscan",
            )
        )

    if not hybrid_comp.empty:
        hybrid_ari = hybrid_comp[hybrid_comp["group"] == "classf"].copy()
        rows.extend(
            _build_gap_rows(
                hybrid_comp,
                metric="MSI",
                method_code=HYBRID_METHOD_CODE,
                method_label=HYBRID_LABEL,
                method_col="msi_rko",
                dbscan_col="msi_dbscan_distk",
            )
        )
        rows.extend(
            _build_gap_rows(
                hybrid_ari,
                metric="ARI_CLASSF",
                method_code=HYBRID_METHOD_CODE,
                method_label=HYBRID_LABEL,
                method_col="ari_rko",
                dbscan_col="ari_dbscan_distk",
            )
        )

    if not rows:
        empty = pd.DataFrame(
            columns=[
                "dataset_id",
                "group",
                "metric",
                "method_code",
                "method_label",
                "comparison_label",
                "gap_raw",
                "gap_pct",
            ]
        )
        return empty, empty

    gap_by_dataset = pd.DataFrame(rows)

    summary = (
        gap_by_dataset
        .groupby(["metric", "method_code", "method_label", "comparison_label"], as_index=False)
        .agg(
            n_pairs=("gap_raw", "count"),
            mean_gap_raw=("gap_raw", "mean"),
            mean_gap_pct=("gap_pct", "mean"),
            n_gap_pct_valid=("gap_pct", lambda s: int(s.notna().sum())),
        )
    )
    summary["mean_gap_raw"] = summary["mean_gap_raw"].round(6)
    summary["mean_gap_pct"] = summary["mean_gap_pct"].round(6)

    metric_order = {"MSI": 0, "ARI_CLASSF": 1}
    summary = (
        summary
        .assign(
            _metric_order=lambda d: d["metric"].map(metric_order).fillna(99),
            _method_order=lambda d: d["method_code"].map(METHOD_ORDER).fillna(99),
        )
        .sort_values(["_metric_order", "_method_order"], kind="stable")
        .drop(columns=["_metric_order", "_method_order"])
        .reset_index(drop=True)
    )

    return gap_by_dataset, summary


# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

def _format_gap_reviewer_table(gap_summary: pd.DataFrame) -> pd.DataFrame:
    family_labels = {"MSI": "ISM", "ARI_CLASSF": "IRA"}
    table = gap_summary.copy()
    table["familia"] = table["metric"].map(family_labels).fillna(table["metric"])
    table = table.rename(
        columns={
            "comparison_label": "comparativo",
            "mean_gap_raw": "gap_medio",
            "mean_gap_pct": "gap_pct_medio",
        }
    )
    return table[[
        "familia",
        "n_pairs",
        "n_gap_pct_valid",
        "comparativo",
        "gap_medio",
        "gap_pct_medio",
    ]]


def _barplot_h(
    table: pd.DataFrame,
    value_col: str,
    xlabel: str,
    metric_label: str,
    out_path: Path,
    reference_line: float | None = None,
) -> None:
    """Horizontal bar plot of a proportion column for the three metaheuristics."""
    if table.empty:
        logger.warning("Bar plot skipped (empty table): %s", out_path.name)
        return

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    labels = table["label"].tolist()
    values = table[value_col].fillna(0).values.astype(float)
    colors = [MH_COLORS[mh] for mh in table["metaheuristic"]]

    y = np.arange(len(labels))
    bars = ax.barh(
        y, values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9,
    )
    for bar, v in zip(bars, values):
        if np.isnan(v):
            continue
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            f"{v:.1%}", va="center", ha="left", fontsize=9,
        )

    if reference_line is not None:
        ax.axvline(reference_line, color="red", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()  # first entry at top
    ax.set_xlim(0, 1.05)
    ax.set_xlabel(xlabel)
    ax.text(
        0.98,
        0.98,
        metric_label,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.grid(axis="x", alpha=0.3)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    argparse.ArgumentParser().parse_args()

    logger.info("Reading MH medians from %s", EXP06_MEDIANS)
    logger.info("Reading DBSCAN-DistK from %s", EXP07_DBSCAN)
    logger.info("Reading hybrid comparison from %s", EXP07_COMPARISON)

    medians = _load_mh_medians()
    dbscan = _load_dbscan()
    hybrid_comp = _load_hybrid_comparison()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # MSI: all datasets.
    msi_table = compute_wlt_vs_dbscan(
        medians, dbscan, metric_prefix="msi", dbscan_col="msi_dbscan",
    )
    msi_table.to_csv(OUT_DIR / "mh_vs_dbscan_msi.csv", index=False)
    _barplot_h(
        msi_table, value_col="overall_prop",
        xlabel="Participa\u00e7\u00e3o nas vit\u00f3rias totais",
        metric_label="M\u00e9trica - ISM",
        out_path=OUT_DIR / "barplot_mh_vs_dbscan_msi_overall.png",
    )
    _barplot_h(
        msi_table, value_col="relative_prop",
        xlabel="Taxa de vit\u00f3ria (V / (V + D + E))",
        metric_label="M\u00e9trica - ISM",
        out_path=OUT_DIR / "barplot_mh_vs_dbscan_msi_relative.png",
        reference_line=0.5,
    )

    # ARI: CLASSF only.
    ari_table = compute_wlt_vs_dbscan(
        medians, dbscan, metric_prefix="ari", dbscan_col="ari_dbscan",
        group_filter="classf",
    )
    ari_table.to_csv(OUT_DIR / "mh_vs_dbscan_ari_classf.csv", index=False)
    _barplot_h(
        ari_table, value_col="overall_prop",
        xlabel="Participa\u00e7\u00e3o nas vit\u00f3rias totais",
        metric_label="M\u00e9trica - IRA",
        out_path=OUT_DIR / "barplot_mh_vs_dbscan_ari_classf_overall.png",
    )
    _barplot_h(
        ari_table, value_col="relative_prop",
        xlabel="Taxa de vit\u00f3ria (V / (V + D + E))",
        metric_label="M\u00e9trica - IRA",
        out_path=OUT_DIR / "barplot_mh_vs_dbscan_ari_classf_relative.png",
        reference_line=0.5,
    )

    # GAP (raw and percent): isolated MH medians + hybrid best-of-20 vs DBSCAN-DistK.
    gap_by_dataset, gap_summary = compute_gap_vs_dbscan(medians, dbscan, hybrid_comp)
    gap_by_dataset.to_csv(OUT_DIR / "mh_vs_dbscan_gap_by_dataset.csv", index=False)
    gap_summary.to_csv(OUT_DIR / "mh_vs_dbscan_gap_summary.csv", index=False)
    logger.info("Saved mh_vs_dbscan_gap_by_dataset.csv")
    logger.info("Saved mh_vs_dbscan_gap_summary.csv")

    gap_reviewer = _format_gap_reviewer_table(gap_summary)
    gap_reviewer.to_csv(OUT_DIR / "mh_vs_dbscan_reviewer_gap.csv", index=False)

    # Statistical tests: paired Wilcoxon (MH vs DBSCAN) + Holm-Bonferroni.
    all_tests = run_statistical_tests(medians, dbscan, OUT_DIR)
    wilcoxon_msi_st = [t for t in all_tests if t["test"] == "wilcoxon" and t["metric"] == "MSI"]
    wilcoxon_ari_st = [
        t for t in all_tests if t["test"] == "wilcoxon" and t["metric"] == "ARI_CLASSF"
    ]

    # Console summary.
    print("\n" + "=" * 80)
    print("EXPERIMENT 03 \u2014 MH vs DBSCAN-DistK (reviewer analysis)")
    print("=" * 80)
    print("\n--- MSI (all datasets) ---")
    print(msi_table.to_string(index=False))
    print("\n--- ARI (18 CLASSF datasets) ---")
    print(ari_table.to_string(index=False))
    print("\n--- GAP m\u00e9dio vs DBSCAN (bruto e percentual) ---")
    if not gap_summary.empty:
        display_gap = gap_summary[[
            "metric",
            "method_label",
            "n_pairs",
            "mean_gap_raw",
            "mean_gap_pct",
            "n_gap_pct_valid",
        ]].rename(columns={
            "metric": "metrica",
            "method_label": "metodo",
            "n_pairs": "N",
            "mean_gap_raw": "gap_medio",
            "mean_gap_pct": "gap_medio_pct",
            "n_gap_pct_valid": "N_gap_pct",
        })
        print(display_gap.to_string(index=False))
    else:
        print("  (sem dados de GAP)")
    print("\n--- Statistical tests: paired Wilcoxon MH vs DBSCAN ---")
    print("Wilcoxon MSI (Holm-Bonferroni m=3):")
    for r in wilcoxon_msi_st:
        print(f"  {r['pair']}: p={r['p_value']}, p_holm={r['p_holm']}, "
              f"sig_holm={r['significant_holm']}")
    print("\nWilcoxon ARI (Holm-Bonferroni m=3):")
    for r in wilcoxon_ari_st:
        print(f"  {r['pair']}: p={r['p_value']}, p_holm={r['p_holm']}, "
              f"sig_holm={r['significant_holm']}")
    print(f"\nArtifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
