"""
Experiment 02 — Publication plots: metaheuristic comparison (BRKGA vs VNS vs ILS).

Reads artifacts/mh_comparison/raw_runs.csv (per-run, 3000 rows) and
artifacts/mh_comparison/per_dataset_medians.csv (one row per dataset) and
produces:

    boxplot_msi.png                      — ISM distribution, 3 MHs side by side
    boxplot_ari_classf.png               — ARI distribution, 3 MHs (CLASSF only)
    raincloud_msi.png                    — ISM half-violin + box + jitter
    raincloud_ari_classf.png             — ARI half-violin + box + jitter (CLASSF)
    raincloud_msi_ari_figure.tex         — LaTeX snippet with both rainclouds side by side
    barplot_sorted_mean_msi.png          — per-dataset mean ISM per MH, sorted

Style:
    Consistent MH palette across plots; serif fonts; 300 DPI; minor gridlines;
    fixed axis limits for cross-panel comparability.

Usage:
    poetry run python -m experiments.02-qlearning-mh-comparison.plot_mh_comparison
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison"
_SMOKE_ARTIFACTS = Path(__file__).parent / "artifacts" / "mh_comparison_smoke"
RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"


def _apply_smoke_overrides() -> None:
    global ARTIFACTS, RAW_RUNS_PATH
    ARTIFACTS = _SMOKE_ARTIFACTS
    RAW_RUNS_PATH = ARTIFACTS / "raw_runs.csv"

METAHEURISTICS: list[str] = ["brkga", "vns", "ils"]
MH_LABELS: dict[str, str] = {"brkga": "BRKGA", "vns": "VNS", "ils": "ILS"}
MH_COLORS: dict[str, str] = {"brkga": "#4C72B0", "vns": "#DD8452", "ils": "#55A868"}

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
# Loaders
# ---------------------------------------------------------------------------

def _load_raw() -> pd.DataFrame:
    if not RAW_RUNS_PATH.exists():
        raise FileNotFoundError(f"Missing {RAW_RUNS_PATH}. Run run_mh_comparison.py first.")
    return pd.read_csv(RAW_RUNS_PATH)


# ---------------------------------------------------------------------------
# Boxplots (per-run distributions)
# ---------------------------------------------------------------------------

def _boxplot(
    values_by_mh: dict[str, np.ndarray],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.0))
    data = [values_by_mh[mh] for mh in METAHEURISTICS]
    labels = [MH_LABELS[mh] for mh in METAHEURISTICS]

    bp = ax.boxplot(
        data, labels=labels, patch_artist=True, showfliers=True,
        flierprops=dict(marker="o", markerfacecolor="gray", markersize=3, alpha=0.4),
        medianprops=dict(color="black", linewidth=1.3),
    )
    for patch, mh in zip(bp["boxes"], METAHEURISTICS):
        patch.set_facecolor(MH_COLORS[mh])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(0.8)

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Rainclouds (half-violin + box + jitter)
# ---------------------------------------------------------------------------

def _half_violin(
    ax: Axes, data: np.ndarray, center: float, color: str, width: float = 0.4,
) -> None:
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return
    kde = gaussian_kde(data, bw_method="scott")
    y_grid = np.linspace(data.min(), data.max(), 200)
    density = kde(y_grid)
    density = density / density.max() * width
    ax.fill_betweenx(y_grid, center - density, center, alpha=0.35, color=color, linewidth=0)
    ax.plot(center - density, y_grid, color=color, linewidth=0.8)


def _raincloud(
    values_by_mh: dict[str, np.ndarray],
    ylabel: str,
    metric_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    rng = np.random.default_rng(42)

    for idx, mh in enumerate(METAHEURISTICS):
        vals = values_by_mh.get(mh)
        if vals is None or len(vals) == 0:
            continue
        color = MH_COLORS[mh]
        center = idx + 1

        _half_violin(ax, vals, center, color, width=0.4)

        clean = vals[~np.isnan(vals)]
        ax.boxplot(
            clean, positions=[center], widths=0.12, vert=True, patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.85),
            whiskerprops=dict(color=color, linewidth=1),
            capprops=dict(color=color, linewidth=1),
        )

        jitter = rng.uniform(0.05, 0.28, size=len(clean))
        ax.scatter(
            center + jitter, clean, s=10, alpha=0.4, color=color,
            edgecolors="none", zorder=3,
        )

    # Mean +/- std annotations.
    labels: list[str] = []
    for mh in METAHEURISTICS:
        vals = values_by_mh.get(mh, np.array([]))
        clean = vals[~np.isnan(vals)] if len(vals) else vals
        if len(clean) >= 2:
            labels.append(f"{MH_LABELS[mh]}\n(m\u00e9dia={clean.mean():.3f} \u00b1 {clean.std(ddof=1):.3f})")
        elif len(clean) == 1:
            labels.append(f"{MH_LABELS[mh]}\n(m\u00e9dia={clean[0]:.3f})")
        else:
            labels.append(MH_LABELS[mh])

    positions = list(range(1, len(METAHEURISTICS) + 1))
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_xlim(0.3, len(METAHEURISTICS) + 0.8)
    ax.text(
        0.02,
        0.98,
        metric_label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="0.7", alpha=0.9),
    )
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


def _write_raincloud_pair_latex(out_path: Path) -> None:
    latex = """\\begin{figure}[ht]
\\centering
\\includegraphics[width=0.48\\linewidth]{raincloud_msi.png}\\hfill
\\includegraphics[width=0.48\\linewidth]{raincloud_ari_classf.png}
\\caption{Figura 01 --- Distribui\\c{c}\\~ao de ISM e IRA por Metaheur\\'istica.}
\\label{fig:mh_comparison_rainclouds}
\\end{figure}
"""
    out_path.write_text(latex, encoding="utf-8")
    logger.info("Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Overall mean MSI bar plot (one bar per MH)
# ---------------------------------------------------------------------------

def _overall_mean_msi_barplot(raw: pd.DataFrame, out_path: Path) -> None:
    """Simple bar chart of overall mean ISM per metaheuristic.

    Overall mean = grand mean across datasets of per-(dataset, MH) mean MSI,
    so each dataset contributes equally regardless of how many runs it has.
    """
    per_ds_means = (
        raw.groupby(["dataset_id", "metaheuristic"])["msi"].mean().reset_index()
    )
    wide = per_ds_means.pivot_table(
        index="dataset_id", columns="metaheuristic", values="msi",
    )
    wide.columns.name = None

    cols = [mh for mh in METAHEURISTICS if mh in wide.columns]
    if not cols:
        logger.warning("Overall mean ISM barplot: no MH columns present")
        return

    means = wide[cols].mean(axis=0, skipna=True)

    fig, ax = plt.subplots(figsize=(6.0, 4.5))
    x = np.arange(len(cols))
    colors = [MH_COLORS[mh] for mh in cols]
    bars = ax.bar(
        x, means.values, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9,
    )
    for bar, v in zip(bars, means.values):
        if np.isnan(v):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
            f"{v:.3f}", ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([MH_LABELS[mh] for mh in cols])
    ax.set_xlabel("Metaheur\u00edstica")
    ax.set_ylabel("ISM m\u00e9dio")
    ax.set_title("Figura XX \u2014 M\u00e9dia de ISM por Metaheur\u00edstica")
    ax.set_ylim(0, max(1.0, float(np.nanmax(means.values)) * 1.1))
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Sorted mean MSI bar plot (reviewer request #2)
# ---------------------------------------------------------------------------

def _sorted_mean_msi_barplot(raw: pd.DataFrame, out_path: Path) -> None:
    """Grouped bar chart of per-dataset mean MSI per MH, sorted by best MH desc.

    Computes per-(dataset, MH) mean MSI from the raw runs. Datasets sorted by
    max(mean_brkga, mean_vns, mean_ils) descending. x-tick labels color-coded
    by group: black (CLUST), navy (CLASSF).
    """
    means_long = (
        raw.groupby(["dataset_id", "group", "metaheuristic"])["msi"]
           .mean().reset_index()
    )
    wide = means_long.pivot_table(
        index=["dataset_id", "group"], columns="metaheuristic", values="msi",
    ).reset_index()
    wide.columns.name = None

    matrix_cols = [mh for mh in METAHEURISTICS if mh in wide.columns]
    if len(matrix_cols) < len(METAHEURISTICS):
        logger.warning(
            "Sorted mean ISM barplot: missing MH columns %s",
            set(METAHEURISTICS) - set(matrix_cols),
        )
        if not matrix_cols:
            return

    work = wide.dropna(subset=matrix_cols, how="all").copy()
    work["_best"] = work[matrix_cols].max(axis=1)
    ordered = work.sort_values("_best", ascending=False).reset_index(drop=True)
    matrix = ordered[matrix_cols].values.astype(float)

    n_ds = len(ordered)
    x = np.arange(n_ds)
    width = 0.27

    fig, ax = plt.subplots(figsize=(max(10.0, 0.22 * n_ds), 5.5))
    for i, mh in enumerate(matrix_cols):
        ax.bar(
            x + (i - 1) * width, matrix[:, i], width=width,
            color=MH_COLORS[mh], label=MH_LABELS[mh], alpha=0.85,
            edgecolor="black", linewidth=0.3,
        )

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["dataset_id"], rotation=90, fontsize=6)

    # Color-code x-tick labels by group (CLUST preto, CLASSF azul-marinho).
    group_colors = {"clust": "black", "classf": "#1f3a60"}
    for tick, grp in zip(ax.get_xticklabels(), ordered["group"].values):
        tick.set_color(group_colors.get(str(grp), "black"))

    ax.set_xlabel("Inst\u00e2ncia")
    ax.set_ylabel("ISM m\u00e9dio")
    ax.set_title("Figura XX \u2014 M\u00e9dia de ISM por inst\u00e2ncia e Metaheur\u00edstica")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info("Saved %s", out_path.name)


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_per_run(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    out: dict[str, np.ndarray] = {}
    for mh in METAHEURISTICS:
        subset = df[df["metaheuristic"] == mh].dropna(subset=[metric])
        out[mh] = subset[metric].values.astype(float)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke", action="store_true",
        help="Read from the smoke-mode artifacts folder (artifacts/mh_comparison_smoke/).",
    )
    args = parser.parse_args()
    if args.smoke:
        _apply_smoke_overrides()
        logger.info("SMOKE MODE: reading from %s", ARTIFACTS)

    raw = _load_raw()

    # Per-run distributions — MSI (all datasets) and ARI (CLASSF only).
    msi_values = _extract_per_run(raw, "msi")
    ari_values = _extract_per_run(raw[raw["group"] == "classf"], "ari")

    _boxplot(
        msi_values, "ISM (por execu\u00e7\u00e3o)",
        "Figura XX \u2014 Distribui\u00e7\u00e3o do ISM por Metaheur\u00edstica",
        ARTIFACTS / "boxplot_msi.png",
    )
    _boxplot(
        ari_values, "IRA (por execu\u00e7\u00e3o)",
        "Figura XX \u2014 Distribui\u00e7\u00e3o do ARI por Metaheur\u00edstica (CLASSF)",
        ARTIFACTS / "boxplot_ari_classf.png",
    )

    _raincloud(
        msi_values, "ISM (por execu\u00e7\u00e3o)",
        "M\u00e9trica ISM",
        ARTIFACTS / "raincloud_msi.png",
    )
    _raincloud(
        ari_values, "IRA (por execu\u00e7\u00e3o)",
        "M\u00e9trica - IRA",
        ARTIFACTS / "raincloud_ari_classf.png",
    )
    _write_raincloud_pair_latex(ARTIFACTS / "raincloud_msi_ari_figure.tex")

    # Overall mean MSI per MH + sorted per-dataset mean MSI bar plots.
    _overall_mean_msi_barplot(raw, ARTIFACTS / "barplot_overall_mean_msi.png")
    _sorted_mean_msi_barplot(raw, ARTIFACTS / "barplot_sorted_mean_msi.png")

    print("All plots written to", ARTIFACTS)


if __name__ == "__main__":
    main()
