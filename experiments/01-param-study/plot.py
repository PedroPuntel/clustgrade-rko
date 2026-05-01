"""
Experiment 01 — Raincloud plots for the 2×2 RKO parameter ablation.

Produces:
    raincloud_msi.png       — MSI comparison across 4 configs (all datasets)
    raincloud_ari_classf.png — ARI comparison across 4 configs (CLASSF datasets)

Usage:
    poetry run python -m experiments.01-param-study.plot
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

_ROOT = Path(__file__).parents[2]
sys.path.insert(0, str(_ROOT / "src"))

from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"

# Config labels in display order.
CONFIG_ORDER = ["QF+FEATURE", "QF+MDS", "noQF+FEATURE", "noQF+MDS"]
CONFIG_LABELS = {
    "QF+FEATURE": "QF + Feature",
    "QF+MDS": "QF + MDS",
    "noQF+FEATURE": "noQF + Feature",
    "noQF+MDS": "noQF + MDS",
}
CONFIG_COLORS = {
    "QF+FEATURE": "#4C72B0",
    "QF+MDS": "#DD8452",
    "noQF+FEATURE": "#55A868",
    "noQF+MDS": "#C44E52",
}


def _config_key(row: pd.Series) -> str:
    """Build config key from raw_runs columns."""
    qf = "QF" if row["quadrat_filter"] else "noQF"
    space = row["msi_space"].upper()
    return f"{qf}+{space}"


def _half_violin(
    ax: Axes,
    data: np.ndarray,
    center: float,
    color: str,
    side: str = "left",
    width: float = 0.35,
) -> None:
    """Draw a half-violin (KDE) on one side of *center*."""
    data = data[~np.isnan(data)]
    if len(data) < 3:
        return
    kde = gaussian_kde(data, bw_method="scott")
    y_grid = np.linspace(data.min(), data.max(), 200)
    density = kde(y_grid)
    density = density / density.max() * width
    if side == "left":
        ax.fill_betweenx(y_grid, center - density, center, alpha=0.35, color=color, linewidth=0)
        ax.plot(center - density, y_grid, color=color, linewidth=0.8)
    else:
        ax.fill_betweenx(y_grid, center, center + density, alpha=0.35, color=color, linewidth=0)
        ax.plot(center + density, y_grid, color=color, linewidth=0.8)


def _raincloud_multi(
    ax: Axes,
    config_values: dict[str, np.ndarray],
    ylabel: str,
    title: str,
) -> None:
    """Draw rainclouds for multiple configs: half-violin + box + jitter strip."""
    rng = np.random.default_rng(42)

    for idx, cfg in enumerate(CONFIG_ORDER):
        vals = config_values.get(cfg)
        if vals is None or len(vals) == 0:
            continue
        color = CONFIG_COLORS[cfg]
        center = idx + 1

        _half_violin(ax, vals, center, color, side="left", width=0.4)

        clean = vals[~np.isnan(vals)]
        ax.boxplot(
            clean,
            positions=[center],
            widths=0.12,
            vert=True,
            patch_artist=True,
            showfliers=False,
            medianprops=dict(color="white", linewidth=1.5),
            boxprops=dict(facecolor=color, edgecolor=color, alpha=0.8),
            whiskerprops=dict(color=color, linewidth=1),
            capprops=dict(color=color, linewidth=1),
        )

        jitter = rng.uniform(0.05, 0.25, size=len(clean))
        ax.scatter(
            center + jitter,
            clean,
            s=12,
            alpha=0.45,
            color=color,
            edgecolors="none",
            zorder=3,
        )

    # Annotate mean ± std below each config label.
    annotations = []
    for idx, cfg in enumerate(CONFIG_ORDER):
        vals = config_values.get(cfg)
        if vals is None or len(vals) == 0:
            annotations.append(CONFIG_LABELS[cfg])
            continue
        clean = vals[~np.isnan(vals)]
        mean, std = clean.mean(), clean.std(ddof=1)
        annotations.append(f"{CONFIG_LABELS[cfg]}\n(mean={mean:.2f} ± {std:.2f})")

    positions = list(range(1, len(CONFIG_ORDER) + 1))
    ax.set_xticks(positions)
    ax.set_xticklabels(annotations, fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0.2, len(CONFIG_ORDER) + 0.8)


def _load_raw() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / "raw_runs.csv")
    df["config"] = df.apply(_config_key, axis=1)
    return df


def _extract_values(df: pd.DataFrame, metric: str) -> dict[str, np.ndarray]:
    """Extract per-config arrays for a given metric column."""
    out: dict[str, np.ndarray] = {}
    for cfg in CONFIG_ORDER:
        subset = df[df["config"] == cfg].dropna(subset=[metric])
        out[cfg] = subset[metric].values.astype(float)
    return out


def plot_raincloud_msi() -> None:
    """Raincloud plot of MSI scores across all datasets (per-run)."""
    df = _load_raw()
    values = _extract_values(df, "msi")

    fig, ax = plt.subplots(figsize=(8, 6))
    _raincloud_multi(
        ax, values, "MSI",
        "Experiment 01 — MSI: RKO Parameter Configurations",
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "raincloud_msi.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raincloud_msi.png")


def plot_raincloud_ari() -> None:
    """Raincloud plot of ARI scores for CLASSF datasets (per-run)."""
    df = _load_raw()
    classf = df[df["group"] == "classf"]
    values = _extract_values(classf, "ari")

    fig, ax = plt.subplots(figsize=(8, 6))
    _raincloud_multi(
        ax, values, "ARI",
        "Experiment 01 — ARI: RKO Parameter Configurations (CLASSF)",
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / "raincloud_ari_classf.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved raincloud_ari_classf.png")


def main() -> None:
    print("Generating Experiment 01 raincloud plots ...")
    plot_raincloud_msi()
    plot_raincloud_ari()
    print("Done.")


if __name__ == "__main__":
    main()
