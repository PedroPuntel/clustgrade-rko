"""Shared helpers for Experiment 01 sensitivity solution plots."""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"
_RKO_FW = _ROOT / "misc" / "rko"

for p in [str(_SRC), str(_RKO_FW), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from experiments._shared.utils import load_features, load_manifest, seed_run
from RKO import RKO
from rko.environment import ClustGradeEnv
from rko.pipeline import preprocess
from spatial.point_pattern import PointPattern
from utils.logging import get_logger

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
N_TOP_DEFAULT = 2
RKO_TIME_BUDGET = 60
RKO_MIX = {"brkga": 1, "vns": 1, "ils": 1}

CONFIG_ORDER = ["QF+FEATURE", "QF+MDS", "noQF+FEATURE", "noQF+MDS"]
CONFIG_LABELS = {
    "QF+FEATURE": "QF + Feature",
    "QF+MDS": "QF + MDS",
    "noQF+FEATURE": "noQF + Feature",
    "noQF+MDS": "noQF + MDS",
}
CONFIG_OPTIONS = {
    "QF+FEATURE": {"quadrat_filter": True, "msi_space": "feature"},
    "QF+MDS": {"quadrat_filter": True, "msi_space": "mds"},
    "noQF+FEATURE": {"quadrat_filter": False, "msi_space": "feature"},
    "noQF+MDS": {"quadrat_filter": False, "msi_space": "mds"},
}

VIBRANT_CLUSTER_COLORS = [
    "#00A6FB",
    "#F7B801",
    "#06D6A0",
    "#8338EC",
    "#FF9F1C",
    "#118AB2",
    "#3A86FF",
    "#FB5607",
    "#2EC4B6",
    "#9B5DE5",
    "#80ED99",
    "#E9C46A",
]
OUTLIER_COLOR = "#FF1E1E"
GRID_COLOR = "#888888"


@dataclass(frozen=True)
class ConfigSpec:
    """Experiment 01 parameter configuration descriptor."""

    name: str
    quadrat_filter: bool
    msi_space: str
    display_label: str


@dataclass(frozen=True)
class SensitivitySelection:
    """A dataset paired with the max-gap configs for a chosen metric."""

    dataset_id: str
    group: str
    metric: Literal["msi", "ari"]
    gap: float
    max_config: ConfigSpec
    min_config: ConfigSpec
    max_value: float
    min_value: float


def _config_rank(config_name: str) -> int:
    return CONFIG_ORDER.index(config_name)


def _config_name(quadrat_filter: Any, msi_space: Any) -> str:
    qf_bool = (
        quadrat_filter
        if isinstance(quadrat_filter, bool)
        else str(quadrat_filter).lower() == "true"
    )
    qf = "QF" if qf_bool else "noQF"
    return f"{qf}+{str(msi_space).upper()}"


def get_config_spec(config_name: str) -> ConfigSpec:
    options = CONFIG_OPTIONS[config_name]
    return ConfigSpec(
        name=config_name,
        quadrat_filter=bool(options["quadrat_filter"]),
        msi_space=str(options["msi_space"]),
        display_label=CONFIG_LABELS[config_name],
    )


def _load_per_dataset_medians() -> pd.DataFrame:
    return pd.read_csv(ARTIFACTS / "per_dataset_medians.csv")


def _load_raw_runs() -> pd.DataFrame:
    df = pd.read_csv(ARTIFACTS / "raw_runs.csv")
    df["config"] = df.apply(
        lambda row: _config_name(row["quadrat_filter"], row["msi_space"]),
        axis=1,
    )
    return df


def _pick_extreme_row(dataset_rows: pd.DataFrame, metric_col: str, pick_max: bool) -> pd.Series:
    ordered = dataset_rows.assign(
        _rank=dataset_rows["config"].map(_config_rank),
    ).sort_values(
        by=[metric_col, "_rank"],
        ascending=[not pick_max, True],
    )
    return ordered.iloc[0]


def select_top_sensitive_datasets(
    metric: Literal["msi", "ari"],
    n_top: int = N_TOP_DEFAULT,
) -> list[SensitivitySelection]:
    """Select the top-N datasets with the largest per-config metric spread."""
    medians = _load_per_dataset_medians()
    metric_col = f"{metric}_median"

    if metric == "ari":
        medians = medians[medians["group"] == "classf"]

    medians = medians.dropna(subset=[metric_col])
    selections: list[SensitivitySelection] = []

    for dataset_id, dataset_rows in medians.groupby("dataset_id", sort=True):
        max_row = _pick_extreme_row(dataset_rows, metric_col, pick_max=True)
        min_row = _pick_extreme_row(dataset_rows, metric_col, pick_max=False)
        max_value = float(max_row[metric_col])
        min_value = float(min_row[metric_col])
        selections.append(
            SensitivitySelection(
                dataset_id=str(dataset_id),
                group=str(dataset_rows.iloc[0]["group"]),
                metric=metric,
                gap=max_value - min_value,
                max_config=get_config_spec(str(max_row["config"])),
                min_config=get_config_spec(str(min_row["config"])),
                max_value=max_value,
                min_value=min_value,
            )
        )

    return sorted(selections, key=lambda item: (-item.gap, item.dataset_id))[:n_top]


def pick_representative_seed(
    raw_runs: pd.DataFrame,
    dataset_id: str,
    config_name: str,
    metric: Literal["msi", "ari"],
) -> int:
    """Pick the seed whose run metric is closest to the config median."""
    subset = raw_runs[
        (raw_runs["dataset_id"] == dataset_id) & (raw_runs["config"] == config_name)
    ].dropna(subset=[metric])
    if subset.empty:
        return 0

    median_value = float(subset[metric].median())
    chosen = subset.assign(
        _abs_delta=(subset[metric] - median_value).abs(),
    ).sort_values(by=["_abs_delta", "seed"])
    return int(chosen.iloc[0]["seed"])


def rerun_config(
    dataset_id: str,
    features: np.ndarray,
    scaled_mds: np.ndarray,
    ppp: PointPattern,
    seed: int,
    config: ConfigSpec,
) -> dict[str, Any]:
    """Rerun one Experiment 01 RKO configuration on shared scaled MDS."""
    seed_run(seed)
    random.seed(seed)
    np.random.seed(seed)

    env = ClustGradeEnv(
        features,
        ppp,
        scaled_mds,
        instance_name=dataset_id,
        max_time=RKO_TIME_BUDGET,
        msi_space=config.msi_space,
        quadrat_filter=config.quadrat_filter,
        q_learning=False,
    )

    solver = RKO(env, print_best=False)
    _, best_keys, _ = solver.solve(
        time_total=RKO_TIME_BUDGET,
        brkga=RKO_MIX["brkga"],
        vns=RKO_MIX["vns"],
        ils=RKO_MIX["ils"],
        runs=1,
    )

    if best_keys is None:
        return {
            "labels": np.full(features.shape[0], -1, dtype=np.int64),
            "xbreaks": None,
            "ybreaks": None,
        }

    result = env.decode_and_evaluate(np.array(best_keys))
    return {
        "labels": np.array(result["labels"], dtype=np.int64),
        "xbreaks": np.array(result["xbreaks"]) if result.get("xbreaks") is not None else None,
        "ybreaks": np.array(result["ybreaks"]) if result.get("ybreaks") is not None else None,
    }


def _format_metric(value: float) -> str:
    return f"{value:.2f}"


def _k_from_labels(labels: np.ndarray) -> int:
    return len(set(labels.tolist()) - {-1})


def _scatter_labels(ax: Axes, scaled_mds: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique_labels = sorted(set(labels.tolist()))
    non_noise = [label for label in unique_labels if label != -1]
    color_map = {
        label: VIBRANT_CLUSTER_COLORS[index % len(VIBRANT_CLUSTER_COLORS)]
        for index, label in enumerate(non_noise)
    }

    for label in unique_labels:
        mask = labels == label
        color = OUTLIER_COLOR if label == -1 else color_map[label]
        ax.scatter(
            scaled_mds[mask, 0],
            scaled_mds[mask, 1],
            c=[color],
            s=15,
            alpha=0.7,
            edgecolors="none",
        )

    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_grid_lines(ax: Axes, xbreaks: np.ndarray, ybreaks: np.ndarray) -> None:
    for xbreak in xbreaks:
        ax.axvline(xbreak, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.6)
    for ybreak in ybreaks:
        ax.axhline(ybreak, color=GRID_COLOR, linewidth=0.6, linestyle="--", alpha=0.6)


def _render_comparison(
    selection: SensitivitySelection,
    scaled_mds: np.ndarray,
    max_seed: int,
    min_seed: int,
    max_result: dict[str, Any],
    min_result: dict[str, Any],
    out_name: str,
) -> None:
    metric_name = selection.metric.upper()
    fig, (ax_max, ax_min) = plt.subplots(1, 2, figsize=(12, 5))

    max_labels = np.array(max_result["labels"], dtype=np.int64)
    min_labels = np.array(min_result["labels"], dtype=np.int64)

    _scatter_labels(
        ax_max,
        scaled_mds,
        max_labels,
        (
            f"{selection.max_config.display_label}  K={_k_from_labels(max_labels)}\n"
            f"{metric_name}={_format_metric(selection.max_value)} (seed={max_seed})"
        ),
    )
    if max_result["xbreaks"] is not None and max_result["ybreaks"] is not None:
        _draw_grid_lines(ax_max, max_result["xbreaks"], max_result["ybreaks"])

    _scatter_labels(
        ax_min,
        scaled_mds,
        min_labels,
        (
            f"{selection.min_config.display_label}  K={_k_from_labels(min_labels)}\n"
            f"{metric_name}={_format_metric(selection.min_value)} (seed={min_seed})"
        ),
    )
    if min_result["xbreaks"] is not None and min_result["ybreaks"] is not None:
        _draw_grid_lines(ax_min, min_result["xbreaks"], min_result["ybreaks"])

    fig.suptitle(
        (
            f"{selection.dataset_id} — {metric_name} gap = {selection.gap:.3f} "
            f"({selection.max_config.display_label} vs {selection.min_config.display_label})"
        ),
        fontsize=11,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(ARTIFACTS / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_name)


def run_and_plot_selection(selection: SensitivitySelection, out_name: str) -> None:
    """Rerun the selected max-vs-min configs and render a grid-overlay figure."""
    raw_runs = _load_raw_runs()
    manifest = load_manifest()
    dataset_row = manifest[manifest["dataset_id"] == selection.dataset_id].iloc[0]

    features = load_features(dataset_row)
    scaled_mds, ppp = preprocess(features)

    max_seed = pick_representative_seed(
        raw_runs, selection.dataset_id, selection.max_config.name, selection.metric
    )
    min_seed = pick_representative_seed(
        raw_runs, selection.dataset_id, selection.min_config.name, selection.metric
    )

    logger.info(
        "%s: %s seed=%d vs %s seed=%d",
        selection.dataset_id,
        selection.max_config.name,
        max_seed,
        selection.min_config.name,
        min_seed,
    )

    max_result = rerun_config(
        selection.dataset_id,
        features,
        scaled_mds,
        ppp,
        max_seed,
        selection.max_config,
    )
    min_result = rerun_config(
        selection.dataset_id,
        features,
        scaled_mds,
        ppp,
        min_seed,
        selection.min_config,
    )

    _render_comparison(selection, scaled_mds, max_seed, min_seed, max_result, min_result, out_name)
