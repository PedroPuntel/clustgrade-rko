"""
Experiment 03 — Visual comparison: best-MH vs DBSCAN-DistK scatter plots.

Identifies the best metaheuristic (BRKGA/VNS/ILS) by highest overall median MSI
across all 50 datasets (from Exp 02 per_dataset_medians.csv), then replicates the
scatter-plot panel structure from plot.py — side-by-side scatter on MinMax-scaled
[0,1]^2 MDS — using that single MH (run in isolation) vs DBSCAN-DistK.

Best-MH selection: highest median MSI across all 50 datasets (primary).
Seed for RKO rerun: argmax MSI per (dataset_id, best_mh) from Exp 02 raw_runs.csv.

Produces (under artifacts/mh_vs_dbscan/):
    plot-01-{mh}-top5-msi-win.png    — best MH on top-5 MSI-win datasets
    plot-02-dbscan-top5-msi-win.png  — DBSCAN-DistK on same datasets
    plot-03-{mh}-top5-ari-win.png    — best MH on top-5 ARI-win datasets (CLASSF)
    plot-04-dbscan-top5-ari-win.png  — DBSCAN-DistK on same datasets
    plot-05-dbscan-top5-msi-loss.png — DBSCAN-DistK on top-5 where DBSCAN > best MH
    plot-06-{mh}-top5-msi-loss.png   — best MH on same datasets

Usage:
    poetry run python -m experiments.03-vs-dbscan-distk.plot_mh_vs_dbscan
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pandas as pd
import yaml
from sklearn.cluster import DBSCAN

_ROOT = Path(__file__).parents[2]
_SRC = _ROOT / "src"
_RKO_FW = _ROOT / "misc" / "rko"

for p in [str(_SRC), str(_RKO_FW), str(_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from RKO import RKO  # noqa: E402

from rko.environment import ClustGradeEnv  # noqa: E402
from rko.pipeline import preprocess  # noqa: E402
from utils.logging import get_logger  # noqa: E402

from experiments._shared.utils import (  # noqa: E402
    load_features,
    load_manifest,
    load_mds,
    seed_run,
)

logger = get_logger(__name__)

ARTIFACTS = Path(__file__).parent / "artifacts"
CONFIGS = _ROOT / "configs"
OUT_DIR = ARTIFACTS / "mh_vs_dbscan"

_EXP06_MH_DIR = (
    _ROOT / "experiments" / "02-qlearning-mh-comparison" / "artifacts" / "mh_comparison"
)
EXP06_MEDIANS = _EXP06_MH_DIR / "per_dataset_medians.csv"
EXP06_RAW_RUNS = _EXP06_MH_DIR / "raw_runs.csv"
EXP07_DBSCAN = ARTIFACTS / "dbscan_distk_runs.csv"
MH_ANALYSIS_DIR = ARTIFACTS / "mh_vs_dbscan"
WLT_MSI_CSV = MH_ANALYSIS_DIR / "mh_vs_dbscan_msi.csv"
WLT_ARI_CSV = MH_ANALYSIS_DIR / "mh_vs_dbscan_ari_classf.csv"
GAP_SUMMARY_CSV = MH_ANALYSIS_DIR / "mh_vs_dbscan_gap_summary.csv"
EXP07_COMPARISON_CSV = ARTIFACTS / "comparison.csv"

PANEL_COUNT = 5
MIN_PTS = 5
RKO_TIME_BUDGET = 60

METAHEURISTICS: list[str] = ["brkga", "vns", "ils"]
MH_LABELS: dict[str, str] = {"brkga": "BRKGA", "vns": "VNS", "ils": "ILS"}

# Inherit RKO base config from Exp 01 (consistent with plot.py and run.py).
_BEST_CONFIG_PATH = (
    Path(__file__).parents[1] / "01-param-study" / "artifacts" / "best_config.json"
)
with open(_BEST_CONFIG_PATH) as f:
    _BEST_CONFIG = json.load(f)
QUADRAT_FILTER: bool = _BEST_CONFIG["quadrat_filter"]
MSI_SPACE: str = _BEST_CONFIG["msi_space"]

VIBRANT_CLUSTER_COLORS = [
    "#00A6FB", "#F7B801", "#06D6A0", "#8338EC", "#FF9F1C",
    "#118AB2", "#3A86FF", "#FB5607", "#2EC4B6", "#9B5DE5",
    "#80ED99", "#E9C46A",
]
OUTLIER_COLOR = "#FF1E1E"

DBSCAN_COLOR = "#C44E52"
HYBRID_METHOD_CODE = "hybrid"
HYBRID_LABEL = "ClustGrade-RKO Híbrido"
DELTA_TOL = 0.01


def _bar_colors_with_hybrid(method_codes: list[str]) -> list[str]:
    color_map = {"brkga": "#4C72B0", "vns": "#DD8452", "ils": "#55A868", "hybrid": "#937860"}
    return [color_map.get(m, "#777777") for m in method_codes]


# ---------------------------------------------------------------------------
# Best-MH selection
# ---------------------------------------------------------------------------

def select_best_mh(medians: pd.DataFrame) -> str:
    """Pick best MH by highest overall median MSI across all 50 datasets."""
    scores: dict[str, float] = {}
    for mh in METAHEURISTICS:
        col = f"msi_{mh}"
        if col in medians.columns:
            scores[mh] = float(medians[col].median())
    best = max(scores, key=lambda m: scores[m])
    logger.info(
        "Best MH by overall median MSI: %s (%.4f) | BRKGA=%.4f VNS=%.4f ILS=%.4f",
        MH_LABELS[best], scores[best],
        scores.get("brkga", float("nan")),
        scores.get("vns", float("nan")),
        scores.get("ils", float("nan")),
    )
    return best


def _build_delta_frame(
    medians: pd.DataFrame, dbscan: pd.DataFrame, best_mh: str,
) -> pd.DataFrame:
    """Merge Exp 02 medians + DBSCAN-DistK; add per-dataset MSI/ARI delta for best_mh."""
    df = medians.merge(dbscan, on=["dataset_id", "group"], how="inner")
    df["msi_delta"] = df[f"msi_{best_mh}"] - df["msi_dbscan"]
    ari_col = f"ari_{best_mh}"
    if ari_col in df.columns and "ari_dbscan" in df.columns:
        df["ari_delta"] = df[ari_col] - df["ari_dbscan"]
    else:
        df["ari_delta"] = float("nan")
    return df


def _get_best_seed(dataset_id: str, best_mh: str, raw_runs: pd.DataFrame) -> int:
    """Return seed with highest MSI for (dataset_id, best_mh) from Exp 02 raw runs."""
    sub = raw_runs[
        (raw_runs["dataset_id"] == dataset_id) & (raw_runs["metaheuristic"] == best_mh)
    ].dropna(subset=["msi"])
    if sub.empty:
        return 0
    return int(sub.loc[sub["msi"].idxmax(), "seed"])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float | None) -> str:
    if v is None or pd.isna(v):
        return "N/A"
    return f"{float(v):.2f}"


def _k_from_labels(labels: np.ndarray) -> int:
    return len(set(labels.tolist()) - {-1})


def _scatter_labels(ax: Axes, mds: np.ndarray, labels: np.ndarray, title: str) -> None:
    unique = sorted(set(labels.tolist()))
    non_noise = [lbl for lbl in unique if lbl != -1]
    color_map = {
        lbl: VIBRANT_CLUSTER_COLORS[i % len(VIBRANT_CLUSTER_COLORS)]
        for i, lbl in enumerate(non_noise)
    }
    for lbl in unique:
        mask = labels == lbl
        color = OUTLIER_COLOR if lbl == -1 else color_map[lbl]
        ax.scatter(
            mds[mask, 0], mds[mask, 1],
            c=[color], s=15, alpha=0.7, edgecolors="none",
        )
    ax.set_title(title, fontsize=9)
    ax.set_xticks([])
    ax.set_yticks([])


def _title_block(rec: pd.Series, k: int, display_algo: str, best_mh: str) -> str:
    dataset_id = str(rec["dataset_id"])
    mh_label = MH_LABELS.get(best_mh, best_mh.upper())
    msi_mh = _fmt(rec.get(f"msi_{best_mh}"))
    msi_db = _fmt(rec.get("msi_dbscan"))

    if str(rec["group"]) == "classf":
        ari_mh = _fmt(rec.get(f"ari_{best_mh}"))
        ari_db = _fmt(rec.get("ari_dbscan"))
        return (
            f"{dataset_id}\n"
            f"MSI {mh_label}={msi_mh} | DDK={msi_db}\n"
            f"ARI {mh_label}={ari_mh} | DDK={ari_db}\n"
            f"{display_algo} K={k}"
        )
    return f"{dataset_id}\nMSI {mh_label}={msi_mh} | DDK={msi_db}\n{display_algo} K={k}"


def load_epsilons() -> dict[str, float | None]:
    path = CONFIGS / "exp03_epsilons_distk.yaml"
    with path.open() as f:
        raw = yaml.safe_load(f)
    out: dict[str, float | None] = {}
    for k, v in raw.items():
        if v is None:
            out[k] = None
        elif isinstance(v, dict):
            eps = v.get("epsilon")
            out[k] = float(eps) if eps is not None else None
        else:
            out[k] = float(v)
    return out


# ---------------------------------------------------------------------------
# Algorithm runners
# ---------------------------------------------------------------------------

def _run_rko_single_mh(
    features: np.ndarray, seed: int, dataset_id: str, best_mh: str,
) -> dict[str, Any]:
    """Run RKO with only best_mh active (others at 0)."""
    scaled_mds, ppp = preprocess(features)
    seed_run(seed)
    env = ClustGradeEnv(
        features, ppp, scaled_mds,
        instance_name=dataset_id,
        max_time=RKO_TIME_BUDGET,
        msi_space=MSI_SPACE,
        quadrat_filter=QUADRAT_FILTER,
        q_learning=False,
    )
    mix = {mh: (1 if mh == best_mh else 0) for mh in METAHEURISTICS}
    solver = RKO(env, print_best=False)
    best_cost, best_keys, _ = solver.solve(
        time_total=RKO_TIME_BUDGET,
        brkga=mix["brkga"],
        vns=mix["vns"],
        ils=mix["ils"],
        runs=1,
    )
    if best_keys is not None:
        result = env.decode_and_evaluate(np.array(best_keys))
        labels = np.array(result["labels"], dtype=np.int64)
    else:
        labels = np.full(features.shape[0], -1, dtype=np.int64)
    return {"labels": labels}


def _run_dbscan(mds: np.ndarray, eps: float) -> dict[str, Any]:
    labels = DBSCAN(eps=eps, min_samples=MIN_PTS).fit_predict(mds)
    return {"labels": labels}


# ---------------------------------------------------------------------------
# Context / caching
# ---------------------------------------------------------------------------

def _build_context(best_mh: str, raw_runs: pd.DataFrame) -> dict[str, Any]:
    manifest = load_manifest()
    epsilons = load_epsilons()
    manifest_map = {str(row["dataset_id"]): row for _, row in manifest.iterrows()}
    return {
        "best_mh": best_mh,
        "raw_runs": raw_runs,
        "manifest_map": manifest_map,
        "epsilons": epsilons,
        "feature_cache": {},
        "mds_cache": {},
        "labels_cache": {},
    }


def _get_features(dataset_id: str, context: dict[str, Any]) -> np.ndarray:
    cache: dict[str, np.ndarray] = context["feature_cache"]
    if dataset_id not in cache:
        row = context["manifest_map"][dataset_id]
        cache[dataset_id] = load_features(row)
    return cache[dataset_id]


def _get_scaled_mds(dataset_id: str, group: str, context: dict[str, Any]) -> np.ndarray:
    cache: dict[tuple[str, str], np.ndarray] = context["mds_cache"]
    key = (dataset_id, group)
    if key not in cache:
        from sklearn.preprocessing import MinMaxScaler
        raw = load_mds(dataset_id, group)
        cache[key] = MinMaxScaler().fit_transform(raw)
    return cache[key]


def _get_labels(
    dataset_id: str, group: str, algo: str, seed: int, context: dict[str, Any],
) -> np.ndarray | None:
    cache: dict[tuple[str, str, str, int], np.ndarray | None] = context["labels_cache"]
    key = (dataset_id, group, algo, seed)
    if key in cache:
        return cache[key]

    features = _get_features(dataset_id, context)
    scaled_mds = _get_scaled_mds(dataset_id, group, context)
    best_mh = context["best_mh"]

    if algo == "MH":
        result = _run_rko_single_mh(features, seed, dataset_id, best_mh)
        labels: np.ndarray | None = result["labels"]
    else:
        eps = context["epsilons"].get(dataset_id)
        if eps is None:
            labels = None
        else:
            result = _run_dbscan(scaled_mds, eps)
            labels = result["labels"]

    cache[key] = labels
    return labels


# ---------------------------------------------------------------------------
# Panel rendering
# ---------------------------------------------------------------------------

def _scatter_ranked_instances(
    records: pd.DataFrame,
    algo: str,
    fig_title: str,
    out_name: str,
    context: dict[str, Any],
) -> None:
    """Render a 1×PANEL_COUNT scatter figure for the given ranked records.

    Args:
        records: subset of delta_frame, already sorted; each row has msi_{best_mh},
                 msi_dbscan, ari_{best_mh}, ari_dbscan columns.
        algo: "MH" to run the best metaheuristic, "DBSCAN" for DBSCAN-DistK.
        fig_title: suptitle string.
        out_name: output filename (written to OUT_DIR).
        context: shared caches + best_mh + raw_runs.
    """
    best_mh = context["best_mh"]
    mh_label = MH_LABELS.get(best_mh, best_mh.upper())
    n = min(PANEL_COUNT, len(records))
    fig, axes = plt.subplots(1, PANEL_COUNT, figsize=(24, 4.8))

    for idx, ax in enumerate(axes):
        if idx >= n:
            ax.axis("off")
            continue

        rec = records.iloc[idx]
        dataset_id = str(rec["dataset_id"])
        group = str(rec["group"])

        if algo == "MH":
            seed = _get_best_seed(dataset_id, best_mh, context["raw_runs"])
            display_algo = mh_label
        else:
            seed = 0
            display_algo = "DBSCAN-DistK"

        labels = _get_labels(dataset_id, group, algo, seed, context)

        if labels is None:
            ax.set_title(f"{dataset_id}\n(no epsilon for DBSCAN-DistK)", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
            continue

        scaled_mds = _get_scaled_mds(dataset_id, group, context)
        k = _k_from_labels(labels)
        _scatter_labels(ax, scaled_mds, labels, _title_block(rec, k, display_algo, best_mh))

    fig.suptitle(fig_title, y=1.05)
    fig.tight_layout()
    fig.savefig(OUT_DIR / out_name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved %s", out_name)


def _load_wlt_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing W/L/T table: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"metaheuristic", "label", "V", "D", "E", "n_total"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"W/L/T table missing columns {missing}: {path}")
    return df


def _load_gap_summary(path: Path) -> pd.DataFrame:
    if not path.exists():
        logger.warning("Missing GAP summary: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    required = {"metric", "method_code", "method_label", "mean_gap_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"GAP summary missing columns {missing}: {path}")
    return df


def _append_hybrid_wlt_row(table: pd.DataFrame, metric_label: str) -> pd.DataFrame:
    """Append ClustGrade-RKO Híbrido W/L/T row if missing.

    Uses Exp 03 hybrid-vs-DBSCAN deltas from comparison.csv and the same
    clinically-meaningful tie threshold (Delta <= 0.01).
    """
    if table.empty:
        return table

    data = table.copy()
    if HYBRID_METHOD_CODE in data["metaheuristic"].astype(str).tolist():
        return data

    if not EXP07_COMPARISON_CSV.exists():
        logger.warning("Missing %s; stacked bars will not include hybrid row", EXP07_COMPARISON_CSV)
        return data

    comp = pd.read_csv(EXP07_COMPARISON_CSV)
    required = {"group", "msi_delta_distk", "ari_delta_distk"}
    missing = required - set(comp.columns)
    if missing:
        logger.warning("comparison.csv missing columns %s; skipping hybrid row", missing)
        return data

    if metric_label.upper() == "ISM":
        deltas = np.asarray(comp["msi_delta_distk"].dropna().to_numpy(), dtype=float)
    elif metric_label.upper() == "IRA":
        deltas = np.asarray(
            comp.loc[comp["group"] == "classf", "ari_delta_distk"].dropna().to_numpy(),
            dtype=float,
        )
    else:
        logger.warning("Unknown metric label %s; skipping hybrid row", metric_label)
        return data

    n_total = int(len(deltas))
    if n_total == 0:
        logger.warning("No hybrid deltas found for metric %s; skipping hybrid row", metric_label)
        return data

    wins = int(np.sum(deltas > DELTA_TOL))
    losses = int(np.sum(deltas < -DELTA_TOL))
    ties = int(n_total - wins - losses)

    hybrid_row = pd.DataFrame([
        {
            "metaheuristic": HYBRID_METHOD_CODE,
            "label": HYBRID_LABEL,
            "n_total": n_total,
            "V": wins,
            "D": losses,
            "E": ties,
            "overall_prop": None,
            "relative_prop": round(wins / n_total, 4),
        }
    ])

    data = pd.concat([data, hybrid_row], ignore_index=True)

    total_wins = int(data["V"].astype(int).sum())
    if total_wins > 0:
        data["overall_prop"] = (data["V"].astype(float) / total_wins).round(4)

    order = {"brkga": 0, "vns": 1, "ils": 2, HYBRID_METHOD_CODE: 3}
    data = (
        data.assign(_order=data["metaheuristic"].map(order).fillna(99))
        .sort_values("_order")
        .drop(columns=["_order"])
        .reset_index(drop=True)
    )

    return data


def _plot_stacked_wins_vs_dbscan(table: pd.DataFrame, metric_label: str, out_name: str) -> None:
    """Stacked bars: method wins (V) + DBSCAN wins (D) + ties (E), with counts inside bars.

    Legend shows only 'Vitórias vs. DBSCAN' (red) and 'Empates (Delta <= 0.01)' (gray).
    Title includes N=<n_total>. V + D + E must equal n_total for every row.
    """
    if table.empty:
        logger.warning("Skipped stacked wins plot (%s): empty table", out_name)
        return

    data = _append_hybrid_wlt_row(table, metric_label)
    data["method_wins"] = data["V"].astype(int)
    data["dbscan_wins"] = data["D"].astype(int)
    data["ties"] = data["E"].astype(int)
    n_total = int(data["n_total"].iloc[0])

    # Integrity check: every row must close to n_total
    for _, row in data.iterrows():
        total = int(row["method_wins"]) + int(row["dbscan_wins"]) + int(row["ties"])
        if total != n_total:
            raise ValueError(
                f"[{out_name}] V+D+E={total} ≠ n_total={n_total} "
                f"for metaheuristic='{row['metaheuristic']}'"
            )

    method_codes = data["metaheuristic"].astype(str).tolist()
    labels = data["label"].astype(str).tolist()

    x = np.arange(len(data))
    method_vals = data["method_wins"].values.astype(float)
    dbscan_vals = data["dbscan_wins"].values.astype(float)
    ties_vals = data["ties"].values.astype(float)

    TIES_COLOR = "#AAAAAA"

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    method_colors = _bar_colors_with_hybrid(method_codes)

    # Segment 1: method wins — no legend entry
    bars_method = ax.bar(
        x,
        method_vals,
        color=method_colors,
        edgecolor="black",
        linewidth=0.5,
    )
    # Segment 2: DBSCAN wins — red, shown in legend
    bars_dbscan = ax.bar(
        x,
        dbscan_vals,
        bottom=method_vals,
        color=DBSCAN_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label="Vitórias DBSCAN",
    )
    # Segment 3: ties — gray, shown in legend
    bars_ties = ax.bar(
        x,
        ties_vals,
        bottom=method_vals + dbscan_vals,
        color=TIES_COLOR,
        edgecolor="black",
        linewidth=0.5,
        label=r"Empates ($\Delta \leq 0{,}01$)",
    )

    # Annotations inside each segment
    for rect, value in zip(bars_method, method_vals):
        if value > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_y() + rect.get_height() / 2,
                f"{int(value)}",
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )
    for rect, base, value in zip(bars_dbscan, method_vals, dbscan_vals):
        if value > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                base + value / 2,
                f"{int(value)}",
                ha="center", va="center",
                fontsize=9, color="white", fontweight="bold",
            )
    for rect, base, value in zip(bars_ties, method_vals + dbscan_vals, ties_vals):
        if value > 0:
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                base + value / 2,
                f"{int(value)}",
                ha="center", va="center",
                fontsize=9, color="black", fontweight="bold",
            )

    max_total = float(np.max(method_vals + dbscan_vals + ties_vals)) if len(method_vals) else 0.0
    ax.set_ylim(0, max_total + 2)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Instâncias")
    ax.set_xlabel("")
    ax.set_title(f"Vitórias e Empates vs. DBSCAN({metric_label}, N={n_total})")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_name)


def _plot_mean_gap_pct(gap_summary: pd.DataFrame, metric: str, title: str, out_name: str) -> None:
    if gap_summary.empty:
        logger.warning("Skipped mean GAP%% plot (%s): empty summary", out_name)
        return

    data = gap_summary[gap_summary["metric"] == metric].copy()
    if data.empty:
        logger.warning("Skipped mean GAP%% plot (%s): metric %s not found", out_name, metric)
        return

    order = {"brkga": 0, "vns": 1, "ils": 2, "hybrid": 3}
    data = data.assign(_order=data["method_code"].map(order).fillna(99)).sort_values("_order")
    labels = data["method_label"].astype(str).tolist()
    values = data["mean_gap_pct"].values.astype(float)
    method_codes = data["method_code"].astype(str).tolist()
    colors = _bar_colors_with_hybrid(method_codes)

    x = np.arange(len(data))
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
    ax.axhline(0.0, color="black", linewidth=0.8)

    for rect, value in zip(bars, values):
        offset = 0.8 if value >= 0 else -0.8
        va = "bottom" if value >= 0 else "top"
        ax.text(
            rect.get_x() + rect.get_width() / 2,
            value + offset,
            f"{value:+.2f}%",
            ha="center",
            va=va,
            fontsize=9,
        )

    span = max(abs(values.min(initial=0.0)), abs(values.max(initial=0.0)))
    ax.set_ylim(-(span + 4.0), span + 4.0)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("GAP médio (%)")
    ax.set_xlabel("")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.grid(axis="x", visible=False)

    fig.tight_layout()
    out_path = OUT_DIR / out_name
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
    logger.info("Saved %s", out_name)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    print("Generating best-MH vs DBSCAN-DistK scatter plots ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    medians = pd.read_csv(EXP06_MEDIANS)
    dbscan_raw = pd.read_csv(EXP07_DBSCAN)
    dbscan = dbscan_raw[["dataset_id", "group", "msi", "ari"]].rename(
        columns={"msi": "msi_dbscan", "ari": "ari_dbscan"},
    )
    raw_runs = pd.read_csv(EXP06_RAW_RUNS)

    best_mh = select_best_mh(medians)
    mh_label = MH_LABELS[best_mh]
    print(f"Best metaheuristic: {mh_label}")

    delta_frame = _build_delta_frame(medians, dbscan, best_mh)
    context = _build_context(best_mh, raw_runs)

    # --- Top-5 datasets where best MH outperforms DBSCAN-DistK (MSI) ---
    msi_valid = delta_frame.dropna(subset=["msi_delta"]).copy()
    msi_mh_wins = msi_valid.sort_values("msi_delta", ascending=False).head(PANEL_COUNT).reset_index(drop=True)

    _scatter_ranked_instances(
        msi_mh_wins, "MH",
        f"Exp 03: Top 5 MSI margins — {mh_label} outperforms DBSCAN-DistK",
        f"plot-01-{best_mh}-top5-msi-win.png", context,
    )
    _scatter_ranked_instances(
        msi_mh_wins, "DBSCAN",
        "Exp 03: DBSCAN-DistK on same datasets (comparison)",
        "plot-02-dbscan-top5-msi-win.png", context,
    )

    # --- Top-5 datasets where DBSCAN-DistK outperforms best MH (MSI) ---
    msi_db_wins = msi_valid.sort_values("msi_delta", ascending=True).head(PANEL_COUNT).reset_index(drop=True)

    _scatter_ranked_instances(
        msi_db_wins, "DBSCAN",
        f"Exp 03: Top 5 MSI margins — DBSCAN-DistK outperforms {mh_label}",
        "plot-05-dbscan-top5-msi-loss.png", context,
    )
    _scatter_ranked_instances(
        msi_db_wins, "MH",
        f"Exp 03: {mh_label} on same datasets (comparison)",
        f"plot-06-{best_mh}-top5-msi-loss.png", context,
    )

    # --- Top-5 datasets where best MH outperforms DBSCAN-DistK (ARI, CLASSF) ---
    classf = delta_frame[delta_frame["group"] == "classf"].copy()
    ari_valid = classf.dropna(subset=["ari_delta"])
    if len(ari_valid) > 0:
        ari_mh_wins = ari_valid.sort_values("ari_delta", ascending=False).head(PANEL_COUNT).reset_index(drop=True)

        _scatter_ranked_instances(
            ari_mh_wins, "MH",
            f"Exp 03: Top 5 ARI margins — {mh_label} outperforms DBSCAN-DistK (CLASSF)",
            f"plot-03-{best_mh}-top5-ari-win.png", context,
        )
        _scatter_ranked_instances(
            ari_mh_wins, "DBSCAN",
            "Exp 03: DBSCAN-DistK on same datasets (comparison)",
            "plot-04-dbscan-top5-ari-win.png", context,
        )

    # --- Reviewer complementary plots: W/L stacked bars + mean GAP% bars ---
    wlt_msi = _load_wlt_table(WLT_MSI_CSV)
    wlt_ari = _load_wlt_table(WLT_ARI_CSV)
    gap_summary = _load_gap_summary(GAP_SUMMARY_CSV)

    _plot_stacked_wins_vs_dbscan(
        wlt_msi,
        metric_label="ISM",
        out_name="barplot_mh_vs_dbscan_stacked_wins_ism.png",
    )
    _plot_stacked_wins_vs_dbscan(
        wlt_ari,
        metric_label="IRA",
        out_name="barplot_mh_vs_dbscan_stacked_wins_ira.png",
    )

    _plot_mean_gap_pct(
        gap_summary,
        metric="MSI",
        title="GAP médio percentual vs DBSCAN (ISM)",
        out_name="barplot_mh_vs_dbscan_mean_gap_pct_ism.png",
    )
    _plot_mean_gap_pct(
        gap_summary,
        metric="ARI_CLASSF",
        title="GAP médio percentual vs DBSCAN (IRA)",
        out_name="barplot_mh_vs_dbscan_mean_gap_pct_ira.png",
    )

    print(f"Done. Artifacts written to {OUT_DIR}")


if __name__ == "__main__":
    main()
