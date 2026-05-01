"""Experiment 01 — plot the largest MSI-sensitivity parameter comparisons.

Produces:
    plot-largest-msi-sensitivity-1.png
    plot-largest-msi-sensitivity-2.png

Usage:
    poetry run python -m experiments.01-param-study.plot_largest_msi_sensitivity
"""
from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from _sensitivity_plot_common import N_TOP_DEFAULT, run_and_plot_selection, select_top_sensitive_datasets


def main() -> None:
    selections = select_top_sensitive_datasets("msi", N_TOP_DEFAULT)

    print(f"Top {len(selections)} MSI-sensitive datasets:")
    for rank, selection in enumerate(selections, start=1):
        print(
            "  "
            f"{selection.dataset_id}: gap={selection.gap:.3f} "
            f"({selection.max_config.name} vs {selection.min_config.name})"
        )
        run_and_plot_selection(selection, f"plot-largest-msi-sensitivity-{rank}.png")

    print("Done.")


if __name__ == "__main__":
    main()
