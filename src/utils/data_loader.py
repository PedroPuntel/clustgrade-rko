from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def load_dataset(path: str | Path, *, has_header: bool = False) -> pd.DataFrame:
    """
    Load a dataset from CSV/TXT with automatic delimiter detection.
    """

    # Resolve the file path once.
    file_path = Path(path)
    # Configure header behavior for raw datasets.
    header = 0 if has_header else None
    # Try auto delimiter detection first.
    try:
        return pd.read_csv(file_path, sep=None, engine="python", header=header)
    except pd.errors.ParserError:
        # Fall back to whitespace-delimited parsing.
        return pd.read_csv(file_path, sep=r"\s+", engine="python", header=header)


def split_features_labels(
    data: pd.DataFrame, label_col: int | str
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split a dataset into features and label series.
    """

    # Extract the label series.
    labels = data[label_col]
    # Drop the label column from the feature matrix.
    features = data.drop(columns=[label_col])
    return features, labels


def to_numpy(data: pd.DataFrame) -> Any:
    """
    Convert a DataFrame to a float NumPy array.
    """

    # Convert to a NumPy array with float dtype.
    return data.to_numpy(dtype=float)
