"""
RKO framework bootstrap.

ClustGrade-RKO depends on the upstream Random-Key Optimizer framework
(`RKO.py` + `Environment.py`). That framework is distributed separately;
the public clustgrade-rko repo does not vendor it. Users clone the
framework alongside this repository and point the `RKO_FRAMEWORK_PATH`
environment variable at the directory that contains `RKO.py`.

This module centralises the resolution and sys.path injection so the
runtime modules (`environment.py`, `run.py`) get a clear, single error
message if the env var is missing or wrong.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ENV_VAR = "RKO_FRAMEWORK_PATH"
_REQUIRED_FILES = ("RKO.py", "Environment.py")


def bootstrap_framework() -> Path:
    """Resolve the RKO framework directory and insert it onto `sys.path`.

    Returns:
        Absolute path to the framework directory.

    Raises:
        ImportError: If the env var is unset or the directory does not
            contain the expected framework files.
    """
    raw = os.environ.get(_ENV_VAR)
    if not raw:
        raise ImportError(
            f"{_ENV_VAR} environment variable is not set. "
            "ClustGrade-RKO requires the upstream RKO framework. "
            "Clone it (see README) and set "
            f"{_ENV_VAR} to the directory that contains RKO.py."
        )

    framework = Path(raw).expanduser().resolve()
    if not framework.is_dir():
        raise ImportError(
            f"{_ENV_VAR}={framework} is not a directory. "
            "Point it at the RKO framework source folder."
        )

    missing = [f for f in _REQUIRED_FILES if not (framework / f).is_file()]
    if missing:
        raise ImportError(
            f"{_ENV_VAR}={framework} is missing required file(s): "
            f"{', '.join(missing)}. "
            "Make sure it points at the RKO framework source folder."
        )

    framework_str = str(framework)
    if framework_str not in sys.path:
        sys.path.insert(0, framework_str)
    return framework
