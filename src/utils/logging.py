from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a module-level logger configured for ClustGrade-RKO.
    """

    # Reuse logger instances by name.
    logger = logging.getLogger(name)
    # Configure once to avoid duplicate handlers.
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
