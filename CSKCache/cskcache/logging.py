"""CSKCache-owned logging that remains visible inside vLLM processes."""

from __future__ import annotations

import logging
import os


def get_log_level() -> int:
    level_name = os.getenv("CSKCACHE_LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


def init_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.handlers.clear()
    logger.propagate = False

    level = get_log_level()
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(
        logging.Formatter(
            "[%(asctime)s] CSKCache %(levelname)s: %(message)s "
            "(%(filename)s:%(lineno)d:%(name)s)"
        )
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger
