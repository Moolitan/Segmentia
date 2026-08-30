"""Lightweight structured runtime events emitted by CSKCache.

The profiler is intentionally owned by CSKCache rather than LMCache.  LMCache
and vLLM may call :func:`profile_event` while executing CSKCache's physical
data-plane operations, but they do not define the event namespace.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from typing import Any


PROFILE_ENABLED = os.getenv("CSKCACHE_PROFILE", "0") == "1"
PROFILE_MARKER = "CSKCACHE_PROFILE_EVENT"
PROFILE_TRACE_PATH = os.getenv("CSKCACHE_PROFILE_TRACE_PATH")
PROFILE_STDOUT_ENABLED = os.getenv(
    "CSKCACHE_PROFILE_STDOUT",
    "0" if PROFILE_TRACE_PATH else "1",
) == "1"

_LOGGER = logging.getLogger("cskcache.profile")

if PROFILE_ENABLED and PROFILE_STDOUT_ENABLED:
    # CSKCache profile records are experimental data rather than ordinary
    # application diagnostics.  Give them an explicit handler so their
    # visibility does not depend on the embedding server's root logger level.
    # vLLM and LMCache configure logging independently, and an INFO record from
    # this otherwise-unconfigured child logger can therefore be silently
    # filtered even though the CSKCache data path executed successfully.
    _handler = logging.StreamHandler(sys.stdout)
    _handler.setFormatter(logging.Formatter("%(message)s"))
    _LOGGER.addHandler(_handler)
    _LOGGER.setLevel(logging.INFO)
    _LOGGER.propagate = False


def profile_event(event: str, request_id: str, **fields: Any) -> None:
    """Emit one machine-readable event when CSKCache profiling is enabled."""

    if not PROFILE_ENABLED:
        return
    payload = {
        "event": event,
        "request_id": request_id,
        "time_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        **fields,
    }
    if PROFILE_STDOUT_ENABLED:
        _LOGGER.info("%s %s", PROFILE_MARKER, json.dumps(payload, sort_keys=True))
    if PROFILE_TRACE_PATH:
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            PROFILE_TRACE_PATH,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
