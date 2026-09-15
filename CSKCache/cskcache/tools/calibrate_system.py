"""Persist one offline system-performance measurement set.

The GPU/SSD benchmark remains an explicit user-run experiment.  This command
validates its compact JSON result and writes the exact profile format consumed
by the runtime profitability gate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Mapping

from ..runtime.system_profile import SystemPerformanceTracker


def _measurement(payload: Mapping[str, object], key: str) -> tuple[int, float]:
    item = payload.get(key)
    if not isinstance(item, Mapping):
        raise ValueError(f"calibration input requires {key}")
    amount = item.get("amount")
    duration_ms = item.get("duration_ms")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        raise ValueError(f"{key}.amount must be a positive integer")
    if (
        isinstance(duration_ms, bool)
        or not isinstance(duration_ms, (int, float))
        or duration_ms <= 0
    ):
        raise ValueError(f"{key}.duration_ms must be positive")
    return amount, float(duration_ms)


def write_calibration(specification: Mapping[str, object]) -> Path:
    metadata_path = specification.get("metadata_path")
    environment = specification.get("environment")
    if not isinstance(metadata_path, str) or not metadata_path:
        raise ValueError("calibration input requires metadata_path")
    if not isinstance(environment, Mapping):
        raise ValueError("calibration input requires environment")
    explicit = specification.get("system_profile_path")
    if explicit is not None and not isinstance(explicit, str):
        raise ValueError("system_profile_path must be a string or null")
    strategy = specification.get("strategy")
    ratio = specification.get("ratio")
    if not isinstance(strategy, str) or not strategy:
        raise ValueError("calibration input requires strategy")
    if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
        raise ValueError("calibration input requires numeric ratio")

    tracker = SystemPerformanceTracker(
        enabled=True,
        environment=dict(environment),
        metadata_path=metadata_path,
        explicit_path=explicit,
    )
    prefill_tokens, prefill_ms = _measurement(specification, "prefill")
    ssd_bytes, ssd_ms = _measurement(specification, "ssd_read")
    h2d_bytes, h2d_ms = _measurement(specification, "h2d")
    composition_tokens, composition_ms = _measurement(
        specification, "composition"
    )
    tracker.record_prefill(prefill_tokens, prefill_ms)
    tracker.record_ssd(ssd_bytes, ssd_ms)
    tracker.record_h2d(h2d_bytes, h2d_ms)
    tracker.record_composition(
        strategy=strategy,
        ratio=float(ratio),
        tokens=composition_tokens,
        duration_ms=composition_ms,
    )
    if not tracker.profile.is_ready(strategy, float(ratio)):
        raise RuntimeError("calibration did not produce a ready profile")
    return tracker.path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Persist one measured CSKCache system calibration"
    )
    parser.add_argument(
        "specification",
        type=Path,
        help="JSON file produced by the explicit offline benchmark",
    )
    args = parser.parse_args()
    payload = json.loads(args.specification.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("calibration specification must be a JSON object")
    path = write_calibration(payload)
    print(f"CSKCache system performance profile saved to {path}")


if __name__ == "__main__":
    main()
