"""Parse one measured CSK request from the JSONL profiler."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSK profile does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise TypeError(f"profile line {line_number} is not an object")
        records.append(payload)
    return records


def _one(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("event") == event]
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one {event} event for measured request, found {len(selected)}"
        )
    return selected[0]


def parse_csk_profile(
    path: Path, *, request_id: str, profile_layer: int
) -> dict[str, int | float]:
    all_records = read_jsonl(path)
    matching_ids = {
        str(record.get("request_id"))
        for record in all_records
        if str(record.get("request_id", "")) == request_id
        or re.fullmatch(
            rf"{re.escape(request_id)}-[0-9a-f]{{8}}",
            str(record.get("request_id", "")),
        )
    }
    if len(matching_ids) != 1:
        raise RuntimeError(
            f"expected one engine request ID below {request_id}, "
            f"found {sorted(matching_ids)}"
        )
    engine_request_id = next(iter(matching_ids))
    records = [
        record
        for record in all_records
        if str(record.get("request_id", "")) == engine_request_id
    ]
    reuse = _one(records, "csk_reuse_registered")
    correction = _one(records, "csk_correction_complete")
    layer_event = _one(records, "cskcache_layer_compute")
    bind = _one(records, "csk_request_bind")
    layer_rows = [
        row
        for row in layer_event.get("calibration_correct_install", [])
        if int(row.get("layer", -1)) == profile_layer
    ]
    if len(layer_rows) != 1:
        raise RuntimeError(
            f"expected layer {profile_layer} once, found {len(layer_rows)}"
        )
    layer = layer_rows[0]
    reuse_start = int(reuse["reuse_start"])
    reuse_end = int(reuse["reuse_end"])
    if reuse_end <= reuse_start:
        raise RuntimeError("profile contains an empty reuse interval")
    forward_ms = float(layer["calibration_forward_ms"])
    residual_ms = float(layer["residual_correction_ms"])
    return {
        "matched_tokens": int(bind["matched_tokens"]),
        "reused_tokens": reuse_end - reuse_start,
        "reuse_start": reuse_start,
        "reuse_end": reuse_end,
        "actual_calibration_tokens": int(correction["calibration_tokens"]),
        "profile_layer": profile_layer,
        "calibration_forward_ms": forward_ms,
        "residual_correction_ms": residual_ms,
        "calibration_compute_ms": forward_ms + residual_ms,
        "layer_gpu_ms": float(layer["gpu_ms"]),
    }
