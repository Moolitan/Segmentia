"""Strict profile validation for the deviation-top-k baseline."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any


def _records_for_request(path: Path, request_id: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"CSK profile does not exist: {path}")
    all_records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not all(isinstance(record, dict) for record in all_records):
        raise TypeError("CSK profile contains a non-object row")
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
    return [
        record
        for record in all_records
        if str(record.get("request_id", "")) == engine_request_id
    ]


def _one(records: list[dict[str, Any]], event: str) -> dict[str, Any]:
    selected = [record for record in records if record.get("event") == event]
    if len(selected) != 1:
        raise RuntimeError(f"expected one {event} event, found {len(selected)}")
    return selected[0]


def parse_deviation_topk_profile(
    path: Path,
    *,
    request_id: str,
    expected_layers: int,
    expected_ratio: float,
    expected_check_layer: int,
) -> dict[str, int | float | str]:
    """Validate check-once deviation selection across every model layer."""

    records = _records_for_request(path, request_id)
    bind = _one(records, "csk_request_bind")
    reuse = _one(records, "csk_reuse_registered")
    correction = _one(records, "csk_correction_complete")
    layers = [
        record
        for record in records
        if record.get("event") == "cskcache_deviation_topk_layer"
    ]
    layers.sort(key=lambda record: int(record["layer"]))
    if len(layers) != expected_layers:
        raise RuntimeError(
            f"expected {expected_layers} deviation layers, found {len(layers)}"
        )
    if [int(record["layer"]) for record in layers] != list(range(expected_layers)):
        raise RuntimeError("deviation layer IDs are not contiguous")
    if correction.get("correction_strategy") != "deviation_topk":
        raise RuntimeError("unexpected correction strategy")
    if correction.get("execution_method") != "deviation_topk":
        raise RuntimeError("unexpected deviation execution method")
    if int(correction.get("calibration_tokens", -1)) != 0:
        raise RuntimeError("deviation_topk must not report prefix calibration")

    reuse_start = int(reuse["reuse_start"])
    reuse_end = int(reuse["reuse_end"])
    candidate_tokens = reuse_end - reuse_start
    if candidate_tokens <= 0:
        raise RuntimeError("profile contains an empty reuse interval")
    selected_tokens = max(1, int(candidate_tokens * expected_ratio))
    for layer_id, record in enumerate(layers):
        if int(record["candidate_tokens"]) != candidate_tokens:
            raise RuntimeError("deviation candidate token count changed across layers")
        if not math.isclose(
            float(record["recompute_ratio"]), expected_ratio, abs_tol=1e-12
        ):
            raise RuntimeError("deviation recompute ratio mismatch")
        if int(record["check_layer"]) != expected_check_layer:
            raise RuntimeError("deviation check layer mismatch")
        expected_tokens = (
            candidate_tokens if layer_id < expected_check_layer else selected_tokens
        )
        if int(record["recomputed_tokens"]) != expected_tokens:
            raise RuntimeError(f"unexpected recompute count at layer {layer_id}")
        if bool(record["selection_applied"]) != (
            layer_id == expected_check_layer
        ):
            raise RuntimeError(f"unexpected selection state at layer {layer_id}")
    return {
        "matched_tokens": int(bind["matched_tokens"]),
        "reused_tokens": candidate_tokens,
        "reuse_start": reuse_start,
        "reuse_end": reuse_end,
        "candidate_tokens": candidate_tokens,
        "selected_tokens": selected_tokens,
        "recompute_ratio": expected_ratio,
        "check_layer": expected_check_layer,
        "correction_strategy": str(correction["correction_strategy"]),
        "execution_method": str(correction["execution_method"]),
    }
