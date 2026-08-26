from __future__ import annotations

import json

from paper_evaluation.common.driver import (
    _benchmark_request_id,
    _fallback,
    _timeline_ttft,
)
from paper_evaluation.common.schema import SAMPLE_COLUMNS, normalize_row


def _write_jsonl(path, records):
    path.write_text(
        "".join(json.dumps(record) + "\n" for record in records),
        encoding="utf-8",
    )


def test_timeline_uses_exact_request_id(tmp_path):
    trace = tmp_path / "timeline.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "request_id": "case-select",
                "event": "api_request_received",
                "monotonic_ns": 1,
            },
            {
                "request_id": "case-select",
                "event": "first_token_ready",
                "monotonic_ns": 2,
            },
            {
                "request_id": "case",
                "event": "api_request_received",
                "monotonic_ns": 1_000_000,
            },
            {
                "request_id": "case",
                "event": "first_token_ready",
                "monotonic_ns": 4_000_000,
                "prompt_tokens": 100,
                "cached_tokens": 75,
            },
        ],
    )
    assert _timeline_ttft(trace, "case") == (3.0, 100, 75)


def test_fallback_uses_exact_request_id(tmp_path):
    trace = tmp_path / "profile.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "request_id": "measure10",
                "event": "csk_fallback",
                "reason": "unrelated",
            },
            {
                "request_id": "measure1",
                "event": "csk_fallback",
                "reason": "authentication",
            },
        ],
    )
    assert _fallback(trace, "measure1") == (True, "authentication")


def test_sample_schema_has_cross_platform_merge_keys():
    row = normalize_row(
        {
            "platform_id": "host-a",
            "model_id": "Qwen3-70B",
            "tensor_parallel_size": 2,
            "system": "CSKCache",
            "skill_tokens": 12518,
            "chunk_tokens": 256,
            "storage_layout": "packed_chunks_single_layer",
            "ttft_ms": 12.5,
        }
    )
    assert tuple(row) == SAMPLE_COLUMNS
    assert row["platform_id"] == "host-a"
    assert row["model_id"] == "Qwen3-70B"
    assert row["tensor_parallel_size"] == 2


def test_benchmark_request_id_triggers_timeline_and_is_stable():
    first = _benchmark_request_id("A6000 CacheBlend-15% case")
    second = _benchmark_request_id("A6000 CacheBlend-15% case")
    assert first == second
    assert first.startswith("cskcache-latency-")
    assert "%" not in first and " " not in first
