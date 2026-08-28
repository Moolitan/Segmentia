from __future__ import annotations

import json

from paper_evaluation.common.csk_config import build_extra_config
from paper_evaluation.common.driver import (
    _benchmark_request_id,
    _fallback,
    _timeline_ttft,
)
from paper_evaluation.common.server import cskcache_environment
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


def test_fallback_accepts_vllm_engine_child_request_id(tmp_path):
    trace = tmp_path / "profile.jsonl"
    _write_jsonl(
        trace,
        [
            {
                "request_id": "measure1-92b02043",
                "event": "csk_fallback",
                "reason": "authentication",
            }
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


def test_raw_catalog_geometry_can_match_a_dedicated_pool(tmp_path):
    catalog = tmp_path / "catalog.json"
    catalog.write_text(
        json.dumps(
            {
                "containers": [
                    {
                        "raw_file_path": "/tmp/pool.bin",
                        "capacity_bytes": 512 * 1024**3,
                        "alignment_bytes": 4096,
                        "header_bytes": 4096,
                        "container_format_version": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    extra = build_extra_config(
        pool_root=tmp_path,
        model_id="Qwen3",
        backend="raw_block",
        chunk_tokens=256,
        storage_layout="packed_chunks_single_layer",
        host_layout="packed_chunks_single_layer",
        execution_order="h2d_first",
        correction_strategy="ratio_prefix",
        calibration_tokens=0,
        calibration_ratio=0.05,
        correction_alpha=0.6,
        minimum_full_recompute_tokens=32,
        minimum_reuse_tokens=256,
        catalog_override=catalog,
        raw_slot_bytes=40 * 1024**2,
        raw_metadata_bytes=256 * 1024**2,
    )
    assert extra["rust_raw_block.slot_bytes"] == 40 * 1024**2
    assert extra["rust_raw_block.meta_total_bytes"] == 256 * 1024**2


def test_cskcache_host_page_tokens_are_independent_from_logical_chunk() -> None:
    extra = {"csk_chunk_size_tokens": 256}
    environment = cskcache_environment(extra, host_page_tokens=512)
    assert environment["LMCACHE_CHUNK_SIZE"] == "512"
    assert json.loads(environment["LMCACHE_EXTRA_CONFIG"])[
        "csk_chunk_size_tokens"
    ] == 256
