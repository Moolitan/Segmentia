from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.profiling import ProfileConfig, ProfileReporter, Profiler
from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.kv_transfer.gpu_connector import VLLMPagedGPUConnector
from cskcache.v1.metadata import CSKCacheEntry, CSKCacheMode, CSKLoadPlan, CSKReqMeta
from cskcache.v1.storage.storage_manager import StorageManager


class _RecordingReporter:
    def __init__(self) -> None:
        self.records: list[dict] = []

    def report(self, record) -> None:
        self.records.append(dict(record))


def _entry() -> CSKCacheEntry:
    key = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    return CSKCacheEntry(
        cache_id="skill",
        source_start=0,
        source_end=4,
        token_ids=[10, 11, 12, 13],
        kv_by_layer={"layer0": (key, key + 100)},
    )


def _load_once(storage: StorageManager, profiler: Profiler) -> dict | None:
    connector = VLLMPagedGPUConnector(block_size=4)
    connector.bind_kv_caches({"layer0": torch.zeros(2, 1, 4, 1, 2)})
    engine = CSKCacheEngine(
        CSKCacheConfig(),
        storage,
        block_size=4,
        gpu_connector=connector,
        profiler=profiler,
    )
    plan = CSKLoadPlan(
        req_id="r1",
        cache_id="skill",
        mode=CSKCacheMode.REUSE,
        start=0,
        end=4,
        token_ids=(10, 11, 12, 13),
        source_offset=0,
    )
    engine.load([CSKReqMeta(plan=plan, block_ids=([0],))])
    return profiler.reporter.records[-1] if profiler.reporter.records else None


def test_profile_config_is_independent_and_disabled_by_default() -> None:
    assert ProfileConfig.from_env({}) == ProfileConfig()
    config = ProfileConfig.from_env(
        {
            "CSKCACHE_PROFILE_ENABLED": "true",
            "CSKCACHE_PROFILE_JSONL": "/tmp/csk-profile.jsonl",
        }
    )
    assert config.enabled
    assert config.jsonl_path == Path("/tmp/csk-profile.jsonl")


def test_reporter_writes_process_isolated_jsonl() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        config = ProfileConfig(
            enabled=True,
            jsonl_path=Path(tmp) / "profile.jsonl",
        )
        reporter = ProfileReporter(config)
        profiler = Profiler(reporter=reporter, config=config)
        trace = profiler.start(
            kind="worker_load",
            req_id="r-json",
            cache_id="skill",
            metadata={"target_start": 0, "target_end": 4, "tokens": 4},
        )
        with trace.cpu_stage("storage_get"):
            pass
        profiler.finish(trace)

        assert reporter.jsonl_path is not None
        assert reporter.jsonl_path.name.startswith("profile.pid")
        record = json.loads(reporter.jsonl_path.read_text())
        assert record["trace_id"] == "worker_load:r-json:1"
        assert record["stage_ms"]["storage_get"] >= 0


def test_disabled_profiler_emits_nothing() -> None:
    reporter = _RecordingReporter()
    profiler = Profiler(ProfileConfig(enabled=False), reporter=reporter)
    storage = StorageManager()
    storage.put(_entry())
    _load_once(storage, profiler)
    assert reporter.records == []


def test_cpu_hit_profiles_existing_function_boundaries() -> None:
    reporter = _RecordingReporter()
    profiler = Profiler(ProfileConfig(enabled=True), reporter=reporter)
    storage = StorageManager()
    storage.put(_entry())

    record = _load_once(storage, profiler)

    assert record is not None
    assert record["kind"] == "worker_load"
    assert record["source_tier"] == "cpu"
    assert record["tokens"] == 4
    assert record["bytes"] == 64
    assert record["expected_layers"] == 1
    assert record["scattered_layers"] == 1
    assert record["skipped_layers"] == 0
    assert set(record["stage_ms"]) >= {
        "storage_get",
        "prepare_reuse_slice",
        "scatter_span",
    }


def test_disk_hit_reports_deserialize_and_promotion() -> None:
    reporter = _RecordingReporter()
    profiler = Profiler(ProfileConfig(enabled=True), reporter=reporter)
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager.with_disk(tmp, cpu_max_bytes=0)
        storage.put(_entry(), persist=True)

        record = _load_once(storage, profiler)

    assert record is not None
    assert record["source_tier"] == "disk"
    assert record["disk_entry_bytes"] == 64
    assert record["stage_ms"]["disk_deserialize"] >= 0
    assert record["stage_ms"]["storage_get"] >= record["stage_ms"]["disk_deserialize"]


def test_scheduler_lookup_is_separate_from_worker_load() -> None:
    reporter = _RecordingReporter()
    profiler = Profiler(ProfileConfig(enabled=True), reporter=reporter)
    storage = StorageManager()
    entry = _entry()
    storage.put(entry)
    engine = CSKCacheEngine(CSKCacheConfig(), storage, block_size=4, profiler=profiler)
    signal = {
        "cskcache": {
            "cache_id": "skill",
            "target_start": 0,
            "target_end": 4,
        }
    }

    assert engine.get_num_new_matched_tokens("r-lookup", entry.token_ids, 0, signal) == (
        4,
        False,
    )

    assert len(reporter.records) == 1
    record = reporter.records[0]
    assert record["kind"] == "scheduler_lookup"
    assert record["source_tier"] == "cpu"
    assert record["target_start"] == 0
    assert record["target_end"] == 4


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
    print("ALL PROFILING TESTS PASSED")
