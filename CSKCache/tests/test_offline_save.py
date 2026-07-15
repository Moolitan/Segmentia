from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cskcache.v1.core.cache_engine import CSKCacheEngine
from cskcache.v1.core.config import CSKCacheConfig
from cskcache.v1.kv_transfer.gpu_connector import VLLMPagedGPUConnector
from cskcache.v1.storage.local_disk_backend import LocalDiskBackend
from cskcache.v1.storage.storage_manager import StorageManager


BLOCK_SIZE = 4
TOKENS = [10, 11, 12, 13]


def _signal() -> dict:
    return {
        "cskcache": {
            "operation": "save",
            "cache_id": "skill-a",
            "source_start": 0,
            "source_end": len(TOKENS),
        }
    }


def test_offline_save_waits_for_full_span_and_persists() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager.with_disk(tmp)
        connector = VLLMPagedGPUConnector(BLOCK_SIZE)
        engine = CSKCacheEngine(
            CSKCacheConfig(),
            storage,
            block_size=BLOCK_SIZE,
            gpu_connector=connector,
        )

        matched, load_async = engine.get_num_new_matched_tokens(
            "r1", TOKENS, 0, _signal()
        )
        assert matched == 0 and load_async is False
        assert "r1" not in engine._pending_saves

        engine.update_save_after_alloc(
            "r1",
            TOKENS,
            0,
            ([0],),
            _signal(),
        )
        assert "r1" in engine._pending_saves
        requests, probes, saves, _ = engine.build_meta({"r1": 2})
        assert not requests and not probes and not saves

        engine.update_save_after_alloc(
            "r1",
            TOKENS,
            2,
            ([0],),
            _signal(),
        )
        requests, probes, saves, _ = engine.build_meta({"r1": 2})
        assert not requests and not probes and len(saves) == 1

        key = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
        value = key + 1000
        packed = torch.stack((key, value), dim=0).reshape(2, 1, 4, 2, 3)
        engine.register_kv_caches({"layer0": packed})
        engine.capture_saves(saves, "layer0", packed)
        assert engine.finalize_saves() == ["skill-a"]

        reopened = LocalDiskBackend(tmp)
        entry = reopened.get("skill-a")
        assert entry is not None
        assert entry.source_start == 0 and entry.source_end == 4
        assert entry.token_ids == TOKENS
        saved_key, saved_value = entry.kv_by_layer["layer0"]
        assert torch.equal(saved_key, key)
        assert torch.equal(saved_value, value)


def test_capture_only_does_not_materialize_existing_catalog() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        storage = StorageManager.with_disk(tmp)
        connector = VLLMPagedGPUConnector(BLOCK_SIZE)
        engine = CSKCacheEngine(
            CSKCacheConfig(capture_only=True),
            storage,
            block_size=BLOCK_SIZE,
            gpu_connector=connector,
        )
        assert engine.catalog.segments == []


if __name__ == "__main__":
    test_offline_save_waits_for_full_span_and_persists()
    print("PASS test_offline_save_waits_for_full_span_and_persists")
    test_capture_only_does_not_materialize_existing_catalog()
    print("PASS test_capture_only_does_not_materialize_existing_catalog")
