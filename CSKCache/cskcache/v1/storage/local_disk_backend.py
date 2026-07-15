from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

import torch

from cskcache.profiling import LoadTrace, NullLoadTrace
from cskcache.v1.metadata import CSKCacheEntry
from cskcache.v1.storage.abstract_backend import (
    StorageBackendInterface,
    entry_nbytes,
)


def entry_to_payload(entry: CSKCacheEntry) -> dict:
    """Serialize an entry into the same dict layout ``registry.load_file`` reads.

    Keeping one on-disk schema means a ``.pt`` produced offline by the capture
    tooling and one spilled here by the disk tier are interchangeable.
    """

    return {
        "cache_id": entry.cache_id,
        "source_start": entry.source_start,
        "source_end": entry.source_end,
        "token_ids": list(entry.token_ids),
        "kv_by_layer": entry.kv_by_layer,
    }


def payload_to_entry(payload: dict, device: torch.device | str | None = None) -> CSKCacheEntry:
    """Rebuild an entry from a ``.pt`` payload (mirror of ``registry.load_file``)."""

    return CSKCacheEntry(
        cache_id=str(payload["cache_id"]),
        source_start=int(payload["source_start"]),
        source_end=int(payload["source_end"]),
        token_ids=[int(value) for value in payload.get("token_ids", [])],
        kv_by_layer={
            str(layer): (
                key.to(device) if device else key,
                value.to(device) if device else value,
            )
            for layer, (key, value) in payload["kv_by_layer"].items()
        },
    )


class LocalDiskBackend(StorageBackendInterface):
    """On-disk cold tier storing one ``.pt`` payload per entry.

    Entries are keyed by ``cache_id`` but filed under a hash so arbitrary
    cache_ids stay filesystem-safe. A small ``.json`` sidecar carries the
    cache_id and byte size, letting the backend enumerate keys and account for
    space at startup without deserializing any tensors. Reads are lazy: an entry
    is only ``torch.load``-ed when actually requested.
    """

    def __init__(
        self,
        root: str | Path,
        device: torch.device | str | None = None,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._device = device
        # cache_id -> (payload_path, nbytes, length), rebuilt from sidecars on
        # startup. length is None for sidecars written before that field
        # existed, which get_metadata() treats as "no cheap answer".
        self._index: dict[str, tuple[Path, int, int | None]] = {}
        self._scan_existing()

    def _paths(self, cache_id: str) -> tuple[Path, Path]:
        digest = hashlib.sha256(cache_id.encode("utf-8")).hexdigest()[:32]
        return self._root / f"{digest}.pt", self._root / f"{digest}.json"

    def _scan_existing(self) -> None:
        for sidecar in self._root.glob("*.json"):
            try:
                meta = json.loads(sidecar.read_text())
                cache_id = str(meta["cache_id"])
                nbytes = int(meta["nbytes"])
                length = int(meta["length"]) if "length" in meta else None
            except Exception:
                continue
            payload_path = sidecar.with_suffix(".pt")
            if payload_path.exists():
                self._index[cache_id] = (payload_path, nbytes, length)

    def contains(self, cache_id: str) -> bool:
        return cache_id in self._index

    def get(
        self,
        cache_id: str,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> CSKCacheEntry | None:
        record = self._index.get(cache_id)
        if record is None:
            return None
        payload_path, nbytes, _length = record
        if trace is None or not trace.enabled:
            payload = torch.load(payload_path, map_location=self._device or "cpu")
            return payload_to_entry(payload, device=self._device)
        trace.set(disk_path=str(payload_path), disk_entry_bytes=nbytes)
        with trace.cpu_stage("disk_deserialize"):
            payload = torch.load(payload_path, map_location=self._device or "cpu")
            return payload_to_entry(payload, device=self._device)

    def get_metadata(
        self,
        cache_id: str,
        trace: LoadTrace | NullLoadTrace | None = None,
    ) -> tuple[int, int] | None:
        """Return (length, nbytes) straight from the sidecar-backed index,
        without torch.load-ing the KV tensors.

        Only sidecars written before the `length` field existed fall back to
        a full get() (via the base-class default), which also means they pay
        the deserialize cost every call since a metadata-only lookup never
        promotes into the CPU tier.
        """
        record = self._index.get(cache_id)
        if record is None:
            return None
        _, nbytes, length = record
        if length is not None:
            return length, nbytes
        return super().get_metadata(cache_id, trace=trace)

    def put(self, entry: CSKCacheEntry) -> None:
        payload_path, sidecar = self._paths(entry.cache_id)
        nbytes = entry_nbytes(entry)
        torch.save(entry_to_payload(entry), payload_path)
        sidecar.write_text(
            json.dumps(
                {
                    "cache_id": entry.cache_id,
                    "nbytes": nbytes,
                    "num_tokens": len(entry.token_ids),
                    "length": entry.length,
                }
            )
        )
        self._index[entry.cache_id] = (payload_path, nbytes, entry.length)

    def remove(self, cache_id: str) -> bool:
        record = self._index.pop(cache_id, None)
        if record is None:
            return False
        payload_path, _, _ = record
        payload_path.unlink(missing_ok=True)
        payload_path.with_suffix(".json").unlink(missing_ok=True)
        return True

    def keys(self) -> Iterable[str]:
        return tuple(self._index.keys())

    def size_bytes(self) -> int:
        return sum(nbytes for _, nbytes, _length in self._index.values())
