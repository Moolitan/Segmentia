#!/usr/bin/env python3
"""Initialize and finalize the direct CSKCache raw Skill KV pool.

This helper is orchestration only.  vLLM/LMCache write KV bytes directly into
the shared raw-block container; CSKCache verifies the resulting 40-layer
objects and atomically publishes the authoritative Catalog.  No per-layer
``.pt`` source object is created.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
import uuid


def repository_root() -> Path:
    for candidate in Path(__file__).resolve().parents:
        if all(
            (candidate / component).is_dir()
            for component in ("CSKCache/cskcache", "LMCache/lmcache", "vllm/vllm")
        ):
            return candidate
    raise RuntimeError("cannot locate CSKCache, LMCache, and vLLM checkout root")


ROOT = repository_root()
for package_root in (ROOT / "CSKCache", ROOT / "LMCache"):
    sys.path.insert(0, str(package_root))

from cskcache import (  # noqa: E402
    ChunkingSpec,
    ContainerMetadata,
    DirectRawCacheBuilder,
    DirectRawCacheObjectBuildInput,
    DirectRawLayerBuildInput,
    KVLayout,
    MetadataManager,
    RawOffsetNotFoundError,
    generation_sidecar_path,
    publish_generation_sidecar,
)
from lmcache.v1.storage_backend.raw_block import (  # noqa: E402
    RawBlockCore,
    RawBlockCoreConfig,
)


POOL_ROOT = Path(
    os.environ.get(
        "SKILL_SAVE_POOL_ROOT", "/mnt/990_pro/skill_save_pool"
    )
).resolve()
MODEL_DIR = os.environ.get("SKILL_POOL_MODEL_DIR_NAME", "Qwen3-14B")
RAW_ROOT = POOL_ROOT / MODEL_DIR / "raw"
RAW_FILE = RAW_ROOT / "skill_kv.bin"
CATALOG = RAW_ROOT / "catalog.json"
PENDING_DIR = RAW_ROOT / ".pending"
BUILD_RECEIPT = RAW_ROOT / "build.json"
COMPLETED = RAW_ROOT / "COMPLETED"

EXPECTED_LAYERS = int(os.environ.get("CSKCACHE_MODEL_NUM_LAYERS", "40"))
CAPACITY_BYTES = int(os.environ.get("CSKCACHE_RAW_CAPACITY_BYTES", str(512 * 1024**3)))
ALIGNMENT_BYTES = 4096
HEADER_BYTES = 4096
SLOT_BYTES = int(os.environ.get("CSKCACHE_RAW_SLOT_BYTES", str(128 * 1024**2)))
METADATA_BYTES = int(
    os.environ.get("CSKCACHE_RAW_METADATA_BYTES", str(64 * 1024**2))
)
CONTAINER_ID = os.environ.get("CSKCACHE_RAW_CONTAINER_ID", "qwen3-14b-skill-kv")
RETAIN_SKILL_VERSIONS = os.environ.get(
    "CSKCACHE_RETAIN_SKILL_VERSIONS", "0"
) == "1"
CONTAINER_FORMAT_VERSION = 1
META_MAGIC = "CSKRAW01"
IO_ENGINE = "io_uring"
IO_QUEUE_DEPTH = 64


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("initialize", "lmcache-config", "finalize"))
    parser.add_argument(
        "--quarantine-unrecoverable",
        action="store_true",
        help=(
            "Move unpublished pending records aside when LMCache has no "
            "recoverable raw index. Intended only for an explicit rebuild."
        ),
    )
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        temporary.unlink(missing_ok=True)


def _container_from_sidecar() -> ContainerMetadata:
    provisional = ContainerMetadata(
        container_id=CONTAINER_ID,
        raw_file_path=str(RAW_FILE),
        container_format_version=CONTAINER_FORMAT_VERSION,
        storage_generation="pending",
        capacity_bytes=CAPACITY_BYTES,
        alignment_bytes=ALIGNMENT_BYTES,
        header_bytes=HEADER_BYTES,
    )
    sidecar = generation_sidecar_path(provisional)
    if not sidecar.is_file():
        raise RuntimeError(f"CSKCache generation sidecar is missing: {sidecar}")
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    expected_static = {
        "container_id": CONTAINER_ID,
        "raw_file_path": str(RAW_FILE),
        "container_format_version": CONTAINER_FORMAT_VERSION,
        "capacity_bytes": CAPACITY_BYTES,
    }
    for field, expected in expected_static.items():
        if payload.get(field) != expected:
            raise ValueError(
                f"generation sidecar {field}={payload.get(field)!r}, expected {expected!r}"
            )
    generation = payload.get("storage_generation")
    if not isinstance(generation, str) or not generation:
        raise ValueError("generation sidecar has no storage_generation")
    return ContainerMetadata(
        container_id=CONTAINER_ID,
        raw_file_path=str(RAW_FILE),
        container_format_version=CONTAINER_FORMAT_VERSION,
        storage_generation=generation,
        capacity_bytes=CAPACITY_BYTES,
        alignment_bytes=ALIGNMENT_BYTES,
        header_bytes=HEADER_BYTES,
    )


def initialize() -> None:
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    if RAW_FILE.exists():
        container = _container_from_sidecar()
        if RAW_FILE.stat().st_size != CAPACITY_BYTES:
            raise ValueError("existing raw file has an incompatible capacity")
        if CATALOG.exists():
            manager = MetadataManager(CATALOG, expected_layers=EXPECTED_LAYERS)
            if manager.list_containers() != (container,):
                raise ValueError("existing Catalog describes a different raw container")
    else:
        descriptor = os.open(RAW_FILE, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            # A sparse regular file exposes the full logical raw-block address
            # space without reserving hundreds of GiB up front.  O_DIRECT
            # writes allocate physical extents only for metadata and live KV
            # slots.
            os.ftruncate(descriptor, CAPACITY_BYTES)
            os.fsync(descriptor)
        except BaseException:
            os.close(descriptor)
            RAW_FILE.unlink(missing_ok=True)
            raise
        else:
            os.close(descriptor)
        container = ContainerMetadata(
            container_id=CONTAINER_ID,
            raw_file_path=str(RAW_FILE),
            container_format_version=CONTAINER_FORMAT_VERSION,
            storage_generation=uuid.uuid4().hex,
            capacity_bytes=CAPACITY_BYTES,
            alignment_bytes=ALIGNMENT_BYTES,
            header_bytes=HEADER_BYTES,
        )
        publish_generation_sidecar(container)

    PENDING_DIR.mkdir(exist_ok=True)
    print(f"[raw] ready file={RAW_FILE} catalog={CATALOG}")


def lmcache_extra_config() -> dict[str, object]:
    return {
        "exact_save_kv_2td": True,
        "storage_plugin.raw_block.module_path": (
            "lmcache.v1.storage_backend.plugins.rust_raw_block_backend"
        ),
        "storage_plugin.raw_block.class_name": "RustRawBlockBackend",
        "rust_raw_block.device_path": str(RAW_FILE),
        "rust_raw_block.capacity_bytes": CAPACITY_BYTES,
        "rust_raw_block.block_align": ALIGNMENT_BYTES,
        "rust_raw_block.header_bytes": HEADER_BYTES,
        "rust_raw_block.slot_bytes": SLOT_BYTES,
        "rust_raw_block.use_odirect": True,
        "rust_raw_block.enable_zero_copy": True,
        "rust_raw_block.meta_total_bytes": METADATA_BYTES,
        "rust_raw_block.meta_magic": META_MAGIC,
        "rust_raw_block.meta_version": CONTAINER_FORMAT_VERSION,
        "rust_raw_block.meta_enable_periodic": False,
        "rust_raw_block.load_checkpoint_on_init": True,
        "rust_raw_block.io_engine": IO_ENGINE,
        "rust_raw_block.iouring_queue_depth": IO_QUEUE_DEPTH,
    }


def raw_core() -> RawBlockCore:
    return RawBlockCore(
        RawBlockCoreConfig(
            device_path=str(RAW_FILE),
            capacity_bytes=CAPACITY_BYTES,
            block_align=ALIGNMENT_BYTES,
            header_bytes=HEADER_BYTES,
            slot_bytes=SLOT_BYTES,
            use_odirect=True,
            enable_zero_copy=True,
            meta_total_bytes=METADATA_BYTES,
            meta_magic=META_MAGIC.encode("ascii"),
            meta_version=CONTAINER_FORMAT_VERSION,
            meta_checkpoint_interval_sec=3600,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            meta_verify_on_load=True,
            load_checkpoint_on_init=True,
            io_engine=IO_ENGINE,
            iouring_queue_depth=IO_QUEUE_DEPTH,
        ),
        key_namespace="legacy",
    )


def _load_pending(path: Path) -> DirectRawCacheObjectBuildInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "cskcache_direct_raw_pending":
        raise ValueError(f"unsupported pending build record: {path}")
    layers = tuple(
        DirectRawLayerBuildInput(
            layer_id=int(layer["layer_id"]),
            backend_key=str(layer["backend_key"]),
            lookup_key=str(layer["lookup_key"]),
            length_bytes=int(layer["length_bytes"]),
            dtype=str(layer["dtype"]),
            shape=tuple(int(dim) for dim in layer["shape"]),
            memory_layout=str(layer["memory_layout"]),
        )
        for layer in payload["layers"]
    )
    return DirectRawCacheObjectBuildInput(
        object_id=str(payload["object_id"]),
        skill_name=str(payload["skill_name"]),
        skill_version=str(payload["skill_version"]),
        model_fingerprint=str(payload["model_fingerprint"]),
        tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
        token_count=int(payload["token_count"]),
        source_position_start=int(payload["source_position_start"]),
        token_ids_sha256=str(payload["token_ids_sha256"]),
        chunk_token_ids_sha256=tuple(
            str(digest)
            for digest in payload.get("chunk_token_ids_sha256", ())
        ),
        start_marker_token_ids=tuple(
            int(token) for token in payload["start_marker_token_ids"]
        ),
        layers=layers,
        chunking=ChunkingSpec.from_dict(payload["chunking"]),
        storage_layout=KVLayout(str(payload["storage_layout"])),
    )


def _quarantine_pending(paths: list[Path], reason: str) -> Path:
    """Preserve an unpublished failed transaction outside the active queue."""

    failed_root = RAW_ROOT / ".failed"
    transaction_dir = failed_root / f"{time.time_ns()}-{uuid.uuid4().hex[:8]}"
    transaction_dir.mkdir(parents=True, exist_ok=False)
    moved_names: list[str] = []
    for path in paths:
        destination = transaction_dir / path.name
        os.replace(path, destination)
        moved_names.append(path.name)
    atomic_write_json(
        transaction_dir / "failure.json",
        {
            "artifact_type": "cskcache_direct_raw_failed_transaction",
            "failed_at_unix_ns": time.time_ns(),
            "reason": reason,
            "pending_records": moved_names,
        },
    )
    return transaction_dir


def finalize(*, quarantine_unrecoverable: bool = False) -> None:
    pending_paths = sorted(PENDING_DIR.glob("*.json"))
    if not pending_paths:
        print("[raw] no new exact-save objects to publish")
        return
    container = _container_from_sidecar()
    sources = tuple(_load_pending(path) for path in pending_paths)
    core = raw_core()
    try:
        builder = DirectRawCacheBuilder(
            core,
            container,
            expected_layers=EXPECTED_LAYERS,
        )
        try:
            built = builder.publish_objects(
                CATALOG,
                sources,
                retain_skill_versions=RETAIN_SKILL_VERSIONS,
            )
        except RawOffsetNotFoundError as exc:
            if not quarantine_unrecoverable:
                raise
            transaction_dir = _quarantine_pending(pending_paths, str(exc))
            print(
                "[raw] quarantined unrecoverable unpublished transaction "
                f"at {transaction_dir}: {exc}"
            )
            return
    finally:
        core.close()

    atomic_write_json(
        BUILD_RECEIPT,
        {
            "artifact_type": "cskcache_direct_raw_build",
            "status": "completed",
            "completed_at_unix_ns": time.time_ns(),
            "raw_file": str(RAW_FILE),
            "catalog": str(CATALOG),
            "storage_generation": container.storage_generation,
            "objects": [
                {
                    "object_id": item.object_id,
                    "skill_name": item.skill_name,
                    "token_count": item.token_count,
                    "layer_count": len(item.layers),
                }
                for item in built
            ],
        },
    )
    COMPLETED.write_text("completed\n", encoding="utf-8")
    for path in pending_paths:
        path.unlink()
    print(
        f"[raw] published objects={len(built)} file={RAW_FILE} catalog={CATALOG}"
    )


def main() -> None:
    args = parse_args()
    if args.command == "initialize":
        if args.quarantine_unrecoverable:
            raise SystemExit(
                "--quarantine-unrecoverable is only valid with finalize"
            )
        initialize()
    elif args.command == "lmcache-config":
        if args.quarantine_unrecoverable:
            raise SystemExit(
                "--quarantine-unrecoverable is only valid with finalize"
            )
        print(json.dumps(lmcache_extra_config(), separators=(",", ":")))
    else:
        finalize(quarantine_unrecoverable=args.quarantine_unrecoverable)


if __name__ == "__main__":
    main()
