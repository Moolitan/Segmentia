#!/usr/bin/env python3
"""Initialize and publish one-layer-per-file LocalDisk Skill objects."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys


ROOT = next(
    candidate
    for candidate in Path(__file__).resolve().parents
    if (candidate / "CSKCache/cskcache").is_dir()
)
for package_root in (ROOT / "CSKCache", ROOT / "LMCache"):
    sys.path.insert(0, str(package_root))

from cskcache import (  # noqa: E402
    ChunkingSpec,
    KVLayout,
    LocalDiskCacheBuilder,
    LocalDiskCacheObjectBuildInput,
    LocalDiskLayerBuildInput,
)


POOL_ROOT = Path(
    os.environ.get("SKILL_SAVE_POOL_ROOT", "/mnt/990_pro/skill_save_pool")
).resolve()
MODEL_DIR = os.environ.get("SKILL_POOL_MODEL_DIR_NAME", "Qwen3-14B")
LOCAL_ROOT = POOL_ROOT / MODEL_DIR / "layer_files"
CATALOG = LOCAL_ROOT / "catalog.json"
PENDING_DIR = LOCAL_ROOT / ".pending"
EXPECTED_LAYERS = int(os.environ.get("CSKCACHE_MODEL_NUM_LAYERS", "40"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("initialize", "lmcache-config", "finalize"),
    )
    return parser.parse_args()


def load_pending(path: Path) -> LocalDiskCacheObjectBuildInput:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("artifact_type") != "cskcache_local_disk_pending":
        raise ValueError(f"unsupported LocalDisk pending record: {path}")
    return LocalDiskCacheObjectBuildInput(
        object_id=str(payload["object_id"]),
        skill_name=str(payload["skill_name"]),
        skill_version=str(payload["skill_version"]),
        model_fingerprint=str(payload["model_fingerprint"]),
        tokenizer_fingerprint=str(payload["tokenizer_fingerprint"]),
        token_count=int(payload["token_count"]),
        source_position_start=int(payload["source_position_start"]),
        token_ids_sha256=str(payload["token_ids_sha256"]),
        start_marker_token_ids=tuple(
            int(token) for token in payload["start_marker_token_ids"]
        ),
        layers=tuple(
            LocalDiskLayerBuildInput(
                layer_id=int(layer["layer_id"]),
                backend_key=str(layer["backend_key"]),
                data_path=str(layer["data_path"]),
                length_bytes=int(layer["length_bytes"]),
                dtype=str(layer["dtype"]),
                shape=tuple(int(dim) for dim in layer["shape"]),
                memory_layout=str(layer["memory_layout"]),
            )
            for layer in payload["layers"]
        ),
        chunking=ChunkingSpec.from_dict(payload["chunking"]),
        storage_layout=KVLayout(str(payload["storage_layout"])),
    )


def main() -> None:
    command = parse_args().command
    if command == "initialize":
        LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
        PENDING_DIR.mkdir(exist_ok=True)
        print(f"[local_disk] ready root={LOCAL_ROOT} catalog={CATALOG}")
        return
    if command == "lmcache-config":
        print(
            json.dumps(
                {
                    "exact_save_kv_2td": True,
                },
                separators=(",", ":"),
            )
        )
        return

    pending_paths = sorted(PENDING_DIR.glob("*.json"))
    if not pending_paths:
        print("[local_disk] no new exact-save objects to publish")
        return
    built = LocalDiskCacheBuilder(expected_layers=EXPECTED_LAYERS).publish_objects(
        CATALOG,
        tuple(load_pending(path) for path in pending_paths),
    )
    for path in pending_paths:
        path.unlink()
    print(f"[local_disk] published objects={len(built)} catalog={CATALOG}")


if __name__ == "__main__":
    main()
