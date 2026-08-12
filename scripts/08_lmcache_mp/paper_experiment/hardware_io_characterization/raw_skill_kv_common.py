#!/usr/bin/env python3
"""Shared raw-block layout helpers for the Skill KV loading experiment."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from lmcache.utils import parse_cache_key
from lmcache.v1.memory_management import (
    MemoryFormat,
    MemoryObjMetadata,
    TensorMemoryObj,
)
from lmcache.v1.storage_backend.raw_block import (
    RawBlockCore,
    RawBlockCoreConfig,
    encode_legacy_key,
)

from common import discover_manifest_layers, resolve_skill_manifest


@dataclass(frozen=True)
class RawLayerSource:
    """One validated layer payload and its durable LMCache metadata."""

    layer_id: int
    path: Path
    cache_key: str
    size_bytes: int
    shape: torch.Size
    dtype: torch.dtype
    memory_format: MemoryFormat
    cached_positions: torch.Tensor


@dataclass(frozen=True)
class RawSkillSource:
    """The ordered 40-layer source object for one Agent Skill."""

    task: str
    skill: str
    token_count: int
    layers: tuple[RawLayerSource, ...]

    @property
    def cache_bytes(self) -> int:
        return sum(layer.size_bytes for layer in self.layers)


def align_up(value: int, alignment: int) -> int:
    """Round *value* upward to an integer multiple of *alignment*."""
    return ((value + alignment - 1) // alignment) * alignment


def selected_pairs(config: dict[str, Any]) -> list[tuple[str, str]]:
    """Return the same 11 task/Skill pairs used by the existing benchmark."""
    cases = config["agent_schedule"]["cases"]
    excluded = set(config["agent_kv_loading_actual"]["excluded_skills"])
    pairs = [
        (str(case["task"]), str(case["skill"]))
        for case in cases
        if str(case["skill"]) not in excluded
    ]
    if excluded != {"docx", "writing-systems-papers"}:
        raise ValueError("raw Skill KV experiment must exclude exactly two Skills")
    if len(pairs) != 11 or len(set(pairs)) != 11:
        raise ValueError(f"expected 11 unique task/Skill pairs, found {pairs}")
    return pairs


def _decode_cached_positions(payload: Any, expected_count: int) -> torch.Tensor:
    if not isinstance(payload, dict):
        raise ValueError("cached_positions must be an encoded object")
    kind = payload.get("kind")
    if kind == "range":
        start = int(payload["start"])
        length = int(payload["length"])
        positions = torch.arange(start, start + length, dtype=torch.long)
    elif kind == "list":
        values = payload.get("values")
        if not isinstance(values, list):
            raise ValueError("cached_positions list is malformed")
        positions = torch.tensor(values, dtype=torch.long)
    else:
        raise ValueError(f"unsupported cached_positions kind: {kind!r}")
    if positions.numel() != expected_count:
        raise ValueError(
            "cached position count "
            f"{positions.numel()} disagrees with token dimension {expected_count}"
        )
    return positions


def discover_raw_sources(config: dict[str, Any]) -> list[RawSkillSource]:
    """Read and validate every selected Skill's layer sidecars."""
    pool_dir = Path(config["fast_ssd_skill_cache"]["pool_dir"]).resolve()
    expected_layers = int(config["skill_cache"]["expected_layers"])
    sources: list[RawSkillSource] = []
    for task, skill in selected_pairs(config):
        manifest_path = resolve_skill_manifest(pool_dir, skill)
        manifest, layer_files = discover_manifest_layers(manifest_path, expected_layers)
        layers: list[RawLayerSource] = []
        for layer_file in layer_files:
            sidecar_path = Path(f"{layer_file.path}.meta.json")
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            cache_key = str(sidecar["cache_key"])
            parsed_key = parse_cache_key(cache_key)
            if getattr(parsed_key, "layer_id", None) != layer_file.layer_id:
                raise ValueError(f"layer key disagrees with filename: {sidecar_path}")
            dtype_name = str(sidecar["dtype"]).removeprefix("torch.")
            dtype = getattr(torch, dtype_name, None)
            if not isinstance(dtype, torch.dtype):
                raise ValueError(f"unsupported dtype {dtype_name}: {sidecar_path}")
            shape = torch.Size(sidecar["shape"])
            logical_bytes = math.prod(shape) * dtype.itemsize
            if logical_bytes != layer_file.size_bytes:
                raise ValueError(
                    f"shape/dtype size disagrees with payload: {sidecar_path}"
                )
            memory_format = MemoryFormat[str(sidecar["memory_format"])]
            # Segmentia persists one layer as [2, tokens, hidden_dim].  LMCache's
            # generic KV_2TD token_dim() currently returns 0, while the local-disk
            # rehydration path and the offline prefill validator both explicitly
            # treat dimension 1 as the token axis for this layout.
            token_dim = (
                1 if memory_format == MemoryFormat.KV_2TD else memory_format.token_dim()
            )
            token_count = shape[token_dim]
            layers.append(
                RawLayerSource(
                    layer_id=layer_file.layer_id,
                    path=layer_file.path,
                    cache_key=cache_key,
                    size_bytes=layer_file.size_bytes,
                    shape=shape,
                    dtype=dtype,
                    memory_format=memory_format,
                    cached_positions=_decode_cached_positions(
                        sidecar["cached_positions"], token_count
                    ),
                )
            )
        token_count = int(manifest["token_count"])
        sources.append(RawSkillSource(task, skill, token_count, tuple(layers)))
    return sources


def build_layout(
    config: dict[str, Any], sources: list[RawSkillSource]
) -> dict[str, Any]:
    """Build the fixed-slot layout shared by preparation and measurement."""
    settings = config["raw_skill_kv"]
    alignment = int(settings["block_alignment_bytes"])
    header_bytes = int(settings["header_bytes"])
    maximum_layer_bytes = max(
        layer.size_bytes for source in sources for layer in source.layers
    )
    slot_bytes = align_up(header_bytes + maximum_layer_bytes, alignment)
    expected_layers = int(config["skill_cache"]["expected_layers"])
    capacity_bytes = int(float(settings["capacity_gib"]) * 1024**3)
    metadata_bytes = int(float(settings["metadata_mib"]) * 1024**2)
    required_bytes = metadata_bytes + slot_bytes * expected_layers * len(sources)
    if required_bytes > capacity_bytes:
        raise ValueError(
            "raw-block file is too small: "
            f"required={required_bytes}, configured={capacity_bytes}"
        )
    return {
        "schema_version": 1,
        "raw_file": str(Path(settings["file"]).resolve()),
        "capacity_bytes": capacity_bytes,
        "block_alignment_bytes": alignment,
        "header_bytes": header_bytes,
        "metadata_bytes": metadata_bytes,
        "required_bytes": required_bytes,
        "slot_bytes": slot_bytes,
        "maximum_layer_bytes": maximum_layer_bytes,
        "io_engine": str(settings["io_engine"]),
        "queue_depth": int(settings["queue_depth"]),
        "expected_layers": expected_layers,
        "expected_skills": len(sources),
    }


def open_core(layout: dict[str, Any]) -> RawBlockCore:
    """Open LMCache RawBlockCore with the experiment's durable layout."""
    return RawBlockCore(
        RawBlockCoreConfig(
            device_path=str(layout["raw_file"]),
            capacity_bytes=int(layout["capacity_bytes"]),
            block_align=int(layout["block_alignment_bytes"]),
            header_bytes=int(layout["header_bytes"]),
            slot_bytes=int(layout["slot_bytes"]),
            use_odirect=True,
            enable_zero_copy=True,
            meta_total_bytes=int(layout["metadata_bytes"]),
            meta_magic=b"CSKRAW01",
            meta_version=1,
            meta_checkpoint_interval_sec=3600,
            meta_idle_quiet_ms=0,
            meta_enable_periodic=False,
            meta_verify_on_load=True,
            io_engine=str(layout["io_engine"]),
            iouring_queue_depth=int(layout["queue_depth"]),
        ),
        key_namespace="legacy",
    )


def key_spec(layer: RawLayerSource):
    """Return the raw-block key specification for one source layer."""
    return encode_legacy_key(parse_cache_key(layer.cache_key))


def memory_object(raw_data: torch.Tensor, layer: RawLayerSource) -> TensorMemoryObj:
    """Wrap a byte tensor with the original layer metadata."""
    metadata = MemoryObjMetadata(
        shape=layer.shape,
        dtype=layer.dtype,
        address=0,
        phy_size=raw_data.numel(),
        ref_count=1,
        fmt=layer.memory_format,
        cached_positions=layer.cached_positions,
    )
    return TensorMemoryObj(raw_data, metadata, parent_allocator=None)


def read_payload(layer: RawLayerSource) -> torch.Tensor:
    """Read one raw layer file into a temporary CPU byte tensor."""
    tensor = torch.empty(layer.size_bytes, dtype=torch.uint8)
    destination = memoryview(tensor.numpy()).cast("B")
    offset = 0
    with layer.path.open("rb", buffering=0) as handle:
        while offset < layer.size_bytes:
            count = handle.readinto(destination[offset:])
            if not count:
                break
            offset += count
    if offset != layer.size_bytes:
        raise IOError(f"short read for {layer.path}: {offset}/{layer.size_bytes}")
    return tensor


def sha256_bytes(buffer: Any) -> str:
    """Hash an arbitrary byte-addressable buffer."""
    view = memoryview(buffer)
    if view.format != "B" or view.itemsize != 1:
        view = view.cast("B")
    return hashlib.sha256(view).hexdigest()
