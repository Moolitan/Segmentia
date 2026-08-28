from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import pytest

from cskcache import (
    CacheBuilder,
    CacheObjectBuildInput,
    ContainerMetadata,
    DirectRawCacheBuilder,
    DirectRawCacheObjectBuildInput,
    DirectRawLayerBuildInput,
    LayerBuildInput,
    LocalDiskCacheBuilder,
    LocalDiskCacheObjectBuildInput,
    LocalDiskLayerBuildInput,
    MetadataManager,
    RawOffsetNotFoundError,
    publish_cache_snapshot,
)


class OffsetBackend:
    def __init__(self, offsets: dict[str, int], *, header_bytes: int = 4096) -> None:
        self.header_bytes = header_bytes
        self._offsets = offsets
        self.lookups: list[str] = []

    def entry_offset(self, key: str) -> int | None:
        self.lookups.append(key)
        return self._offsets.get(key)


class DirectOffsetBackend(OffsetBackend):
    def __init__(
        self,
        container: ContainerMetadata,
        offsets: dict[str, int],
    ) -> None:
        super().__init__(offsets, header_bytes=container.header_bytes)
        self.device_path = container.raw_file_path
        self.capacity_bytes = container.capacity_bytes
        self.block_align = container.alignment_bytes


def make_container(tmp_path: Path) -> ContainerMetadata:
    raw_path = tmp_path / "skill-kv.raw"
    raw_path.write_bytes(b"\0" * (4096 * 64))
    return ContainerMetadata(
        container_id="qwen3-14b-raw-v1",
        raw_file_path=str(raw_path.resolve()),
        container_format_version=1,
        storage_generation="generation-1",
        capacity_bytes=raw_path.stat().st_size,
        alignment_bytes=4096,
        header_bytes=4096,
    )


def make_source(layer_count: int = 40) -> CacheObjectBuildInput:
    return CacheObjectBuildInput(
        object_id="internal-comms:v1:qwen3-14b",
        skill_name="internal-comms",
        skill_version="v1",
        model_fingerprint="model-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
        token_count=339,
        source_position_start=0,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1, 2, 3),
        layers=tuple(
            LayerBuildInput(
                layer_id=layer,
                backend_key=f"persistent-layer-{layer}",
                lookup_key=f"lookup-layer-{layer}",
                length_bytes=512,
                dtype="bfloat16",
                shape=(2, 4, 32),
                memory_layout="KV_2TD",
                payload_sha256=f"{layer + 1:064x}"[-64:],
            )
            for layer in range(layer_count)
        ),
    )


def make_backend(layer_count: int = 40) -> OffsetBackend:
    return OffsetBackend(
        {
            f"lookup-layer-{layer}": 4096 * (layer + 1)
            for layer in range(layer_count)
        }
    )


def make_direct_source(
    *,
    skill_name: str = "internal-comms",
    object_id: str = "internal-comms:direct",
    layer_count: int = 4,
) -> DirectRawCacheObjectBuildInput:
    return DirectRawCacheObjectBuildInput(
        object_id=object_id,
        skill_name=skill_name,
        skill_version="v1",
        model_fingerprint="model-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
        token_count=4,
        source_position_start=0,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1, 2, 3),
        layers=tuple(
            DirectRawLayerBuildInput(
                layer_id=layer,
                backend_key=f"persistent-layer-{layer}",
                lookup_key=f"lookup-layer-{layer}",
                length_bytes=512,
                dtype="bfloat16",
                shape=(2, 4, 32),
                memory_layout="KV_2TD",
            )
            for layer in range(layer_count)
        ),
    )


def make_direct_builder(
    tmp_path: Path, *, layer_count: int = 4
) -> tuple[DirectRawCacheBuilder, ContainerMetadata, dict[int, bytes]]:
    container = make_container(tmp_path)
    offsets = {
        f"lookup-layer-{layer}": 4096 * (layer + 1)
        for layer in range(layer_count)
    }
    payloads = {
        layer: bytes([layer + 1]) * 512 for layer in range(layer_count)
    }
    with Path(container.raw_file_path).open("r+b") as handle:
        for layer in range(layer_count):
            handle.seek(offsets[f"lookup-layer-{layer}"] + container.header_bytes)
            handle.write(payloads[layer])
    backend = DirectOffsetBackend(container, offsets)
    return (
        DirectRawCacheBuilder(
            backend,
            container,
            expected_layers=layer_count,
        ),
        container,
        payloads,
    )


def test_builder_resolves_all_offsets_only_during_offline_build(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    backend = make_backend()
    metadata = CacheBuilder(
        backend, container, expected_layers=40
    ).build_object(make_source())

    assert backend.lookups == [f"lookup-layer-{layer}" for layer in range(40)]
    assert tuple(layer.layer_id for layer in metadata.layers) == tuple(range(40))
    assert metadata.layers[0].offset_bytes == 8192
    assert metadata.layers[-1].offset_bytes == 4096 * 41
    assert metadata.source_position_start == 0


def test_direct_raw_builder_verifies_payloads_without_pt_intermediate(
    tmp_path: Path,
) -> None:
    builder, _, payloads = make_direct_builder(tmp_path)

    metadata = builder.build_object(make_direct_source())

    assert tuple(layer.layer_id for layer in metadata.layers) == (0, 1, 2, 3)
    assert [layer.payload_sha256 for layer in metadata.layers] == [
        hashlib.sha256(payloads[layer]).hexdigest() for layer in range(4)
    ]


def test_local_disk_builder_publishes_complete_layer_files(tmp_path: Path) -> None:
    layer_paths = []
    for layer_id in range(4):
        path = tmp_path / f"local-layer-{layer_id}.pt"
        path.write_bytes(bytes([layer_id + 1]) * 512)
        layer_paths.append(path)
    source = LocalDiskCacheObjectBuildInput(
        object_id="internal-comms:local",
        skill_name="internal-comms",
        skill_version="v1",
        model_fingerprint="model-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
        token_count=4,
        source_position_start=0,
        token_ids_sha256="a" * 64,
        start_marker_token_ids=(1, 2, 3),
        layers=tuple(
            LocalDiskLayerBuildInput(
                layer_id=layer_id,
                backend_key=f"local-layer-{layer_id}",
                data_path=str(path.resolve()),
                length_bytes=512,
                dtype="bfloat16",
                shape=(2, 4, 32),
                memory_layout="KV_2TD",
            )
            for layer_id, path in enumerate(layer_paths)
        ),
    )
    catalog = tmp_path / "local-catalog.json"

    built = LocalDiskCacheBuilder(expected_layers=4).publish_objects(
        catalog, [source]
    )

    reloaded = MetadataManager(catalog, expected_layers=4)
    assert not reloaded.list_containers()
    assert reloaded.list_objects() == built
    assert all(layer.offset_bytes is None for layer in built[0].layers)

    replacement = replace(
        source,
        object_id="internal-comms:local-v2",
        skill_version="v2",
        token_ids_sha256="b" * 64,
    )
    LocalDiskCacheBuilder(expected_layers=4).publish_objects(
        catalog, [replacement]
    )
    assert [
        item.object_id
        for item in MetadataManager(catalog, expected_layers=4).list_objects()
    ] == ["internal-comms:local-v2"]


def test_direct_raw_builder_reports_missing_checkpoint_offset(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    backend = DirectOffsetBackend(
        container,
        {
            f"lookup-layer-{layer}": 4096 * (layer + 1)
            for layer in range(1, 4)
        },
    )
    builder = DirectRawCacheBuilder(backend, container, expected_layers=4)

    with pytest.raises(
        RawOffsetNotFoundError,
        match="no direct raw offset for layer 0",
    ):
        builder.build_object(make_direct_source())
    assert not list(tmp_path.rglob("*.pt"))


def test_direct_raw_publication_is_all_or_nothing(tmp_path: Path) -> None:
    builder, _, _ = make_direct_builder(tmp_path)
    catalog = tmp_path / "catalog.json"
    incomplete = replace(make_direct_source(), layers=make_direct_source().layers[:-1])

    with pytest.raises(ValueError, match="expected 4 direct raw layers"):
        builder.publish_objects(catalog, [incomplete])

    assert not catalog.exists()


def test_direct_raw_publication_replaces_only_same_skill_identity(
    tmp_path: Path,
) -> None:
    builder, _, _ = make_direct_builder(tmp_path)
    catalog = tmp_path / "catalog.json"
    first = make_direct_source()
    other = make_direct_source(skill_name="docx", object_id="docx:direct")
    builder.publish_objects(catalog, [first, other])

    replacement = replace(
        first,
        object_id="internal-comms:replacement",
        skill_version="v2",
        token_ids_sha256="b" * 64,
    )
    builder.publish_objects(catalog, [replacement])

    manager = MetadataManager(catalog, expected_layers=4)
    assert [item.object_id for item in manager.list_objects()] == [
        "docx:direct",
        "internal-comms:replacement",
    ]


def test_direct_raw_publication_can_retain_skill_body_versions(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    offsets = {
        **{
            f"lookup-layer-{layer}": 4096 * (layer + 1)
            for layer in range(4)
        },
        **{
            f"lookup-v2-layer-{layer}": 4096 * (layer + 9)
            for layer in range(4)
        },
    }
    builder = DirectRawCacheBuilder(
        DirectOffsetBackend(container, offsets),
        container,
        expected_layers=4,
    )
    first = make_direct_source()
    second = replace(
        first,
        object_id="internal-comms:direct-v2",
        skill_version="v2",
        token_ids_sha256="b" * 64,
        layers=tuple(
            replace(
                layer,
                backend_key=f"persistent-v2-layer-{layer.layer_id}",
                lookup_key=f"lookup-v2-layer-{layer.layer_id}",
            )
            for layer in first.layers
        ),
    )
    catalog = tmp_path / "versioned-catalog.json"

    builder.publish_objects(
        catalog,
        [first, second],
        retain_skill_versions=True,
    )

    manager = MetadataManager(catalog, expected_layers=4)
    assert len(manager.list_objects()) == 2
    with pytest.raises(ValueError, match="ambiguous"):
        manager.resolve_object(
            skill_name="internal-comms",
            model_fingerprint="model-fingerprint",
            tokenizer_fingerprint="tokenizer-fingerprint",
        )
    assert manager.resolve_object(
        skill_name="internal-comms",
        skill_version="v2",
        model_fingerprint="model-fingerprint",
        tokenizer_fingerprint="tokenizer-fingerprint",
    ).object_id == "internal-comms:direct-v2"


def test_builder_rejects_missing_raw_layer(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    backend = make_backend()
    del backend._offsets["lookup-layer-17"]
    with pytest.raises(RawOffsetNotFoundError, match="no raw offset for layer 17"):
        CacheBuilder(backend, container, expected_layers=40).build_object(
            make_source()
        )


def test_snapshot_is_complete_atomic_and_idempotent(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    metadata = CacheBuilder(
        make_backend(), container, expected_layers=40
    ).build_object(make_source())
    destination = tmp_path / "catalog.json"

    publish_cache_snapshot(
        destination,
        container,
        [metadata],
        expected_layers=40,
    )
    publish_cache_snapshot(
        destination,
        container,
        [metadata],
        expected_layers=40,
    )

    reloaded = MetadataManager(destination, expected_layers=40)
    assert reloaded.list_containers() == (container,)
    assert reloaded.list_objects() == (metadata,)
    assert not list(tmp_path.glob(".*.tmp"))


def test_snapshot_refuses_to_replace_different_metadata(tmp_path: Path) -> None:
    container = make_container(tmp_path)
    builder = CacheBuilder(make_backend(), container, expected_layers=40)
    metadata = builder.build_object(make_source())
    destination = tmp_path / "catalog.json"
    publish_cache_snapshot(
        destination, container, [metadata], expected_layers=40
    )

    builder = CacheBuilder(make_backend(), container, expected_layers=40)
    different = builder.build_object(
        replace(make_source(), object_id="different-object", skill_version="v2")
    )
    with pytest.raises(ValueError, match="different cache objects"):
        publish_cache_snapshot(
            destination, container, [different], expected_layers=40
        )


def test_rebuilt_container_can_atomically_replace_stale_generation(
    tmp_path: Path,
) -> None:
    container = make_container(tmp_path)
    metadata = CacheBuilder(
        make_backend(), container, expected_layers=40
    ).build_object(make_source())
    destination = tmp_path / "catalog.json"
    publish_cache_snapshot(
        destination, container, [metadata], expected_layers=40
    )

    rebuilt_container = replace(
        container, storage_generation="rebuilt-generation"
    )
    rebuilt_metadata = CacheBuilder(
        make_backend(), rebuilt_container, expected_layers=40
    ).build_object(make_source())
    publish_cache_snapshot(
        destination,
        rebuilt_container,
        [rebuilt_metadata],
        expected_layers=40,
        replace_existing=True,
    )

    reloaded = MetadataManager(destination, expected_layers=40)
    assert reloaded.list_containers() == (rebuilt_container,)
