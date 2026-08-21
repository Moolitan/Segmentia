from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from cskcache import (
    BindingState,
    CacheObjectMetadata,
    CacheObjectStatus,
    ContainerMetadata,
    HostLoadState,
    LayerExtent,
    MetadataManager,
    ReadStrategy,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def make_container() -> ContainerMetadata:
    return ContainerMetadata(
        container_id="qwen3-14b-raw-v1",
        raw_file_path="/mnt/990_pro/cskcache-qwen3-14b.raw",
        container_format_version=1,
        storage_generation="generation-1",
        capacity_bytes=1 << 20,
        alignment_bytes=4096,
        header_bytes=4096,
    )


def make_object(
    object_id: str = "internal-comms:v1:qwen3-14b",
    *,
    skill_name: str = "internal-comms",
    skill_version: str = "v1",
    layer_count: int = 4,
    gap_bytes: int = 0,
) -> CacheObjectMetadata:
    length = 4096
    stride = length + gap_bytes
    layers = tuple(
        LayerExtent(
            layer_id=layer,
            backend_key=f"layer-key-{layer}",
            offset_bytes=layer * stride,
            length_bytes=length,
            dtype="bfloat16",
            shape=(2, 128, 64),
            memory_layout="KV_2TD",
            payload_sha256=SHA_A,
        )
        for layer in range(layer_count)
    )
    strategy = ReadStrategy.CONTIGUOUS if gap_bytes == 0 else ReadStrategy.BATCHED
    return CacheObjectMetadata(
        object_id=object_id,
        skill_name=skill_name,
        skill_version=skill_version,
        model_fingerprint="qwen3-14b@revision",
        tokenizer_fingerprint="qwen3-tokenizer@revision",
        token_count=128,
        source_position_start=0,
        token_ids_sha256=SHA_B,
        start_marker_token_ids=(10, 20, 30),
        container_id=make_container().container_id,
        read_strategy=strategy,
        layers=layers,
    )


@pytest.fixture
def manager(tmp_path: Path) -> MetadataManager:
    instance = MetadataManager(tmp_path / "metadata.json", expected_layers=4)
    instance.publish_container(make_container())
    instance.publish_object(make_object())
    return instance


def test_publish_round_trip_and_resolve(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    original = make_object(gap_bytes=4096)
    first = MetadataManager(path, expected_layers=4)
    first.publish_container(make_container())
    first.publish_object(original)

    reloaded = MetadataManager(path, expected_layers=4)
    assert reloaded.get_container(make_container().container_id) == make_container()
    assert reloaded.get_object(original.object_id) == original
    assert (
        reloaded.resolve_object(
            skill_name="internal-comms",
            skill_version="v1",
            model_fingerprint=original.model_fingerprint,
            tokenizer_fingerprint=original.tokenizer_fingerprint,
        )
        == original
    )


def test_publish_rejects_missing_layer_and_bad_strategy(tmp_path: Path) -> None:
    manager = MetadataManager(tmp_path / "metadata.json", expected_layers=4)
    manager.publish_container(make_container())
    missing = replace(make_object(), layers=make_object().layers[:-1])
    with pytest.raises(ValueError, match="expected 4"):
        manager.publish_object(missing)

    bad_strategy = replace(make_object(), read_strategy=ReadStrategy.BATCHED)
    with pytest.raises(ValueError, match="read_strategy"):
        manager.publish_object(bad_strategy)

    bad_position = replace(make_object(), source_position_start=-1)
    with pytest.raises(ValueError, match="source_position_start"):
        manager.publish_object(bad_position)


def test_container_is_required_and_immutable(tmp_path: Path) -> None:
    manager = MetadataManager(tmp_path / "metadata.json", expected_layers=4)
    with pytest.raises(KeyError, match="unknown raw container"):
        manager.publish_object(make_object())
    manager.publish_container(make_container())
    with pytest.raises(ValueError, match="container_id already exists"):
        manager.publish_container(make_container())


def test_publish_rejects_duplicate_identity_and_object_id(
    manager: MetadataManager,
) -> None:
    with pytest.raises(ValueError, match="object_id already exists"):
        manager.publish_object(make_object())
    same_identity = replace(make_object(), object_id="different-object")
    with pytest.raises(ValueError, match="active object already owns identity"):
        manager.publish_object(same_identity)


def test_invalidation_is_persistent_and_allows_new_version(
    tmp_path: Path,
) -> None:
    path = tmp_path / "metadata.json"
    manager = MetadataManager(path, expected_layers=4)
    manager.publish_container(make_container())
    old = make_object()
    manager.publish_object(old)
    invalidated = manager.invalidate_object(old.object_id)
    assert invalidated.status is CacheObjectStatus.INVALIDATED
    with pytest.raises(KeyError):
        manager.resolve_object(
            skill_name=old.skill_name,
            skill_version=old.skill_version,
            model_fingerprint=old.model_fingerprint,
            tokenizer_fingerprint=old.tokenizer_fingerprint,
        )

    new = replace(old, object_id="internal-comms:v2", skill_version="v2")
    manager.publish_object(new)
    reloaded = MetadataManager(path, expected_layers=4)
    assert len(reloaded.list_objects(include_invalidated=True)) == 2
    assert reloaded.resolve_object(
        skill_name=new.skill_name,
        skill_version="v2",
        model_fingerprint=new.model_fingerprint,
        tokenizer_fingerprint=new.tokenizer_fingerprint,
    ) == new


def test_request_can_bind_before_host_load_completes(
    manager: MetadataManager,
) -> None:
    state = manager.create_ticket("call-1", make_object().object_id, now_ns=100)
    assert state.host_load_state is HostLoadState.NOT_STARTED
    manager.mark_observation_verified("call-1")
    state = manager.bind_request(
        "call-1",
        request_id="request-1",
        verified_cache_object_id=make_object().object_id,
        segment_start=100,
        segment_end=164,
    )
    assert state.binding_state is BindingState.VERIFIED
    assert state.host_load_state is HostLoadState.NOT_STARTED
    with pytest.raises(ValueError, match="host data"):
        manager.activate("call-1")

    manager.start_host_load("call-1", io_operation_id="io-1")
    manager.mark_host_ready("call-1")
    assert manager.activate("call-1").binding_state is BindingState.ACTIVE


def test_host_can_be_ready_before_request_binding(manager: MetadataManager) -> None:
    manager.create_ticket("call-2", make_object().object_id)
    manager.start_host_load("call-2", io_operation_id="io-2")
    ready = manager.mark_host_ready("call-2")
    assert ready.binding_state is BindingState.UNBOUND
    manager.mark_observation_verified("call-2")
    manager.bind_request(
        "call-2",
        request_id="request-2",
        verified_cache_object_id=make_object().object_id,
        segment_start=100,
        segment_end=164,
    )
    assert manager.activate("call-2").binding_state is BindingState.ACTIVE


def test_binding_rejects_wrong_object_and_duplicate_request(
    manager: MetadataManager,
) -> None:
    manager.create_ticket("call-3", make_object().object_id)
    manager.mark_observation_verified("call-3")
    with pytest.raises(ValueError, match="does not match"):
        manager.bind_request(
            "call-3",
            request_id="request-3",
            verified_cache_object_id="wrong",
            segment_start=100,
            segment_end=164,
        )
    manager.bind_request(
        "call-3",
        request_id="request-3",
        verified_cache_object_id=make_object().object_id,
        segment_start=100,
        segment_end=164,
    )
    manager.create_ticket("call-4", make_object().object_id)
    manager.mark_observation_verified("call-4")
    with pytest.raises(ValueError, match="already bound"):
        manager.bind_request(
            "call-4",
            request_id="request-3",
            verified_cache_object_id=make_object().object_id,
            segment_start=100,
            segment_end=164,
        )


def test_host_failure_forces_fallback(manager: MetadataManager) -> None:
    manager.create_ticket("call-5", make_object().object_id)
    manager.start_host_load("call-5", io_operation_id="io-5")
    failed = manager.mark_host_failed("call-5", "short_read")
    assert failed.host_load_state is HostLoadState.FAILED
    assert failed.binding_state is BindingState.FALLBACK
    assert failed.fallback_reason == "short_read"


def test_layer_progress_is_consecutive_and_load_precedes_correction(
    manager: MetadataManager,
) -> None:
    manager.create_ticket("call-6", make_object().object_id)
    manager.start_host_load("call-6", io_operation_id="io-6")
    manager.mark_host_ready("call-6")
    manager.mark_observation_verified("call-6")
    manager.bind_request(
        "call-6",
        request_id="request-6",
        verified_cache_object_id=make_object().object_id,
        segment_start=100,
        segment_end=164,
    )
    manager.activate("call-6")

    with pytest.raises(ValueError, match="consecutive"):
        manager.mark_layer_loaded("call-6", 1)
    with pytest.raises(ValueError, match="loaded before"):
        manager.mark_layer_corrected("call-6", 0)
    manager.mark_layer_loaded("call-6", 0)
    manager.mark_layer_corrected("call-6", 0)
    assert manager.get_runtime("call-6").corrected_through_layer == 0


def test_deadline_expiration_and_release_remove_request_index(
    manager: MetadataManager,
) -> None:
    manager.create_ticket(
        "call-7", make_object().object_id, now_ns=100, deadline_ns=200
    )
    manager.mark_observation_verified("call-7")
    manager.bind_request(
        "call-7",
        request_id="request-7",
        verified_cache_object_id=make_object().object_id,
        segment_start=100,
        segment_end=164,
    )
    expired = manager.expire(now_ns=200)
    assert len(expired) == 1
    assert expired[0].binding_state is BindingState.FALLBACK
    released = manager.release("call-7")
    assert released.binding_state is BindingState.RELEASED
    with pytest.raises(KeyError, match="unknown request_id"):
        manager.get_runtime_for_request("request-7")


def test_runtime_state_is_not_persisted(tmp_path: Path) -> None:
    path = tmp_path / "metadata.json"
    first = MetadataManager(path, expected_layers=4)
    first.publish_container(make_container())
    item = make_object()
    first.publish_object(item)
    first.create_ticket("ephemeral", item.object_id)

    second = MetadataManager(path, expected_layers=4)
    assert second.get_object(item.object_id) == item
    with pytest.raises(KeyError, match="unknown ticket"):
        second.get_runtime("ephemeral")


def test_legacy_manifest_is_rejected_instead_of_silently_misread(
    tmp_path: Path,
) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        '{"schema_version": 1, "status": "completed", "skills": {}}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected top-level"):
        MetadataManager(path, expected_layers=4)
