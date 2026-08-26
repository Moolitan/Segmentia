from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Sequence
import threading
import time

from cskcache import (
    BindingState,
    CacheObjectMetadata,
    ChunkingSpec,
    CorrectionStrategy,
    ContainerMetadata,
    HostLoadState,
    LayerExtent,
    MetadataManager,
    ReadStrategy,
    ReusePolicy,
    ReuseReadiness,
    RequestManager,
    SkillMatchMode,
    StorageManager,
    fingerprint_full_token_chunks,
    publish_generation_sidecar,
)


SKILL_TOKENS = (10, 11, 12, 13)


def token_sha256(token_ids: Sequence[int]) -> str:
    digest = hashlib.sha256()
    for token_id in token_ids:
        digest.update(token_id.to_bytes(4, "little", signed=False))
    return digest.hexdigest()


@dataclass
class Destination:
    byte_array: bytearray


class BlockingBackend:
    def __init__(self, container: ContainerMetadata) -> None:
        self.device_path = container.raw_file_path
        self.capacity_bytes = container.capacity_bytes
        self.block_align = container.alignment_bytes
        self.header_bytes = container.header_bytes
        self.started = threading.Event()
        self.complete = threading.Event()
        self.read_calls = 0

    def read_extents_into(
        self,
        offsets: Sequence[int],
        lengths: Sequence[int],
        objs: Sequence[Any],
    ) -> list[bool]:
        self.read_calls += 1
        self.started.set()
        if not self.complete.wait(timeout=5):
            raise TimeoutError("test did not complete the storage read")
        return [True] * len(offsets)


class RecordingPool:
    def __init__(self) -> None:
        self.acquire_calls = 0
        self.release_calls = 0

    def acquire(self, extents: Sequence[LayerExtent]) -> Sequence[Destination]:
        self.acquire_calls += 1
        return tuple(Destination(bytearray(item.length_bytes)) for item in extents)

    def acquire_persistent(
        self, extents: Sequence[LayerExtent]
    ) -> Sequence[Destination]:
        return self.acquire(extents)

    def arrange_loaded_layers(
        self,
        _extents: Sequence[LayerExtent],
        persistent_layer_regions: Sequence[Any],
    ) -> Sequence[Any]:
        return tuple(persistent_layer_regions)

    def release(self, memory_objects: Sequence[Any]) -> None:
        assert memory_objects
        self.release_calls += 1


def build_runtime(
    tmp_path: Path,
    *,
    ttl_seconds: float = 60.0,
    skill_tokens: Sequence[int] = SKILL_TOKENS,
    chunk_size_tokens: int = 256,
):
    raw_path = tmp_path / "skill.raw"
    raw_path.write_bytes(b"\0" * 16384)
    container = ContainerMetadata(
        container_id="container-v1",
        raw_file_path=str(raw_path.resolve()),
        container_format_version=1,
        storage_generation="generation-v1",
        capacity_bytes=16384,
        alignment_bytes=4096,
        header_bytes=4096,
    )
    publish_generation_sidecar(container)
    extents = tuple(
        LayerExtent(
            layer_id=layer,
            backend_key=f"key-{layer}",
            offset_bytes=4096 * (layer + 1),
            length_bytes=512,
            dtype="bfloat16",
            shape=(2, 2, 64),
            memory_layout="KV_2TD",
            payload_sha256=f"{layer + 1:064x}",
        )
        for layer in range(2)
    )
    cache_object = CacheObjectMetadata(
        object_id="internal-comms:v1:model-a",
        skill_name="internal-comms",
        skill_version="v1",
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        token_count=len(skill_tokens),
        source_position_start=0,
        token_ids_sha256=token_sha256(skill_tokens),
        chunk_token_ids_sha256=fingerprint_full_token_chunks(
            skill_tokens, chunk_size_tokens
        ),
        start_marker_token_ids=tuple(skill_tokens[:2]),
        container_id=container.container_id,
        read_strategy=ReadStrategy.BATCHED,
        layers=extents,
        chunking=ChunkingSpec(chunk_size_tokens),
    )
    metadata = MetadataManager(tmp_path / "metadata.json", expected_layers=2)
    metadata.publish_container(container)
    metadata.publish_object(cache_object)
    backend = BlockingBackend(container)
    pool = RecordingPool()
    storage = StorageManager(metadata, backend, host_buffer_pool=pool)
    requests = RequestManager(
        metadata,
        storage,
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        ticket_ttl_seconds=ttl_seconds,
    )
    return metadata, backend, pool, storage, requests


def bind_verified_request(
    requests: RequestManager,
    *,
    skill_tokens: Sequence[int],
    prompt_prefix: Sequence[int] = (),
    ticket: str = "call-1",
    request_id: str = "request-1",
) -> None:
    content = (
        '<context_segment skill_name="internal-comms">\n'
        "body\n</context_segment>\n"
    )
    assert requests.inspect_tool_observation(ticket, "skill", content)
    prompt = [*prompt_prefix, *skill_tokens, 9999]
    assert requests.authenticate_and_bind(ticket, request_id, prompt)


def test_select_is_nonblocking_and_duplicate_is_idempotent(tmp_path: Path) -> None:
    metadata, backend, pool, _storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.read_calls == 1
        state = metadata.get_runtime("call-1")
        assert state.host_load_state is HostLoadState.LOADING
        backend.complete.set()
        deadline = time.monotonic() + 5
        while requests.poll("call-1").host_load_state is not HostLoadState.READY:
            assert time.monotonic() < deadline
            time.sleep(0.005)
        requests.release("call-1")
        assert pool.acquire_calls == pool.release_calls == 1
    finally:
        backend.complete.set()
        requests.close()


def test_two_tickets_load_and_release_independent_host_buffers(
    tmp_path: Path,
) -> None:
    metadata, backend, pool, storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        assert requests.select_skill("call-2", "internal-comms")
        assert backend.read_calls == 1
        assert metadata.get_runtime("call-1").io_operation_id != metadata.get_runtime(
            "call-2"
        ).io_operation_id
        backend.complete.set()
        deadline = time.monotonic() + 5
        while any(
            requests.poll(ticket).host_load_state is not HostLoadState.READY
            for ticket in ("call-1", "call-2")
        ):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert backend.read_calls == 2
        assert storage.get_ready_buffers("call-1") is not storage.get_ready_buffers(
            "call-2"
        )
        requests.release("call-1")
        assert pool.release_calls == 1
        requests.release("call-2")
        assert pool.release_calls == 2
    finally:
        backend.complete.set()
        requests.close()


def test_miss_and_deployment_mismatch_fail_without_io(tmp_path: Path) -> None:
    metadata, backend, pool, storage, requests = build_runtime(tmp_path)
    wrong_deployment = RequestManager(
        metadata,
        storage,
        model_fingerprint="model-b",
        tokenizer_fingerprint="tokenizer-a",
    )
    try:
        assert not requests.select_skill("call-1", "missing-skill")
        assert not wrong_deployment.select_skill("call-2", "internal-comms")
        assert backend.read_calls == 0
        assert pool.acquire_calls == 0
    finally:
        wrong_deployment.close()


def test_cancel_unknown_ticket_is_idempotent(tmp_path: Path) -> None:
    _metadata, backend, _pool, _storage, requests = build_runtime(tmp_path)
    try:
        requests.cancel("unknown-ticket")
        assert backend.read_calls == 0
    finally:
        requests.close()


def test_same_ticket_cannot_select_another_object(tmp_path: Path) -> None:
    metadata, backend, _pool, _storage, requests = build_runtime(tmp_path)
    original = metadata.resolve_object(
        skill_name="internal-comms",
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
    )
    second = CacheObjectMetadata(
        object_id="docx:v1:model-a",
        skill_name="docx",
        skill_version="v1",
        model_fingerprint="model-a",
        tokenizer_fingerprint="tokenizer-a",
        token_count=original.token_count,
        source_position_start=0,
        token_ids_sha256="b" * 64,
        start_marker_token_ids=(3, 4),
        container_id=original.container_id,
        read_strategy=original.read_strategy,
        layers=tuple(
            LayerExtent(
                layer_id=item.layer_id,
                backend_key=f"docx-{item.backend_key}",
                offset_bytes=item.offset_bytes,
                length_bytes=item.length_bytes,
                dtype=item.dtype,
                shape=item.shape,
                memory_layout=item.memory_layout,
                payload_sha256=item.payload_sha256,
            )
            for item in original.layers
        ),
    )
    metadata.publish_object(second)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        assert not requests.select_skill("call-1", "docx")
        assert backend.read_calls == 1
    finally:
        backend.complete.set()
        requests.close()


def test_expired_ticket_releases_storage_lease(tmp_path: Path) -> None:
    metadata, backend, pool, _storage, requests = build_runtime(
        tmp_path, ttl_seconds=0.001
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        time.sleep(0.005)
        state = requests.poll("call-1")
        assert state.binding_state is BindingState.FALLBACK
        backend.complete.set()
        deadline = time.monotonic() + 5
        while pool.release_calls == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        backend.complete.set()
        requests.close()


def test_tool_observation_check_and_token_binding_run_before_host_ready(
    tmp_path: Path,
) -> None:
    metadata, backend, _pool, _storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        content = (
            '<context_segment skill_name="internal-comms">\n'
            "skill body\n"
            "</context_segment>\n"
            "--- Skill Resources ---\n"
        )
        assert requests.inspect_tool_observation("call-1", "skill", content)
        assert metadata.get_runtime("call-1").binding_state is BindingState.OBSERVED

        prompt = [90, *SKILL_TOKENS, 91]
        binding = requests.authenticate_and_bind("call-1", "request-1", prompt)
        assert binding is not None
        assert (binding.segment_start, binding.segment_end) == (1, 5)
        state = metadata.get_runtime("call-1")
        assert state.binding_state is BindingState.VERIFIED
        assert state.host_load_state is HostLoadState.LOADING
        assert (state.segment_start, state.segment_end) == (1, 5)
    finally:
        backend.complete.set()
        requests.close()


def test_binding_selects_newest_authenticated_occurrence(tmp_path: Path) -> None:
    _metadata, backend, _pool, _storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        content = (
            '<context_segment skill_name="internal-comms">\n'
            "skill body\n</context_segment>\n"
        )
        assert requests.inspect_tool_observation("call-1", "skill", content)
        prompt = [*SKILL_TOKENS, 99, *SKILL_TOKENS, 100]
        binding = requests.authenticate_and_bind("call-1", "request-1", prompt)
        assert binding is not None
        assert (binding.segment_start, binding.segment_end) == (5, 9)
    finally:
        backend.complete.set()
        requests.close()


def test_binding_reuses_only_longest_unchanged_chunk_prefix(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 2024))
    metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path,
        skill_tokens=skill_tokens,
        chunk_size_tokens=64,
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        content = (
            '<context_segment skill_name="internal-comms">\n'
            "changed body\n</context_segment>\n"
        )
        assert requests.inspect_tool_observation("call-1", "skill", content)
        variant = list(skill_tokens)
        variant[6 * 64] = 999_999
        prompt_prefix = tuple(range(13))

        binding = requests.authenticate_and_bind(
            "call-1",
            "request-1",
            [*prompt_prefix, *variant, 55],
        )

        assert binding is not None
        assert binding.match_mode is SkillMatchMode.PARTIAL_PREFIX
        assert binding.matched_chunk_count == 6
        assert (binding.segment_start, binding.segment_end) == (13, 397)
        state = metadata.get_runtime("call-1")
        assert state.match_mode is SkillMatchMode.PARTIAL_PREFIX
        assert state.matched_chunk_count == 6

        plan = requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        )
        assert plan is not None
        assert plan.source_object_token_count == len(skill_tokens)
        assert (plan.segment_start, plan.segment_end) == (13, 397)
        assert (plan.reuse_start, plan.reuse_end) == (80, 384)
        assert (plan.source_reuse_start, plan.source_reuse_end) == (67, 371)
    finally:
        backend.complete.set()
        requests.close()


def test_newest_variant_does_not_fall_back_to_older_exact_skill(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 1512))
    _metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path,
        skill_tokens=skill_tokens,
        chunk_size_tokens=64,
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        content = (
            '<context_segment skill_name="internal-comms">\n'
            "changed body\n</context_segment>\n"
        )
        assert requests.inspect_tool_observation("call-1", "skill", content)
        variant = list(skill_tokens)
        variant[3 * 64] = 999_999
        newest_start = len(skill_tokens) + 1
        prompt = [*skill_tokens, 77, *variant, 88]

        binding = requests.authenticate_and_bind("call-1", "request-1", prompt)

        assert binding is not None
        assert binding.segment_start == newest_start
        assert binding.segment_end == newest_start + 3 * 64
        assert binding.matched_chunk_count == 3
        assert binding.match_mode is SkillMatchMode.PARTIAL_PREFIX
    finally:
        backend.complete.set()
        requests.close()


def test_partial_authentication_stops_before_an_unchanged_suffix(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 1384))
    _metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path,
        skill_tokens=skill_tokens,
        chunk_size_tokens=64,
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        content = (
            '<context_segment skill_name="internal-comms">\n'
            "changed body\n</context_segment>\n"
        )
        assert requests.inspect_tool_observation("call-1", "skill", content)
        variant = list(skill_tokens)
        variant[2 * 64] = 999_999

        binding = requests.authenticate_and_bind(
            "call-1", "request-1", [*variant, 55]
        )

        assert binding is not None
        assert binding.matched_chunk_count == 2
        assert binding.segment_end == 2 * 64
    finally:
        backend.complete.set()
        requests.close()


def test_failed_skill_observation_cancels_prefetch(tmp_path: Path) -> None:
    metadata, backend, pool, _storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        assert not requests.inspect_tool_observation(
            "call-1", "skill", "Error reading SKILL.md"
        )
        state = metadata.get_runtime("call-1")
        assert state.binding_state is BindingState.FALLBACK
        assert state.fallback_reason == "invalid_skill_observation"
        backend.complete.set()
        deadline = time.monotonic() + 5
        while pool.release_calls == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        backend.complete.set()
        requests.close()


def test_wrong_skill_and_wrong_prompt_tokens_fail_closed(tmp_path: Path) -> None:
    metadata, backend, _pool, _storage, requests = build_runtime(tmp_path)
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        wrong = '<context_segment skill_name="docx">\nbody\n</context_segment>\n'
        assert not requests.inspect_tool_observation("call-1", "skill", wrong)
        assert metadata.get_runtime("call-1").binding_state is BindingState.FALLBACK

        assert requests.select_skill("call-2", "internal-comms")
        correct = (
            '<context_segment skill_name="internal-comms">\n'
            "body\n</context_segment>\n"
        )
        assert requests.inspect_tool_observation("call-2", "skill", correct)
        assert requests.authenticate_and_bind(
            "call-2", "request-2", [10, 11, 12, 999]
        ) is None
        assert metadata.get_runtime("call-2").binding_state is BindingState.FALLBACK
    finally:
        backend.complete.set()
        requests.close()


def test_prepare_reuse_aligns_online_and_source_ranges(tmp_path: Path) -> None:
    skill_tokens = tuple(range(1000, 2024))
    metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path, skill_tokens=skill_tokens
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(
            requests, skill_tokens=skill_tokens, prompt_prefix=tuple(range(13))
        )

        plan = requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        )

        assert plan is not None
        assert (plan.segment_start, plan.segment_end) == (13, 1037)
        assert (plan.reuse_start, plan.reuse_end) == (80, 1024)
        assert (plan.source_reuse_start, plan.source_reuse_end) == (67, 1011)
        assert (plan.calibration_start, plan.calibration_end) == (48, 80)
        assert plan.calibration_start - plan.segment_start >= 32
        assert plan.correction_alpha == 0.6
        state = metadata.get_runtime("call-1")
        assert state.reuse_start == 80
        assert state.reuse_end == 1024
        assert requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        ) == plan
        assert requests.prepare_reuse(
            "call-1", "request-1", block_alignment=32
        ) is None
    finally:
        backend.complete.set()
        requests.close()


def test_prepare_reuse_resolves_ratio_and_direct_strategies(tmp_path: Path) -> None:
    skill_tokens = tuple(range(1000, 2024))

    def prepare(ticket: str, request_id: str, policy: ReusePolicy):
        (tmp_path / ticket).mkdir()
        _metadata, backend, _pool, _storage, requests = build_runtime(
            tmp_path / ticket, skill_tokens=skill_tokens
        )
        assert requests.select_skill(ticket, "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(
            requests,
            ticket=ticket,
            request_id=request_id,
            skill_tokens=skill_tokens,
        )
        result = requests.prepare_reuse(
            ticket, request_id, block_alignment=16, policy=policy
        )
        backend.complete.set()
        requests.close()
        return result

    ratio = prepare(
        "ratio-call",
        "ratio-request",
        ReusePolicy(
            correction_strategy=CorrectionStrategy.RATIO_PREFIX,
            calibration_ratio=0.15,
        ),
    )
    assert ratio is not None
    assert ratio.correction_strategy is CorrectionStrategy.RATIO_PREFIX
    assert ratio.calibration_end - ratio.calibration_start == 154

    direct = prepare(
        "direct-call",
        "direct-request",
        ReusePolicy(correction_strategy=CorrectionStrategy.DIRECT),
    )
    assert direct is not None
    assert direct.correction_strategy is CorrectionStrategy.DIRECT
    assert direct.calibration_start == direct.calibration_end == direct.reuse_start

    deviation = prepare(
        "deviation-call",
        "deviation-request",
        ReusePolicy(
            correction_strategy=CorrectionStrategy.DEVIATION_TOPK,
            deviation_recompute_ratio=0.15,
            deviation_check_layer=1,
        ),
    )
    assert deviation is not None
    assert deviation.correction_strategy is CorrectionStrategy.DEVIATION_TOPK
    assert deviation.calibration_start == deviation.calibration_end
    assert deviation.deviation_recompute_ratio == 0.15
    assert deviation.deviation_check_layer == 1


def test_readiness_is_orthogonal_to_verified_request(tmp_path: Path) -> None:
    skill_tokens = tuple(range(1000, 2024))
    _metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path, skill_tokens=skill_tokens
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(requests, skill_tokens=skill_tokens)
        plan = requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        )
        assert plan is not None
        loading = requests.query_reuse_readiness("call-1", "request-1")
        assert loading.status is ReuseReadiness.LOADING
        assert loading.plan == plan

        backend.complete.set()
        deadline = time.monotonic() + 5
        while True:
            ready = requests.query_reuse_readiness("call-1", "request-1")
            if ready.status is ReuseReadiness.READY:
                break
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert ready.plan == plan
        assert requests.query_reuse_readiness(
            "call-1", "wrong-request"
        ).status is ReuseReadiness.FALLBACK
    finally:
        backend.complete.set()
        requests.close()


def test_activation_exposes_only_the_bound_requests_complete_layer_group(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 2024))
    metadata, backend, pool, _storage, requests = build_runtime(
        tmp_path, skill_tokens=skill_tokens
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(requests, skill_tokens=skill_tokens)
        plan = requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        )
        assert plan is not None
        assert requests.activate_reuse("call-1", "request-1") is None

        backend.complete.set()
        deadline = time.monotonic() + 5
        while (
            requests.query_reuse_readiness("call-1", "request-1").status
            is not ReuseReadiness.READY
        ):
            assert time.monotonic() < deadline
            time.sleep(0.005)

        assert requests.activate_reuse("call-1", "wrong-request") is None
        assert requests.activate_reuse("call-1", "request-1") == plan
        assert requests.activate_reuse("call-1", "request-1") == plan
        buffers = requests.get_active_layer_buffers("call-1", "request-1")
        assert len(buffers) == 2
        assert requests.mark_layer_loaded(
            "call-1", "request-1", 0
        ).loaded_through_layer == 0
        assert requests.mark_layer_corrected(
            "call-1", "request-1", 0
        ).corrected_through_layer == 0
        assert requests.mark_layer_loaded(
            "call-1", "request-1", 1
        ).loaded_through_layer == 1
        assert requests.mark_layer_corrected(
            "call-1", "request-1", 1
        ).corrected_through_layer == 1

        requests.release("call-1")
        assert metadata.get_runtime("call-1").binding_state is BindingState.RELEASED
        assert pool.release_calls == 1
    finally:
        backend.complete.set()
        requests.close()


def test_active_layer_progress_rejects_wrong_request_and_out_of_order_layer(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 2024))
    _metadata, backend, _pool, _storage, requests = build_runtime(
        tmp_path, skill_tokens=skill_tokens
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(requests, skill_tokens=skill_tokens)
        assert requests.prepare_reuse(
            "call-1", "request-1", block_alignment=16
        ) is not None
        backend.complete.set()
        deadline = time.monotonic() + 5
        while (
            requests.query_reuse_readiness("call-1", "request-1").status
            is not ReuseReadiness.READY
        ):
            assert time.monotonic() < deadline
            time.sleep(0.005)
        assert requests.activate_reuse("call-1", "request-1") is not None

        try:
            requests.get_active_layer_buffers("call-1", "wrong-request")
        except ValueError as exc:
            assert "does not own" in str(exc)
        else:
            raise AssertionError("wrong request unexpectedly accessed buffers")
        try:
            requests.mark_layer_loaded("call-1", "request-1", 1)
        except ValueError as exc:
            assert "layer" in str(exc)
        else:
            raise AssertionError("out-of-order layer unexpectedly succeeded")
    finally:
        backend.complete.set()
        requests.close()


def test_short_reusable_suffix_falls_back_and_releases_load(
    tmp_path: Path,
) -> None:
    skill_tokens = tuple(range(1000, 1400))
    metadata, backend, pool, _storage, requests = build_runtime(
        tmp_path, skill_tokens=skill_tokens
    )
    try:
        assert requests.select_skill("call-1", "internal-comms")
        assert backend.started.wait(timeout=5)
        bind_verified_request(requests, skill_tokens=skill_tokens)
        assert requests.prepare_reuse(
            "call-1",
            "request-1",
            block_alignment=16,
            policy=ReusePolicy(minimum_reuse_tokens=352),
        ) is None
        state = metadata.get_runtime("call-1")
        assert state.binding_state is BindingState.FALLBACK
        assert state.fallback_reason == "reusable_suffix_too_short"
        backend.complete.set()
        deadline = time.monotonic() + 5
        while pool.release_calls == 0:
            assert time.monotonic() < deadline
            time.sleep(0.005)
    finally:
        backend.complete.set()
        requests.close()
