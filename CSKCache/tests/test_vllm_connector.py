from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from cskcache.integrations.vllm.base import (
    CSKCacheConnectorMetadata,
    FINALIZE_PROGRESSIVE_REUSE,
    PREPARE_REUSE,
    QUERY_PROGRESSIVE_REQUIREMENT,
)
from cskcache.integrations.vllm.connector import CSKCacheConnectorV1
from cskcache.runtime.base import CorrectionStrategy, ReuseAllocation, ReusePlan
from cskcache.runtime.coordinator import SchedulerReuseCoordinator
from cskcache.runtime.transport import PlanTransportCoordinator


def make_connector() -> CSKCacheConnectorV1:
    connector = object.__new__(CSKCacheConnectorV1)
    connector._lmcache_engine = MagicMock()
    connector._lmcache_engine.lookup_client = MagicMock()
    connector._csk_transport = MagicMock()
    connector._csk_rank_control = MagicMock()
    return connector


def test_connector_owns_cskcache_control_dispatch() -> None:
    connector = make_connector()
    lookup = connector._lmcache_engine.lookup_client
    lookup.submit_external_control.return_value = True
    lookup.execute_external_control.side_effect = [True, {"bound": True}]
    connector._csk_transport = PlanTransportCoordinator()

    assert connector.execute_connector_control(
        "cskcache.submit_prefetch",
        {"ticket": "call-1", "skill_name": "docx"},
    )
    assert connector.execute_connector_control(
        "cskcache.inspect_tool_observation",
        {"ticket": "call-1", "tool_name": "skill", "content": "body"},
    )
    assert connector.execute_connector_control(
        "cskcache.authenticate_request",
        {
            "ticket": "call-1",
            "request_id": "req-1",
            "prompt_token_ids": [1, 2],
        },
    ) == {"bound": True}
    assert connector.execute_connector_control(
        "cskcache.cancel_prefetch",
        {"ticket": "call-1", "reason": "request_failed"},
    ) is None

    lookup.submit_external_control.assert_any_call(
        "cskcache.submit_prefetch",
        {"ticket": "call-1", "skill_name": "docx"},
    )
    lookup.submit_external_control.assert_any_call(
        "cskcache.cancel_prefetch",
        {"ticket": "call-1", "reason": "request_failed"},
    )


def test_connector_rejects_unknown_control_command() -> None:
    connector = make_connector()
    with pytest.raises(ValueError, match="unknown CSKCache control command"):
        connector.execute_connector_control("unknown", {})


def test_request_arrival_prefetch_waits_until_ticket_is_created() -> None:
    connector = make_connector()
    lookup = connector._lmcache_engine.lookup_client
    lookup.execute_external_control.return_value = True

    assert connector.execute_connector_control(
        "cskcache.submit_prefetch",
        {
            "ticket": "call-request",
            "skill_name": "result-to-claim",
            "wait": True,
        },
    )

    lookup.execute_external_control.assert_called_once_with(
        "cskcache.submit_prefetch",
        {"ticket": "call-request", "skill_name": "result-to-claim"},
        default=False,
    )
    lookup.submit_external_control.assert_not_called()


def test_worker_runtime_initializes_after_lmcache_physical_resources(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        LMCacheConnectorV1,
        "register_kv_caches",
        lambda _self, _caches: calls.append("lmcache_post_init"),
    )
    connector = object.__new__(CSKCacheConnectorV1)
    connector._role = KVConnectorRole.WORKER
    connector._initialize_csk_worker = lambda: calls.append("csk_runtime")

    connector.register_kv_caches({"layer": MagicMock()})

    assert calls == ["lmcache_post_init", "csk_runtime"]


def make_plan() -> ReusePlan:
    return ReusePlan(
        ticket="call-1",
        cache_object_id="skill-v1",
        request_id="request-1",
        segment_start=96,
        segment_end=112,
        reuse_start=104,
        reuse_end=108,
        source_reuse_start=8,
        source_reuse_end=12,
        calibration_start=100,
        calibration_end=104,
        correction_alpha=0.6,
        block_alignment=4,
    )


def test_allocation_builds_csk_owned_worker_metadata() -> None:
    connector = make_connector()
    plan = make_plan()
    allocation = ReuseAllocation(
        plan=plan,
        computed_start=100,
        computed_end=108,
        block_ids=tuple(range(27)),
    )
    connector._csk_transport.bind_allocation.return_value = allocation
    connector._vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(block_size=4)
    )
    connector._pending_worker_requests = {}
    request = SimpleNamespace(
        request_id="request-1", all_token_ids=list(range(120))
    )
    blocks = SimpleNamespace()

    connector.update_state_after_alloc(request, blocks, 8)

    worker_request = connector._pending_worker_requests["request-1"]
    assert worker_request.plan is plan
    assert worker_request.token_ids == tuple(range(108))
    assert worker_request.slot_mapping.tolist() == list(range(108))
    assert worker_request.failed_block_ids == frozenset({25, 26})
    connector._lmcache_engine.register_external_materialization.assert_called_once_with(
        request,
        computed_end=108,
        block_ids=list(range(27)),
    )


def test_composite_metadata_filters_csk_from_lmcache_physical_requests() -> None:
    connector = make_connector()
    plan = make_plan()
    worker_request = SimpleNamespace(plan=plan)
    connector._pending_worker_requests = {"request-1": worker_request}
    physical = SimpleNamespace(
        requests=[
            SimpleNamespace(req_id="request-1"),
            SimpleNamespace(req_id="ordinary"),
        ]
    )
    connector._lmcache_engine.build_connector_meta.return_value = physical

    metadata = connector.build_connector_meta(SimpleNamespace())

    assert isinstance(metadata, CSKCacheConnectorMetadata)
    assert metadata.requests == [worker_request]
    assert [request.req_id for request in physical.requests] == ["ordinary"]
    assert connector._pending_worker_requests == {}


def test_progressive_readiness_freezes_the_max_rank_requirement() -> None:
    connector = make_connector()
    connector._csk_transport = PlanTransportCoordinator()
    lookup = connector._lmcache_engine.lookup_client
    provisional = ReusePlan(
        ticket="call-progressive",
        cache_object_id="skill-v1",
        request_id="request-progressive",
        segment_start=96,
        segment_end=164,
        reuse_start=112,
        reuse_end=160,
        source_reuse_start=16,
        source_reuse_end=64,
        calibration_start=100,
        calibration_end=112,
        correction_alpha=0.6,
        block_alignment=4,
        source_token_count=68,
    )
    frozen = ReusePlan.from_dict(
        {
            **provisional.to_dict(),
            "reuse_start": 124,
            "source_reuse_start": 28,
            "calibration_end": 124,
        }
    )

    def execute(command, payload, **_kwargs):
        if command == PREPARE_REUSE:
            return provisional.to_dict()
        if command == FINALIZE_PROGRESSIVE_REUSE:
            assert payload["calibration_tokens"] == 24
            return frozen.to_dict()
        raise AssertionError(f"unexpected command: {command}")

    lookup.execute_external_control.side_effect = execute
    connector._csk_rank_control.execute_all.return_value = [
        {"status": "ready", "required_calibration_tokens": 16},
        {"status": "ready", "required_calibration_tokens": 24},
    ]

    prepared = connector.prepare_csk_reuse(
        provisional.ticket,
        provisional.request_id,
        provisional.block_alignment,
    )
    readiness = connector.query_csk_readiness(
        provisional.ticket, provisional.request_id
    )

    assert prepared == provisional
    assert readiness == {
        "status": "ready",
        "plan": frozen.to_dict(),
        "reason": None,
    }
    connector._csk_rank_control.execute_all.assert_called_once_with(
        QUERY_PROGRESSIVE_REQUIREMENT,
        {
            "ticket": provisional.ticket,
            "request_id": provisional.request_id,
        },
    )


def test_progressive_readiness_fails_closed_on_rank_capability_mismatch() -> None:
    connector = make_connector()
    lookup = connector._lmcache_engine.lookup_client
    connector._csk_rank_control.execute_all.return_value = [
        {"status": "ready", "required_calibration_tokens": 16},
        {"status": "unsupported", "reason": "not_progressive"},
    ]

    assert connector.query_csk_readiness("call-1", "request-1") == {
        "status": "fallback",
        "plan": None,
        "reason": "progressive_rank_capability_mismatch",
    }
    lookup.execute_external_control.assert_not_called()


def test_static_layerwise_readiness_requires_same_plan_on_every_rank() -> None:
    connector = make_connector()
    connector._csk_transport = PlanTransportCoordinator()
    lookup = connector._lmcache_engine.lookup_client
    plan = ReusePlan(
        ticket="call-layerwise",
        cache_object_id="skill-v1",
        request_id="request-layerwise",
        segment_start=96,
        segment_end=1160,
        reuse_start=128,
        reuse_end=1152,
        source_reuse_start=32,
        source_reuse_end=1056,
        calibration_start=128,
        calibration_end=128,
        correction_alpha=0.6,
        block_alignment=16,
        source_token_count=1064,
        correction_strategy=CorrectionStrategy.DEVIATION_TOPK,
        deviation_recompute_ratio=0.15,
        deviation_check_layer=1,
    )
    lookup.execute_external_control.return_value = plan.to_dict()
    assert connector.prepare_csk_reuse(
        plan.ticket, plan.request_id, plan.block_alignment
    ) == plan
    connector._csk_rank_control.execute_all.return_value = [
        {"status": "ready", "mode": "static_layerwise", "plan": plan.to_dict()},
        {"status": "ready", "mode": "static_layerwise", "plan": plan.to_dict()},
    ]

    assert connector.query_csk_readiness(plan.ticket, plan.request_id) == {
        "status": "ready",
        "plan": plan.to_dict(),
        "reason": None,
    }


def test_progressive_readiness_fails_closed_on_empty_rank_response() -> None:
    connector = make_connector()
    lookup = connector._lmcache_engine.lookup_client
    connector._csk_rank_control.execute_all.return_value = []

    assert connector.query_csk_readiness("call-1", "request-1") == {
        "status": "fallback",
        "plan": None,
        "reason": "progressive_rank_control_failed",
    }
    lookup.execute_external_control.assert_not_called()


def test_readiness_uses_existing_control_when_rank_client_is_unavailable() -> None:
    connector = make_connector()
    connector._csk_rank_control = None
    lookup = connector._lmcache_engine.lookup_client
    lookup.execute_external_control.return_value = {
        "status": "ready",
        "plan": None,
        "reason": None,
    }

    assert connector.query_csk_readiness("call-1", "request-1") == {
        "status": "ready",
        "plan": None,
        "reason": None,
    }
    lookup.execute_external_control.assert_called_once()


def test_shutdown_closes_cskcache_rank_control() -> None:
    connector = make_connector()
    rank_control = connector._csk_rank_control
    connector._lmcache_engine.lmcache_engine = None
    connector._csk_runtime = None
    connector._runtime_control_handler = None

    connector.shutdown()

    rank_control.close.assert_called_once_with()
    assert connector._csk_rank_control is None
    connector._lmcache_engine.shutdown.assert_called_once_with()


def test_scheduler_freezes_expanded_plan_before_activation() -> None:
    provisional = ReusePlan(
        ticket="call-progressive",
        cache_object_id="skill-v1",
        request_id="request-progressive",
        segment_start=96,
        segment_end=164,
        reuse_start=112,
        reuse_end=160,
        source_reuse_start=16,
        source_reuse_end=64,
        calibration_start=100,
        calibration_end=112,
        correction_alpha=0.6,
        block_alignment=4,
        source_token_count=68,
    )
    frozen = ReusePlan.from_dict(
        {
            **provisional.to_dict(),
            "reuse_start": 124,
            "source_reuse_start": 28,
            "calibration_end": 124,
        }
    )
    control = MagicMock()
    control.prepare_csk_reuse.return_value = provisional
    control.query_csk_readiness.return_value = {
        "status": "ready",
        "plan": frozen.to_dict(),
        "reason": None,
    }
    control.activate_csk_reuse.return_value = frozen
    coordinator = SchedulerReuseCoordinator()
    verified = {
        "ticket": provisional.ticket,
        "cache_object_id": provisional.cache_object_id,
        "request_id": provisional.request_id,
        "segment_start": provisional.segment_start,
        "segment_end": provisional.segment_end,
    }

    assert coordinator.register(
        verified,
        request_id=provisional.request_id,
        prompt_tokens=200,
        block_alignment=4,
        async_scheduling=False,
        control=control,
    )
    assert coordinator.limit_prefill(
        provisional.request_id,
        num_computed_tokens=0,
        num_new_tokens=200,
        control=control,
    ) == provisional.calibration_start
    coordinator.limit_prefill(
        provisional.request_id,
        num_computed_tokens=0,
        num_new_tokens=10,
        control=control,
    )
    control.mark_csk_execution_selected.assert_called_once_with(
        provisional.ticket, provisional.request_id
    )
    assert coordinator.reaches_calibration_boundary(
        provisional.request_id,
        num_computed_tokens=provisional.calibration_start,
        stopped=False,
        produced_tokens=False,
        request_running=True,
        in_flight_tokens=0,
    )
    assert coordinator.poll(provisional.request_id, control)

    directive = coordinator.activate(
        provisional.request_id,
        num_computed_tokens=provisional.calibration_start,
        control=control,
    )

    assert directive.activated
    assert directive.external_tokens == (
        frozen.reuse_end - frozen.calibration_start
    )
    assert coordinator.state(provisional.request_id).plan == frozen
