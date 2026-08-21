from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)

from cskcache.integrations.vllm.base import CSKCacheConnectorMetadata
from cskcache.integrations.vllm.connector import CSKCacheConnectorV1
from cskcache.runtime.base import ReuseAllocation, ReusePlan
from cskcache.runtime.transport import PlanTransportCoordinator


def make_connector() -> CSKCacheConnectorV1:
    connector = object.__new__(CSKCacheConnectorV1)
    connector._lmcache_engine = MagicMock()
    connector._lmcache_engine.lookup_client = MagicMock()
    connector._csk_transport = MagicMock()
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
