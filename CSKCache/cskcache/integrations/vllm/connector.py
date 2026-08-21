"""External vLLM connector that owns CSKCache scheduling semantics."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
    LMCacheConnectorV1,
)
from vllm.logger import init_logger
from lmcache.integration.vllm.utils import ENGINE_NAME

from ...integrations.lmcache import (
    LMCacheRuntimeBridge,
    LMCacheWorkerIntegration,
    lmcache_integration_enabled,
)
from ...runtime.transport import PlanTransportCoordinator

from .base import (
    ACTIVATE_REUSE,
    AUTHENTICATE_REQUEST,
    CANCEL_PREFETCH,
    CSKCacheConnectorMetadata,
    CSKCacheWorkerRequest,
    INSPECT_TOOL_OBSERVATION,
    PREPARE_REUSE,
    QUERY_READINESS,
    RELEASE_REUSE,
    SUBMIT_PREFETCH,
)
from .scheduler import CSKCacheSchedulerExtension

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


logger = init_logger(__name__)


class _CSKLookupControl:
    """Translate CSK lifecycle operations onto LMCache's opaque RPC."""

    def __init__(self, lookup: Any) -> None:
        self._lookup = lookup

    def prepare_csk_reuse(
        self, ticket: str, request_id: str, block_alignment: int
    ) -> Mapping[str, object] | None:
        return self._lookup.execute_external_control(
            PREPARE_REUSE,
            {
                "ticket": ticket,
                "request_id": request_id,
                "block_alignment": block_alignment,
            },
        )

    def query_csk_readiness(
        self, ticket: str, request_id: str
    ) -> dict[str, Any]:
        result = self._lookup.execute_external_control(
            QUERY_READINESS,
            {"ticket": ticket, "request_id": request_id},
            default={
                "status": "fallback",
                "plan": None,
                "reason": "cskcache_unavailable",
            },
        )
        if isinstance(result, dict):
            return result
        return {
            "status": "fallback",
            "plan": None,
            "reason": "invalid_control_response",
        }

    def activate_csk_reuse(
        self, ticket: str, request_id: str
    ) -> Mapping[str, object] | None:
        return self._lookup.execute_external_control(
            ACTIVATE_REUSE,
            {"ticket": ticket, "request_id": request_id},
        )

    def release_csk_reuse(self, ticket: str) -> bool:
        return bool(
            self._lookup.execute_external_control(
                RELEASE_REUSE, {"ticket": ticket}, default=False
            )
        )

    def cancel_csk_prefetch(self, ticket: str, reason: str) -> None:
        self._lookup.submit_external_control(
            CANCEL_PREFETCH,
            {"ticket": ticket, "reason": reason},
        )


class CSKCacheConnectorV1(LMCacheConnectorV1):
    """Compose CSKCache control with LMCache's existing physical connector."""

    def __init__(
        self,
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        super().__init__(vllm_config, role, kv_cache_config)
        self._csk_transport = PlanTransportCoordinator()
        self._pending_worker_requests: dict[str, CSKCacheWorkerRequest] = {}
        self._bound_csk_requests: tuple[CSKCacheWorkerRequest, ...] = ()
        self._csk_invalid_block_ids: set[int] = set()
        self._csk_scheduler = (
            CSKCacheSchedulerExtension(self)
            if role == KVConnectorRole.SCHEDULER
            else None
        )
        self._role = role
        self._csk_runtime = None
        self._runtime_control_handler = None
        self._csk_worker = None

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        super().register_kv_caches(kv_caches)
        if self._role == KVConnectorRole.WORKER:
            self._initialize_csk_worker()

    def _initialize_csk_worker(self) -> None:
        if self._csk_runtime is not None:
            return
        engine = self._lmcache_engine.lmcache_engine
        if (
            engine is None
            or not lmcache_integration_enabled(engine.config)
            or engine.gpu_connector is None
        ):
            return
        try:
            self._csk_runtime = LMCacheRuntimeBridge(engine)
            self._runtime_control_handler = self._execute_runtime_control
            self._csk_worker = LMCacheWorkerIntegration(
                self._csk_runtime,
                engine.gpu_connector,
                ENGINE_NAME,
                execution_order=str(
                    self._lmcache_engine.config.get_extra_config_value(
                        "csk_execution_order", "h2d_first"
                    )
                ),
            )
            engine.gpu_connector.set_layerwise_model_provider(
                lambda: self._csk_worker.layerwise_model
            )
            engine.register_external_control_handler(
                self._runtime_control_handler
            )
        except Exception as error:
            if self._runtime_control_handler is not None:
                engine.unregister_external_control_handler(
                    self._runtime_control_handler
                )
            if self._csk_runtime is not None:
                self._csk_runtime.close()
            self._csk_runtime = None
            self._runtime_control_handler = None
            self._csk_worker = None
            logger.error("CSKCache runtime is unavailable: %s", error)

    def get_scheduler_extension(self):
        return self._csk_scheduler

    def execute_connector_control(
        self, command: str, payload: Mapping[str, Any]
    ) -> Any:
        if command == SUBMIT_PREFETCH:
            return self.submit_csk_prefetch(
                payload["ticket"], payload["skill_name"]
            )
        if command == INSPECT_TOOL_OBSERVATION:
            return self.inspect_csk_tool_observation(
                payload["ticket"], payload["tool_name"], payload["content"]
            )
        if command == AUTHENTICATE_REQUEST:
            return self.authenticate_csk_request(
                payload["ticket"],
                payload["request_id"],
                payload["prompt_token_ids"],
            )
        if command == CANCEL_PREFETCH:
            self.cancel_csk_prefetch(payload["ticket"], payload["reason"])
            return None
        raise ValueError(f"unknown CSKCache control command: {command}")

    def _execute_runtime_control(
        self, command: str, payload: Mapping[str, Any]
    ) -> Any:
        runtime = self._csk_runtime
        if runtime is None:
            return None
        if command == SUBMIT_PREFETCH:
            return runtime.submit_prefetch(payload["ticket"], payload["skill_name"])
        if command == INSPECT_TOOL_OBSERVATION:
            return runtime.inspect_tool_observation(
                payload["ticket"], payload["tool_name"], payload["content"]
            )
        if command == AUTHENTICATE_REQUEST:
            return runtime.authenticate_request(
                payload["ticket"],
                payload["request_id"],
                payload["prompt_token_ids"],
            )
        if command == PREPARE_REUSE:
            return runtime.prepare_reuse(
                payload["ticket"],
                payload["request_id"],
                payload["block_alignment"],
            )
        if command == QUERY_READINESS:
            return runtime.query_readiness(
                payload["ticket"], payload["request_id"]
            )
        if command == ACTIVATE_REUSE:
            return runtime.activate_reuse(
                payload["ticket"], payload["request_id"]
            )
        if command == RELEASE_REUSE:
            return runtime.release(payload["ticket"])
        if command == CANCEL_PREFETCH:
            runtime.cancel(payload["ticket"], payload["reason"])
            return None
        raise ValueError(f"unknown CSKCache runtime command: {command}")

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        allocation = self._csk_transport.bind_allocation(
            request.request_id,
            num_external_tokens=num_external_tokens,
            blocks=blocks,
        )
        if allocation is None:
            return super().update_state_after_alloc(
                request, blocks, num_external_tokens
            )

        token_ids = tuple(request.all_token_ids[: allocation.computed_end])
        block_ids = allocation.block_ids
        block_size = self._vllm_config.cache_config.block_size
        block_tensor = torch.tensor(block_ids, dtype=torch.long)
        offsets = torch.arange(block_size, dtype=torch.long)
        slot_mapping = (
            block_tensor.reshape(-1, 1) * block_size
            + offsets.reshape(1, -1)
        ).flatten()[: allocation.computed_end]
        first = allocation.computed_start // block_size
        last = (allocation.computed_end + block_size - 1) // block_size
        self._pending_worker_requests[request.request_id] = (
            CSKCacheWorkerRequest(
                plan=allocation.plan,
                token_ids=token_ids,
                slot_mapping=slot_mapping,
                failed_block_ids=frozenset(block_ids[first:last]),
            )
        )
        self._lmcache_engine.register_external_materialization(
            request,
            computed_end=allocation.computed_end,
            block_ids=list(block_ids),
        )

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> CSKCacheConnectorMetadata:
        lmcache_metadata = super().build_connector_meta(scheduler_output)
        request_ids = set(self._pending_worker_requests)
        physical_requests = getattr(lmcache_metadata, "requests", None)
        if physical_requests is not None and request_ids:
            physical_requests[:] = [
                request
                for request in physical_requests
                if request.req_id not in request_ids
            ]
        requests = list(self._pending_worker_requests.values())
        self._pending_worker_requests.clear()
        return CSKCacheConnectorMetadata(
            lmcache_metadata=lmcache_metadata,
            requests=requests,
        )

    def bind_connector_metadata(self, connector_metadata) -> None:
        if not isinstance(connector_metadata, CSKCacheConnectorMetadata):
            raise TypeError("CSKCacheConnectorV1 requires its own metadata")
        self._bound_csk_requests = tuple(connector_metadata.requests)
        super().bind_connector_metadata(connector_metadata.lmcache_metadata)

    def clear_connector_metadata(self) -> None:
        self._bound_csk_requests = ()
        super().clear_connector_metadata()

    def start_load_kv(
        self, forward_context: "ForwardContext", **kwargs: Any
    ) -> None:
        if self._bound_csk_requests:
            self._execute_csk_requests(forward_context)
        super().start_load_kv(forward_context, **kwargs)

    def get_block_ids_with_load_errors(self) -> set[int]:
        failed = super().get_block_ids_with_load_errors()
        failed.update(self._csk_invalid_block_ids)
        self._csk_invalid_block_ids.clear()
        return failed

    def _execute_csk_requests(self, forward_context: "ForwardContext") -> None:
        engine = self._lmcache_engine.lmcache_engine
        kvcaches = list(self._lmcache_engine.kv_caches.values())
        for request in self._bound_csk_requests:
            try:
                if forward_context.attn_metadata is None:
                    raise RuntimeError("attention metadata is unavailable")
                if engine is None or self._csk_worker is None:
                    raise RuntimeError("CSKCache worker is unavailable")
                result = self._csk_worker.execute(
                    request.plan,
                    token_ids=request.token_ids,
                    kvcaches=kvcaches,
                    slot_mapping=request.slot_mapping.to(
                        self._lmcache_engine.device
                    ),
                )
                self._csk_runtime.release(result.ticket)
            except Exception:
                try:
                    torch.cuda.synchronize(self._lmcache_engine.device)
                except Exception:
                    pass
                self._csk_invalid_block_ids.update(request.failed_block_ids)
                if self._csk_runtime is not None:
                    self._csk_runtime.cancel(
                        request.plan.ticket, "worker_load_failed"
                    )

    def submit_csk_prefetch(self, ticket: str, skill_name: str) -> bool:
        lookup = self._lmcache_engine.lookup_client
        return False if lookup is None else lookup.submit_external_control(
            SUBMIT_PREFETCH,
            {"ticket": ticket, "skill_name": skill_name},
        )

    def inspect_csk_tool_observation(
        self, ticket: str, tool_name: str, content: str
    ) -> bool:
        lookup = self._lmcache_engine.lookup_client
        return False if lookup is None else bool(
            lookup.execute_external_control(
                INSPECT_TOOL_OBSERVATION,
                {"ticket": ticket, "tool_name": tool_name, "content": content},
                default=False,
            )
        )

    def authenticate_csk_request(
        self, ticket: str, request_id: str, prompt_token_ids: list[int]
    ) -> dict[str, Any] | None:
        lookup = self._lmcache_engine.lookup_client
        return None if lookup is None else lookup.execute_external_control(
            AUTHENTICATE_REQUEST,
            {
                "ticket": ticket,
                "request_id": request_id,
                "prompt_token_ids": prompt_token_ids,
            },
        )

    def prepare_csk_reuse(
        self, ticket: str, request_id: str, block_alignment: int
    ) -> Any | None:
        lookup = self._lmcache_engine.lookup_client
        if lookup is None:
            return None
        return self._csk_transport.prepare(
            _CSKLookupControl(lookup), ticket, request_id, block_alignment
        )

    def query_csk_readiness(
        self, ticket: str, request_id: str
    ) -> dict[str, Any]:
        lookup = self._lmcache_engine.lookup_client
        if lookup is None:
            return {
                "status": "fallback",
                "plan": None,
                "reason": "cskcache_unavailable",
            }
        return _CSKLookupControl(lookup).query_csk_readiness(ticket, request_id)

    def activate_csk_reuse(
        self, ticket: str, request_id: str
    ) -> Any | None:
        lookup = self._lmcache_engine.lookup_client
        if lookup is None:
            return None
        return self._csk_transport.activate(
            _CSKLookupControl(lookup), ticket, request_id
        )

    def release_csk_reuse(self, ticket: str) -> bool:
        lookup = self._lmcache_engine.lookup_client
        return False if lookup is None else self._csk_transport.release(
            _CSKLookupControl(lookup), ticket
        )

    def cancel_csk_prefetch(self, ticket: str, reason: str) -> None:
        lookup = self._lmcache_engine.lookup_client
        if lookup is not None:
            self._csk_transport.cancel(
                _CSKLookupControl(lookup), ticket, reason
            )

    def shutdown(self) -> None:
        engine = self._lmcache_engine.lmcache_engine
        if (
            engine is not None
            and self._runtime_control_handler is not None
        ):
            engine.unregister_external_control_handler(
                self._runtime_control_handler
            )
        if self._csk_runtime is not None:
            self._csk_runtime.close()
            self._csk_runtime = None
        self._lmcache_engine.shutdown()
