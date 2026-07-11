from __future__ import annotations

import sys
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.v1.core.sched.output import SchedulerOutput

from cskcache.integration.vllm.v1_adapter import CSKCacheConnectorV1Impl

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request


class CSKCacheConnectorV1(KVConnectorBase_V1):
    """vLLM v1 connector entrypoint for CSKCache.

    vLLM can load this class without patching the connector factory:

    --kv-transfer-config '{"kv_connector":"CSKCacheConnectorV1",
      "kv_connector_module_path":"cskcache.integration.vllm.v1_connector",
      "kv_role":"kv_both"}'
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig | None" = None,
    ) -> None:
        print(
            "CSKCacheConnectorV1 __init__ from "
            f"{__file__} role={role} extra_config="
            f"{vllm_config.kv_transfer_config.kv_connector_extra_config}",
            file=sys.stderr,
            flush=True,
        )
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._engine = CSKCacheConnectorV1Impl(vllm_config, role, self)
        print(
            "CSKCacheConnectorV1 engine ready",
            file=sys.stderr,
            flush=True,
        )

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self._engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self._engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self) -> None:
        self._engine.wait_for_save()

    def build_connector_worker_meta(self):
        return self._engine.build_connector_worker_meta()

    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._engine.get_finished(finished_req_ids)

    def update_connector_output(self, connector_output) -> None:
        self._engine.update_connector_output(connector_output)

    def shutdown(self) -> None:
        self._engine.shutdown()

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._engine.get_num_new_matched_tokens(request, num_computed_tokens)

    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._engine.update_state_after_alloc(request, blocks, num_external_tokens)

    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return self._engine.build_connector_meta(scheduler_output)

    def cap_prefill_before_reuse(
        self,
        request: "Request",
        base_num_computed_tokens: int,
        num_new_tokens: int,
    ) -> int:
        return self._engine.cap_prefill_before_reuse(
            request,
            base_num_computed_tokens,
            num_new_tokens,
        )

    def get_boundary_reuse_load_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        return self._engine.get_boundary_reuse_load_tokens(
            request, num_computed_tokens
        )
