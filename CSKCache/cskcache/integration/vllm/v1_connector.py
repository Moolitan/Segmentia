from __future__ import annotations

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
from cskcache.logging import init_logger


logger = init_logger(__name__)

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
        logger.info(
            "connector init role=%s module=%s extra_config=%s",
            role,
            __file__,
            vllm_config.kv_transfer_config.kv_connector_extra_config,
        )
        super().__init__(
            vllm_config=vllm_config,
            role=role,
            kv_cache_config=kv_cache_config,
        )
        self._engine = CSKCacheConnectorV1Impl(vllm_config, role, self)
        logger.info("connector ready role=%s", role)

    # Called by GPUModelRunner.initialize_kv_cache()
    # (vllm/v1/worker/gpu_model_runner.py), when the worker allocates its
    # paged KV cache tensors.
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        self._engine.register_kv_caches(kv_caches)

    # Called by GPUModelRunner.initialize_kv_cache()
    # (vllm/v1/worker/gpu_model_runner.py), right alongside
    # register_kv_caches(), to hand the connector the model itself.
    def register_model(self, model: torch.nn.Module) -> None:
        """Bind model-owned state needed by KV-only connector operations."""

        self._engine.register_model(model)

    # Called by KVConnectorModelRunnerMixin._get_kv_connector_output()
    # (vllm/v1/worker/kv_connector_model_runner_mixin.py), the context
    # manager wrapping each step's model forward call.
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs: Any) -> None:
        self._engine.start_load_kv(forward_context, **kwargs)

    # Called on entry by the maybe_transfer_kv_layer() decorator
    # (vllm/model_executor/layers/attention/kv_transfer_utils.py), which
    # wraps unified_attention() (attention.py:612, the actual per-layer
    # attention forward) -- fires once per layer per step, before that
    # layer's real attention computation runs.
    def wait_for_layer_load(self, layer_name: str) -> None:
        self._engine.wait_for_layer_load(layer_name)

    # Called by the same maybe_transfer_kv_layer() decorator wrapping
    # unified_attention() as wait_for_layer_load() above, but after that
    # layer's real attention computation returns instead of before.
    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs: Any,
    ) -> None:
        self._engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    # Called by KVConnectorModelRunnerMixin.finalize_kv_connector()
    # (vllm/v1/worker/kv_connector_model_runner_mixin.py; also invoked from
    # inside that same file's _get_kv_connector_output() cleanup).
    def wait_for_save(self) -> None:
        self._engine.wait_for_save()

    # Called by KVConnectorModelRunnerMixin._get_kv_connector_output()
    # (vllm/v1/worker/kv_connector_model_runner_mixin.py), after the step's
    # forward pass, to collect worker->scheduler metadata.
    def build_connector_worker_meta(self):
        return self._engine.build_connector_worker_meta()

    # Called by KVConnectorModelRunnerMixin._get_kv_connector_output()
    # (vllm/v1/worker/kv_connector_model_runner_mixin.py), alongside
    # build_connector_worker_meta().
    def get_finished(
        self,
        finished_req_ids: set[str],
    ) -> tuple[set[str] | None, set[str] | None]:
        return self._engine.get_finished(finished_req_ids)

    # Called by Scheduler._connector_finished()
    # (vllm/v1/core/sched/scheduler.py).
    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self._engine.request_finished(request, block_ids)

    # Called by Scheduler._update_from_kv_xfer_finished()
    # (vllm/v1/core/sched/scheduler.py).
    def update_connector_output(self, connector_output) -> None:
        self._engine.update_connector_output(connector_output)

    # Called by Scheduler.shutdown() (vllm/v1/core/sched/scheduler.py).
    def shutdown(self) -> None:
        self._engine.shutdown()

    # Called by Scheduler.schedule() (vllm/v1/core/sched/scheduler.py),
    # vLLM's main per-step scheduling loop.
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        return self._engine.get_num_new_matched_tokens(request, num_computed_tokens)

    # Called by Scheduler.schedule() (vllm/v1/core/sched/scheduler.py),
    # right after get_num_new_matched_tokens() (or CSKCache's own
    # get_boundary_reuse_load_tokens()) allocates blocks.
    def update_state_after_alloc(
        self,
        request: "Request",
        blocks: "KVCacheBlocks",
        num_external_tokens: int,
    ) -> None:
        self._engine.update_state_after_alloc(request, blocks, num_external_tokens)

    # Called by Scheduler.schedule() (vllm/v1/core/sched/scheduler.py),
    # once per step, to package this step's scheduler decisions for the
    # worker.
    def build_connector_meta(
        self,
        scheduler_output: SchedulerOutput,
    ) -> KVConnectorMetadata:
        return self._engine.build_connector_meta(scheduler_output)

    # Not part of vLLM's official KVConnectorBase_V1 contract -- CSKCache's
    # own patch to Scheduler.schedule() (vllm/v1/core/sched/scheduler.py in
    # the /home/wsh/vllm fork), guarded with hasattr() so vanilla vLLM
    # still works without it. Called right alongside the standard hooks
    # above.
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

    # Same as cap_prefill_before_reuse(): CSKCache's own patch to
    # Scheduler.schedule() (vllm/v1/core/sched/scheduler.py in the
    # /home/wsh/vllm fork), not part of vLLM's official contract.
    def get_boundary_reuse_load_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        return self._engine.get_boundary_reuse_load_tokens(
            request, num_computed_tokens
        )
