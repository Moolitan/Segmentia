"""LMCache-side T0 lifecycle wired to CSKCache runtime ownership."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from ...metadata.fingerprint import fingerprint_model, fingerprint_tokenizer
from ...metadata.manager import MetadataManager
from ...profile import profile_event
from ...runtime.base import CorrectionStrategy, ReusePolicy
from ...runtime.request_manager import RequestManager
from ...runtime.validator import validate_catalog_layout
from ...host_memory.pool import LMCacheHostBufferPool
from ...storage.manager import StorageManager
from ...storage.backends.local_disk import LMCacheLayerObjectReader
from lmcache.logging import init_logger

from .base import LMCacheRuntimeSettings


logger = init_logger(__name__)


def lmcache_integration_enabled(config: Any) -> bool:
    """Parse the single LMCache feature gate at the CSKCache boundary."""

    enabled = config.get_extra_config_value("csk_t0_prefetch", False)
    if not isinstance(enabled, bool):
        raise ValueError("csk_t0_prefetch must be a boolean")
    return enabled


class LMCacheRuntimeBridge:
    """Own CSKCache policy and lifecycle while borrowing LMCache resources."""

    def __init__(self, engine: Any) -> None:
        config = engine.config
        metadata_path = config.get_extra_config_value(
            "cskcache_metadata_path", None
        )
        if isinstance(metadata_path, str):
            for placeholder, attribute in (
                ("{worker_id}", "worker_id"),
                ("{world_size}", "world_size"),
            ):
                if placeholder not in metadata_path:
                    continue
                value = getattr(engine.metadata, attribute, None)
                if value is None:
                    raise ValueError(
                        f"cskcache_metadata_path uses {placeholder}, but "
                        f"LMCache metadata has no {attribute}"
                    )
                metadata_path = metadata_path.replace(placeholder, str(value))
        ticket_ttl_seconds = config.get_extra_config_value(
            "csk_prefetch_handle_ttl_seconds", 60.0
        )
        retain_last_host_object = config.get_extra_config_value(
            "csk_retain_last_host_object", False
        )
        settings = LMCacheRuntimeSettings(
            metadata_path=metadata_path,
            tokenizer_path=config.get_extra_config_value(
                "cskcache_tokenizer_path", None
            ),
            storage_backend=str(
                config.get_extra_config_value(
                    "csk_storage_backend", "raw_block"
                )
            ),
            storage_layout=str(
                config.get_extra_config_value(
                    "csk_storage_layout", "packed_chunks_single_layer"
                )
            ),
            host_layout=str(
                config.get_extra_config_value(
                    "csk_host_layout", "packed_chunks_single_layer"
                )
            ),
            chunk_size_tokens=int(
                config.get_extra_config_value("csk_chunk_size_tokens", 256)
            ),
            ticket_ttl_seconds=(
                None
                if ticket_ttl_seconds is None
                else float(ticket_ttl_seconds)
            ),
            reuse_policy=ReusePolicy(
                correction_strategy=CorrectionStrategy(
                    str(
                        config.get_extra_config_value(
                            "csk_correction_strategy", "fixed_prefix"
                        )
                    )
                ),
                minimum_full_recompute_tokens=int(
                    config.get_extra_config_value(
                        "csk_minimum_full_recompute_tokens", 32
                    )
                ),
                calibration_tokens=int(
                    config.get_extra_config_value(
                        "csk_calibration_tokens", 32
                    )
                ),
                calibration_ratio=(
                    None
                    if config.get_extra_config_value(
                        "csk_calibration_ratio", None
                    ) is None
                    else float(
                        config.get_extra_config_value(
                            "csk_calibration_ratio", None
                        )
                    )
                ),
                deviation_recompute_ratio=float(
                    config.get_extra_config_value(
                        "csk_deviation_recompute_ratio", 0.15
                    )
                ),
                deviation_check_layer=int(
                    config.get_extra_config_value(
                        "csk_deviation_check_layer", 1
                    )
                ),
                minimum_reuse_tokens=int(
                    config.get_extra_config_value(
                        "csk_minimum_reuse_tokens", 256
                    )
                ),
                correction_alpha=float(
                    config.get_extra_config_value(
                        "csk_correction_alpha", 0.6
                    )
                ),
            ),
            retain_last_host_object=retain_last_host_object,
        )
        if not config.local_cpu or not engine.use_layerwise:
            raise ValueError(
                "CSKCache requires LMCache local CPU and layer-wise modes"
            )
        if engine.storage_manager is None:
            raise RuntimeError("LMCache storage manager is not initialized")

        backends = engine.storage_manager.storage_backends
        raw_backend = backends.get("raw_block")
        local_disk_backend = backends.get("LocalDiskBackend")
        local_cpu_backend = backends.get("LocalCPUBackend")
        selected_backend = (
            raw_backend
            if settings.storage_backend == "raw_block"
            else local_disk_backend
        )
        if selected_backend is None or local_cpu_backend is None:
            raise ValueError(
                "selected CSKCache storage backend or LocalCPUBackend is unavailable"
            )
        metadata_manager = MetadataManager(
            settings.metadata_path,
            expected_layers=engine.num_layers,
        )
        validate_catalog_layout(
            metadata_manager.list_objects(),
            chunk_size_tokens=settings.chunk_size_tokens,
            storage_layout=settings.storage_layout,
        )
        host_pool = LMCacheHostBufferPool(
            local_cpu_backend,
            layout=settings.host_layout,
            chunk_size_tokens=settings.chunk_size_tokens,
        )
        local_disk_reader = (
            LMCacheLayerObjectReader(
                engine.storage_manager,
                location="LocalDiskBackend",
            )
            if settings.storage_backend == "local_disk"
            else None
        )
        if local_disk_reader is not None:
            local_disk_reader.register_catalog_objects(
                metadata_manager.list_objects()
            )
        storage_manager = StorageManager(
            metadata_manager,
            raw_backend if settings.storage_backend == "raw_block" else None,
            storage_backend=settings.storage_backend,
            local_disk_backend=local_disk_reader,
            host_buffer_pool=host_pool,
            max_inflight_loads=4,
            retain_last_host_object=settings.retain_last_host_object,
        )
        model_path = engine.metadata.model_name
        self._settings = settings
        self._engine = engine
        self._manager = RequestManager(
            metadata_manager,
            storage_manager,
            model_fingerprint=fingerprint_model(model_path),
            tokenizer_fingerprint=fingerprint_tokenizer(
                settings.tokenizer_path or model_path
            ),
            ticket_ttl_seconds=settings.ticket_ttl_seconds,
        )
        logger.info(
            "CSKCache T0 enabled: metadata=%s backend=%s model=%s",
            settings.metadata_path,
            settings.storage_backend,
            model_path,
        )

    def submit_prefetch(self, ticket: str, skill_name: str) -> bool:
        if not self._engine.is_healthy():
            return False
        profile_event(
            "csk_t0_prefetch_begin",
            ticket,
            skill_name=skill_name,
            owner="cskcache",
        )
        accepted = self._manager.select_skill(ticket, skill_name)
        profile_event(
            "csk_t0_prefetch_submit",
            ticket,
            skill_name=skill_name,
            accepted=accepted,
            owner="cskcache",
        )
        return accepted

    def inspect_tool_observation(
        self, ticket: str, tool_name: str, content: str
    ) -> bool:
        return self._manager.inspect_tool_observation(
            ticket, tool_name, content
        )

    def authenticate_request(
        self, ticket: str, request_id: str, prompt_token_ids: Any
    ) -> dict[str, object] | None:
        tokens = (
            prompt_token_ids.tolist()
            if isinstance(prompt_token_ids, torch.Tensor)
            else list(prompt_token_ids)
        )
        binding = self._manager.authenticate_and_bind(
            ticket, request_id, tokens
        )
        if binding is None:
            return None
        profile_event(
            "csk_request_bind",
            request_id,
            ticket=ticket,
            cache_object_id=binding.cache_object_id,
            segment_start=binding.segment_start,
            segment_end=binding.segment_end,
            matched_tokens=binding.segment_end - binding.segment_start,
            match_mode=binding.match_mode.value,
            matched_chunk_count=binding.matched_chunk_count,
        )
        return {
            "ticket": binding.ticket,
            "cache_object_id": binding.cache_object_id,
            "request_id": binding.request_id,
            "segment_start": binding.segment_start,
            "segment_end": binding.segment_end,
            "match_mode": binding.match_mode.value,
            "matched_chunk_count": binding.matched_chunk_count,
        }

    def prepare_reuse(
        self, ticket: str, request_id: str, block_alignment: int
    ) -> dict[str, object] | None:
        plan = self._manager.prepare_reuse(
            ticket,
            request_id,
            block_alignment=block_alignment,
            policy=self._settings.reuse_policy,
        )
        profile_event(
            "csk_reuse_plan",
            request_id,
            ticket=ticket,
            accepted=plan is not None,
            block_alignment=block_alignment,
        )
        return None if plan is None else plan.to_dict()

    def query_readiness(
        self, ticket: str, request_id: str
    ) -> dict[str, object]:
        return self._manager.query_reuse_readiness(
            ticket, request_id
        ).to_dict()

    def activate_reuse(
        self, ticket: str, request_id: str
    ) -> dict[str, object] | None:
        plan = self._manager.activate_reuse(ticket, request_id)
        profile_event(
            "csk_reuse_activate",
            request_id,
            ticket=ticket,
            accepted=plan is not None,
        )
        return None if plan is None else plan.to_dict()

    def release(self, ticket: str) -> bool:
        try:
            self._manager.release(ticket)
        except (KeyError, ValueError):
            released = False
        else:
            released = True
        profile_event(
            "csk_reuse_release",
            ticket,
            ticket=ticket,
            released=released,
            owner="cskcache",
        )
        return released

    def cancel(self, ticket: str, reason: str) -> None:
        self._manager.cancel(ticket, reason)

    def get_active_layer_buffers(
        self, ticket: str, request_id: str
    ) -> Sequence[Any]:
        return self._manager.get_active_layer_buffers(ticket, request_id)

    def mark_layer_loaded(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None:
        self._manager.mark_layer_loaded(ticket, request_id, layer_id)

    def mark_layer_corrected(
        self, ticket: str, request_id: str, layer_id: int
    ) -> None:
        self._manager.mark_layer_corrected(ticket, request_id, layer_id)

    def close(self) -> None:
        self._manager.close()
