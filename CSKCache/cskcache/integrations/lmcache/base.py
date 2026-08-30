"""Declarative settings for the LMCache integration."""

from __future__ import annotations

from dataclasses import dataclass

from ...layouts import KVLayout
from ...runtime.base import ReusePolicy


@dataclass(frozen=True)
class LMCacheRuntimeSettings:
    """All LMCache-facing CSKCache settings, validated at one boundary."""

    metadata_path: str
    tokenizer_path: str | None
    storage_backend: str
    storage_layout: str
    host_layout: str
    chunk_size_tokens: int
    ticket_ttl_seconds: float | None
    reuse_policy: ReusePolicy
    retain_last_host_object: bool = False

    def __post_init__(self) -> None:
        if not self.metadata_path:
            raise ValueError("CSKCache requires cskcache_metadata_path")
        if self.tokenizer_path is not None and not self.tokenizer_path:
            raise ValueError("cskcache_tokenizer_path must not be empty")
        if self.storage_backend not in ("raw_block", "local_disk"):
            raise ValueError(
                "csk_storage_backend must be 'raw_block' or 'local_disk'"
            )
        storage_layout = KVLayout(self.storage_layout)
        host_layout = KVLayout(self.host_layout)
        if self.chunk_size_tokens <= 0:
            raise ValueError("csk_chunk_size_tokens must be positive")
        if storage_layout not in (
            KVLayout.CHUNK_SINGLE_LAYER,
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
        ):
            raise ValueError(
                "the current offline encoder supports only single-layer "
                "persistent layouts"
            )
        if host_layout not in (
            KVLayout.CHUNK_SINGLE_LAYER,
            KVLayout.PACKED_CHUNKS_SINGLE_LAYER,
        ):
            raise ValueError(
                "the current LMCache layerwise connector supports only "
                "single-layer host layouts"
            )
        if (
            self.ticket_ttl_seconds is not None
            and self.ticket_ttl_seconds <= 0
        ):
            raise ValueError("csk_prefetch_handle_ttl_seconds must be positive")
        if not isinstance(self.retain_last_host_object, bool):
            raise ValueError("csk_retain_last_host_object must be a boolean")
