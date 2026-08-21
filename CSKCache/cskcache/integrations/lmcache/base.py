"""Declarative settings for the LMCache integration."""

from __future__ import annotations

from dataclasses import dataclass

from ...runtime.base import ReusePolicy


@dataclass(frozen=True)
class LMCacheRuntimeSettings:
    """All LMCache-facing CSKCache settings, validated at one boundary."""

    metadata_path: str
    tokenizer_path: str | None
    storage_backend: str
    host_layout: str
    host_chunk_tokens: int
    ticket_ttl_seconds: float
    reuse_policy: ReusePolicy

    def __post_init__(self) -> None:
        if not self.metadata_path:
            raise ValueError("CSKCache requires cskcache_metadata_path")
        if self.tokenizer_path is not None and not self.tokenizer_path:
            raise ValueError("cskcache_tokenizer_path must not be empty")
        if self.storage_backend not in ("raw_block", "local_disk"):
            raise ValueError(
                "csk_storage_backend must be 'raw_block' or 'local_disk'"
            )
        if self.host_layout not in ("full_layer", "chunk_major"):
            raise ValueError(
                "csk_host_layout must be 'full_layer' or 'chunk_major'"
            )
        if self.host_chunk_tokens <= 0:
            raise ValueError("csk_host_chunk_tokens must be positive")
        if self.ticket_ttl_seconds <= 0:
            raise ValueError("csk_prefetch_handle_ttl_seconds must be positive")
