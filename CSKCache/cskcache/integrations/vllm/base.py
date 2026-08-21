"""Declarative constants for the vLLM integration."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata,
)

from ...runtime.base import ReusePlan


VERIFIED_REQUEST_FIELD = "cskcache_verified"

SUBMIT_PREFETCH = "cskcache.submit_prefetch"
INSPECT_TOOL_OBSERVATION = "cskcache.inspect_tool_observation"
AUTHENTICATE_REQUEST = "cskcache.authenticate_request"
CANCEL_PREFETCH = "cskcache.cancel_prefetch"
PREPARE_REUSE = "cskcache.prepare_reuse"
QUERY_READINESS = "cskcache.query_readiness"
ACTIVATE_REUSE = "cskcache.activate_reuse"
RELEASE_REUSE = "cskcache.release_reuse"


@dataclass(frozen=True)
class CSKCacheWorkerRequest:
    """One validated CSK materialization handed from scheduler to worker."""

    plan: ReusePlan
    token_ids: tuple[int, ...]
    slot_mapping: torch.Tensor
    failed_block_ids: frozenset[int]


@dataclass
class CSKCacheConnectorMetadata(KVConnectorMetadata):
    """CSK request metadata composed with untouched LMCache metadata."""

    lmcache_metadata: KVConnectorMetadata
    requests: list[CSKCacheWorkerRequest] = field(default_factory=list)
