"""CSKCache metadata, runtime lifecycle, and storage primitives."""

from .metadata.base import (
    CacheObjectMetadata,
    CacheObjectStatus,
    ContainerMetadata,
    CONTEXT_SEGMENT_FORMAT,
    ContextSegmentTokenIdentity,
    LayerExtent,
    ReadStrategy,
    StorageBackend,
    RAW_BUILD_CHECKPOINT_TYPE,
    SOURCE_ARTIFACT_TYPE,
)
from .metadata.builders import (
    CacheBuilder,
    CacheObjectBuildInput,
    DirectRawCacheBuilder,
    DirectRawCacheObjectBuildInput,
    DirectRawLayerBuildInput,
    DirectRawOffsetBackend,
    LayerBuildInput,
    LocalDiskCacheBuilder,
    LocalDiskCacheObjectBuildInput,
    LocalDiskLayerBuildInput,
    OfflineOffsetBackend,
    RawOffsetNotFoundError,
    publish_cache_snapshot,
    publish_local_disk_snapshot,
)
from .metadata.manager import MetadataManager
from .metadata.context_segment import (
    ParsedContextSegment,
    build_context_segment_token_identity,
    parse_context_segment,
    render_context_segment,
)
from .metadata.fingerprint import (
    fingerprint_model,
    fingerprint_token_ids,
    fingerprint_tokenizer,
)
from .profile import PROFILE_ENABLED, PROFILE_MARKER, profile_event
from .execution.corrector import ContextAwareKVCorrector
from .storage.buffer_pool import (
    LMCacheHostBufferPool,
    LMCacheLayerObjectReader,
)
from .storage.layouts.base import ChunkedLayerBuffer, HostLayout, LayerChunk
from .runtime.request_manager import RequestManager
from .runtime.coordinator import SchedulerReuseCoordinator
from .runtime.transport import PlanTransportCoordinator
from .execution.base import (
    ExecutionOrder,
    LayerwiseCalibrationModel,
    LayerwiseReuseStream,
    ReuseDataPlane,
    ReuseExecutionResult,
)
from .execution.executor import CSKCacheReuseExecutor
from .runtime.base import (
    BindingState,
    HostLoadState,
    ReusePlan,
    ReuseFailure,
    ReusePolicy,
    ReuseReadiness,
    ReuseReadinessResult,
    RuntimeReuseState,
    VerifiedRequestBinding,
)
from .storage.base import (
    CSKReadBatch,
    CSKReadResult,
    ExtentReadBackend,
    HostBufferPool,
    LayerObjectReadBackend,
    StorageLoader,
)
from .storage.manager import (
    StorageManager,
)
from .storage.raw_block import generation_sidecar_path, publish_generation_sidecar

__all__ = [
    "BindingState",
    "CacheBuilder",
    "CacheObjectBuildInput",
    "CacheObjectMetadata",
    "CacheObjectStatus",
    "ContainerMetadata",
    "CONTEXT_SEGMENT_FORMAT",
    "ContextAwareKVCorrector",
    "ContextSegmentTokenIdentity",
    "DirectRawCacheBuilder",
    "DirectRawCacheObjectBuildInput",
    "DirectRawLayerBuildInput",
    "DirectRawOffsetBackend",
    "CSKReadBatch",
    "CSKReadResult",
    "CSKCacheReuseExecutor",
    "ExtentReadBackend",
    "ExecutionOrder",
    "HostBufferPool",
    "LayerObjectReadBackend",
    "HostLoadState",
    "LayerExtent",
    "ChunkedLayerBuffer",
    "HostLayout",
    "LayerChunk",
    "LayerwiseReuseStream",
    "LayerwiseCalibrationModel",
    "LayerBuildInput",
    "LMCacheHostBufferPool",
    "LMCacheLayerObjectReader",
    "LocalDiskCacheBuilder",
    "LocalDiskCacheObjectBuildInput",
    "LocalDiskLayerBuildInput",
    "MetadataManager",
    "OfflineOffsetBackend",
    "ParsedContextSegment",
    "PROFILE_ENABLED",
    "PROFILE_MARKER",
    "ReadStrategy",
    "RAW_BUILD_CHECKPOINT_TYPE",
    "RawOffsetNotFoundError",
    "ReusePlan",
    "ReuseFailure",
    "ReuseDataPlane",
    "ReuseExecutionResult",
    "ReusePolicy",
    "ReuseReadiness",
    "ReuseReadinessResult",
    "RuntimeReuseState",
    "RequestManager",
    "SchedulerReuseCoordinator",
    "PlanTransportCoordinator",
    "VerifiedRequestBinding",
    "StorageManager",
    "StorageLoader",
    "StorageBackend",
    "SOURCE_ARTIFACT_TYPE",
    "build_context_segment_token_identity",
    "generation_sidecar_path",
    "fingerprint_model",
    "fingerprint_token_ids",
    "fingerprint_tokenizer",
    "parse_context_segment",
    "publish_generation_sidecar",
    "publish_cache_snapshot",
    "publish_local_disk_snapshot",
    "render_context_segment",
    "profile_event",
]
