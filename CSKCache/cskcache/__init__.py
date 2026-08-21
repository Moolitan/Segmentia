"""CSKCache metadata, runtime lifecycle, and storage primitives."""

from .chunking import (
    ChunkingMode,
    ChunkingSpec,
    ChunkSpan,
    SkillChunkPlan,
    build_chunk_plan,
)
from .layouts import KVLayout, KVLayoutPlan, KVRegion, build_layout_plan

from .metadata.base import (
    CacheObjectMetadata,
    CacheObjectStatus,
    ContainerMetadata,
    SKILL_PAYLOAD_FORMAT,
    SkillTokenIdentity,
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
from .metadata.skill_format import (
    ParsedSkillPayload,
    build_skill_token_identity,
    parse_skill_payload,
    render_skill_payload,
)
from .metadata.fingerprint import (
    fingerprint_model,
    fingerprint_token_ids,
    fingerprint_tokenizer,
)
from .profile import PROFILE_ENABLED, PROFILE_MARKER, profile_event
from .execution.corrector import ContextAwareKVCorrector
from .host_memory.pool import LMCacheHostBufferPool
from .storage.backends.local_disk import LMCacheLayerObjectReader
from .host_memory.base import (
    ChunkLayerBuffer,
    SingleLayerChunkBuffers,
    SingleLayerKVBuffer,
)
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
from .storage.backends.raw_block import (
    generation_sidecar_path,
    publish_generation_sidecar,
)

__all__ = [
    "BindingState",
    "ChunkingMode",
    "ChunkingSpec",
    "ChunkSpan",
    "SkillChunkPlan",
    "build_chunk_plan",
    "KVLayout",
    "KVLayoutPlan",
    "KVRegion",
    "build_layout_plan",
    "CacheBuilder",
    "CacheObjectBuildInput",
    "CacheObjectMetadata",
    "CacheObjectStatus",
    "ContainerMetadata",
    "SKILL_PAYLOAD_FORMAT",
    "ContextAwareKVCorrector",
    "SkillTokenIdentity",
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
    "ChunkLayerBuffer",
    "SingleLayerChunkBuffers",
    "SingleLayerKVBuffer",
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
    "ParsedSkillPayload",
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
    "build_skill_token_identity",
    "generation_sidecar_path",
    "fingerprint_model",
    "fingerprint_token_ids",
    "fingerprint_tokenizer",
    "parse_skill_payload",
    "publish_generation_sidecar",
    "publish_cache_snapshot",
    "publish_local_disk_snapshot",
    "render_skill_payload",
    "profile_event",
]
