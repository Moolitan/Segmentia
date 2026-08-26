"""Runtime reuse lifecycle state owned by CSKCache."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from collections.abc import Mapping, Sequence
from typing import Any, Protocol


class HostLoadState(str, Enum):
    """Progress of raw-block data from SSD into long-lived pinned CPU memory."""

    NOT_STARTED = "not_started"
    LOADING = "loading"
    READY = "ready"
    FAILED = "failed"


class BindingState(str, Enum):
    """Progress of binding a prefetched object to an online request."""

    UNBOUND = "unbound"
    OBSERVED = "observed"
    VERIFIED = "verified"
    ACTIVE = "active"
    FALLBACK = "fallback"
    RELEASED = "released"


class ReuseReadiness(str, Enum):
    """Whether a verified request may enter the GPU reuse stage."""

    LOADING = "loading"
    READY = "ready"
    FALLBACK = "fallback"


class SchedulerReusePhase(str, Enum):
    """Scheduler-visible phase of one CSKCache reuse transaction."""

    INITIAL = "initial"
    WAITING = "waiting"
    ACTIVATED = "activated"
    FALLBACK = "fallback"


class LeaseOwner(str, Enum):
    """Owner responsible for releasing one request-local host lease."""

    SCHEDULER = "scheduler"
    WORKER = "worker"
    RELEASED = "released"


class SkillMatchMode(str, Enum):
    """How much of an online Skill was authenticated against one object."""

    EXACT = "exact"
    PARTIAL_PREFIX = "partial_prefix"


class CorrectionStrategy(str, Enum):
    """Online contextualization policy for one authenticated Skill span."""

    DIRECT = "direct"
    FIXED_PREFIX = "fixed_prefix"
    RATIO_PREFIX = "ratio_prefix"
    DEVIATION_TOPK = "deviation_topk"


@dataclass(frozen=True)
class ReusePolicy:
    """CSKCache token-axis correction policy for one deployment.

    All token counts are relative to the beginning of the authenticated Skill
    span.  They are converted to absolute prompt positions only when a request
    is prepared for scheduling.
    """

    minimum_full_recompute_tokens: int = 32
    calibration_tokens: int = 32
    calibration_ratio: float | None = None
    deviation_recompute_ratio: float = 0.15
    deviation_check_layer: int = 1
    minimum_reuse_tokens: int = 256
    correction_alpha: float = 0.6
    correction_strategy: CorrectionStrategy = CorrectionStrategy.FIXED_PREFIX

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        try:
            strategy = CorrectionStrategy(self.correction_strategy)
        except ValueError as exc:
            raise ValueError(
                f"unsupported correction_strategy: {self.correction_strategy}"
            ) from exc
        if self.minimum_full_recompute_tokens <= 0:
            raise ValueError("minimum_full_recompute_tokens must be > 0")
        if strategy is CorrectionStrategy.FIXED_PREFIX and self.calibration_tokens <= 0:
            raise ValueError("calibration_tokens must be > 0")
        if strategy is CorrectionStrategy.RATIO_PREFIX and (
            self.calibration_ratio is None
            or isinstance(self.calibration_ratio, bool)
            or not 0.0 < self.calibration_ratio <= 1.0
        ):
            raise ValueError(
                "ratio_prefix requires calibration_ratio in (0, 1]"
            )
        if strategy is CorrectionStrategy.DEVIATION_TOPK and (
            isinstance(self.deviation_recompute_ratio, bool)
            or not 0.0 < self.deviation_recompute_ratio <= 1.0
        ):
            raise ValueError(
                "deviation_topk requires deviation_recompute_ratio in (0, 1]"
            )
        if strategy is CorrectionStrategy.DEVIATION_TOPK and (
            isinstance(self.deviation_check_layer, bool)
            or self.deviation_check_layer < 0
        ):
            raise ValueError(
                "deviation_topk requires deviation_check_layer >= 0"
            )
        if self.minimum_reuse_tokens <= 0:
            raise ValueError("minimum_reuse_tokens must be > 0")
        if not 0.0 <= self.correction_alpha <= 1.0:
            raise ValueError("correction_alpha must be in [0, 1]")

    def resolve_calibration_tokens(self, authenticated_tokens: int) -> int:
        """Resolve the concrete contiguous budget for one request."""

        if authenticated_tokens <= 0:
            raise ValueError("authenticated_tokens must be positive")
        strategy = CorrectionStrategy(self.correction_strategy)
        if strategy in (
            CorrectionStrategy.DIRECT,
            CorrectionStrategy.DEVIATION_TOPK,
        ):
            return 0
        if strategy is CorrectionStrategy.FIXED_PREFIX:
            return self.calibration_tokens
        assert self.calibration_ratio is not None
        return max(1, math.ceil(authenticated_tokens * self.calibration_ratio))


@dataclass(frozen=True)
class ReusePlan:
    """Request-local execution plan produced after token authentication.

    ``reuse_start`` and ``reuse_end`` are absolute prompt-token positions.
    ``source_reuse_*`` address the corresponding range in the context-free
    offline object.  This is runtime state, not persistent cache metadata.
    """

    ticket: str
    cache_object_id: str
    request_id: str
    segment_start: int
    segment_end: int
    reuse_start: int
    reuse_end: int
    source_reuse_start: int
    source_reuse_end: int
    calibration_start: int
    calibration_end: int
    correction_alpha: float
    block_alignment: int
    source_token_count: int | None = None
    correction_strategy: CorrectionStrategy = CorrectionStrategy.FIXED_PREFIX
    deviation_recompute_ratio: float = 0.15
    deviation_check_layer: int = 1

    def __post_init__(self) -> None:
        """Validate once when the immutable execution plan is constructed."""

        if self.source_token_count is None:
            object.__setattr__(
                self,
                "source_token_count",
                self.segment_end - self.segment_start,
            )
        self.validate()

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ReusePlan":
        """Rebuild and validate one plan transported through LMCache.

        The scheduler/worker boundary serializes :class:`ReusePlan` as a
        plain mapping.  Reconstructing it here keeps interpretation of CSKCache
        policy fields inside CSKCache rather than duplicating it in LMCache.
        """

        if not isinstance(payload, Mapping):
            raise TypeError("CSKCache reuse plan must be a mapping")
        string_fields = ("ticket", "cache_object_id", "request_id")
        integer_fields = (
            "segment_start",
            "segment_end",
            "reuse_start",
            "reuse_end",
            "source_reuse_start",
            "source_reuse_end",
            "calibration_start",
            "calibration_end",
            "block_alignment",
        )
        values: dict[str, object] = {}
        for field_name in string_fields:
            value = payload.get(field_name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"CSKCache reuse plan requires {field_name}")
            values[field_name] = value
        for field_name in integer_fields:
            value = payload.get(field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(
                    f"CSKCache reuse plan requires integer {field_name}"
                )
            values[field_name] = value
        alpha = payload.get("correction_alpha")
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
            raise ValueError(
                "CSKCache reuse plan requires numeric correction_alpha"
            )
        values["correction_alpha"] = float(alpha)
        source_token_count = payload.get("source_token_count")
        if source_token_count is not None and (
            not isinstance(source_token_count, int)
            or isinstance(source_token_count, bool)
        ):
            raise ValueError(
                "CSKCache reuse plan requires integer source_token_count"
            )
        values["source_token_count"] = source_token_count
        strategy = payload.get(
            "correction_strategy", CorrectionStrategy.FIXED_PREFIX.value
        )
        if not isinstance(strategy, str):
            raise ValueError(
                "CSKCache reuse plan requires string correction_strategy"
            )
        try:
            values["correction_strategy"] = CorrectionStrategy(strategy)
        except ValueError as exc:
            raise ValueError(
                f"unsupported CSKCache correction_strategy: {strategy}"
            ) from exc
        ratio = payload.get("deviation_recompute_ratio", 0.15)
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError(
                "CSKCache reuse plan requires numeric deviation_recompute_ratio"
            )
        values["deviation_recompute_ratio"] = float(ratio)
        check_layer = payload.get("deviation_check_layer", 1)
        if not isinstance(check_layer, int) or isinstance(check_layer, bool):
            raise ValueError(
                "CSKCache reuse plan requires integer deviation_check_layer"
            )
        values["deviation_check_layer"] = check_layer
        return cls(**values)  # type: ignore[arg-type]

    @property
    def source_object_token_count(self) -> int:
        """Full offline object length, including an unmatched online suffix."""

        assert self.source_token_count is not None
        return self.source_token_count

    def validate(self) -> None:
        """Fail closed on malformed request-local execution ranges."""

        if not self.ticket or not self.cache_object_id or not self.request_id:
            raise ValueError("CSKCache reuse plan identities must be non-empty")
        try:
            strategy = CorrectionStrategy(self.correction_strategy)
        except ValueError as exc:
            raise ValueError(
                f"unsupported CSKCache correction_strategy: {self.correction_strategy}"
            ) from exc
        if not (
            0
            <= self.segment_start
            <= self.calibration_start
            <= self.calibration_end
            <= self.reuse_start
            < self.reuse_end
            <= self.segment_end
        ):
            raise ValueError("CSKCache reuse plan token ranges are invalid")
        if self.calibration_end != self.reuse_start:
            raise ValueError("CSKCache calibration must end at the reuse boundary")
        calibration_tokens = self.calibration_end - self.calibration_start
        no_calibration_strategies = (
            CorrectionStrategy.DIRECT,
            CorrectionStrategy.DEVIATION_TOPK,
        )
        if strategy in no_calibration_strategies and calibration_tokens != 0:
            raise ValueError(
                f"{strategy.value} reuse cannot contain calibration tokens"
            )
        if strategy not in no_calibration_strategies and calibration_tokens <= 0:
            raise ValueError("prefix correction requires calibration tokens")
        if strategy is CorrectionStrategy.DEVIATION_TOPK:
            if not 0.0 < self.deviation_recompute_ratio <= 1.0:
                raise ValueError(
                    "deviation_topk recompute ratio must be in (0, 1]"
                )
            if self.deviation_check_layer < 0:
                raise ValueError("deviation_topk check layer must be >= 0")
        if self.source_reuse_start < 0:
            raise ValueError("CSKCache source reuse range must be non-negative")
        source_position_start = self.source_reuse_start - (
            self.reuse_start - self.segment_start
        )
        if (
            self.source_object_token_count <= 0
            or source_position_start < 0
            or self.source_reuse_end
            > source_position_start + self.source_object_token_count
        ):
            raise ValueError("CSKCache source object range is invalid")
        if (
            self.source_reuse_end - self.source_reuse_start
            != self.reuse_end - self.reuse_start
        ):
            raise ValueError("CSKCache source and target reuse lengths differ")
        if self.block_alignment <= 0:
            raise ValueError("CSKCache block alignment must be positive")
        if (
            self.reuse_start % self.block_alignment
            or self.reuse_end % self.block_alignment
        ):
            raise ValueError("CSKCache reuse range must be block aligned")
        if not 0.0 <= self.correction_alpha <= 1.0:
            raise ValueError("CSKCache correction alpha must be in [0, 1]")

    def to_dict(self) -> dict[str, str | int | float]:
        return {
            "ticket": self.ticket,
            "cache_object_id": self.cache_object_id,
            "request_id": self.request_id,
            "segment_start": self.segment_start,
            "segment_end": self.segment_end,
            "reuse_start": self.reuse_start,
            "reuse_end": self.reuse_end,
            "source_reuse_start": self.source_reuse_start,
            "source_reuse_end": self.source_reuse_end,
            "calibration_start": self.calibration_start,
            "calibration_end": self.calibration_end,
            "correction_alpha": self.correction_alpha,
            "block_alignment": self.block_alignment,
            "source_token_count": self.source_object_token_count,
            "correction_strategy": CorrectionStrategy(
                self.correction_strategy
            ).value,
            "deviation_recompute_ratio": self.deviation_recompute_ratio,
            "deviation_check_layer": self.deviation_check_layer,
        }

    def failure(self, reason: str) -> "ReuseFailure":
        """Describe the exact range that must be recomputed after failure."""

        return ReuseFailure(
            ticket=self.ticket,
            request_id=self.request_id,
            token_start=self.calibration_start,
            token_end=self.reuse_end,
            reason=reason,
        )


class SchedulerControlPort(Protocol):
    """Control operations supplied by the serving-runtime integration."""

    def prepare_csk_reuse(
        self, ticket: str, request_id: str, block_alignment: int
    ) -> ReusePlan | None: ...

    def query_csk_readiness(
        self, ticket: str, request_id: str
    ) -> dict[str, Any]: ...

    def activate_csk_reuse(
        self, ticket: str, request_id: str
    ) -> ReusePlan | None: ...

    def release_csk_reuse(self, ticket: str) -> bool: ...

    def cancel_csk_prefetch(self, ticket: str, reason: str) -> None: ...


class LookupControlPort(Protocol):
    """Raw JSON control transport used by the LMCache integration."""

    def prepare_csk_reuse(
        self, ticket: str, request_id: str, block_alignment: int
    ) -> Mapping[str, object] | None: ...

    def query_csk_readiness(
        self, ticket: str, request_id: str
    ) -> dict[str, Any]: ...

    def activate_csk_reuse(
        self, ticket: str, request_id: str
    ) -> Mapping[str, object] | None: ...

    def release_csk_reuse(self, ticket: str) -> bool: ...

    def cancel_csk_prefetch(self, ticket: str, reason: str) -> None: ...


class KVBlockAllocation(Protocol):
    """Structural subset of a serving runtime's allocated KV blocks."""

    def get_block_ids(self) -> Sequence[Sequence[int]]: ...


@dataclass
class SchedulerReuseState:
    """Mutable scheduler state owned exclusively by CSKCache."""

    ticket: str
    plan: ReusePlan
    phase: SchedulerReusePhase = SchedulerReusePhase.INITIAL
    readiness: dict[str, Any] | None = None
    lease_owner: LeaseOwner = LeaseOwner.SCHEDULER


@dataclass(frozen=True)
class ActivationDirective:
    """Result of trying to activate a waiting external reuse range."""

    activated: bool
    external_tokens: int = 0


@dataclass(frozen=True)
class ReuseAllocation:
    """Validated binding between one ReusePlan and allocated PagedKV blocks."""

    plan: ReusePlan
    computed_start: int
    computed_end: int
    block_ids: tuple[int, ...]


@dataclass(frozen=True)
class ReuseFailure:
    """CSKCache decision consumed by a serving runtime's block reporter."""

    ticket: str
    request_id: str
    token_start: int
    token_end: int
    reason: str


@dataclass(frozen=True)
class FailedReuseRange:
    """One externally installed range invalidated by the worker."""

    request_id: str
    recompute_from: int
    block_ids: frozenset[int]


@dataclass(frozen=True)
class ReuseReadinessResult:
    """Scheduler-facing status of one already prepared reuse plan."""

    status: ReuseReadiness
    plan: ReusePlan | None = None
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "plan": None if self.plan is None else self.plan.to_dict(),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class RuntimeReuseState:
    """One ticket's request-independent prefetch and request-binding state."""

    ticket: str
    cache_object_id: str
    host_load_state: HostLoadState
    binding_state: BindingState
    created_at_ns: int
    deadline_ns: int | None = None
    request_id: str | None = None
    segment_start: int | None = None
    segment_end: int | None = None
    match_mode: SkillMatchMode | None = None
    matched_chunk_count: int | None = None
    source_token_count: int | None = None
    reuse_start: int | None = None
    reuse_end: int | None = None
    source_reuse_start: int | None = None
    source_reuse_end: int | None = None
    calibration_start: int | None = None
    calibration_end: int | None = None
    correction_alpha: float | None = None
    correction_strategy: CorrectionStrategy | None = None
    deviation_recompute_ratio: float | None = None
    deviation_check_layer: int | None = None
    block_alignment: int | None = None
    io_operation_id: str | None = None
    loaded_through_layer: int = -1
    corrected_through_layer: int = -1
    fallback_reason: str | None = None

    def updated(self, **changes: Any) -> "RuntimeReuseState":
        return replace(self, **changes)


@dataclass(frozen=True)
class VerifiedRequestBinding:
    """Authenticated online location of one offline Skill cache object."""

    ticket: str
    cache_object_id: str
    request_id: str
    segment_start: int
    segment_end: int
    match_mode: SkillMatchMode
    matched_chunk_count: int
