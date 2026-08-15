"""Runtime reuse lifecycle state owned by CSKCache."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from collections.abc import Mapping
from typing import Any


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


@dataclass(frozen=True)
class ReusePolicy:
    """Fixed CSKCache token-axis correction policy for one deployment.

    All token counts are relative to the beginning of the authenticated Skill
    span.  They are converted to absolute prompt positions only when a request
    is prepared for scheduling.
    """

    prefix_tokens: int = 256
    calibration_start: int = 132
    calibration_end: int = 256
    minimum_reuse_tokens: int = 256
    correction_alpha: float = 0.6

    def validate(self) -> None:
        if self.prefix_tokens <= 0:
            raise ValueError("prefix_tokens must be > 0")
        if not 0 <= self.calibration_start < self.calibration_end:
            raise ValueError("calibration interval is invalid")
        if self.calibration_end > self.prefix_tokens:
            raise ValueError("calibration interval must be inside the prefix")
        if self.minimum_reuse_tokens <= 0:
            raise ValueError("minimum_reuse_tokens must be > 0")
        if not 0.0 <= self.correction_alpha <= 1.0:
            raise ValueError("correction_alpha must be in [0, 1]")


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
        plan = cls(**values)  # type: ignore[arg-type]
        plan.validate()
        return plan

    def validate(self) -> None:
        """Fail closed on malformed request-local execution ranges."""

        if not self.ticket or not self.cache_object_id or not self.request_id:
            raise ValueError("CSKCache reuse plan identities must be non-empty")
        if not (
            0
            <= self.segment_start
            <= self.calibration_start
            < self.calibration_end
            <= self.reuse_start
            < self.reuse_end
            <= self.segment_end
        ):
            raise ValueError("CSKCache reuse plan token ranges are invalid")
        if self.source_reuse_start < 0:
            raise ValueError("CSKCache source reuse range must be non-negative")
        if (
            self.source_reuse_end - self.source_reuse_start
            != self.reuse_end - self.reuse_start
        ):
            raise ValueError("CSKCache source and target reuse lengths differ")
        if self.block_alignment <= 0:
            raise ValueError("CSKCache block alignment must be positive")
        if self.reuse_start % self.block_alignment or self.reuse_end % self.block_alignment:
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
        }


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
    reuse_start: int | None = None
    reuse_end: int | None = None
    source_reuse_start: int | None = None
    source_reuse_end: int | None = None
    calibration_start: int | None = None
    calibration_end: int | None = None
    correction_alpha: float | None = None
    block_alignment: int | None = None
    io_operation_id: str | None = None
    storage_lease_id: str | None = None
    loaded_through_layer: int = -1
    corrected_through_layer: int = -1
    fallback_reason: str | None = None

    def updated(self, **changes: Any) -> "RuntimeReuseState":
        return replace(self, **changes)
