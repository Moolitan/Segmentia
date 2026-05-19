"""Skill-aware cache reuse interfaces and scheduling policy.

The goal of this module is to make the paper framework executable at the
decision layer. It models the interface we eventually want to wire into
LMCache/vLLM while staying dependency-free for offline analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Mapping, Sequence


class ReuseMode(str, Enum):
    """Available reuse decisions for one skill segment."""

    FULL_RECOMPUTE = "full_recompute"
    NAIVE_REUSE = "naive_reuse"
    ROPE_REUSE = "rope_reuse"
    SELECTIVE_RECOMPUTE = "selective_recompute"


@dataclass(frozen=True)
class SkillSegment:
    """Stable identity and token range for a reusable skill document."""

    skill_id: str
    version_hash: str
    token_hash: str
    token_range: tuple[int, int]
    skill_type: str = "generic"
    length: int | None = None

    def __post_init__(self) -> None:
        start, end = self.token_range
        if start < 0 or end <= start:
            raise ValueError(f"invalid token_range={self.token_range}")
        actual_len = end - start
        if self.length is None:
            object.__setattr__(self, "length", actual_len)
        elif self.length != actual_len:
            raise ValueError(
                f"length={self.length} does not match token_range={self.token_range}"
            )

    @property
    def start(self) -> int:
        return self.token_range[0]

    @property
    def end(self) -> int:
        return self.token_range[1]


@dataclass(frozen=True)
class SkillReusePlan:
    """Scheduler output for one skill segment."""

    segment: SkillSegment
    reuse_mode: ReuseMode
    rope_relocate: bool
    recompute_ratio: float
    check_layers: tuple[int, ...]
    query_anchor_len: int
    storage_location: str | None
    prefetch_deadline_ms: float | None
    should_admit: bool
    quality_risk: float
    estimated_load_ms: float | None
    estimated_recompute_ms: float | None
    reason: str

    @property
    def uses_cached_kv(self) -> bool:
        return self.reuse_mode is not ReuseMode.FULL_RECOMPUTE


@dataclass(frozen=True)
class SkillBlendConfig:
    """Policy knobs mirroring the proposed LMCache/vLLM configuration."""

    enable_skill_blending: bool = True
    skill_special_str: str = " # # "
    skill_min_tokens: int = 64
    skill_query_anchor_threshold: int = 64
    skill_recompute_ratios: tuple[float, float, float] = (0.05, 0.12, 0.18)
    skill_critical_mode: bool = False
    check_layers: tuple[int, ...] = (1,)
    low_risk_threshold: float = 0.25
    high_risk_threshold: float = 0.65
    cache_worthwhile_speedup: float = 1.10
    default_prefetch_deadline_ms: float | None = None

    def __post_init__(self) -> None:
        if self.skill_min_tokens < 0:
            raise ValueError("skill_min_tokens must be non-negative")
        if self.skill_query_anchor_threshold < 0:
            raise ValueError("skill_query_anchor_threshold must be non-negative")
        if len(self.skill_recompute_ratios) != 3:
            raise ValueError("skill_recompute_ratios must be (low, medium, high)")
        if any(r < 0.0 or r > 1.0 for r in self.skill_recompute_ratios):
            raise ValueError("recompute ratios must be in [0, 1]")
        if sorted(self.skill_recompute_ratios) != list(self.skill_recompute_ratios):
            raise ValueError("recompute ratios must be monotonically increasing")


@dataclass(frozen=True)
class SkillRequestContext:
    """Runtime signals used to plan reuse for one skill segment."""

    segment: SkillSegment
    cache_hit: bool
    query_anchor_len: int
    position_gap: int = 0
    storage_location: str | None = None
    estimated_load_ms: float | None = None
    estimated_full_recompute_ms: float | None = None
    estimated_selective_recompute_ms: float | None = None
    rope_supported: bool = True
    hit_count: int = 0
    request_is_critical: bool = False
    token_deviation_score: float | None = None
    gpu_busy_ratio: float | None = None
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.query_anchor_len < 0:
            raise ValueError("query_anchor_len must be non-negative")
        if self.hit_count < 0:
            raise ValueError("hit_count must be non-negative")
        if self.gpu_busy_ratio is not None and not (0.0 <= self.gpu_busy_ratio <= 1.0):
            raise ValueError("gpu_busy_ratio must be in [0, 1]")


class SkillCachePolicy:
    """Query-anchor aware SkillBlend scheduler.

    The policy is deliberately conservative. It only chooses raw reuse for
    low-risk, cache-hit cases; otherwise it moves to selective recompute or full
    recompute when cache loading is unlikely to pay off.
    """

    def __init__(self, config: SkillBlendConfig | None = None) -> None:
        self.config = config or SkillBlendConfig()

    def plan(self, ctx: SkillRequestContext) -> SkillReusePlan:
        cfg = self.config
        segment = ctx.segment
        should_admit = self._should_admit(ctx)
        prefetch_deadline = self._prefetch_deadline(ctx)

        if not cfg.enable_skill_blending:
            return self._full_recompute(
                ctx,
                should_admit=should_admit,
                risk=1.0,
                reason="skill blending disabled",
                prefetch_deadline_ms=prefetch_deadline,
            )

        if segment.length is None or segment.length < cfg.skill_min_tokens:
            return self._full_recompute(
                ctx,
                should_admit=should_admit,
                risk=0.0,
                reason=f"skill shorter than min_tokens={cfg.skill_min_tokens}",
                prefetch_deadline_ms=prefetch_deadline,
            )

        if not ctx.cache_hit:
            return self._full_recompute(
                ctx,
                should_admit=should_admit,
                risk=0.0,
                reason="cache miss; compute and optionally admit",
                prefetch_deadline_ms=prefetch_deadline,
            )

        risk = self.quality_risk(ctx)
        if self._cache_is_slower_than_recompute(ctx):
            return self._full_recompute(
                ctx,
                should_admit=should_admit,
                risk=risk,
                reason="estimated cache load is not faster than recompute",
                prefetch_deadline_ms=prefetch_deadline,
            )

        if cfg.skill_critical_mode and ctx.request_is_critical and risk >= cfg.high_risk_threshold:
            return self._full_recompute(
                ctx,
                should_admit=should_admit,
                risk=risk,
                reason="critical request with high reuse risk",
                prefetch_deadline_ms=prefetch_deadline,
            )

        if risk < cfg.low_risk_threshold:
            if ctx.rope_supported and ctx.position_gap != 0:
                mode = ReuseMode.ROPE_REUSE
                reason = "low risk with position relocation; use RoPE-corrected reuse"
                rope_relocate = True
            else:
                mode = ReuseMode.NAIVE_REUSE
                reason = "low risk and no useful RoPE relocation needed"
                rope_relocate = False
            return SkillReusePlan(
                segment=segment,
                reuse_mode=mode,
                rope_relocate=rope_relocate,
                recompute_ratio=0.0,
                check_layers=(),
                query_anchor_len=ctx.query_anchor_len,
                storage_location=ctx.storage_location,
                prefetch_deadline_ms=prefetch_deadline,
                should_admit=should_admit,
                quality_risk=risk,
                estimated_load_ms=ctx.estimated_load_ms,
                estimated_recompute_ms=0.0,
                reason=reason,
            )

        ratio = self.recompute_ratio(ctx, risk)
        return SkillReusePlan(
            segment=segment,
            reuse_mode=ReuseMode.SELECTIVE_RECOMPUTE,
            rope_relocate=ctx.rope_supported and ctx.position_gap != 0,
            recompute_ratio=ratio,
            check_layers=cfg.check_layers,
            query_anchor_len=ctx.query_anchor_len,
            storage_location=ctx.storage_location,
            prefetch_deadline_ms=prefetch_deadline,
            should_admit=should_admit,
            quality_risk=risk,
            estimated_load_ms=ctx.estimated_load_ms,
            estimated_recompute_ms=self._estimate_selective_ms(ctx, ratio),
            reason=self._selective_reason(ctx, risk, ratio),
        )

    def quality_risk(self, ctx: SkillRequestContext) -> float:
        cfg = self.config
        risk = 0.05

        if cfg.skill_query_anchor_threshold > 0:
            anchor_ratio = min(ctx.query_anchor_len / cfg.skill_query_anchor_threshold, 1.0)
            risk += 0.45 * (1.0 - anchor_ratio)

        if ctx.query_anchor_len <= 4:
            risk += 0.20
        if ctx.request_is_critical:
            risk += 0.15
        if ctx.segment.skill_type in {"tool", "code", "safety", "compliance"}:
            risk += 0.10
        if ctx.position_gap:
            risk += min(abs(ctx.position_gap) / 4096.0, 1.0) * 0.08
        if ctx.token_deviation_score is not None:
            risk += min(max(ctx.token_deviation_score, 0.0), 1.0) * 0.25
        if ctx.gpu_busy_ratio is not None and ctx.gpu_busy_ratio > 0.85:
            # Under high GPU load, extra recompute is more expensive; nudge the
            # planner away from aggressive recompute only through risk.
            risk += 0.05

        return min(risk, 1.0)

    def recompute_ratio(self, ctx: SkillRequestContext, risk: float | None = None) -> float:
        cfg = self.config
        risk = self.quality_risk(ctx) if risk is None else risk
        low, medium, high = cfg.skill_recompute_ratios
        if risk >= cfg.high_risk_threshold or ctx.query_anchor_len <= 4:
            return high
        if risk >= cfg.low_risk_threshold:
            return medium
        return low

    def _should_admit(self, ctx: SkillRequestContext) -> bool:
        length = ctx.segment.length or 0
        if length < self.config.skill_min_tokens:
            return False
        if ctx.cache_hit:
            return True
        return ctx.hit_count > 0 or length >= self.config.skill_min_tokens * 2

    def _prefetch_deadline(self, ctx: SkillRequestContext) -> float | None:
        if self.config.default_prefetch_deadline_ms is not None:
            return self.config.default_prefetch_deadline_ms
        if ctx.estimated_full_recompute_ms is None:
            return None
        return max(ctx.estimated_full_recompute_ms * 0.8, 0.0)

    def _cache_is_slower_than_recompute(self, ctx: SkillRequestContext) -> bool:
        if ctx.estimated_load_ms is None or ctx.estimated_full_recompute_ms is None:
            return False
        return ctx.estimated_load_ms * self.config.cache_worthwhile_speedup >= (
            ctx.estimated_full_recompute_ms
        )

    def _estimate_selective_ms(self, ctx: SkillRequestContext, ratio: float) -> float | None:
        if ctx.estimated_selective_recompute_ms is not None:
            return ctx.estimated_selective_recompute_ms
        if ctx.estimated_full_recompute_ms is None:
            return None
        return ctx.estimated_full_recompute_ms * ratio

    def _selective_reason(
        self,
        ctx: SkillRequestContext,
        risk: float,
        ratio: float,
    ) -> str:
        parts: list[str] = [f"risk={risk:.2f}", f"recompute_ratio={ratio:.2f}"]
        if ctx.query_anchor_len < self.config.skill_query_anchor_threshold:
            parts.append("weak query anchor")
        if ctx.request_is_critical:
            parts.append("critical request")
        if ctx.rope_supported and ctx.position_gap != 0:
            parts.append("RoPE relocate cached K before selective update")
        return "; ".join(parts)

    def _full_recompute(
        self,
        ctx: SkillRequestContext,
        *,
        should_admit: bool,
        risk: float,
        reason: str,
        prefetch_deadline_ms: float | None,
    ) -> SkillReusePlan:
        return SkillReusePlan(
            segment=ctx.segment,
            reuse_mode=ReuseMode.FULL_RECOMPUTE,
            rope_relocate=False,
            recompute_ratio=1.0,
            check_layers=(),
            query_anchor_len=ctx.query_anchor_len,
            storage_location=ctx.storage_location,
            prefetch_deadline_ms=prefetch_deadline_ms,
            should_admit=should_admit,
            quality_risk=risk,
            estimated_load_ms=ctx.estimated_load_ms,
            estimated_recompute_ms=ctx.estimated_full_recompute_ms,
            reason=reason,
        )


def summarize_plans(plans: Sequence[SkillReusePlan]) -> dict[str, object]:
    """Aggregate plans for quick CLI/report use."""

    mode_counts: dict[str, int] = {}
    ratios: list[float] = []
    risks: list[float] = []
    for plan in plans:
        mode_counts[plan.reuse_mode.value] = mode_counts.get(plan.reuse_mode.value, 0) + 1
        ratios.append(plan.recompute_ratio)
        risks.append(plan.quality_risk)
    return {
        "count": len(plans),
        "mode_counts": mode_counts,
        "mean_recompute_ratio": sum(ratios) / len(ratios) if ratios else 0.0,
        "mean_quality_risk": sum(risks) / len(risks) if risks else 0.0,
    }
