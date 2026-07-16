from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cskcache.v1.compute import CSKProbeDecision


class CSKReuseStage(str, Enum):
    """Scheduler-visible state machine for probe-gated reuse.

    The whole cached span is scattered into the paged cache immediately, as
    soon as the span is discovered -- before any judgement about whether it
    is still fresh. vLLM's own real forward pass then runs normally over the
    probe prefix (and, if the gate fails, on to the recompute prefix),
    naturally overwriting whatever was scattered at those positions. Once
    that resolves, the scheduler frontier is advanced the rest of the way;
    nothing beyond the (small, bounded) probe/recompute prefix is ever
    reloaded or recomputed.
    """

    LOADING = "loading"
    """Span just discovered; bulk preload not dispatched yet."""

    PROBING = "probing"
    """Bulk preload dispatched; waiting for vLLM to really prefill the first
    probe_len tokens."""

    GATING = "gating"
    """Probe tokens computed; waiting for the worker to turn the residual
    into a pass/fail gate decision."""

    RECOMPUTING = "recomputing"
    """Gate failed; waiting for vLLM to really prefill on out to anchor_len
    tokens."""

    READY = "ready"
    """Gate resolved (passed, or failed-and-recomputed). The frontier still
    needs to be advanced to state.end; nothing left needs real computation
    or a fresh load, since the rest of the span was already scattered in."""

    DONE = "done"
    """Frontier confirmed at state.end."""


@dataclass
class CSKReuseState:
    """Per-request scheduler state for a candidate probe-gated skill span."""

    req_id: str
    cache_id: str
    start: int
    end: int
    probe_len: int
    anchor_len: int
    tau: float
    gate_metric: str
    stage: CSKReuseStage = CSKReuseStage.LOADING
    pending_capture: bool = False
    decision: CSKProbeDecision | None = None
    gap_completed_logged: bool = False
    probe_scheduled_logged: bool = False
    recompute_scheduled_logged: bool = False
    prefetch_hint_sent: bool = False

    @property
    def length(self) -> int:
        return self.end - self.start

    @property
    def probe_end(self) -> int:
        return self.start + self.probe_len

    @property
    def anchor_end(self) -> int:
        return self.start + self.anchor_len
