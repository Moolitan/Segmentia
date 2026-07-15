from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from cskcache.v1.compute import CSKProbeDecision


class CSKProbePhase(str, Enum):
    """Scheduler-visible state machine for probe-gated reuse.

    Probe mode does not immediately trust the offline skill K/V. It first lets
    vLLM normally prefill a short probe prefix, gathers the real K/V from that
    prefill on the worker, compares it to RoPE-corrected cached K/V, and then
    either loads the remaining tail or recomputes an anchor prefix before
    loading the tail.
    """

    NEED_PROBE = "need_probe"
    WAIT_PROBE = "wait_probe"
    NEED_ANCHOR = "need_anchor"
    NEED_LOAD = "need_load"
    DONE = "done"


@dataclass
class CSKProbeState:
    """Per-request scheduler state for a candidate probe-gated skill span."""

    req_id: str
    cache_id: str
    start: int
    end: int
    probe_len: int
    anchor_len: int
    tau: float
    gate_metric: str
    phase: CSKProbePhase = CSKProbePhase.NEED_PROBE
    pending_capture: str | None = None
    load_start: int | None = None
    decision: CSKProbeDecision | None = None
    gap_completed_logged: bool = False
    probe_scheduled_logged: bool = False
    anchor_scheduled_logged: bool = False
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
