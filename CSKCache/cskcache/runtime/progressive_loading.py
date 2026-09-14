"""CSKCache-owned progress tracking for asynchronous Host loading."""

from __future__ import annotations

from dataclasses import dataclass, field
from queue import Queue
import threading
import time
from typing import Any, Callable


@dataclass(frozen=True)
class ProgressiveLoadingConfig:
    """Runtime parameters for passive transfer-rate instrumentation."""

    ssd_bandwidth_bytes_per_ms: float = 7_000_000.0
    h2d_bandwidth_bytes_per_ms: float = 22_000_000.0
    bandwidth_ewma_alpha: float = 0.2

    def __post_init__(self) -> None:
        if self.ssd_bandwidth_bytes_per_ms <= 0:
            raise ValueError("SSD bandwidth seed must be positive")
        if self.h2d_bandwidth_bytes_per_ms <= 0:
            raise ValueError("H2D bandwidth seed must be positive")
        if not 0.0 < self.bandwidth_ewma_alpha <= 1.0:
            raise ValueError("bandwidth EWMA alpha must be in (0, 1]")


@dataclass(frozen=True)
class ProgressiveLoadSnapshot:
    """One barrier-consistent view of a ticket's physical progress."""

    ticket: str
    total_layers: int
    host_ready_layers: tuple[int, ...]
    host_ready_prefix: int
    total_ssd_bytes: int
    completed_ssd_bytes: int
    completed_h2d_bytes: int
    ssd_bandwidth_bytes_per_ms: float
    h2d_bandwidth_bytes_per_ms: float
    execution_selected_at_ns: int | None
    failure_reason: str | None

    @property
    def remaining_ssd_bytes(self) -> int:
        """Return bytes not yet published as immutable Host data."""

        return max(0, self.total_ssd_bytes - self.completed_ssd_bytes)

    def to_dict(self) -> dict[str, object]:
        """Serialize scalar progress for connector control transport."""

        return {
            "ticket": self.ticket,
            "total_layers": self.total_layers,
            "host_ready_layers": list(self.host_ready_layers),
            "host_ready_prefix": self.host_ready_prefix,
            "total_ssd_bytes": self.total_ssd_bytes,
            "completed_ssd_bytes": self.completed_ssd_bytes,
            "remaining_ssd_bytes": self.remaining_ssd_bytes,
            "completed_h2d_bytes": self.completed_h2d_bytes,
            "ssd_bandwidth_bytes_per_ms": self.ssd_bandwidth_bytes_per_ms,
            "h2d_bandwidth_bytes_per_ms": self.h2d_bandwidth_bytes_per_ms,
            "execution_selected_at_ns": self.execution_selected_at_ns,
            "failure_reason": self.failure_reason,
        }


@dataclass
class _ProgressiveSession:
    ticket: str
    layer_bytes: tuple[int, ...]
    load_started_at_ns: int
    host_ready_layers: set[int] = field(default_factory=set)
    completed_ssd_bytes: int = 0
    completed_h2d_bytes: int = 0
    ssd_bandwidth_bytes_per_ms: float = 0.0
    h2d_bandwidth_bytes_per_ms: float = 0.0
    execution_selected_at_ns: int | None = None
    failure_reason: str | None = None


@dataclass(frozen=True)
class _CoordinatorEvent:
    operation: Callable[[], Any]
    done: threading.Event | None = None
    result: dict[str, Any] | None = None


class ProgressiveLoadCoordinator:
    """Track all progressive tickets on one worker-owned control thread."""

    def __init__(self, config: ProgressiveLoadingConfig | None = None) -> None:
        self.config = config or ProgressiveLoadingConfig()
        self._sessions: dict[str, _ProgressiveSession] = {}
        self._ssd_bandwidth_bytes_per_ms = (
            self.config.ssd_bandwidth_bytes_per_ms
        )
        self._h2d_bandwidth_bytes_per_ms = (
            self.config.h2d_bandwidth_bytes_per_ms
        )
        self._events: Queue[_CoordinatorEvent | None] = Queue()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            daemon=True,
            name="cskcache-progressive-load",
        )
        self._thread.start()

    def register(self, ticket: str, layer_bytes: tuple[int, ...]) -> None:
        """Register one ticket before its first layer I/O is submitted."""

        if not ticket or not layer_bytes or any(size <= 0 for size in layer_bytes):
            raise ValueError("progressive registration requires positive layer sizes")

        def operation() -> None:
            if ticket in self._sessions:
                raise ValueError(f"progressive ticket already exists: {ticket}")
            self._sessions[ticket] = _ProgressiveSession(
                ticket=ticket,
                layer_bytes=tuple(layer_bytes),
                load_started_at_ns=time.monotonic_ns(),
                ssd_bandwidth_bytes_per_ms=self._ssd_bandwidth_bytes_per_ms,
                h2d_bandwidth_bytes_per_ms=self._h2d_bandwidth_bytes_per_ms,
            )

        self._submit(operation, wait=True)

    def mark_host_ready(self, ticket: str, layer_id: int) -> None:
        """Publish one completely loaded immutable Host layer."""

        def operation() -> None:
            session = self._require_session(ticket)
            if not 0 <= layer_id < len(session.layer_bytes):
                raise ValueError("progressive Host layer is outside the model")
            if layer_id in session.host_ready_layers:
                return
            session.host_ready_layers.add(layer_id)
            session.completed_ssd_bytes += session.layer_bytes[layer_id]
            elapsed_ms = max(
                (time.monotonic_ns() - session.load_started_at_ns) / 1_000_000,
                0.001,
            )
            observed = session.completed_ssd_bytes / elapsed_ms
            session.ssd_bandwidth_bytes_per_ms = self._ewma(
                session.ssd_bandwidth_bytes_per_ms, observed
            )
            self._ssd_bandwidth_bytes_per_ms = self._ewma(
                self._ssd_bandwidth_bytes_per_ms, observed
            )

        self._submit(operation)

    def mark_h2d_complete(
        self,
        ticket: str,
        *,
        transferred_bytes: int,
        duration_ms: float,
    ) -> None:
        """Record one completed Pinned-to-GPU layer transfer."""

        if transferred_bytes <= 0 or duration_ms <= 0:
            raise ValueError("H2D samples require positive bytes and duration")

        def operation() -> None:
            session = self._require_session(ticket)
            session.completed_h2d_bytes += transferred_bytes
            observed = transferred_bytes / duration_ms
            session.h2d_bandwidth_bytes_per_ms = self._ewma(
                session.h2d_bandwidth_bytes_per_ms, observed
            )
            self._h2d_bandwidth_bytes_per_ms = self._ewma(
                self._h2d_bandwidth_bytes_per_ms, observed
            )

        self._submit(operation)

    def mark_execution_selected(self, ticket: str) -> None:
        """Record the first scheduler execution opportunity for one ticket."""

        def operation() -> None:
            session = self._require_session(ticket)
            if session.execution_selected_at_ns is None:
                session.execution_selected_at_ns = time.monotonic_ns()

        self._submit(operation)

    def mark_failed(self, ticket: str, reason: str) -> None:
        """Publish a terminal physical-load failure for one ticket."""

        if not reason:
            raise ValueError("progressive failure reason must be non-empty")

        def operation() -> None:
            self._require_session(ticket).failure_reason = reason

        self._submit(operation)

    def snapshot(self, ticket: str) -> ProgressiveLoadSnapshot:
        """Return a view ordered after every previously submitted event."""

        def operation() -> ProgressiveLoadSnapshot:
            session = self._require_session(ticket)
            prefix = 0
            while prefix in session.host_ready_layers:
                prefix += 1
            return ProgressiveLoadSnapshot(
                ticket=ticket,
                total_layers=len(session.layer_bytes),
                host_ready_layers=tuple(sorted(session.host_ready_layers)),
                host_ready_prefix=prefix,
                total_ssd_bytes=sum(session.layer_bytes),
                completed_ssd_bytes=session.completed_ssd_bytes,
                completed_h2d_bytes=session.completed_h2d_bytes,
                ssd_bandwidth_bytes_per_ms=(
                    session.ssd_bandwidth_bytes_per_ms
                ),
                h2d_bandwidth_bytes_per_ms=(
                    session.h2d_bandwidth_bytes_per_ms
                ),
                execution_selected_at_ns=session.execution_selected_at_ns,
                failure_reason=session.failure_reason,
            )

        return self._submit(operation, wait=True)

    def release(self, ticket: str) -> None:
        """Forget one terminal session after its buffers are released."""

        self._submit(lambda: self._sessions.pop(ticket, None), wait=True)

    def close(self) -> None:
        """Drain prior events and stop the worker-owned control thread."""

        if self._closed:
            return
        self._closed = True
        self._events.put(None)
        self._thread.join()

    def _submit(self, operation: Callable[[], Any], *, wait: bool = False) -> Any:
        if self._closed:
            raise RuntimeError("progressive coordinator is closed")
        if not wait:
            self._events.put(_CoordinatorEvent(operation))
            return None
        done = threading.Event()
        result: dict[str, Any] = {}
        self._events.put(_CoordinatorEvent(operation, done, result))
        done.wait()
        error = result.get("error")
        if error is not None:
            raise error
        return result.get("value")

    def _run(self) -> None:
        while True:
            event = self._events.get()
            if event is None:
                return
            try:
                value = event.operation()
            except Exception as error:
                if event.result is not None:
                    event.result["error"] = error
            else:
                if event.result is not None:
                    event.result["value"] = value
            finally:
                if event.done is not None:
                    event.done.set()

    def _require_session(self, ticket: str) -> _ProgressiveSession:
        try:
            return self._sessions[ticket]
        except KeyError as exc:
            raise KeyError(f"unknown progressive ticket: {ticket}") from exc

    def _ewma(self, previous: float, observed: float) -> float:
        alpha = self.config.bandwidth_ewma_alpha
        return alpha * observed + (1.0 - alpha) * previous
