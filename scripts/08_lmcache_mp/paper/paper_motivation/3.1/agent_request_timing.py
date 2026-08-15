"""Shared client-side timing probe for Recompute and CSKCache Agent runs."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def _boot_id() -> str:
    try:
        return Path("/proc/sys/kernel/random/boot_id").read_text(
            encoding="utf-8"
        ).strip()
    except OSError:
        return "unavailable"


class AgentRequestTimingProbe:
    """Tag LLM requests and identify the request immediately after a Skill result."""

    def __init__(self) -> None:
        self.case_id = os.getenv("CSKCACHE_LATENCY_CASE_ID", "adhoc")
        self.trace_path = os.getenv("CSKCACHE_AGENT_TIMELINE_PATH")
        self.boot_id = _boot_id()
        self._pending_skill_results: list[dict[str, str]] = []
        self._request_ordinal = 0

    def on_event(self, event: Any) -> None:
        if (
            event.__class__.__name__ != "ObservationEvent"
            or getattr(event, "tool_name", None) != "skill"
        ):
            return
        observation = getattr(event, "observation", None)
        skill_name = str(getattr(observation, "skill_name", "")).strip()
        if not skill_name or skill_name == "list":
            return
        self._pending_skill_results.append(
            {
                "skill_name": skill_name,
                "tool_call_id": str(getattr(event, "tool_call_id", "")),
                "observation_event_id": str(getattr(event, "id", "")),
            }
        )

    def attach(self, llm: Any) -> None:
        original = getattr(llm, "_transport_call")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            self._request_ordinal += 1
            observations = self._pending_skill_results
            self._pending_skill_results = []
            post_skill = len(observations) == 1
            request_token = (
                f"cskcache-latency-{self.case_id}-q{self._request_ordinal}-"
                f"{uuid.uuid4().hex[:8]}"
            )
            request_id = f"chatcmpl-{request_token}"
            headers = dict(kwargs.get("extra_headers") or {})
            if any(key.lower() == "x-request-id" for key in headers):
                raise RuntimeError("X-Request-Id is already set on the LLM request")
            headers["X-Request-Id"] = request_token
            kwargs["extra_headers"] = headers
            self._record(
                "client_request_start",
                request_id,
                request_ordinal=self._request_ordinal,
                post_skill=post_skill,
                skill_observations=observations,
            )
            try:
                return original(*args, **kwargs)
            finally:
                self._record(
                    "client_response_received",
                    request_id,
                    request_ordinal=self._request_ordinal,
                    post_skill=post_skill,
                )

        object.__setattr__(llm, "_transport_call", wrapped)

    def _record(self, event: str, request_id: str, **fields: Any) -> None:
        if not self.trace_path:
            return
        payload = {
            "event": event,
            "request_id": request_id,
            "case_id": self.case_id,
            "boot_id": self.boot_id,
            "monotonic_ns": time.monotonic_ns(),
            "unix_ns": time.time_ns(),
            "pid": os.getpid(),
            **fields,
        }
        encoded = (
            json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        descriptor = os.open(
            self.trace_path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o644,
        )
        try:
            os.write(descriptor, encoded)
        finally:
            os.close(descriptor)
