from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Any, Mapping

from cskcache.logging import init_logger
from cskcache.profiling.config import ProfileConfig


logger = init_logger(__name__)


class ProfileReporter:
    """Emit one compact log line and an optional process-isolated JSONL file."""

    def __init__(self, config: ProfileConfig) -> None:
        self._lock = threading.Lock()
        self._jsonl_path = self._process_path(config.jsonl_path)

    @staticmethod
    def _process_path(path: Path | None) -> Path | None:
        if path is None:
            return None
        suffix = path.suffix or ".jsonl"
        stem = path.stem if path.suffix else path.name
        return path.with_name(f"{stem}.pid{os.getpid()}{suffix}")

    @property
    def jsonl_path(self) -> Path | None:
        return self._jsonl_path

    def report(self, record: Mapping[str, Any]) -> None:
        stages = record.get("stage_ms", {})
        stage_text = " ".join(
            f"{name}_ms={float(value):.3f}" for name, value in sorted(stages.items())
        )
        logger.info(
            "PROFILE kind=%s trace_id=%s req_id=%s cache_id=%s "
            "reuse_index=%s source_tier=%s target=[%s,%s) tokens=%s bytes=%s "
            "total_ms=%.3f effective_gbps=%.3f status=%s %s",
            record.get("kind"),
            record.get("trace_id"),
            record.get("req_id"),
            record.get("cache_id"),
            record.get("reuse_index"),
            record.get("source_tier", "unknown"),
            record.get("target_start"),
            record.get("target_end"),
            record.get("tokens"),
            record.get("bytes", 0),
            float(record.get("total_ms", 0.0)),
            float(record.get("effective_gbps", 0.0)),
            record.get("status"),
            stage_text,
        )

        if self._jsonl_path is None:
            return
        self._jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        line = json.dumps(dict(record), sort_keys=True) + "\n"
        with self._lock, self._jsonl_path.open("a", encoding="utf-8") as output:
            output.write(line)
