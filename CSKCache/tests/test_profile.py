"""Subprocess tests for profiling output routing."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


PROFILE_MODULE = Path(__file__).parents[1] / "cskcache" / "profile.py"
EMIT_SCRIPT = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("cskcache_profile_test", {str(PROFILE_MODULE)!r})
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.profile_event("test_event", "test-request", value=7)
"""


def _emit(tmp_path: Path, *, trace: bool, stdout: bool | None):
    trace_path = tmp_path / "profile.jsonl"
    environment = os.environ.copy()
    environment["CSKCACHE_PROFILE"] = "1"
    if trace:
        environment["CSKCACHE_PROFILE_TRACE_PATH"] = str(trace_path)
    else:
        environment.pop("CSKCACHE_PROFILE_TRACE_PATH", None)
    if stdout is None:
        environment.pop("CSKCACHE_PROFILE_STDOUT", None)
    else:
        environment["CSKCACHE_PROFILE_STDOUT"] = "1" if stdout else "0"
    result = subprocess.run(
        [sys.executable, "-c", EMIT_SCRIPT],
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return result, trace_path


def test_trace_path_suppresses_duplicate_stdout_by_default(tmp_path: Path) -> None:
    result, trace_path = _emit(tmp_path, trace=True, stdout=None)

    assert "CSKCACHE_PROFILE_EVENT" not in result.stdout
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    assert payload["event"] == "test_event"
    assert payload["request_id"] == "test-request"
    assert payload["value"] == 7


def test_profile_stdout_can_be_enabled_with_trace_file(tmp_path: Path) -> None:
    result, trace_path = _emit(tmp_path, trace=True, stdout=True)

    assert "CSKCACHE_PROFILE_EVENT" in result.stdout
    assert trace_path.is_file()


def test_profile_without_trace_keeps_stdout_default(tmp_path: Path) -> None:
    result, trace_path = _emit(tmp_path, trace=False, stdout=None)

    assert "CSKCACHE_PROFILE_EVENT" in result.stdout
    assert not trace_path.exists()
