from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "09_run_agent_skill_locator_batch.py"


def load_module():
    module_name = "segmentia_agent_skill_locator_batch"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_analyze_case_uses_executed_action_when_model_repeats_skill(
    tmp_path,
) -> None:
    module = load_module()
    request_a = "chatcmpl-segmentia-window-a"
    request_b = "chatcmpl-segmentia-window-b"
    tool_call_id = "tool-call-1"
    write_jsonl(
        tmp_path / module.TRACE_FILES["skill_actions"],
        [
            {
                "boundary": "structured_skill_action_ready",
                "request_id": request_a,
                "tool_call_id": tool_call_id,
                "skill_name": "internal-comms",
                "skill_action_ready_unix_ns": 1_000_000,
                "skill_action_ready_monotonic_ns": 1_000_000,
                "boot_id": "boot-1",
            },
            {
                "boundary": "structured_skill_action_ready",
                "request_id": request_b,
                "tool_call_id": "unexecuted-repeat",
                "skill_name": "internal-comms",
                "skill_action_ready_unix_ns": 9_000_000,
                "skill_action_ready_monotonic_ns": 9_000_000,
                "boot_id": "boot-1",
            },
        ],
    )
    write_jsonl(
        tmp_path / module.TRACE_FILES["locator"],
        [
            {
                "request_id": request_b,
                "source_tool_call_id": tool_call_id,
                "skill_name": "internal-comms",
                "status": "ok",
                "boot_id": "boot-1",
                "request_received_monotonic_ns": 3_000_000,
                "tokenization_completed_monotonic_ns": 5_000_000,
                "locator_start_monotonic_ns": 5_100_000,
                "locator_end_monotonic_ns": 5_200_000,
                "locator_start_unix_ns": 6_000_000,
                "locator_end_unix_ns": 7_000_000,
                "segment_start": 100,
                "segment_end": 439,
            }
        ],
    )
    write_jsonl(
        tmp_path / module.TRACE_FILES["scheduler"],
        [
            {
                "request_id": request_b,
                "source_tool_call_id": tool_call_id,
                "boot_id": "boot-1",
                "boundary": "immediately_before_scheduler_add_request",
                "scheduler_admission_unix_ns": 8_000_000,
                "scheduler_admission_monotonic_ns": 8_000_000,
            }
        ],
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")

    result = module.analyze_case(
        {"task": "incident", "skill": "internal-comms"},
        tmp_path,
        {
            "token_count": 339,
            "cache_bytes": 1024,
            "manifest_path": manifest_path,
        },
        "fingerprint",
    )

    assert result["t0_t1_ms"] == 2.0
    assert result["t1_t2_ms"] == 2.0
    assert result["t2_t3_ms"] == 3.0
    assert result["prefetch_window_ms"] == 7.0
    assert result["request_a_id"] == request_a
    assert result["source_tool_call_id"] == tool_call_id


def test_parse_args_accepts_analysis_only() -> None:
    module = load_module()
    assert module.parse_args([]).analysis_only is False
    assert module.parse_args(["--analysis-only"]).analysis_only is True
