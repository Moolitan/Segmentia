from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "10_run_mermaid_skill_timeline.py"


def load_module():
    name = "segmentia_mermaid_skill_timeline"
    spec = importlib.util.spec_from_file_location(name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


def test_analyze_workspace_joins_all_fine_grained_boundaries(tmp_path) -> None:
    module = load_module()
    tool_call_id = "tool-call-mermaid"
    request_a = "chatcmpl-segmentia-window-a"
    request_b = "chatcmpl-segmentia-window-b"
    boot_id = "boot-1"

    write_jsonl(
        tmp_path / module.TRACE_FILES["skill_actions"],
        [{
            "boundary": "structured_skill_action_ready",
            "request_id": request_a,
            "tool_call_id": tool_call_id,
            "skill_name": "mermaid-diagram",
            "skill_action_ready_monotonic_ns": 1_000_000,
            "boot_id": boot_id,
        }],
    )
    write_jsonl(
        tmp_path / module.TRACE_FILES["locator"],
        [{
            "request_id": request_b,
            "source_tool_call_id": tool_call_id,
            "skill_name": "mermaid-diagram",
            "status": "ok",
            "request_received_monotonic_ns": 10_000_000,
            "tokenization_completed_monotonic_ns": 11_000_000,
            "boot_id": boot_id,
        }],
    )
    write_jsonl(
        tmp_path / module.TRACE_FILES["scheduler"],
        [{
            "request_id": request_b,
            "source_tool_call_id": tool_call_id,
            "boundary": "immediately_before_scheduler_add_request",
            "scheduler_admission_monotonic_ns": 12_000_000,
            "boot_id": boot_id,
        }],
    )
    write_jsonl(
        tmp_path / module.TRACE_FILES["execution"],
        [
            {
                "boundary": "skill_tool_execution_start",
                "tool_call_id": tool_call_id,
                "monotonic_ns": 4_000_000,
                "boot_id": boot_id,
            },
            {
                "boundary": "skill_tool_execution_returned",
                "tool_call_id": tool_call_id,
                "monotonic_ns": 5_000_000,
                "boot_id": boot_id,
            },
            {
                "boundary": "skill_observation_event_created",
                "tool_call_id": tool_call_id,
                "monotonic_ns": 6_000_000,
                "boot_id": boot_id,
            },
        ],
    )
    (tmp_path / module.TRACE_FILES["agent"]).write_text(
        json.dumps(
            [
                {
                    "boundary": "skill_action_event_callback",
                    "tool_call_id": tool_call_id,
                    "monotonic_ns": 3_000_000,
                    "boot_id": boot_id,
                },
                {
                    "boundary": "skill_observation_event_callback",
                    "tool_call_id": tool_call_id,
                    "monotonic_ns": 7_000_000,
                    "boot_id": boot_id,
                },
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / module.TRACE_FILES["transport"]).write_text(
        json.dumps([
            {
                "request_id": request_a,
                "boundary": "client_transport_response_received",
                "client_response_received_monotonic_ns": 2_000_000,
                "boot_id": boot_id,
            },
            {
                "request_id": request_b,
                "boundary": "client_transport_response_received",
                "request_wrapper_enter_monotonic_ns": 8_000_000,
                "client_transport_handoff_monotonic_ns": 9_000_000,
                "boot_id": boot_id,
            },
        ]),
        encoding="utf-8",
    )

    result = module.analyze_workspace(
        tmp_path,
        repetition=1,
        task="mermaid-diagram-skill-reuse-pipeline",
        skill="mermaid-diagram",
        skill_tokens=4245,
        fingerprint="fingerprint",
    )

    assert result["t0_t3_ms"] == 11.0
    for field, _, _, _ in module.INTERVALS:
        assert result[field] == 1.0
    assert result["request_a_id"] == request_a
    assert result["request_b_id"] == request_b
    assert result["source_tool_call_id"] == tool_call_id
