from __future__ import annotations

import importlib.util
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("run_probe.py")
SPEC = importlib.util.spec_from_file_location("secondary_lookup_run_probe", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
RUN_PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUN_PROBE)


def _events() -> list[dict]:
    names = RUN_PROBE.EXPECTED_EVENT_ORDER
    events = [
        {
            "event": name,
            "request_id": "probe",
            "log_line": index,
        }
        for index, name in enumerate(names, start=1)
    ]
    by_name = {event["event"]: event for event in events}
    by_name["secondary_lookup_boundary"].update(
        segment_start=33, lookup_cursor=48, alignment=16
    )
    by_name["secondary_lookup_initial_probe"].update(
        lookup_start=0, matched_end=0
    )
    by_name["secondary_lookup_forward_complete"].update(
        num_computed_tokens=48, num_in_flight_tokens=0
    )
    by_name["secondary_lookup_pinned"].update(pinned_block_count=3)
    by_name["secondary_lookup_requeued"].update(
        num_computed_tokens=0, num_in_flight_tokens=0
    )
    by_name["secondary_lookup_external_probe"].update(
        lookup_start=33, matched_end=801, external_tokens_applied=0
    )
    by_name["secondary_lookup_local_reattach"].update(
        local_apc_reattached=True, local_apc_hit_tokens=48
    )
    by_name["secondary_lookup_unpinned"].update(pinned_block_count=0)
    return events


def test_summarize_probe_go():
    summary = RUN_PROBE.summarize_probe(_events(), expected_segment_start=33)

    assert summary["status"] == "go"
    assert summary["overlap_tokens"] == 15
    assert all(summary["checks"].values())


def test_summarize_probe_rejects_external_kv_application():
    events = _events()
    next(
        event
        for event in events
        if event["event"] == "secondary_lookup_external_probe"
    )["external_tokens_applied"] = 12

    summary = RUN_PROBE.summarize_probe(events, expected_segment_start=33)

    assert summary["status"] == "no_go"
    assert summary["checks"]["external_kv_not_applied"] is False


def test_parse_probe_events_accepts_vllm_internal_request_suffix(tmp_path):
    log_path = tmp_path / "vllm.log"
    log_path.write_text(
        "INFO SEGMENTIA_SECONDARY_LOOKUP_EVENT "
        '{"event":"secondary_lookup_boundary",'
        '"request_id":"cmpl-secondary-probe-run-0-deadbeef"}\n',
        encoding="utf-8",
    )

    events = RUN_PROBE.parse_probe_events(
        log_path, "cmpl-secondary-probe-run"
    )

    assert len(events) == 1
    assert events[0]["log_line"] == 1
