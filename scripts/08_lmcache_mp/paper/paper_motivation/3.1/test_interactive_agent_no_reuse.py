from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "interactive_agent_no_reuse.py"


def load_module():
    spec = importlib.util.spec_from_file_location(
        "interactive_agent_no_reuse", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_no_reuse_llm_options_have_no_kv_transfer_signal() -> None:
    module = load_module()
    args = argparse.Namespace(
        served_model="Qwen3",
        api_key="EMPTY",
        base_url="http://127.0.0.1:8014",
    )

    options = module.build_llm_options(args)
    rendered = repr(options)

    assert "kv_transfer_params" not in rendered
    assert "lmcache_segmentia_lookup" not in rendered


def test_no_reuse_entry_has_no_segmentia_injection_path() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")

    assert "attach_skill_kv_injector" not in source
    assert "install_skill_boundary" not in source
    assert "lmcache_segmentia_lookup" not in source
    assert '"/tokenize"' not in source


def test_no_reuse_catalog_exposes_plain_skill_directories(tmp_path: Path) -> None:
    module = load_module()
    primary = tmp_path / "primary"
    extra = tmp_path / "extra"
    catalog = tmp_path / "catalog"
    primary.mkdir()
    extra.mkdir()
    for index in range(99):
        skill_dir = primary / f"skill_{index:03d}"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            f"# Skill {index}\n", encoding="utf-8"
        )

    catalog_path, selector, count = module.build_skill_catalog(
        primary,
        extra,
        catalog,
    )

    assert catalog_path == catalog
    assert selector == "default:auto+standalone"
    assert count == 99
    assert len(list(catalog.iterdir())) == 99
    assert all(path.is_symlink() for path in catalog.iterdir())


def test_no_reuse_parse_args_accumulates_repeated_skills(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--skill",
            "paper-writing",
            "--skill",
            "paper-plan",
        ],
    )

    args = module.parse_args()

    assert args.skill == ["paper-writing", "paper-plan"]
    assert args.collection is None


def test_schedule_probe_tags_request_a_and_links_request_b() -> None:
    module = load_module()

    class FakeLLM:
        def __init__(self) -> None:
            self.calls = []

        def _transport_call(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return "ok"

    llm = FakeLLM()
    probe = module.NormalPrefillScheduleProbe()
    module.attach_normal_prefill_schedule_probe(llm, probe)

    assert llm._transport_call(messages=[]) == "ok"
    request_a_headers = llm.calls[0][1]["extra_headers"]
    assert request_a_headers["X-Request-Id"].startswith(
        module.SCHEDULE_REQUEST_PREFIX
    )
    assert probe.events == []
    assert probe.transport_events[0]["request_id"] == (
        f"chatcmpl-{request_a_headers['X-Request-Id']}"
    )
    assert probe.transport_events[0]["boundary"] == (
        "client_transport_response_received"
    )

    observation = module.SkillObservationTimestamp(
        skill_name="docx",
        event_id="event-1",
        event_timestamp="2026-08-11T00:00:00Z",
        action_id="action-1",
        tool_call_id="tool-call-1",
        callback_unix_ns=1,
    )
    downstream = module.SkillObservationTimestamp(
        skill_name="paper-compile",
        event_id="event-2",
        event_timestamp="2026-08-11T00:00:00Z",
        action_id="action-2",
        tool_call_id="tool-call-2",
        callback_unix_ns=2,
    )
    probe._pending.extend((observation, downstream))
    assert llm._transport_call(messages=[]) == "ok"

    request_b_headers = llm.calls[1][1]["extra_headers"]
    assert request_b_headers["X-Request-Id"].startswith(
        module.SCHEDULE_REQUEST_PREFIX
    )
    assert request_b_headers != request_a_headers
    assert [
        event["schedule_timing"]["tool_call_id"] for event in probe.events
    ] == ["tool-call-1", "tool-call-2"]
    assert {event["request_id"] for event in probe.events} == {
        f"chatcmpl-{request_b_headers['X-Request-Id']}"
    }
    assert len(probe.transport_events) == 2
