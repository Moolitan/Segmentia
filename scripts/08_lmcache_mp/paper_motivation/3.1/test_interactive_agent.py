from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "interactive_agent.py"
PREFILL_MODULE_PATH = SCRIPT_DIR / "prefill_skill_pool.py"
UPGRADE_MODULE_PATH = SCRIPT_DIR / "upgrade_skill_pool_manifests.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module():
    module_name = "segmentia_interactive_agent"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_prefill_module():
    module_name = "segmentia_prefill_skill_pool"
    spec = importlib.util.spec_from_file_location(module_name, PREFILL_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_upgrade_module():
    module_name = "segmentia_upgrade_skill_pool_manifests"
    spec = importlib.util.spec_from_file_location(module_name, UPGRADE_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def _transport_call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "ok"


def seed_skill_observation(probe, skill_name: str, tool_call_id: str) -> None:
    event_type = type("ObservationEvent", (), {})
    event = event_type()
    event.id = "observation-event"
    event.timestamp = "2026-08-11T00:00:00"
    event.action_id = "action-id"
    event.tool_call_id = tool_call_id
    event.tool_name = "skill"
    event.observation = type(
        "SkillObservation", (), {"skill_name": skill_name}
    )()
    probe.on_event(event)


def test_reuse_prompt_does_not_install_skill_boundaries() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8")
    launcher = (SCRIPT_DIR / "run_interactive_agent.sh").read_text(
        encoding="utf-8"
    )
    prefill_launcher = (SCRIPT_DIR / "run.sh").read_text(encoding="utf-8")

    assert "install_skill_boundary" not in source
    assert "separator-config" not in source
    assert "<|repo_name|>" not in source
    assert "<|repo_name|>" not in launcher
    assert "<|repo_name|>" not in prefill_launcher


def test_injector_sends_the_exact_cached_object_span(monkeypatch, tmp_path) -> None:
    module = load_module()
    from skill_cache_tokens import context_segment_text

    skill_tokens = (10, 11, 12)
    marker_tokens = (10,)
    skill = module.CachedSkill(
        name="doc-coauthoring",
        skill_path=tmp_path / "SKILL.md",
        cache_id="cache-id",
        text="# Doc coauthoring\nUse the workflow.",
        token_ids=skill_tokens,
        token_sha256=module.sha256_tokens(skill_tokens),
        start_marker_token_ids=marker_tokens,
        start_marker_token_sha256=module.sha256_tokens(marker_tokens),
    )
    llm = FakeLLM()
    probe = module.SkillScheduleWindowProbe()
    seed_skill_observation(probe, skill.name, "tool-call-doc")
    module.attach_skill_kv_injector(
        llm,
        {skill.name: skill},
        probe,
    )
    wrapped = context_segment_text(skill.name, skill.text)
    messages = [{"role": "tool", "content": wrapped}]

    result = llm._transport_call(messages=messages, tools=[])

    assert result == "ok"
    assert messages[0]["content"] == wrapped
    request_kwargs = llm.calls[0][1]
    assert request_kwargs["extra_body"]["kv_transfer_params"] == {
        "lmcache_segmentia_lookup": {
            "cache_id": skill.cache_id,
            "skill_name": skill.name,
            "source_tool_call_id": "tool-call-doc",
            "token_count": len(skill_tokens),
            "token_ids_sha256": skill.token_sha256,
            "locator": {
                "kind": module.LOCATOR_KIND,
                "start_marker_token_ids": list(marker_tokens),
                "start_marker_token_count": len(marker_tokens),
                "start_marker_token_ids_sha256": (
                    skill.start_marker_token_sha256
                ),
            },
        }
    }
    event = llm._segmentia_skill_events[0]
    assert event["skill"] == skill.name
    assert event["segment_start"] is None
    assert event["segment_end"] is None
    assert event["span_owner"] == "vllm_post_tokenization_locator"
    assert "/tokenize" not in MODULE_PATH.read_text(encoding="utf-8")


def test_prefix_injector_sends_frozen_section_3_2_policy(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    skill_tokens = tuple(range(600))
    marker_tokens = skill_tokens[:4]
    skill = module.CachedSkill(
        name="doc-coauthoring",
        skill_path=tmp_path / "SKILL.md",
        cache_id="cache-id",
        text="skill body",
        token_ids=skill_tokens,
        token_sha256=module.sha256_tokens(skill_tokens),
        start_marker_token_ids=marker_tokens,
        start_marker_token_sha256=module.sha256_tokens(marker_tokens),
    )
    llm = FakeLLM()
    probe = module.SkillScheduleWindowProbe()
    seed_skill_observation(probe, skill.name, "tool-call-prefix")
    module.attach_skill_kv_injector(
        llm,
        {skill.name: skill},
        probe,
        "prefix_correction",
    )

    llm._transport_call(
        messages=[{"role": "tool", "content": "skill body"}], tools=[]
    )

    lookup = llm.calls[0][1]["extra_body"]["kv_transfer_params"][
        "lmcache_segmentia_lookup"
    ]
    assert "segment_start" not in lookup
    assert "segment_end" not in lookup
    assert "cache_end" not in lookup
    assert lookup["correction_mode"] == "prefix_k_headwise"
    assert lookup["prefix_tokens"] == 256
    assert lookup["calibration_start"] == 132
    assert lookup["calibration_end"] == 256
    assert lookup["minimum_reuse_tokens"] == 256
    assert lookup["correction_alpha"] == 0.6


def test_cache_object_is_only_the_named_context_segment() -> None:
    from skill_cache_tokens import (
        context_segment_text,
        qwen_context_segment_start_marker_token_ids,
        qwen_context_segment_token_ids,
    )

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert text == (
                '<context_segment skill_name="internal-comms">\n'
                "skill body\n"
                "</context_segment>\n"
            )
            assert add_special_tokens is False
            return [10, 11, 12]

    wrapped = context_segment_text("internal-comms", "skill body\n")
    assert "<tool_response>" not in wrapped
    assert qwen_context_segment_token_ids(
        FakeTokenizer(), "internal-comms", "skill body\n"
    ) == [10, 11, 12]

    class MarkerTokenizer:
        def encode(self, text, add_special_tokens):
            assert text == '<context_segment skill_name="internal-comms">\n'
            assert add_special_tokens is False
            return [7, 8]

    assert qwen_context_segment_start_marker_token_ids(
        MarkerTokenizer(), "internal-comms"
    ) == [7, 8]


def test_parse_args_accumulates_repeated_skills(monkeypatch) -> None:
    module = load_module()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(MODULE_PATH),
            "--skill",
            "idea-discovery",
            "--skill",
            "research-lit",
        ],
    )

    args = module.parse_args()

    assert args.skill == ["idea-discovery", "research-lit"]
    assert args.collection is None


def test_catalog_selects_skills_or_one_collection(tmp_path) -> None:
    module = load_module()
    auto_dir = tmp_path / "Auto-claude-code-research-in-sleep" / "skills"
    extra_dir = tmp_path / "skills"
    superpowers_dir = extra_dir / "superpowers" / "skills"

    def add_skill(root: Path, name: str) -> None:
        skill_dir = root / name
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {name}\n---\n", encoding="utf-8"
        )

    add_skill(auto_dir, "idea-discovery")
    add_skill(auto_dir, "research-lit")
    add_skill(superpowers_dir, "systematic-debugging")
    add_skill(extra_dir, "internal-comms")
    catalog_dir = tmp_path / "catalog"

    _, selector, count = module.build_skill_catalog(
        auto_dir,
        extra_dir,
        catalog_dir,
        skills=["internal-comms"],
    )
    assert selector == "skill:internal-comms"
    assert count == 1
    assert [path.name for path in catalog_dir.iterdir()] == ["internal-comms"]

    _, selector, count = module.build_skill_catalog(
        auto_dir,
        extra_dir,
        catalog_dir,
        skills=["idea-discovery", "research-lit"],
    )
    assert selector == "skills:idea-discovery,research-lit"
    assert count == 2
    assert sorted(path.name for path in catalog_dir.iterdir()) == [
        "idea-discovery",
        "research-lit",
    ]

    _, selector, count = module.build_skill_catalog(
        auto_dir,
        extra_dir,
        catalog_dir,
        collection=module.AUTO_RESEARCH_COLLECTION,
    )
    assert selector == "collection:Auto-claude-code-research-in-sleep"
    assert count == 2
    assert sorted(path.name for path in catalog_dir.iterdir()) == [
        "idea-discovery",
        "research-lit",
    ]

    _, selector, count = module.build_skill_catalog(
        auto_dir,
        extra_dir,
        catalog_dir,
        collection=module.SUPERPOWERS_COLLECTION,
    )
    assert selector == "collection:superpowers"
    assert count == 1
    assert [path.name for path in catalog_dir.iterdir()] == [
        "systematic-debugging"
    ]

    _, selector, count = module.build_skill_catalog(
        auto_dir,
        extra_dir,
        catalog_dir,
    )
    assert selector == "default:auto+standalone"
    assert count == 3
    assert sorted(path.name for path in catalog_dir.iterdir()) == [
        "idea-discovery",
        "internal-comms",
        "research-lit",
    ]


def test_catalog_rejects_collection_or_duplicate_passed_as_skills(tmp_path) -> None:
    module = load_module()
    with pytest.raises(RuntimeError, match="use --collection superpowers"):
        module.build_skill_catalog(
            tmp_path / "auto",
            tmp_path / "extra",
            tmp_path / "catalog",
            skills=["superpowers"],
        )
    with pytest.raises(RuntimeError, match="duplicate --skill selection"):
        module.build_skill_catalog(
            tmp_path / "auto",
            tmp_path / "extra",
            tmp_path / "catalog",
            skills=["research-lit", "research-lit"],
        )


def test_prefill_collection_excludes_exact_cache_id(tmp_path) -> None:
    module = load_prefill_module()

    def add_skill(relative_dir: str) -> None:
        skill_dir = tmp_path / relative_dir
        skill_dir.mkdir(parents=True)
        (skill_dir / "SKILL.md").write_text(
            f"---\nname: {skill_dir.name}\n---\n", encoding="utf-8"
        )

    add_skill("bundle/skills/meta-apply")
    add_skill("bundle/skills/skills-codex/meta-apply")
    add_skill("bundle/skills/research-lit")

    specs = module.discover_skills(
        tmp_path,
        collection="bundle",
        selected_skill=None,
        excluded_skills={"bundle/skills-codex/meta-apply"},
    )

    assert [spec.cache_id for spec in specs] == [
        "bundle/meta-apply",
        "bundle/research-lit",
    ]


def test_schema3_manifest_upgrade_adds_locator_without_touching_kv(tmp_path) -> None:
    module = load_upgrade_module()

    class ByteTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode("utf-8"))

    skill_path = tmp_path / "skills" / "internal-comms" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text("skill body\n", encoding="utf-8")
    skill_name = skill_path.parent.name
    full_tokens = module.qwen_context_segment_token_ids(
        ByteTokenizer(), skill_name, skill_path.read_text(encoding="utf-8")
    )
    cache_dir = tmp_path / "pool" / skill_name
    kv_dir = cache_dir / "kv"
    kv_dir.mkdir(parents=True)
    data_files = []
    for layer in range(40):
        data_name = f"layer-{layer}.pt"
        data_files.append(data_name)
        (kv_dir / data_name).write_bytes(b"unchanged-kv")
        (kv_dir / f"layer-{layer}.pt.meta.json").write_text(
            json.dumps(
                {
                    "cached_positions": {
                        "kind": "range",
                        "start": 0,
                        "length": len(full_tokens),
                    },
                    "shape": [2, len(full_tokens), 4],
                    "data_file": data_name,
                    "cache_key": f"same-cache-key@{layer}",
                }
            ),
            encoding="utf-8",
        )
    (cache_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")
    manifest_path = cache_dir / "manifest.json"
    record = {
        "schema_version": 3,
        "status": "completed",
        "cache_object": module.CACHE_OBJECT_TYPE,
        "cache_id": skill_name,
        "skill_name": skill_name,
        "skill_path": str(skill_path),
        "token_count": len(full_tokens),
        "token_ids_sha256": module.token_hash(full_tokens),
        "data_files": data_files,
    }
    manifest_path.write_text(json.dumps(record), encoding="utf-8")
    before = {
        path.name: (path.stat().st_size, path.read_bytes())
        for path in kv_dir.glob("*.pt")
    }

    upgrade = module.prepare_upgrade(
        manifest_path, record, ByteTokenizer(), expected_layers=40
    )

    assert upgrade is not None
    assert upgrade.record["schema_version"] == 4
    locator = upgrade.record["locator"]
    assert locator["kind"] == module.LOCATOR_KIND
    assert full_tokens[: locator["start_marker_token_count"]] == locator[
        "start_marker_token_ids"
    ]
    assert before == {
        path.name: (path.stat().st_size, path.read_bytes())
        for path in kv_dir.glob("*.pt")
    }

    stale = dict(record, token_count=len(full_tokens) + 1)
    with pytest.raises(RuntimeError, match="stale token_count"):
        module.prepare_upgrade(
            manifest_path, stale, ByteTokenizer(), expected_layers=40
        )

    schema2 = dict(record, schema_version=2, cache_object="qwen_tool_response")
    assert (
        module.prepare_upgrade(
            manifest_path, schema2, ByteTokenizer(), expected_layers=40
        )
        is None
    )


def test_skill_tool_wraps_only_skill_body_in_context_segment(tmp_path) -> None:
    from openhands.tools.skill.definition import SkillAction
    from openhands.tools.skill.impl import SkillExecutor

    skill_dir = tmp_path / "internal-comms"
    references_dir = skill_dir / "references"
    references_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text("skill body\n", encoding="utf-8")
    (references_dir / "guide.md").write_text("details\n", encoding="utf-8")

    executor = SkillExecutor(
        skills_dir=str(tmp_path),
        context_segment_wrapper=True,
    )
    observation = executor(SkillAction(name="internal-comms"))

    assert observation.text == (
        '<context_segment skill_name="internal-comms">\n'
        "skill body\n"
        "</context_segment>\n"
        "--- Skill Resources ---\n"
        "  references/: guide.md"
    )


def test_partial_pool_skips_schema2_but_strictly_checks_schema3(
    tmp_path,
    capsys,
) -> None:
    module = load_module()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode("utf-8"))

    skills_dir = tmp_path / "skills"
    pool_dir = tmp_path / "pool"
    valid_skill = skills_dir / "internal-comms" / "SKILL.md"
    old_skill = skills_dir / "doc-coauthoring" / "SKILL.md"
    valid_skill.parent.mkdir(parents=True)
    old_skill.parent.mkdir(parents=True)
    valid_skill.write_text("internal body\n", encoding="utf-8")
    old_skill.write_text("doc body\n", encoding="utf-8")

    valid_tokens = module.qwen_context_segment_token_ids(
        FakeTokenizer(),
        "internal-comms",
        valid_skill.read_text(encoding="utf-8"),
    )
    valid_dir = pool_dir / "internal-comms"
    valid_kv_dir = valid_dir / "kv"
    valid_kv_dir.mkdir(parents=True)
    for layer in range(40):
        (valid_kv_dir / f"layer-{layer}.pt.meta.json").write_text("{}")
    marker_tokens = module.qwen_context_segment_start_marker_token_ids(
        FakeTokenizer(), "internal-comms"
    )
    valid_record = {
        "schema_version": module.CACHE_SCHEMA_VERSION,
        "cache_object": module.CACHE_OBJECT_TYPE,
        "skill_name": "internal-comms",
        "skill_path": str(valid_skill.resolve()),
        "status": "completed",
        "cache_id": "internal-comms",
        "token_count": len(valid_tokens),
        "token_ids_sha256": module.sha256_tokens(valid_tokens),
        "locator": {
            "kind": module.LOCATOR_KIND,
            "start_marker_text": module.context_segment_start_marker_text(
                "internal-comms"
            ),
            "start_marker_token_ids": marker_tokens,
            "start_marker_token_count": len(marker_tokens),
            "start_marker_token_ids_sha256": module.sha256_tokens(marker_tokens),
        },
    }
    valid_manifest = valid_dir / "manifest.json"
    valid_manifest.write_text(json.dumps(valid_record), encoding="utf-8")
    (valid_dir / "COMPLETED").write_text("completed\n", encoding="utf-8")

    old_dir = pool_dir / "doc-coauthoring"
    old_dir.mkdir(parents=True)
    (old_dir / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 2,
                "cache_object": "qwen_tool_response",
                "skill_path": str(old_skill.resolve()),
                "status": "completed",
            }
        ),
        encoding="utf-8",
    )

    cached = module.load_cached_skills(skills_dir, pool_dir, FakeTokenizer())
    assert list(cached) == ["internal-comms"]
    output = capsys.readouterr().out
    assert "cached schema4 Skills=1" in output
    assert "uncached/incompatible Skills=1" in output
    assert "doc-coauthoring:schema-2" in output

    with pytest.raises(
        RuntimeError,
        match="explicit Skill selection requires compatible offline KV",
    ):
        module.load_cached_skills(
            skills_dir,
            pool_dir,
            FakeTokenizer(),
            require_all=True,
        )

    valid_record["skill_name"] = "wrong-name"
    valid_manifest.write_text(json.dumps(valid_record), encoding="utf-8")
    with pytest.raises(RuntimeError, match="cache metadata is invalid"):
        module.load_cached_skills(skills_dir, pool_dir, FakeTokenizer())
