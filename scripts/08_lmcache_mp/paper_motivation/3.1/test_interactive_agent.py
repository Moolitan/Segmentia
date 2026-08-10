from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "interactive_agent.py"
PREFILL_MODULE_PATH = SCRIPT_DIR / "prefill_skill_pool.py"
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


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def _transport_call(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return "ok"


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
    skill = module.CachedSkill(
        name="doc-coauthoring",
        skill_path=tmp_path / "SKILL.md",
        cache_id="cache-id",
        text="# Doc coauthoring\nUse the workflow.",
        token_ids=skill_tokens,
        token_sha256=module.sha256_tokens(skill_tokens),
    )
    prompt_ids = [1, 2, *skill_tokens, 20, 21]
    monkeypatch.setattr(
        module,
        "tokenize_openhands_request",
        lambda *_args, **_kwargs: prompt_ids,
    )
    llm = FakeLLM()
    module.attach_skill_kv_injector(
        llm,
        {skill.name: skill},
        "http://127.0.0.1:8014",
        "EMPTY",
        "Qwen3",
    )
    wrapped = context_segment_text(skill.name, skill.text)
    messages = [{"role": "tool", "content": wrapped}]

    result = llm._transport_call(messages=messages, tools=[])

    assert result == "ok"
    assert messages[0]["content"] == wrapped
    request_kwargs = llm.calls[0][1]
    assert request_kwargs["extra_body"]["kv_transfer_params"] == {
        "lmcache_segmentia_lookup": {
            "segment_start": 2,
            "segment_end": 5,
        }
    }
    assert llm._segmentia_skill_events == [
        {
            "skill": skill.name,
            "cache_id": skill.cache_id,
            "segment_start": 2,
            "segment_end": 5,
            "token_count": len(skill_tokens),
            "segmentia_mode": "direct_reuse",
        }
    ]


def test_prefix_injector_sends_frozen_section_3_2_policy(
    monkeypatch, tmp_path
) -> None:
    module = load_module()
    skill_tokens = tuple(range(600))
    skill = module.CachedSkill(
        name="doc-coauthoring",
        skill_path=tmp_path / "SKILL.md",
        cache_id="cache-id",
        text="skill body",
        token_ids=skill_tokens,
        token_sha256=module.sha256_tokens(skill_tokens),
    )
    prompt_ids = [900, *skill_tokens, 901]
    monkeypatch.setattr(
        module,
        "tokenize_openhands_request",
        lambda *_args, **_kwargs: prompt_ids,
    )
    llm = FakeLLM()
    module.attach_skill_kv_injector(
        llm,
        {skill.name: skill},
        "http://127.0.0.1:8014",
        "EMPTY",
        "Qwen3",
        "prefix_correction",
    )

    llm._transport_call(
        messages=[{"role": "tool", "content": "skill body"}], tools=[]
    )

    lookup = llm.calls[0][1]["extra_body"]["kv_transfer_params"][
        "lmcache_segmentia_lookup"
    ]
    assert lookup == {
        "segment_start": 1,
        "segment_end": 601,
        "cache_end": 601,
        "correction_mode": "prefix_k_headwise",
        "prefix_tokens": 256,
        "calibration_start": 132,
        "calibration_end": 256,
        "minimum_reuse_tokens": 256,
        "correction_alpha": 0.6,
    }


def test_cache_object_is_only_the_named_context_segment() -> None:
    from skill_cache_tokens import (
        context_segment_text,
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
            return [len(text), sum(text.encode("utf-8")) % 100_000]

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
    valid_record = {
        "schema_version": module.CACHE_SCHEMA_VERSION,
        "cache_object": module.CACHE_OBJECT_TYPE,
        "skill_name": "internal-comms",
        "skill_path": str(valid_skill.resolve()),
        "status": "completed",
        "cache_id": "internal-comms",
        "token_count": len(valid_tokens),
        "token_ids_sha256": module.sha256_tokens(valid_tokens),
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
    assert "cached schema3 Skills=1" in output
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
