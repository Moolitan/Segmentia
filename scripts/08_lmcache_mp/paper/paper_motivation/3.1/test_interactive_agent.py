from __future__ import annotations

import importlib.util
import hashlib
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_PATH = SCRIPT_DIR / "interactive_agent.py"
PREFILL_MODULE_PATH = SCRIPT_DIR / "prefill_skill_to_raw.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


def load_module():
    module_name = "cskcache_interactive_agent"
    spec = importlib.util.spec_from_file_location(module_name, MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def load_prefill_module():
    module_name = "cskcache_prefill_skill_pool"
    spec = importlib.util.spec_from_file_location(module_name, PREFILL_MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


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
    assert 'rglob("manifest.json")' not in source
    assert ' / "kv"' not in source
    assert "cskcache_raw" not in launcher
    assert "serve-lmcache-config" in launcher
    assert 'Qwen3-14B/raw' in source


def test_cache_object_is_only_the_named_context_segment() -> None:
    from cskcache import (
        build_context_segment_token_identity,
        render_context_segment,
    )

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            if text == (
                '<context_segment skill_name="internal-comms">\n'
                "skill body\n"
                "</context_segment>\n"
            ):
                return [10, 11, 12]
            if text == '<context_segment skill_name="internal-comms">\n':
                return [10]
            raise AssertionError(text)

    wrapped = render_context_segment("internal-comms", "skill body\n")
    assert "<tool_response>" not in wrapped
    identity = build_context_segment_token_identity(
        FakeTokenizer(), "internal-comms", "skill body\n"
    )
    assert identity.token_ids == (10, 11, 12)
    assert identity.start_marker_token_ids == (10,)


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


def test_catalog_preflight_selects_current_skill_and_rejects_stale_text(
    tmp_path,
    capsys,
    monkeypatch,
) -> None:
    module = load_module()

    class FakeTokenizer:
        def encode(self, text, add_special_tokens):
            assert add_special_tokens is False
            return list(text.encode("utf-8"))

    skills_dir = tmp_path / "skills"
    pool_dir = tmp_path / "raw"
    pool_dir.mkdir()
    model_path = tmp_path / "model"
    model_path.mkdir()
    valid_skill = skills_dir / "internal-comms" / "SKILL.md"
    missing_skill = skills_dir / "doc-coauthoring" / "SKILL.md"
    valid_skill.parent.mkdir(parents=True)
    missing_skill.parent.mkdir(parents=True)
    valid_skill.write_text("internal body\n", encoding="utf-8")
    missing_skill.write_text("doc body\n", encoding="utf-8")

    from cskcache import (
        CacheObjectMetadata,
        ContainerMetadata,
        LayerExtent,
        MetadataManager,
        ReadStrategy,
        build_context_segment_token_identity,
    )

    identity = build_context_segment_token_identity(
        FakeTokenizer(),
        "internal-comms",
        valid_skill.read_text(encoding="utf-8"),
    )
    raw_file = pool_dir / "skill_kv.bin"
    capacity = 1024 * 1024
    with raw_file.open("wb") as stream:
        stream.truncate(capacity)
    manager = MetadataManager(pool_dir / "catalog.json", expected_layers=40)
    manager.publish_container(
        ContainerMetadata(
            container_id="test-container",
            raw_file_path=str(raw_file),
            container_format_version=1,
            storage_generation="generation-1",
            capacity_bytes=capacity,
            alignment_bytes=4096,
            header_bytes=4096,
        )
    )
    manager.publish_object(
        CacheObjectMetadata(
            object_id="internal-comms:test",
            skill_name="internal-comms",
            skill_version=hashlib.sha256(
                identity.cache_text.encode("utf-8")
            ).hexdigest(),
            model_fingerprint="model-fingerprint",
            tokenizer_fingerprint="tokenizer-fingerprint",
            token_count=len(identity.token_ids),
            source_position_start=0,
            token_ids_sha256=identity.token_ids_sha256,
            start_marker_token_ids=identity.start_marker_token_ids,
            container_id="test-container",
            read_strategy=ReadStrategy.BATCHED,
            layers=tuple(
                LayerExtent(
                    layer_id=layer,
                    backend_key=f"layer-{layer}",
                    offset_bytes=4096 + layer * 8192,
                    length_bytes=256,
                    dtype="bfloat16",
                    shape=(2, len(identity.token_ids), 1),
                    memory_layout="KV_2TD",
                    payload_sha256="0" * 64,
                )
                for layer in range(40)
            ),
        )
    )
    monkeypatch.setattr(module, "fingerprint_model", lambda _path: "model-fingerprint")
    monkeypatch.setattr(
        module,
        "fingerprint_tokenizer",
        lambda _path: "tokenizer-fingerprint",
    )

    cached = module.load_cached_skills(
        skills_dir, pool_dir, FakeTokenizer(), model_path
    )
    assert list(cached) == ["internal-comms"]
    output = capsys.readouterr().out
    assert "cached CSKCache Skills=1" in output
    assert "uncached/incompatible Skills=1" in output
    assert "doc-coauthoring:missing" in output

    with pytest.raises(
        RuntimeError,
        match="explicit Skill selection requires compatible offline KV",
    ):
        module.load_cached_skills(
            skills_dir,
            pool_dir,
            FakeTokenizer(),
            model_path,
            require_all=True,
        )

    valid_skill.write_text("changed body\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Skill version is stale"):
        module.load_cached_skills(
            valid_skill.parent.parent,
            pool_dir,
            FakeTokenizer(),
            model_path,
        )
