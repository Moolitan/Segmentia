from __future__ import annotations

from pathlib import Path
import sys


OFFLINE_DIR = (
    Path(__file__).parents[1] / "example" / "offline_skill_kv"
)
sys.path.insert(0, str(OFFLINE_DIR))

from prefill_skill_to_raw import cache_id_for_path, discover_skills  # noqa: E402


def test_discovery_preserves_bundle_path_for_symlinked_skill(tmp_path: Path) -> None:
    source = tmp_path / "original" / "SKILL.md"
    source.parent.mkdir()
    source.write_text("authenticated Skill body", encoding="utf-8")
    bundle = tmp_path / "bundle"
    link = bundle / "00-task" / "example-skill" / "SKILL.md"
    link.parent.mkdir(parents=True)
    link.symlink_to(source)

    specs = discover_skills(bundle, None, {"example-skill"})

    assert len(specs) == 1
    assert specs[0].source_path == link.absolute()
    assert specs[0].source_path.read_text(encoding="utf-8") == source.read_text(
        encoding="utf-8"
    )
    assert cache_id_for_path(bundle.absolute(), specs[0].source_path) == (
        "00-task/example-skill"
    )


def test_fixed_length_launcher_uses_large_ephemeral_pages_without_hot_cache() -> None:
    launcher = (
        Path(__file__).parents[1]
        / "example/paper_evaluation/section_6_3_latency_scaling/offline_cache/run.sh"
    ).read_text(encoding="utf-8")
    assert 'LMCACHE_CHUNK_SIZE="${LMCACHE_CHUNK_SIZE:-512}"' in launcher
    assert 'LMCACHE_LOCAL_CPU="${LMCACHE_LOCAL_CPU:-False}"' in launcher
    assert "FIXED_LENGTH_OFFLINE_OVERWRITE=1" in launcher
    assert '[[ ! -f "$RAW_DIR/catalog.json" ]]' in launcher
