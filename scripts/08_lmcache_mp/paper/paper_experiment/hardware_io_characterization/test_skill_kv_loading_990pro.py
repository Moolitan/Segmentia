from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace


SCRIPT_DIR = Path(__file__).resolve().parent


def load_script(filename: str, module_name: str):
    path = SCRIPT_DIR / filename
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_complete_skill_layout_uses_distinct_aligned_slices(tmp_path) -> None:
    module = load_script(
        "08_measure_agent_skill_kv_loading.py", "skill_kv_loading_990pro"
    )
    layers = [
        module.LayerFile(0, tmp_path / "layer0.pt", 5),
        module.LayerFile(1, tmp_path / "layer1.pt", 7),
        module.LayerFile(2, tmp_path / "layer2.pt", 3),
    ]
    resident, used = module.layout_layers(layers, alignment=8)

    assert [item.offset for item in resident] == [0, 8, 16]
    assert used == 19
    for previous, current in zip(resident, resident[1:]):
        assert previous.offset + previous.layer.size_bytes <= current.offset


def test_loading_cases_exclude_only_two_configured_skills() -> None:
    module = load_script(
        "08_measure_agent_skill_kv_loading.py", "skill_kv_loading_cases"
    )
    skills = [
        "doc-coauthoring",
        "docx",
        "experiment-plan",
        "frontend-design",
        "idea-discovery",
        "internal-comms",
        "mcp-builder",
        "mermaid-diagram",
        "paper-writing",
        "research-refine-pipeline",
        "using-superpowers",
        "writing-systems-papers",
        "xlsx",
    ]
    config = {
        "agent_schedule": {
            "cases": [
                {"task": f"task-{index}", "skill": skill}
                for index, skill in enumerate(skills)
            ]
        },
        "agent_kv_loading_actual": {
            "excluded_skills": ["docx", "writing-systems-papers"]
        },
    }

    pairs = module.configured_pairs(config)

    assert len(pairs) == 11
    assert {skill for _, skill in pairs}.isdisjoint({"docx", "writing-systems-papers"})


def test_loading_plot_writes_pdf_and_png(tmp_path) -> None:
    module = load_script(
        "08_measure_agent_skill_kv_loading.py", "skill_kv_loading_plot"
    )
    rows = [
        {
            "skill": "short-skill",
            "skill_tokens": 400,
            "cold_ssd_to_pinned_ms": 12.0,
            "warm_page_cache_to_pinned_ms": 4.0,
            "standalone_pinned_to_gpu_ms": 2.0,
        },
        {
            "skill": "long-skill",
            "skill_tokens": 4000,
            "cold_ssd_to_pinned_ms": 90.0,
            "warm_page_cache_to_pinned_ms": 35.0,
            "standalone_pinned_to_gpu_ms": 18.0,
        },
    ]

    module.plot_loading(rows, tmp_path)

    assert (tmp_path / f"{module.OUTPUT_STEM}.pdf").is_file()
    assert (tmp_path / f"{module.OUTPUT_STEM}.png").is_file()


def test_raw_block_layout_is_aligned_and_capacity_checked() -> None:
    module = load_script("raw_skill_kv_common.py", "raw_skill_kv_layout")
    config = {
        "skill_cache": {"expected_layers": 2},
        "raw_skill_kv": {
            "file": "/tmp/raw-skill-kv.bin",
            "capacity_gib": 1,
            "block_alignment_bytes": 4096,
            "header_bytes": 4096,
            "metadata_mib": 1,
            "io_engine": "io_uring",
            "queue_depth": 64,
        },
    }
    sources = [
        SimpleNamespace(
            layers=(
                SimpleNamespace(size_bytes=5000),
                SimpleNamespace(size_bytes=9000),
            )
        )
    ]

    layout = module.build_layout(config, sources)

    assert layout["slot_bytes"] == 16384
    assert layout["required_bytes"] == 1024**2 + 2 * 16384
    assert layout["required_bytes"] <= layout["capacity_bytes"]

    config["raw_skill_kv"]["capacity_gib"] = 0.0001
    try:
        module.build_layout(config, sources)
    except ValueError as error:
        assert "raw-block file is too small" in str(error)
    else:
        raise AssertionError("undersized raw-block file was accepted")


def test_common_discovers_new_capture_manifest_without_sidecars(tmp_path) -> None:
    module = load_script("common.py", "hardware_io_common_capture_manifest")
    object_dir = tmp_path / "skill"
    kv_dir = object_dir / "kv"
    kv_dir.mkdir(parents=True)
    data_files = []
    layers = []
    for layer_id in range(4):
        filename = f"model@1@0@abc@bfloat16@{layer_id}.pt"
        data_files.append(filename)
        (kv_dir / filename).write_bytes(b"x" * 24)
        layers.append(
            {
                "layer_id": layer_id,
                "data_file": filename,
                "cache_key": f"/model@1@0@abc@bfloat16@{layer_id}",
                "size_bytes": 24,
                "dtype": "bfloat16",
                "shape": [2, 3, 2],
                "memory_layout": "KV_2TD",
                "cached_positions": {"kind": "range", "start": 0, "length": 3},
                "payload_sha256": "a" * 64,
            }
        )
    manifest_path = object_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "artifact_type": "cskcache_source_object",
                "status": "completed",
                "layer_count": 4,
                "data_files": data_files,
                "cache_dir": str(kv_dir),
                "layers": layers,
            }
        )
    )

    manifest, discovered = module.discover_manifest_layers(manifest_path, 4)

    assert manifest["layers"] == layers
    assert [layer.layer_id for layer in discovered] == [0, 1, 2, 3]
    assert all(layer.metadata is not None for layer in discovered)
    assert not list(kv_dir.glob("*.meta.json"))
