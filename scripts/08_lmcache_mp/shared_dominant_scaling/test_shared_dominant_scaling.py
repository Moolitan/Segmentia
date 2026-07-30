from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = SCRIPT_ROOT / "cross_request_kv_capture"
sys.path[:0] = [str(SCRIPT_ROOT), str(CAPTURE_DIR)]

from shared_dominant_scaling.prepare_shapes import (  # noqa: E402
    SHAPES,
    main,
    theoretical_kv_gain,
)


def test_theoretical_kv_gain_reaches_two_only_after_rho_two() -> None:
    assert theoretical_kv_gain(4, 1.0) == pytest.approx(1.6)
    assert theoretical_kv_gain(4, 2.0) == pytest.approx(2.0)
    assert theoretical_kv_gain(4, 3.0) == pytest.approx(16 / 7)
    assert theoretical_kv_gain(4, 8.0) == pytest.approx(3.0)


def test_prepare_shapes_preserves_exact_geometry_and_arm_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seed = {
        "segment_token_ids": [*range(100, 807), 99],
        "effective_separator_tokens": [99],
        "request": {"prompt": list(range(1000, 5000))},
    }
    seed_path = tmp_path / "seed.json"
    seed_path.write_text(json.dumps(seed), encoding="utf-8")
    output_dir = tmp_path / "requests"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "prepare_shapes.py",
            "--seed-spec",
            str(seed_path),
            "--output-dir",
            str(output_dir),
        ],
    )
    main()

    top_manifest = json.loads((output_dir / "manifest.json").read_text())
    assert top_manifest["generator_version"] == 3

    expected = {
        "long-6k": (2048, 6144, 3.0, 16 / 7),
        "long-8k": (1024, 8192, 8.0, 3.0),
    }
    for shape in SHAPES:
        geometry = json.loads(
            (output_dir / shape.name / "manifest.json").read_text()
        )
        p, b1, rho, gain = expected[shape.name]
        assert geometry["target_p"] == p
        assert geometry["shared_b1_tokens"] == b1
        assert geometry["rho_shared_over_private"] == rho
        assert geometry["theoretical_kv_gain_n4"] == pytest.approx(gain)
        assert geometry["natural_skill_case"] is False

        source = json.loads(
            (output_dir / shape.name / "source.json").read_text()
        )
        owner = json.loads(
            (output_dir / shape.name / "reuse" / "owner.json").read_text()
        )
        full = json.loads(
            (output_dir / shape.name / "full" / "owner.json").read_text()
        )
        assert source["segment_token_hash"] == owner["segment_token_hash"]
        assert source["segment_start"] != owner["segment_start"]
        assert source["prompt_tokens"] > source["segment_end"]
        assert owner["prompt_tokens"] > owner["segment_end"]
        assert source["cache_end"] + 1 == source["segment_end"]
        assert source["request"]["prompt"][source["segment_start"] - 1] == 99
        assert source["request"]["prompt"][source["cache_end"]] == 99
        lookup = owner["request"]["kv_transfer_params"]["lmcache_segmentia_lookup"]
        assert lookup["segment_start"] + 256 == p
        assert lookup["cache_end"] - p == b1
        assert lookup["segment_end"] == lookup["cache_end"] + 1
        assert "kv_transfer_params" not in full["request"]

        follower_hashes = {
            json.loads(
                (output_dir / shape.name / "reuse" / f"follower-{i:03d}.json").read_text()
            )["request"]["prompt"][: shape.segment_start][0]
            for i in range(4)
        }
        assert len(follower_hashes) > 1
