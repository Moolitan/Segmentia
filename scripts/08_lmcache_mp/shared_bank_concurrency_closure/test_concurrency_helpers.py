from __future__ import annotations

import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPT_ROOT = SCRIPT_DIR.parent
CAPTURE_DIR = SCRIPT_ROOT / "cross_request_kv_capture"
sys.path[:0] = [str(SCRIPT_ROOT), str(CAPTURE_DIR)]

from shared_bank_concurrency_closure.send_concurrent import load_specs  # noqa: E402
from shared_bank_gpu_closure.prepare_requests import perturb_prefix  # noqa: E402


def test_follower_prefixes_are_distinct_and_leave_skill_unchanged() -> None:
    record = {
        "prompt_token_ids": list(range(24)),
        "effective_separator_tokens": [91, 92],
        "segment_start": 8,
    }

    variants = [perturb_prefix(record, index) for index in range(4)]

    assert len({tuple(row[:8]) for row in variants}) == 4
    assert all(row[8:] == record["prompt_token_ids"][8:] for row in variants)
    assert all(row[6:8] == record["prompt_token_ids"][6:8] for row in variants)


def test_load_specs_requires_dense_ordered_follower_roles(tmp_path: Path) -> None:
    for index in range(4):
        (tmp_path / f"follower-{index:03d}.json").write_text(
            json.dumps(
                {
                    "status": "prepared",
                    "role": f"follower-{index:03d}",
                    "request": {},
                }
            ),
            encoding="utf-8",
        )

    records = load_specs(tmp_path, 4)

    assert [row["role"] for row in records] == [
        "follower-000",
        "follower-001",
        "follower-002",
        "follower-003",
    ]
