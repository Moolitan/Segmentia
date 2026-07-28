from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from validate_capture import file_has_nonzero_bytes


SCRIPT_DIR = Path(__file__).resolve().parent


class CaptureValidationTest(unittest.TestCase):
    def test_raw_kv_zero_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            zero_path = Path(temporary) / "zero.pt"
            nonzero_path = Path(temporary) / "nonzero.pt"
            zero_path.write_bytes(b"\0" * 32)
            nonzero_path.write_bytes(b"\0" * 31 + b"\1")
            self.assertFalse(file_has_nonzero_bytes(zero_path, chunk_bytes=7))
            self.assertTrue(file_has_nonzero_bytes(nonzero_path, chunk_bytes=7))

    def test_valid_triplet_writes_completed_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            case_dir = Path(temporary) / "case"
            phases = ("source", "target_reuse", "target_full")
            for phase in phases:
                phase_dir = case_dir / phase
                phase_dir.mkdir(parents=True)
                response_id = f"chatcmpl-{phase}"
                record = {
                    "case_id": "case",
                    "phase": phase,
                    "status": "completed",
                    "skill": "skill",
                    "skill_sha256": "digest",
                    "task": "source_task" if phase == "source" else "target_task",
                    "turn": 1,
                    "invocation": 2,
                    "response_id": response_id,
                    "segment_start": 4,
                    "segment_end": 7,
                    "segment_token_count": 3,
                    "segment_token_ids": [11, 12, 13],
                    "effective_separator_tokens": [13],
                    "prompt_token_ids": (
                        [1, 2, 3, 4, 11, 12, 13]
                        if phase == "source"
                        else [8, 9, 10, 4, 11, 12, 13]
                    ),
                }
                (phase_dir / "request.json").write_text(json.dumps(record))
                if phase == "target_reuse":
                    event = {
                        "event": "segmentia_lookup_external_apply",
                        "request_id": f"{response_id}-engine",
                        "lookup_start": 4,
                        "lookup_cursor": 5,
                        "matched_end": 6,
                        "external_tokens_applied": 1,
                    }
                else:
                    event = {
                        "event": "segmentia_lookup_complete",
                        "request_id": f"{response_id}-engine",
                        "lookup_cursor": 5,
                        "external_tokens": 0,
                        "phase": "local_fallback",
                        "retained_local_tokens": 5,
                    }
                log_path = phase_dir / "vllm.log"
                with log_path.open("a") as handle:
                    recovered = 2 if phase == "target_reuse" else 0
                    groups = 1 if phase == "target_reuse" else 0
                    recovered_bytes = 24 if phase == "target_reuse" else 0
                    handle.write(
                        "Local disk rehydration complete: "
                        f"recovered_groups={groups} recovered_layers={recovered} "
                        f"recovered_bytes={recovered_bytes} invalid_sidecars=0 "
                        "incomplete_groups=0 skipped_capacity_groups=0\n"
                    )
                    handle.write(
                        "LMCache INFO: SEGMENTIA_EVENT " + json.dumps(event) + "\n"
                    )
                    if phase == "target_reuse":
                        for layer in range(2):
                            profile = {
                                "event": "segmentia_storage_read",
                                "request_id": f"{response_id}-engine",
                                "storage_tier": "ssd",
                                "key_count": 1,
                                "bytes": 8,
                                "layer": layer,
                            }
                            handle.write(
                                "LMCache INFO: SEGMENTIA_PROFILE_EVENT "
                                + json.dumps(profile)
                                + "\n"
                            )

            expected_bytes = 2 * 2 * 1 * 2 * 1
            for cache_name in ("shared_ssd", "target_full_ssd"):
                cache_dir = case_dir / cache_name
                cache_dir.mkdir()
                for layer in range(2):
                    (cache_dir / f"model@hash@bfloat16@{layer}.pt").write_bytes(
                        b"x" * expected_bytes
                    )
                    (
                        cache_dir
                        / f"model@hash@bfloat16@{layer}.pt.meta.json"
                    ).write_text(
                        json.dumps(
                            {
                                "data_file": f"model@hash@bfloat16@{layer}.pt",
                                "size": expected_bytes,
                                "shape": [2, 2, 1024],
                                "cached_positions": {
                                    "kind": "range",
                                    "start": 4,
                                    "length": 2,
                                },
                            }
                        )
                    )

            subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT_DIR / "validate_capture.py"),
                    "--case-dir",
                    str(case_dir),
                    "--layers",
                    "2",
                    "--kv-heads",
                    "1",
                    "--head-dim",
                    "2",
                    "--dtype-bytes",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            manifest = json.loads((case_dir / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "completed")
            self.assertEqual(manifest["cached_skill_token_count"], 2)
            self.assertEqual(manifest["kv_shape_per_layer"], [2, 2, 1, 2])
            self.assertEqual(manifest["rehydration"]["target_reuse"]["layers"], 2)


if __name__ == "__main__":
    unittest.main()
