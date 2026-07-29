from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import analyze_latency
from benchmark_common import (
    PREFIX_TOKENS,
    SEPARATOR_TOKEN_ID,
    SUFFIX_TOKENS,
    build_prompt,
    lookup_params,
    skill_tokens,
)
from validate_run import inspect_ssd


class WorkloadTest(unittest.TestCase):
    def test_source_and_target_share_only_the_skill(self) -> None:
        source, source_start, source_end, source_cache_end = build_prompt(
            skill_length=768, phase="source", arm="direct", replica=0, nonce=0
        )
        target, target_start, target_end, target_cache_end = build_prompt(
            skill_length=768, phase="measure", arm="prefix_256", replica=0, nonce=1
        )
        self.assertEqual(source_start, PREFIX_TOKENS + 1)
        self.assertEqual(source_start, target_start)
        self.assertEqual(source_end, target_end)
        self.assertEqual(source_cache_end, target_cache_end)
        self.assertNotEqual(source[:source_start], target[:target_start])
        self.assertEqual(
            source[source_start:source_cache_end],
            target[target_start:target_cache_end],
        )
        self.assertEqual(len(source), PREFIX_TOKENS + 2 + 768 + SUFFIX_TOKENS)
        self.assertEqual(source[PREFIX_TOKENS], SEPARATOR_TOKEN_ID)
        self.assertNotIn(SEPARATOR_TOKEN_ID, skill_tokens(768))

    def test_prefix_policy_is_frozen(self) -> None:
        lookup = lookup_params("prefix_256", 4097, 4867, 4866)
        assert lookup is not None
        self.assertEqual(lookup["prefix_tokens"], 256)
        self.assertEqual(lookup["calibration_start"], 132)
        self.assertEqual(lookup["calibration_end"], 256)
        self.assertEqual(lookup["minimum_reuse_tokens"], 256)
        self.assertEqual(lookup["cache_end"], 4866)
        self.assertIsNone(lookup_params("full", 4097, 4867, 4866))

    def test_target_prompt_is_identical_across_arms(self) -> None:
        prompts = []
        for arm in ("full", "direct", "prefix_no_correction", "prefix_256"):
            prompt, start, end, cache_end = build_prompt(
                skill_length=1536,
                phase="measure",
                arm=arm,
                replica=2,
                nonce=12345,
            )
            prompts.append((prompt, start, end, cache_end))
        self.assertTrue(all(candidate == prompts[0] for candidate in prompts[1:]))


class SSDValidationTest(unittest.TestCase):
    def test_multiple_complete_groups(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            for group, length in (("a", 512), ("b", 768)):
                for layer in range(2):
                    name = f"{group}@{layer}.pt"
                    data = cache / name
                    size = 2 * length * 4
                    data.write_bytes(b"x" * size)
                    payload = {
                        "data_file": name,
                        "size": size,
                        "shape": [2, length, 4],
                        "cached_positions": {
                            "kind": "range",
                            "start": 4097,
                            "length": length,
                        },
                    }
                    (cache / f"{name}.meta.json").write_text(json.dumps(payload))
            result = inspect_ssd(cache, [512, 768], layers=2)
            self.assertEqual(result["groups"], 2)
            self.assertEqual(result["layers"], 4)
            self.assertEqual(result["token_lengths"], [512, 768])


class AnalysisTest(unittest.TestCase):
    def test_cpu_pipeline_join_uses_probe_and_request_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            leaf = Path(temporary)
            log_rows = [
                (
                    "SEGMENTIA_PROFILE_EVENT",
                    {
                        "event": "segmentia_cpu_prefetch_complete",
                        "request_id": "resp:segmentia",
                        "duration_ms": 12.5,
                    },
                ),
                (
                    "SEGMENTIA_PROFILE_EVENT",
                    {
                        "event": "segmentia_cpu_activate",
                        "request_id": "resp",
                        "source_tier": "ssd",
                    },
                ),
                (
                    "SEGMENTIA_PROFILE_EVENT",
                    {
                        "event": "segmentia_storage_read",
                        "request_id": "resp",
                        "storage_tier": "cpu",
                        "duration_ms": 0.25,
                    },
                ),
                (
                    "SEGMENTIA_PROFILE_EVENT",
                    {
                        "event": "segmentia_h2d_breakdown",
                        "request_id": "resp",
                        "pure_h2d_gpu_ms": 3.5,
                    },
                ),
                (
                    "SEGMENTIA_EVENT",
                    {
                        "event": "segmentia_lookup_waiting",
                        "request_id": "resp",
                        "monotonic_ns": 1_000_000,
                    },
                ),
                (
                    "SEGMENTIA_EVENT",
                    {
                        "event": "segmentia_lookup_complete",
                        "request_id": "resp",
                        "monotonic_ns": 3_500_000,
                    },
                ),
            ]
            (leaf / "vllm.log").write_text(
                "\n".join(f"INFO {marker} {json.dumps(payload)}" for marker, payload in log_rows)
            )
            rows = analyze_latency.collect_cpu_pipeline_rows(
                leaf,
                [
                    {
                        "response_id": "resp",
                        "arm": "direct",
                        "skill_tokens": 768,
                        "kind": "warmup",
                        "ordinal": 0,
                    }
                ],
            )
            self.assertEqual(rows[0]["source_tier"], "ssd")
            self.assertEqual(rows[0]["ssd_to_cpu_ms"], 12.5)
            self.assertEqual(rows[0]["p_boundary_wait_ms"], 2.5)
            self.assertEqual(rows[0]["cpu_read_ms"], 0.25)
            self.assertEqual(rows[0]["h2d_gpu_ms"], 3.5)

    def test_sustained_break_even(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            run = root / "run"
            output = root / "result"
            latencies = {
                "full": 100.0,
                "direct": 80.0,
                "prefix_no_correction": 88.0,
                "prefix_256": 90.0,
            }
            for replica in range(3):
                for arm, latency in latencies.items():
                    leaf = run / f"replica_{replica}" / arm
                    leaf.mkdir(parents=True)
                    (leaf / "manifest.json").write_text(
                        json.dumps({"status": "completed", "arm": arm})
                    )
                    with (leaf / "timings.jsonl").open("w") as handle:
                        for length in (640, 768, 1024):
                            handle.write(
                                json.dumps(
                                    {
                                        "kind": "measure",
                                        "skill_tokens": length,
                                        "elapsed_ms": latency + replica,
                                    }
                                )
                                + "\n"
                            )
            argv = [
                "analyze_latency.py",
                "--run-dir",
                str(run),
                "--output-dir",
                str(output),
                "--replicas",
                "3",
            ]
            with patch("sys.argv", argv):
                analyze_latency.main()
            decision = json.loads(
                (output / "tables" / "break_even.json").read_text()
            )
            self.assertEqual(decision["gate"], "go")
            self.assertEqual(decision["break_even_tokens"], 640)


if __name__ == "__main__":
    unittest.main()
