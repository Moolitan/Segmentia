#!/usr/bin/env python3
"""Lightweight tests for the latency workload and result guards."""
from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path

import analyze_latency
import validate_leaf
import workload


def fake_skill() -> workload.CachedSkill:
    tokens = tuple(range(2000, 2020))
    return workload.CachedSkill(
        cache_id="collection/example",
        name="example",
        tokens=tokens,
        text_sha256="text",
        token_ids_sha256=workload.token_sha256(tokens),
        manifest_path=Path("/offline/manifest.json"),
    )


class WorkloadTest(unittest.TestCase):
    def test_prompt_is_mode_independent_and_sample_dependent(self) -> None:
        skill = fake_skill()
        rows = []
        payloads = []
        for mode in workload.MODES:
            row, payload = workload.build_row(
                skill=skill,
                mode=mode,
                replica=1,
                kind="measure",
                ordinal=3,
                prefix_tokens=16,
                suffix_tokens=4,
                model="Qwen3",
            )
            rows.append(row)
            payloads.append(payload)
        self.assertEqual(len({row["prompt_sha256"] for row in rows}), 1)
        self.assertNotIn("kv_transfer_params", payloads[0])
        self.assertIn("kv_transfer_params", payloads[1])
        correction = payloads[2]["kv_transfer_params"]["lmcache_segmentia_lookup"]
        self.assertEqual(correction["prefix_tokens"], 256)
        self.assertEqual(correction["calibration_start"], 132)
        self.assertEqual(correction["correction_alpha"], 0.6)

        other, _ = workload.build_row(
            skill=skill,
            mode="full",
            replica=1,
            kind="measure",
            ordinal=4,
            prefix_tokens=16,
            suffix_tokens=4,
            model="Qwen3",
        )
        self.assertNotEqual(rows[0]["prefix_sha256"], other["prefix_sha256"])


class ValidationTest(unittest.TestCase):
    def test_direct_leaf_requires_external_apply_per_request(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            leaf = Path(temporary)
            skill = fake_skill()
            rows = []
            event_lines = []
            for kind, ordinal in (("cold", 0), ("measure", 0)):
                row, _ = workload.build_row(
                    skill=skill,
                    mode="direct",
                    replica=0,
                    kind=kind,
                    ordinal=ordinal,
                    prefix_tokens=16,
                    suffix_tokens=4,
                    model="Qwen3",
                )
                row.update(
                    status="completed",
                    elapsed_ms=10.0 + ordinal,
                    response_id=f"cmpl-{row['request_id']}",
                    completion_tokens=1,
                )
                rows.append(row)
                event = {
                    "event": "segmentia_lookup_external_apply",
                    "request_id": row["response_id"],
                    "lookup_cursor": row["segment_start"],
                    "matched_end": row["segment_end"],
                    "external_tokens_applied": row["skill_tokens"],
                }
                event_lines.append(f"INFO SEGMENTIA_EVENT {json.dumps(event)}")
            (leaf / "timings.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            (leaf / "manifest.json").write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "mode": "direct",
                        "warmups": 0,
                        "measurements": 1,
                        "skill_token_ids_sha256": skill.token_ids_sha256,
                    }
                ),
                encoding="utf-8",
            )
            log = leaf / "vllm.log"
            log.write_text("\n".join(event_lines), encoding="utf-8")
            validate_leaf.validate(
                argparse.Namespace(leaf=leaf, log=log, mode="direct")
            )
            result = json.loads((leaf / "validation.json").read_text(encoding="utf-8"))
            self.assertEqual(result["external_apply_rows"], 2)


class AnalysisTest(unittest.TestCase):
    def test_cross_mode_guard_and_aggregates(self) -> None:
        rows = []
        latencies = {"full": 100.0, "direct": 40.0, "correction": 50.0}
        for mode in workload.MODES:
            for ordinal in range(2):
                rows.append(
                    {
                        "mode": mode,
                        "replica": 0,
                        "kind": "measure",
                        "ordinal": ordinal,
                        "prompt_sha256": f"prompt-{ordinal}",
                        "prefix_sha256": f"prefix-{ordinal}",
                        "skill_sha256": "skill",
                        "prompt_tokens": 100,
                        "segment_start": 10,
                        "segment_end": 90,
                        "elapsed_ms": latencies[mode] + ordinal,
                    }
                )
        analyze_latency.validate_cross_mode(rows, replicas=1)
        summary, derived = analyze_latency.aggregate(rows, replicas=1)
        self.assertEqual(len(summary), 3)
        self.assertGreater(derived["direct_speedup_vs_full"], 2.0)
        self.assertEqual(derived["direct_faster_replicas"], 1)


if __name__ == "__main__":
    unittest.main()
