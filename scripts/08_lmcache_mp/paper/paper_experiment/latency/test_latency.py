#!/usr/bin/env python3
"""CPU-only tests for the CSKCache latency analyzer and timing probe."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_request_timing import AgentRequestTimingProbe
from analyze_latency import (
    belongs_to_request,
    delta_ms,
    exactly_one,
    make_gpu_pipeline_figure,
    paired_prompt_deltas,
    parse_gpu_pipeline,
    percentile,
)


class FakeLLM:
    def __init__(self) -> None:
        self.calls = []

    def _transport_call(self, *args, **kwargs):
        self.calls.append(kwargs)
        return {"ok": True}


class FakeObservation:
    skill_name = "doc-coauthoring"


class FakeObservationEvent:
    __class_name__ = "ObservationEvent"
    tool_name = "skill"
    observation = FakeObservation()
    tool_call_id = "call-1"
    id = "observation-1"


class LatencyHelpersTest(unittest.TestCase):
    def test_percentile(self) -> None:
        self.assertEqual(percentile([1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertAlmostEqual(percentile([1.0, 2.0], 0.95), 1.95)

    def test_exactly_one_and_clock_delta(self) -> None:
        records = [{"event": "x", "boot_id": "b", "monotonic_ns": 1_000_000}]
        self.assertEqual(exactly_one(records, "x"), records[0])
        self.assertEqual(
            delta_ms(
                {"boot_id": "b", "monotonic_ns": 3_500_000}, records[0]
            ),
            2.5,
        )

    def test_request_b_matching_excludes_request_a(self) -> None:
        case_id = "r0-recompute-measure-0"
        request_a = f"chatcmpl-cskcache-latency-{case_id}-q1-aaaa"
        request_b = f"chatcmpl-cskcache-latency-{case_id}-q2-bbbb"

        self.assertFalse(belongs_to_request(request_a, request_b))
        self.assertTrue(belongs_to_request(request_b, request_b))
        self.assertTrue(
            belongs_to_request(f"{request_b}-engine-core-suffix", request_b)
        )
        self.assertFalse(
            belongs_to_request(f"{request_b}unexpected-suffix", request_b)
        )

    def test_paired_gate_uses_request_b_growth(self) -> None:
        recompute = {
            "request_a_prompt_tokens": 8582,
            "request_b_added_tokens": 3366,
        }
        cskcache = {
            "request_a_prompt_tokens": 8594,
            "request_b_added_tokens": 3370,
        }
        self.assertEqual(paired_prompt_deltas(recompute, cskcache), (12, 4))

        cskcache["request_b_added_tokens"] = 3375
        with self.assertRaisesRegex(ValueError, "differs by 9 tokens"):
            paired_prompt_deltas(recompute, cskcache)

    def test_shared_cuda_pipeline_requires_complete_40_layer_groups(self) -> None:
        h2d_records = []
        correction = []
        commit = []
        for layer in range(40):
            h2d_start = 0.0 if layer == 0 else 1.2 + (layer - 1) * 1.5
            h2d_end = h2d_start + 1.0
            correction_start = h2d_end
            correction_end = correction_start + 1.0
            commit_end = correction_end + 0.5
            h2d_records.append(
                {
                    "event": "cskcache_h2d_layer",
                    "layer": layer,
                    "start_ms": h2d_start,
                    "end_ms": h2d_end,
                }
            )
            correction.append(
                {
                    "layer": layer,
                    "start_ms": correction_start,
                    "end_ms": correction_end,
                }
            )
            commit.append(
                {
                    "layer": layer,
                    "start_ms": correction_end,
                    "end_ms": commit_end,
                }
            )
        intervals, metrics = parse_gpu_pipeline(
            h2d_records,
            {
                "shared_cuda_timeline": True,
                "correction_per_layer": correction,
                "commit_per_layer": commit,
            },
            {"layers": 40},
        )
        self.assertEqual(len(intervals), 120)
        self.assertGreater(metrics["gpu_pair_overlap_ms"], 0.0)
        self.assertAlmostEqual(metrics["gpu_overlap_ratio"], 1.0)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pipeline"
            representative = make_gpu_pipeline_figure(
                [
                    {
                        "mode": "cskcache",
                        "case_id": "synthetic-request",
                        "gpu_pipeline_span_ms": metrics[
                            "gpu_pipeline_span_ms"
                        ],
                        "_gpu_pipeline_intervals": intervals,
                    }
                ],
                output,
            )
            self.assertEqual(representative["case_id"], "synthetic-request")
            self.assertTrue(output.with_suffix(".pdf").is_file())
            self.assertTrue(output.with_suffix(".png").is_file())

        with self.assertRaisesRegex(ValueError, "missing=\\[39\\]"):
            parse_gpu_pipeline(
                h2d_records[:-1],
                {
                    "shared_cuda_timeline": True,
                    "correction_per_layer": correction,
                    "commit_per_layer": commit,
                },
                {"layers": 40},
            )

    def test_old_gpu_trace_requires_rerun(self) -> None:
        with self.assertRaisesRegex(ValueError, "rerun the latency experiment"):
            parse_gpu_pipeline([], {}, {"layers": 40})

    def test_probe_tags_post_skill_request(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            trace = Path(directory) / "agent.jsonl"
            with patch.dict(
                os.environ,
                {
                    "CSKCACHE_LATENCY_CASE_ID": "case-1",
                    "CSKCACHE_AGENT_TIMELINE_PATH": str(trace),
                },
            ):
                probe = AgentRequestTimingProbe()
            # The production callback checks the runtime class name. Build a
            # tiny dynamic class with the expected name rather than depending
            # on OpenHands in this CPU-only unit test.
            event_type = type("ObservationEvent", (), {})
            event = event_type()
            event.tool_name = "skill"
            event.observation = FakeObservation()
            event.tool_call_id = "call-1"
            event.id = "observation-1"
            probe.on_event(event)
            llm = FakeLLM()
            probe.attach(llm)
            llm._transport_call()
            records = [json.loads(line) for line in trace.read_text().splitlines()]
            start = exactly_one(records, "client_request_start")
            self.assertTrue(start["post_skill"])
            self.assertEqual(start["skill_observations"][0]["tool_call_id"], "call-1")
            self.assertIn("cskcache-latency-case-1", start["request_id"])


if __name__ == "__main__":
    unittest.main()
