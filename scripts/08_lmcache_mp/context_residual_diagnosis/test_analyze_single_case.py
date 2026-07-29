#!/usr/bin/env python3
from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

import torch

from analyze_single_case import (
    offset_predictions,
    relocate_neox_rope,
    require_populated_raw_kv,
    singular_spectrum_metrics,
    tensor_metrics,
)
from calibration_ablation import (
    deployable_windows,
    diagnostic_windows,
    offset_predictions_for_window,
    summarize_rows,
)


class ContextResidualAnalysisTest(unittest.TestCase):
    def test_analysis_rejects_zero_raw_kv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "layer.pt"
            path.write_bytes(b"\0" * 16)
            with self.assertRaisesRegex(ValueError, "only zero bytes"):
                require_populated_raw_kv(path, chunk_bytes=5)

    def test_rope_relocation_matches_direct_target_rotation(self) -> None:
        torch.manual_seed(7)
        tokens, heads, head_dim = 9, 2, 8
        unrotated = torch.randn(tokens, heads, head_dim)
        zeros = torch.zeros(tokens, dtype=torch.int64)
        source_positions = torch.arange(13, 13 + tokens)
        target_positions = torch.arange(47, 47 + tokens)
        source = relocate_neox_rope(unrotated, zeros, source_positions, 10000.0)
        expected = relocate_neox_rope(unrotated, zeros, target_positions, 10000.0)
        relocated = relocate_neox_rope(
            source, source_positions, target_positions, 10000.0
        )
        self.assertTrue(torch.allclose(relocated, expected, atol=2e-5, rtol=2e-5))

    def test_boundary_offsets_are_evaluated_only_on_suffix(self) -> None:
        source = torch.zeros(6, 2, 3)
        target = source.clone()
        target[:, 0] += 2.0
        target[:, 1] -= 1.0
        predictions = offset_predictions(source, target, boundary_tokens=2)
        suffix = target[2:]
        direct = tensor_metrics(predictions["direct"], suffix)
        shared = tensor_metrics(predictions["layer_shared_offset"], suffix)
        headwise = tensor_metrics(predictions["headwise_offset"], suffix)
        self.assertGreater(direct["relative_l2"], 0.0)
        self.assertGreater(shared["relative_l2"], 0.0)
        self.assertAlmostEqual(headwise["relative_l2"], 0.0, places=7)

    def test_spectrum_reports_rank_one_residual(self) -> None:
        left = torch.arange(1, 8, dtype=torch.float32).unsqueeze(1)
        right = torch.tensor([[1.0, -2.0, 0.5]])
        metrics = singular_spectrum_metrics(left @ right, ranks=(1, 2))
        self.assertAlmostEqual(metrics["energy_rank_1"], 1.0, places=6)
        self.assertAlmostEqual(
            metrics["oracle_relative_error_rank_1"], 0.0, places=6
        )

    def test_deployable_windows_exclude_header_when_possible(self) -> None:
        self.assertEqual(deployable_windows(8, 8), [("full_prefix", 0, 8)])
        self.assertEqual(
            deployable_windows(32, 8),
            [
                ("full_prefix", 0, 32),
                ("body_prefix", 8, 32),
                ("tail_body", 20, 32),
            ],
        )

    def test_diagnostic_windows_are_bounded_and_deterministic(self) -> None:
        first = diagnostic_windows(100, 16, 8, 19)
        second = diagnostic_windows(100, 16, 8, 19)
        self.assertEqual(first, second)
        for _, start, end in first:
            self.assertEqual(end - start, 16)
            self.assertGreaterEqual(start, 0)
            self.assertLessEqual(end, 100)

    def test_arbitrary_calibration_window_predicts_only_masked_tokens(self) -> None:
        source = torch.zeros(10, 2, 3)
        target = source.clone()
        target[:, 0] += 2.0
        target[:, 1] -= 1.0
        evaluation_mask = torch.arange(10) >= 6
        evaluation_target, predictions = offset_predictions_for_window(
            source, target, 2, 6, evaluation_mask
        )
        self.assertEqual(evaluation_target.shape[0], 4)
        headwise = tensor_metrics(
            predictions["headwise_offset"], evaluation_target
        )
        self.assertAlmostEqual(headwise["relative_l2"], 0.0, places=7)

    def test_summary_counts_improved_layers(self) -> None:
        rows = [
            {
                "kv_type": "K",
                "baseline": "headwise_offset",
                "relative_l2": 0.2,
                "rmse": 0.1,
                "cosine": 0.9,
                "squared_error": 1.0 - improvement,
                "direct_squared_error": 1.0,
                "target_squared_norm": 4.0,
            }
            for improvement in (0.25, -0.5)
        ]
        summary = summarize_rows(rows, ("kv_type", "baseline"))
        self.assertEqual(summary[0]["layers"], 2)
        self.assertEqual(summary[0]["improved_layers"], 1)
        self.assertAlmostEqual(
            summary[0]["aggregate_improvement_vs_direct"], -0.125
        )


if __name__ == "__main__":
    unittest.main()
