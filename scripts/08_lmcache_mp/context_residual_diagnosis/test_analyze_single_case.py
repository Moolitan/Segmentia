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


if __name__ == "__main__":
    unittest.main()
