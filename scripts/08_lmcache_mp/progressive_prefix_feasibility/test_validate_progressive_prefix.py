#!/usr/bin/env python3
from __future__ import annotations

import unittest

import torch

from validate_progressive_prefix import (
    compute_layer_head_rows,
    evaluate_feasibility_gate,
    spearman_correlation,
    summarize_case_budgets,
)


class ProgressivePrefixStatisticsTest(unittest.TestCase):
    def _rows(
        self,
        target: torch.Tensor,
        *,
        case_id: str = "case",
    ) -> list[dict]:
        return compute_layer_head_rows(
            source=torch.zeros_like(target),
            target=target,
            case_id=case_id,
            split="heldout",
            layer=0,
            prefix_endpoints=(4, 6, 8),
            calibration_start=2,
            common_evaluation_start=8,
        )

    def test_shared_shift_is_recovered(self) -> None:
        target = torch.full((12, 2, 4), 3.0)
        rows = self._rows(target)
        final = [row for row in rows if row["prefix_end"] == 8]
        self.assertEqual(len(final), 2)
        for row in final:
            self.assertAlmostEqual(row["common_corrected_squared_error"], 0.0)
            self.assertAlmostEqual(row["tail_offset_relative_error"], 0.0)
            self.assertAlmostEqual(row["observable_relative_change"], 0.0)

    def test_tail_cannot_change_observable_prefix_signal(self) -> None:
        target_a = torch.ones((12, 1, 4))
        target_b = target_a.clone()
        target_b[8:] = 9.0
        rows_a = self._rows(target_a)
        rows_b = self._rows(target_b)
        for left, right in zip(rows_a, rows_b, strict=True):
            self.assertEqual(left["prefix_end"], right["prefix_end"])
            self.assertEqual(
                left["observable_relative_change"],
                right["observable_relative_change"],
            )
            self.assertEqual(left["observable_cosine"], right["observable_cosine"])
        self.assertNotEqual(
            rows_a[-1]["tail_offset_relative_error"],
            rows_b[-1]["tail_offset_relative_error"],
        )

    def test_heterogeneous_prefix_is_less_stable(self) -> None:
        stable = torch.ones((12, 1, 4))
        unstable = stable.clone()
        unstable[4:6] = -1.0
        stable_rows = self._rows(stable)
        unstable_rows = self._rows(unstable)
        stable_at_six = next(row for row in stable_rows if row["prefix_end"] == 6)
        unstable_at_six = next(row for row in unstable_rows if row["prefix_end"] == 6)
        self.assertGreater(
            unstable_at_six["observable_relative_change"],
            stable_at_six["observable_relative_change"],
        )

    def test_invalid_endpoint_crossing_common_tail_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot cross"):
            compute_layer_head_rows(
                source=torch.zeros((12, 1, 4)),
                target=torch.ones((12, 1, 4)),
                case_id="case",
                split="design",
                layer=0,
                prefix_endpoints=(4, 10),
                calibration_start=2,
                common_evaluation_start=8,
            )


class GateTest(unittest.TestCase):
    def test_spearman_handles_ties_and_direction(self) -> None:
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [4.0, 5.0, 6.0]), 1.0
        )
        self.assertAlmostEqual(
            spearman_correlation([1.0, 2.0, 3.0], [6.0, 5.0, 4.0]), -1.0
        )
        self.assertIsNone(spearman_correlation([1.0, 1.0], [2.0, 3.0]))

    def test_gate_reports_go_for_transfer_observability_and_early_budget(self) -> None:
        rows: list[dict] = []
        summaries: list[dict] = []
        for case_index in range(3):
            case_id = f"case-{case_index}"
            for endpoint, change, error in (
                (160, None, 0.8),
                (192, 0.6, 0.6),
                (224, 0.3, 0.3),
                (256, 0.1, 0.1),
            ):
                rows.append(
                    {
                        "case_id": case_id,
                        "observable_relative_change": change,
                        "tail_offset_relative_error": error,
                    }
                )
                summaries.append(
                    {
                        "case_id": case_id,
                        "split": "heldout",
                        "prefix_end": endpoint,
                        "common_improvement_vs_direct": (
                            0.85 if endpoint < 256 else 1.0
                        ),
                    }
                )
        gate = evaluate_feasibility_gate(rows, summaries)
        self.assertEqual(gate["status"], "go")

    def test_gate_reports_weak_go_when_stability_is_uninformative(self) -> None:
        rows: list[dict] = []
        summaries: list[dict] = []
        for case_index in range(3):
            case_id = f"case-{case_index}"
            for endpoint, change, error in (
                (160, None, 0.2),
                (192, 0.1, 0.6),
                (224, 0.2, 0.4),
                (256, 0.3, 0.2),
            ):
                rows.append(
                    {
                        "case_id": case_id,
                        "observable_relative_change": change,
                        "tail_offset_relative_error": error,
                    }
                )
                summaries.append(
                    {
                        "case_id": case_id,
                        "split": "heldout",
                        "prefix_end": endpoint,
                        "common_improvement_vs_direct": (
                            0.1 if endpoint < 256 else 0.5
                        ),
                    }
                )
        gate = evaluate_feasibility_gate(rows, summaries)
        self.assertEqual(gate["status"], "weak_go")

    def test_real_summary_aggregation_uses_summed_squared_error(self) -> None:
        target = torch.ones((12, 1, 4))
        rows = self._synthetic_rows(target)
        summary = summarize_case_budgets(rows)
        fixed = next(row for row in summary if row["prefix_end"] == 8)
        self.assertAlmostEqual(fixed["common_improvement_vs_direct"], 1.0)

    @staticmethod
    def _synthetic_rows(target: torch.Tensor) -> list[dict]:
        return compute_layer_head_rows(
            source=torch.zeros_like(target),
            target=target,
            case_id="aggregate-case",
            split="design",
            layer=0,
            prefix_endpoints=(4, 6, 8),
            calibration_start=2,
            common_evaluation_start=8,
        )


if __name__ == "__main__":
    unittest.main()
