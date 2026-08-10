from __future__ import annotations

import importlib.util
import math
from pathlib import Path
import sys

import torch


MODULE_PATH = Path(__file__).with_name("analyze_token_axis.py")
SPEC = importlib.util.spec_from_file_location("token_axis_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

PLOT_PATH = Path(__file__).with_name("plot_token_axis.py")
PLOT_SPEC = importlib.util.spec_from_file_location("token_axis_plot", PLOT_PATH)
assert PLOT_SPEC is not None and PLOT_SPEC.loader is not None
PLOT = importlib.util.module_from_spec(PLOT_SPEC)
PLOT_SPEC.loader.exec_module(PLOT)

PUBLISH_PATH = Path(__file__).with_name("publish_token_axis.py")
PUBLISH_SPEC = importlib.util.spec_from_file_location(
    "token_axis_publish", PUBLISH_PATH
)
assert PUBLISH_SPEC is not None and PUBLISH_SPEC.loader is not None
PUBLISH = importlib.util.module_from_spec(PUBLISH_SPEC)
PUBLISH_SPEC.loader.exec_module(PUBLISH)


def test_same_layer_prefix_mean_recovers_common_offset() -> None:
    torch.manual_seed(0)
    offset = torch.randn(8, 128)
    residual = torch.randn(400, 8, 128)
    residual[132:256] = offset
    estimate = MODULE.estimate_token_offset(residual, 132, 256)
    torch.testing.assert_close(estimate, offset)


def test_estimator_does_not_read_suffix() -> None:
    torch.manual_seed(1)
    prefix = torch.randn(256, 8, 128)
    first = torch.cat((prefix, torch.zeros(64, 8, 128)))
    second = torch.cat((prefix, torch.randn(64, 8, 128) * 1000))
    first_estimate = MODULE.estimate_token_offset(first, 132, 256)
    second_estimate = MODULE.estimate_token_offset(second, 132, 256)
    torch.testing.assert_close(first_estimate, second_estimate)


def test_each_layer_is_estimated_independently() -> None:
    first = torch.zeros(320, 8, 128)
    second = torch.zeros(320, 8, 128)
    first[132:256] = 1.0
    second[132:256] = -2.0
    first_estimate = MODULE.estimate_token_offset(first, 132, 256)
    second_estimate = MODULE.estimate_token_offset(second, 132, 256)
    torch.testing.assert_close(first_estimate, torch.ones(8, 128))
    torch.testing.assert_close(second_estimate, torch.full((8, 128), -2.0))


def test_common_offset_improves_suffix_cosine_and_l2() -> None:
    torch.manual_seed(2)
    direct = torch.randn(80, 8, 128)
    offset = torch.randn(8, 128) * 0.3
    recompute = direct + offset.unsqueeze(0)
    direct_metrics = MODULE.fidelity(direct, recompute)
    corrected_metrics = MODULE.fidelity(direct + offset.unsqueeze(0), recompute)
    assert corrected_metrics["cosine_mean"] > direct_metrics["cosine_mean"]
    assert corrected_metrics["sse"] < direct_metrics["sse"] * 1e-10


def test_fixed_alpha_scales_offset_before_suffix_correction() -> None:
    direct = torch.zeros(3, 8, 128)
    offset = torch.full((8, 128), 2.0)
    corrected = MODULE.apply_token_offset(direct, offset, 0.6)
    torch.testing.assert_close(corrected, torch.full_like(direct, 1.2))


def test_direction_cosine_reports_alignment() -> None:
    direction = torch.tensor([1.0, -2.0, 3.0])
    assert abs(MODULE.direction_cosine(direction, direction) - 1.0) < 1e-12
    assert abs(MODULE.direction_cosine(direction, -direction) + 1.0) < 1e-12
    assert math.isnan(MODULE.direction_cosine(direction, torch.zeros_like(direction)))


def test_plot_reads_direct_and_corrected_token_axis_fields() -> None:
    rows = []
    for layer in range(40):
        rows.append(
            {
                "skill": "skill-a",
                "component": "K",
                "layer": str(layer),
                "direct_to_recompute_cosine_mean": "0.8",
                "corrected_to_recompute_cosine_mean": "0.9",
                "direct_to_recompute_sse": "4.0",
                "corrected_to_recompute_sse": "1.0",
                "recompute_sq_norm": "4.0",
                "alpha": "0.6",
            }
        )
    _, direct_cosine, corrected_cosine = PLOT.layer_values(
        rows, "skill-a", "K", "cosine"
    )
    _, direct_l2, corrected_l2 = PLOT.layer_values(rows, "skill-a", "K", "l2")
    assert direct_cosine == [0.8] * 40
    assert corrected_cosine == [0.9] * 40
    assert direct_l2 == [1.0] * 40
    assert corrected_l2 == [0.5] * 40
    assert PLOT.fixed_alpha(rows) == 0.6
    lower, upper = PLOT.metric_limits(rows, "K", "cosine")
    assert abs(lower - 0.793) < 1e-12
    assert abs(upper - 0.907) < 1e-12


def test_publisher_aggregates_only_rows_from_each_skill_key() -> None:
    rows = []
    for skill, corrected_sse in (("skill-a", "1.0"), ("skill-b", "9.0")):
        rows.append(
            {
                "skill": skill,
                "component": "K",
                "direct_to_recompute_cosine_mean": "0.8",
                "corrected_to_recompute_cosine_mean": "0.9",
                "direct_to_recompute_sse": "4.0",
                "corrected_to_recompute_sse": corrected_sse,
                "recompute_sq_norm": "4.0",
            }
        )
    aggregate = PUBLISH.aggregate_fidelity(rows)
    assert aggregate[("K", "skill-a")]["corrected_l2"] == 0.5
    assert aggregate[("K", "skill-b")]["corrected_l2"] == 1.5
