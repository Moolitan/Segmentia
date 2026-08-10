from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


MODULE_PATH = Path(__file__).with_name("analyze_context_free_residual.py")
SPEC = importlib.util.spec_from_file_location("context_free_analysis", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

CALIBRATED_PATH = Path(__file__).with_name("analyze_calibrated_residual.py")
CALIBRATED_SPEC = importlib.util.spec_from_file_location(
    "calibrated_context_free_analysis", CALIBRATED_PATH
)
assert CALIBRATED_SPEC is not None and CALIBRATED_SPEC.loader is not None
CALIBRATED = importlib.util.module_from_spec(CALIBRATED_SPEC)
CALIBRATED_SPEC.loader.exec_module(CALIBRATED)

LAYER_AXIS_PATH = Path(__file__).with_name("analyze_layer_axis.py")
LAYER_AXIS_SPEC = importlib.util.spec_from_file_location(
    "layer_axis_analysis", LAYER_AXIS_PATH
)
assert LAYER_AXIS_SPEC is not None and LAYER_AXIS_SPEC.loader is not None
LAYER_AXIS = importlib.util.module_from_spec(LAYER_AXIS_SPEC)
sys.modules[LAYER_AXIS_SPEC.name] = LAYER_AXIS
LAYER_AXIS_SPEC.loader.exec_module(LAYER_AXIS)

DIRECTION_PATH = Path(__file__).with_name("analyze_shallow_deep_direction.py")
DIRECTION_SPEC = importlib.util.spec_from_file_location(
    "shallow_deep_direction_analysis", DIRECTION_PATH
)
assert DIRECTION_SPEC is not None and DIRECTION_SPEC.loader is not None
DIRECTION = importlib.util.module_from_spec(DIRECTION_SPEC)
DIRECTION_SPEC.loader.exec_module(DIRECTION)

CORRECTED_PLOT_PATH = Path(__file__).with_name(
    "plot_corrected_recompute_cosine.py"
)
CORRECTED_PLOT_SPEC = importlib.util.spec_from_file_location(
    "corrected_recompute_plot", CORRECTED_PLOT_PATH
)
assert CORRECTED_PLOT_SPEC is not None and CORRECTED_PLOT_SPEC.loader is not None
CORRECTED_PLOT = importlib.util.module_from_spec(CORRECTED_PLOT_SPEC)
CORRECTED_PLOT_SPEC.loader.exec_module(CORRECTED_PLOT)

L2_PLOT_PATH = Path(__file__).with_name("plot_layerwise_normalized_l2.py")
L2_PLOT_SPEC = importlib.util.spec_from_file_location(
    "layerwise_l2_plot", L2_PLOT_PATH
)
assert L2_PLOT_SPEC is not None and L2_PLOT_SPEC.loader is not None
L2_PLOT = importlib.util.module_from_spec(L2_PLOT_SPEC)
L2_PLOT_SPEC.loader.exec_module(L2_PLOT)


def test_rope_relocation_round_trip() -> None:
    torch.manual_seed(0)
    key = torch.randn(7, 8, 128)
    moved = MODULE.relocate_neox_rope(key, 123, 1_000_000.0)
    restored = MODULE.relocate_neox_rope(moved, -123, 1_000_000.0)
    torch.testing.assert_close(restored, key, rtol=1e-5, atol=1e-5)


def test_shared_offset_reduces_tail_error() -> None:
    torch.manual_seed(1)
    offline = torch.randn(300, 8, 128)
    offset = torch.randn(8, 128) * 0.1
    online = offline + offset
    estimate = (online[:64] - offline[:64]).mean(dim=0)
    direct = (offline[256:] - online[256:]).square().sum()
    corrected = (offline[256:] + estimate - online[256:]).square().sum()
    assert float(corrected) < float(direct) * 1e-10


def test_held_out_calibration_recovers_scale() -> None:
    torch.manual_seed(2)
    direction = torch.randn(8, 128)
    residual = torch.empty(320, 8, 128)
    residual[:128] = direction
    residual[128:256] = 0.4 * direction
    residual[256:] = 0.4 * direction
    offset, alpha, _, _ = CALIBRATED.estimate_calibrated_offset(residual)
    torch.testing.assert_close(alpha, torch.full((8,), 0.4), rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(offset, 0.4 * direction, rtol=1e-5, atol=1e-5)


def test_calibration_does_not_read_suffix() -> None:
    torch.manual_seed(3)
    prefix = torch.randn(256, 8, 128)
    first = torch.cat((prefix, torch.zeros(64, 8, 128)))
    second = torch.cat((prefix, torch.randn(64, 8, 128) * 1000))
    first_offset, first_alpha, _, _ = CALIBRATED.estimate_calibrated_offset(first)
    second_offset, second_alpha, _, _ = CALIBRATED.estimate_calibrated_offset(second)
    torch.testing.assert_close(first_alpha, second_alpha)
    torch.testing.assert_close(first_offset, second_offset)


def test_fidelity_cosine_and_normalized_l2_are_complementary() -> None:
    full = torch.ones(12, 8, 128)
    scaled = 2.0 * full
    metrics = CALIBRATED.fidelity_metrics(scaled, full)
    assert abs(metrics["cosine_mean"] - 1.0) < 1e-6
    assert abs(metrics["cosine_median"] - 1.0) < 1e-6
    assert abs(metrics["normalized_l2"] - 1.0) < 1e-6


def test_exact_full_has_unit_cosine_and_zero_error() -> None:
    torch.manual_seed(4)
    full = torch.randn(12, 8, 128)
    metrics = CALIBRATED.fidelity_metrics(full, full)
    assert abs(metrics["cosine_mean"] - 1.0) < 1e-6
    assert metrics["normalized_l2"] == 0.0
    assert metrics["sse"] == 0.0


def test_self_shallow_offset_averages_only_observed_layers_and_tokens() -> None:
    torch.manual_seed(5)
    expected = torch.randn(8, 128)
    shallow = expected.view(1, 1, 8, 128).expand(4, 23, 8, 128).clone()
    estimate = LAYER_AXIS.estimate_self_shallow_offset(shallow)
    torch.testing.assert_close(estimate, expected)


def test_self_shallow_offset_does_not_read_deep_layers_or_other_skills() -> None:
    torch.manual_seed(6)
    shallow = torch.randn(4, 17, 8, 128)
    first = LAYER_AXIS.estimate_self_shallow_offset(shallow)
    unrelated_deep = torch.randn(36, 17, 8, 128) * 1000
    unrelated_skill = torch.randn(40, 31, 8, 128) * 1000
    second = LAYER_AXIS.estimate_self_shallow_offset(shallow.clone())
    assert unrelated_deep.numel() > 0 and unrelated_skill.numel() > 0
    torch.testing.assert_close(first, second)


def test_self_shallow_offset_corrects_matching_deep_shift() -> None:
    torch.manual_seed(7)
    direct = torch.randn(29, 8, 128)
    offset = torch.randn(8, 128) * 0.1
    shallow = offset.view(1, 1, 8, 128).expand(4, 29, 8, 128)
    estimate = LAYER_AXIS.estimate_self_shallow_offset(shallow)
    recompute = direct + offset.unsqueeze(0)
    direct_error = LAYER_AXIS.fidelity(direct, recompute)["sse"]
    corrected_error = LAYER_AXIS.fidelity(
        direct + estimate.unsqueeze(0), recompute
    )["sse"]
    assert corrected_error < direct_error * 1e-10


def test_direction_cosine_uses_one_head_direction_vector() -> None:
    residual = torch.tensor([1.0, 0.0, 2.0])
    assert abs(DIRECTION.direction_cosine(residual, residual) - 1.0) < 1e-12
    assert abs(DIRECTION.direction_cosine(-residual, residual) + 1.0) < 1e-12


def test_corrected_plot_rows_use_self_only_fidelity_field() -> None:
    fidelity = []
    for component in ("K", "V"):
        for layer in range(4, 40):
            fidelity.append(
                {
                    "component": component,
                    "skill": "skill-a",
                    "cutoff": "4",
                    "target_layer": str(layer),
                    "corrected_to_recompute_cosine_mean": "0.75",
                }
            )
    rows = CORRECTED_PLOT.corrected_rows(fidelity, ["skill-a"], cutoff=4)
    assert len(rows) == 80
    assert all(
        row["value_source"] == "recomputed_shallow_layer"
        for row in rows
        if row["layer"] < 4
    )
    assert all(
        row["value_source"] == "self_shallow_offset_deep_layer"
        for row in rows
        if row["layer"] >= 4
    )


def test_l2_plot_rows_use_self_only_sse_field() -> None:
    direct = []
    fidelity = []
    for component in ("K", "V"):
        for layer in range(40):
            direct.append(
                {
                    "component": component,
                    "skill": "skill-a",
                    "layer": str(layer),
                    "direct_to_full_normalized_l2": "0.5",
                }
            )
            if layer >= 4:
                fidelity.append(
                    {
                        "component": component,
                        "skill": "skill-a",
                        "cutoff": "4",
                        "target_layer": str(layer),
                        "corrected_to_recompute_sse": "1.0",
                        "recompute_sq_norm": "4.0",
                    }
                )
    rows = L2_PLOT.build_rows(direct, fidelity, cutoff=4)
    assert len(rows) == 80
    assert all(
        row["normalized_l2_to_recompute"] == 0.0
        for row in rows
        if row["layer"] < 4
    )
    assert all(
        row["normalized_l2_to_recompute"] == 0.5
        for row in rows
        if row["layer"] >= 4
    )
