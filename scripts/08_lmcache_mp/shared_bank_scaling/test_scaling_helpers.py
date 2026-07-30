from __future__ import annotations

import sys
from pathlib import Path

import pytest

SCRIPT_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = SCRIPT_ROOT / "cross_request_kv_capture"
sys.path[:0] = [str(SCRIPT_ROOT), str(CAPTURE_DIR)]

from shared_bank_scaling.run_point import percentile  # noqa: E402


@pytest.mark.parametrize(
    ("values", "quantile", "expected"),
    [
        ([4.0], 0.95, 4.0),
        ([4.0, 1.0, 3.0, 2.0], 0.5, 3.0),
        ([4.0, 1.0, 3.0, 2.0], 0.95, 4.0),
    ],
)
def test_percentile_uses_observed_order_statistic(
    values: list[float], quantile: float, expected: float
) -> None:
    assert percentile(values, quantile) == expected
