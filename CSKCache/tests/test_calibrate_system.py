from __future__ import annotations

import json
from pathlib import Path

from cskcache.tools.calibrate_system import write_calibration


PROFITABILITY_ENABLED = True


def test_offline_measurements_write_runtime_profile(tmp_path: Path) -> None:
    output = tmp_path / "profile.json"
    path = write_calibration(
        {
            "metadata_path": str(tmp_path / "pool" / "catalog.json"),
            "system_profile_path": str(output),
            "environment": {"model": "model-a", "gpu": "gpu-a"},
            "strategy": "ratio_prefix",
            "ratio": 0.05,
            "prefill": {"amount": 50, "duration_ms": 5.0},
            "ssd_read": {"amount": 1000, "duration_ms": 10.0},
            "h2d": {"amount": 1000, "duration_ms": 2.0},
            "composition": {"amount": 1000, "duration_ms": 40.0},
        }
    )
    assert path == output
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["composition_ms_per_token"]["ratio_prefix:0.05000000"][
        "samples"
    ] == 1
