from __future__ import annotations

import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CAPTURE_DIR = SCRIPT_DIR.parent / "cross_request_kv_capture"
sys.path[:0] = [str(SCRIPT_DIR), str(CAPTURE_DIR)]

from validate_closure import structured_events  # noqa: E402


def test_structured_events_accepts_lmcache_source_suffix() -> None:
    log_text = "\n".join(
        (
            'INFO SEGMENTIA_PROFILE_EVENT {"event":"clean","layers":40}',
            "INFO SEGMENTIA_PROFILE_EVENT "
            '{"event":"suffixed","layers":40} '
            "\x1b[3m(segmentia_profile.py:71:record)\x1b[0m",
            "INFO SEGMENTIA_PROFILE_EVENT not-json",
        )
    )

    assert structured_events(log_text, "SEGMENTIA_PROFILE_EVENT") == [
        {"event": "clean", "layers": 40},
        {"event": "suffixed", "layers": 40},
    ]
