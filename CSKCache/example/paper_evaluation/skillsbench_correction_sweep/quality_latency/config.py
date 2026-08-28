"""Frozen settings for the SkillsBench quality--latency sweep."""

from __future__ import annotations

from pathlib import Path

from paper_evaluation.config import ACTIVE_PLATFORMS
from paper_evaluation.common.driver import SystemVariant


SKILLSBENCH_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench"
)
SKILLSBENCH_COMMIT = "9a1f4dd5f7659f75707435da3ce854b6e48321d1"

POOL_ROOT = Path(
    "/mnt/990_pro/skill_save_pool/Qwen3-14B-SkillsBench-9a1f4dd5-v2"
)
MANIFEST_PATH = POOL_ROOT / "skillsbench_manifest.json"
MASTER_CATALOG_PATH = POOL_ROOT / "raw/catalog.json"
EXPECTED_CATALOG_SHA256 = (
    "13e407ef24ac9331d148f4ab84c69656d76d4ea21779fbc331ca8e31fea607ce"
)
RAW_SLOT_BYTES = 40 * 1024**2
RAW_METADATA_BYTES = 256 * 1024**2

WORKLOAD_FILE = Path(__file__).with_name("workloads.json")
PLATFORM_IDS = ACTIVE_PLATFORMS
MAX_TOKENS = 384
SEED = 0
CHUNK_TOKENS = 256
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
PROFILE_LAYER = 8
SMOKE_TASK_ID = "data-to-d3"

SYSTEMS = (
    SystemVariant("Full", "full"),
    SystemVariant(
        "Ratio-1%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.01,
    ),
    SystemVariant(
        "Ratio-3%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.03,
    ),
    SystemVariant(
        "Ratio-5%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.05,
    ),
    SystemVariant(
        "Ratio-7%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.07,
    ),
    SystemVariant(
        "Ratio-10%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.10,
    ),
)
