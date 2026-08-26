"""Fixed systems and generation settings for Section 6.2."""

from pathlib import Path

from paper_evaluation.config import ACTIVE_PLATFORMS
from common.driver import SystemVariant


WORKLOAD_FILE = Path(__file__).with_name("workloads.json")
PLATFORM_IDS = ACTIVE_PLATFORMS
MAX_TOKENS = 384
CHUNK_TOKENS = 256
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
REPETITIONS = 1

SYSTEMS = (
    SystemVariant("Full", "full"),
    SystemVariant("Direct", "cskcache", correction_strategy="direct"),
    SystemVariant(
        "Fixed-64", "cskcache", correction_strategy="fixed_prefix",
        calibration_tokens=64,
    ),
    SystemVariant(
        "Fixed-128", "cskcache", correction_strategy="fixed_prefix",
        calibration_tokens=128,
    ),
    SystemVariant(
        "Fixed-256", "cskcache", correction_strategy="fixed_prefix",
        calibration_tokens=256,
    ),
    SystemVariant(
        "Ratio-5%", "cskcache", correction_strategy="ratio_prefix",
        calibration_ratio=0.05,
    ),
    SystemVariant(
        "Ratio-15%", "cskcache", correction_strategy="ratio_prefix",
        calibration_ratio=0.15,
    ),
    SystemVariant(
        "Ratio-30%", "cskcache", correction_strategy="ratio_prefix",
        calibration_ratio=0.30,
    ),
    SystemVariant("CacheBlend-15%", "cacheblend", cacheblend_ratio=0.15),
)
