"""Fixed-concurrency steady-state throughput matrix."""

from pathlib import Path

from paper_evaluation.config import ROOT
from common.driver import SystemVariant


PLATFORM_IDS = ("a6000_qwen3_14b",)
FIXED_LENGTH_POOL_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/cache_pools/"
    "Qwen3-14B-fixed-length-v1"
)
SELECTED_LENGTH_BUCKETS = ("5K-8K", ">10K")

SYSTEMS = (
    SystemVariant("Full Prefill", "full"),
    SystemVariant(
        "SkillCache-5%",
        "cskcache",
        correction_strategy="ratio_prefix",
        calibration_ratio=0.05,
    ),
)
CONCURRENCIES = (1, 2, 4, 8)
REPLICAS = 3
WARMUP_SECONDS = 20.0
MEASUREMENT_SECONDS = 120.0

CHUNK_TOKENS = 256
HOST_PAGE_TOKENS = 512
HOST_POOL_GIB = 40.0
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.2
MAX_TOKENS = 1
CORRECTION_ALPHA = 0.6
MINIMUM_REUSE_TOKENS = 256
VLLM_BLOCK_ALIGNMENT_TOKENS = 16
MAX_CALIBRATION_RATIO = 0.05

# Include the workload loader in the run fingerprint so a selection-code edit
# cannot silently resume an incompatible active run.
WORKLOAD_MODULE = (
    ROOT
    / "CSKCache/example/paper_evaluation/section_6_3_latency_scaling/workload.py"
)
