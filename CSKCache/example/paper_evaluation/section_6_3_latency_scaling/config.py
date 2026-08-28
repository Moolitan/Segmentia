"""Frozen settings for fixed-length host-resident TTFT scaling."""

from __future__ import annotations

from pathlib import Path

from paper_evaluation.common.driver import SystemVariant


SKILLSBENCH_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench"
)
SKILLSBENCH_COMMIT = "9a1f4dd5f7659f75707435da3ce854b6e48321d1"

QWEN3_14B_POOL_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/cache_pools/"
    "Qwen3-14B-fixed-length-v1"
)
PLATFORM_CACHE_POOLS: dict[str, Path | None] = {
    "a6000_qwen3_14b": QWEN3_14B_POOL_ROOT,
    # Filled only after the same six source workloads are encoded and verified
    # with each model's own tokenizer and KV geometry on the A100 host.
    "a100_qwen3_32b": None,
    "2xa100_qwen3_70b": None,
}
ACTIVE_PLATFORM_IDS = ("a6000_qwen3_14b",)
MODEL_PANEL_ORDER = (
    "a6000_qwen3_14b",
    "a100_qwen3_32b",
    "2xa100_qwen3_70b",
)
FUTURE_PLATFORM_IDS = MODEL_PANEL_ORDER[1:]

SYSTEMS = (
    SystemVariant("Full", "full"),
    SystemVariant(
        "CacheBlend-15%",
        "cskcache",
        correction_strategy="deviation_topk",
        cacheblend_ratio=0.15,
    ),
    SystemVariant(
        "CSKCache-5%", "cskcache", correction_strategy="ratio_prefix",
        calibration_ratio=0.05,
    ),
)
SMOKE_SYSTEM_NAMES = ("Full", "CacheBlend-15%", "CSKCache-5%")

SELECTION_SEED = "20260827"
LENGTH_BUCKETS = (
    ("<1K", 0, 1_000),
    ("1K-3K", 1_000, 3_000),
    ("3K-5K", 3_000, 5_000),
    ("5K-8K", 5_000, 8_000),
    ("8K-10K", 8_000, 10_000),
    (">10K", 10_000, None),
)
BUCKET_ORDER = tuple(item[0] for item in LENGTH_BUCKETS)
SMOKE_BUCKET = ">10K"
REPETITIONS = 5
CHUNK_TOKENS = 256
# LMCache's paged pinned allocator sizes one physical page from this value.
# The >10K workload needs 54.5 MiB per persisted layer, so a 256-token
# (40-MiB) page cannot hold one layer; 512 tokens yields an 80-MiB page.
HOST_PAGE_TOKENS = 512
VLLM_BLOCK_ALIGNMENT_TOKENS = 16
MAX_CALIBRATION_RATIO = 0.05
MAX_TOKENS = 1
CORRECTION_ALPHA = 0.6
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
PROFILE_LAYER = 8
DEVIATION_RECOMPUTE_RATIO = 0.15
DEVIATION_CHECK_LAYER = 1
RAW_SLOT_BYTES = 64 * 1024**2
RAW_METADATA_BYTES = 64 * 1024**2
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20260827
