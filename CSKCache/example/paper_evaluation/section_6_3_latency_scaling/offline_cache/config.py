"""Dedicated Qwen3-14B pool for the six fixed-length workloads."""

from __future__ import annotations

import os
from pathlib import Path


STORAGE_BACKEND = "raw_block"
CHUNK_SIZE_TOKENS = 256
STORAGE_LAYOUT = "packed_chunks_single_layer"
SKILLS = (
    "marker",
    "erlang-concurrency",
    "sympy",
    "pptx",
    "citation-management",
    "proof-checker",
)
COLLECTION = None
EXCLUDED_SKILLS = ()
DEDUPLICATE_CONTENT = False
RETAIN_SKILL_VERSIONS = True
OVERWRITE = os.environ.get("FIXED_LENGTH_OFFLINE_OVERWRITE", "0") == "1"
DRY_RUN = os.environ.get("FIXED_LENGTH_OFFLINE_DRY_RUN", "0") == "1"

MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
SERVED_MODEL = "Qwen3"
POOL_ROOT = Path(
    os.environ.get(
        "FIXED_LENGTH_CSKCACHE_POOL_ROOT",
        "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/cache_pools",
    )
)
POOL_MODEL_DIR = os.environ.get(
    "FIXED_LENGTH_CSKCACHE_POOL_NAME", "Qwen3-14B-fixed-length-v1"
)
SKILLS_DIR = POOL_ROOT / POOL_MODEL_DIR / "sources"

PORT = int(os.environ.get("FIXED_LENGTH_OFFLINE_PORT", "8013"))
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("FIXED_LENGTH_OFFLINE_GPU_UTIL", "0.9")
)
MAX_MODEL_LEN = 32768
EXPECTED_LAYERS = 40
READINESS_ATTEMPTS = 450
READINESS_INTERVAL_SECONDS = 2
SHUTDOWN_TIMEOUT_SECONDS = 30

# Six objects x 40 layers require 240 slots.  The 64-MiB slot also fits the
# 13,314-token proof-checker layer (about 54.5 MiB for Qwen3-14B).
RAW_CAPACITY_BYTES = 32 * 1024**3
RAW_SLOT_BYTES = 64 * 1024**2
RAW_METADATA_BYTES = 64 * 1024**2
RAW_CONTAINER_ID = "qwen3-14b-fixed-length-v1"
