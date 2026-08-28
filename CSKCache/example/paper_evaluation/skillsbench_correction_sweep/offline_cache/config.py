"""Frozen configuration for the SkillsBench CSKCache offline pool."""

from __future__ import annotations

import os
from pathlib import Path


SKILLSBENCH_COMMIT = "9a1f4dd5f7659f75707435da3ce854b6e48321d1"
SKILLSBENCH_ROOT = Path(
    os.environ.get(
        "SKILLSBENCH_ROOT",
        "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench",
    )
)
TASKS_ROOT = SKILLSBENCH_ROOT / "tasks"

# The generic offline encoder discovers every selected name recursively below
# TASKS_ROOT. Exact repeated (name, body) pairs are collapsed, while same-name
# bodies with different content remain separate Catalog versions.
STORAGE_BACKEND = "raw_block"
CHUNK_SIZE_TOKENS = 256
STORAGE_LAYOUT = "packed_chunks_single_layer"
SKILLS = tuple(
    sorted({path.parent.name for path in TASKS_ROOT.glob("*/environment/skills/*/SKILL.md")})
)
COLLECTION = None
EXCLUDED_SKILLS = ()
DEDUPLICATE_CONTENT = True
RETAIN_SKILL_VERSIONS = True
OVERWRITE = os.environ.get("SKILLSBENCH_OFFLINE_OVERWRITE", "0") == "1"
DRY_RUN = os.environ.get("SKILLSBENCH_OFFLINE_DRY_RUN", "0") == "1"

SKILLS_DIR = TASKS_ROOT
MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
SERVED_MODEL = "Qwen3"
POOL_ROOT = Path(
    os.environ.get("SKILLSBENCH_CSKCACHE_POOL_ROOT", "/mnt/990_pro/skill_save_pool")
)
POOL_MODEL_DIR = os.environ.get(
    "SKILLSBENCH_CSKCACHE_POOL_NAME",
    "Qwen3-14B-SkillsBench-9a1f4dd5-v2",
)

PORT = int(os.environ.get("SKILLSBENCH_OFFLINE_PORT", "8013"))
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = float(
    os.environ.get("SKILLSBENCH_OFFLINE_GPU_UTIL", "0.9")
)
MAX_MODEL_LEN = 32768
EXPECTED_LAYERS = 40
READINESS_ATTEMPTS = 450
READINESS_INTERVAL_SECONDS = 2
SHUTDOWN_TIMEOUT_SECONDS = 30

# One exact-save layer occupies one fixed raw slot. The longest SkillsBench
# body needs about 32.4 MiB per layer, so 40 MiB safely fits it and lets the
# 512-GiB sparse container hold all 202 * 40 layer objects.
RAW_CAPACITY_BYTES = 512 * 1024**3
RAW_SLOT_BYTES = 40 * 1024**2
# RawBlockCore keeps two durable checkpoint copies. The failed v1 run proved
# that 64 MiB total gives only about 32 MiB per copy, while 8,080 keys require
# about 55.5 MiB. 256 MiB provides about 128 MiB per copy and ample headroom.
RAW_METADATA_BYTES = 256 * 1024**2
RAW_CONTAINER_ID = "qwen3-14b-skillsbench-9a1f4dd5-v2"

EXPECTED_TASKS = 87
EXPECTED_OCCURRENCES = 232
EXPECTED_SKILL_NAMES = 195
EXPECTED_BODY_VERSIONS = 202
