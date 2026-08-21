"""Edit these variables, then run ``bash run.sh``."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

# Workload.
SKILL_NAME = "doc-coauthoring"
SKILLS_DIR = ROOT / "skills"
TASK_PROMPT = "Use the requested Skill and briefly acknowledge that it is loaded."
WARMUP_PAIRS = 1
MEASURE_PAIRS = 5
MAX_TOKENS = 1

# Model and service.
MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
SERVED_MODEL = "Qwen3"
PORT = 8015
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768
REQUEST_TIMEOUT_SECONDS = 720.0
PREFETCH_TIMEOUT_SECONDS = 30.0

# Existing offline Skill KV pool.
STORAGE_BACKEND = "raw_block"  # raw_block | local_disk
POOL_ROOT = Path("/mnt/990_pro/skill_save_pool")
POOL_MODEL_DIR = "Qwen3-14B"

# CSKCache online restoration policy.
HOST_LAYOUT = "full_layer"  # full_layer | chunk_major
HOST_CHUNK_TOKENS = 256
EXECUTION_ORDER = "h2d_first"  # h2d_first | compute_first
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
CALIBRATION_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
CORRECTION_ALPHA = 0.6
PREFETCH_HANDLE_TTL_SECONDS = 60.0

# LMCache physical backend.
LMCACHE_MAX_LOCAL_CPU_SIZE_GB = 5
LMCACHE_MAX_LOCAL_DISK_SIZE_GB = 1000
RAW_SLOT_BYTES = 128 * 1024**2
RAW_METADATA_BYTES = 64 * 1024**2
RAW_METADATA_MAGIC = "CSKRAW01"
RAW_IO_ENGINE = "io_uring"
RAW_QUEUE_DEPTH = 64
