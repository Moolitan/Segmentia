"""Edit these variables, then run ``bash run_interactive_agent.sh``."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

# Online mode and offline object selection.
MODE = "cskcache"  # cskcache | no_reuse
STORAGE_BACKEND = "raw_block"  # raw_block | local_disk
SKILLS = ("doc-coauthoring",)
COLLECTION = None

# Skill, model, pool, and output paths.
SKILLS_DIR = ROOT / "skills" / "Auto-claude-code-research-in-sleep" / "skills"
EXTRA_SKILLS_DIR = ROOT / "skills"
MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
SERVED_MODEL = "Qwen3"
POOL_ROOT = Path("/mnt/990_pro/skill_save_pool")
POOL_MODEL_DIR = "Qwen3-14B"
WORKSPACE = ROOT / "workspace" / "08_lmcache_mp" / "interactive_agent"

# vLLM and Agent settings.
PORT = 8014
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768
TOOL_CALL_PARSER = "hermes"
REASONING_PARSER = "qwen3"
MAX_ITERATIONS = 2
PROMPT_FILE = None

# CSKCache reuse policy and physical execution.
HOST_LAYOUT = "full_layer"  # full_layer | chunk_major
HOST_CHUNK_TOKENS = 256
EXECUTION_ORDER = "h2d_first"  # h2d_first | compute_first
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
CALIBRATION_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
CORRECTION_ALPHA = 0.6
PREFETCH_HANDLE_TTL_SECONDS = 60.0

# Profiling.
PROFILE = True
FINE_TIMELINE = False
DISABLE_VISUALIZER = False

# LMCache capacity and raw-block settings.
LMCACHE_MAX_LOCAL_CPU_SIZE_GB = 5
LMCACHE_MAX_LOCAL_DISK_SIZE_GB = 1000
RAW_SLOT_BYTES = 128 * 1024**2
RAW_METADATA_BYTES = 64 * 1024**2
RAW_METADATA_MAGIC = "CSKRAW01"
RAW_IO_ENGINE = "io_uring"
RAW_QUEUE_DEPTH = 64
