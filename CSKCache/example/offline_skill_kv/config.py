"""Edit these variables, then run ``bash run.sh``."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

# Offline object selection.
STORAGE_BACKEND = "raw_block"  # raw_block | local_disk
CHUNK_SIZE_TOKENS = 256
STORAGE_LAYOUT = "packed_chunks_single_layer"
SKILLS = ("doc-coauthoring",)
COLLECTION = None
EXCLUDED_SKILLS = ()
OVERWRITE = False
DRY_RUN = False

# Model and output paths.
SKILLS_DIR = ROOT / "skills"
MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
SERVED_MODEL = "Qwen3"
POOL_ROOT = Path("/mnt/990_pro/skill_save_pool")
POOL_MODEL_DIR = "Qwen3-14B"

# Temporary vLLM service used for exact prefill.
PORT = 8013
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768
EXPECTED_LAYERS = 40
READINESS_ATTEMPTS = 450
READINESS_INTERVAL_SECONDS = 2
SHUTDOWN_TIMEOUT_SECONDS = 30

# Raw-block container geometry.
RAW_CAPACITY_BYTES = 512 * 1024**3
RAW_SLOT_BYTES = 128 * 1024**2
RAW_CONTAINER_ID = "qwen3-14b-skill-kv"
