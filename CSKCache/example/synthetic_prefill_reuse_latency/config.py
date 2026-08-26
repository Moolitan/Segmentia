"""Fixed configuration for the synthetic prefill/reuse latency sweep."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]

MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/"
    "Qwen3-14B/Qwen/Qwen3-14B"
)
OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/"
    "synthetic_prefill_reuse_latency"
)
PUBLISH_ROOT = (
    ROOT
    / "results/problem_exploration/"
    "cskcache_synthetic_prefill_reuse_latency"
)

TOKEN_LENGTHS = (512, 1024, 2048, 4096, 8192)
WARMUPS_PER_MODE = 1
REPETITIONS = 5

PREFIX_TOKENS = 256
TAIL_TOKENS = 32
CALIBRATION_TOKENS = 32
DEVIATION_RECOMPUTE_RATIO = 0.15
DEVIATION_CHECK_LAYER = 1
CHUNK_SIZE_TOKENS = 256
MINIMUM_FULL_RECOMPUTE_TOKENS = 32
MINIMUM_REUSE_TOKENS = 256
CORRECTION_ALPHA = 0.6
STORAGE_LAYOUT = "packed_chunks_single_layer"
HOST_LAYOUT = "packed_chunks_single_layer"
EXECUTION_ORDER = "h2d_first"

PROMPT_FILL_TOKEN_ID = 100
KV_FILL_VALUE = 0.0
MAX_TOKENS = 1
MAX_MODEL_LEN = 32768
GPU_MEMORY_UTILIZATION = 0.9
LMCACHE_MAX_LOCAL_CPU_SIZE_GB = 5
PROFILE_WAIT_TIMEOUT_SECONDS = 30.0

METHODS = (
    "normal_prefill",
    "direct_reuse",
    "deviation_topk",
)
