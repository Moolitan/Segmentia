"""Configuration for the auxiliary-forward versus native-vLLM microbenchmark."""

from pathlib import Path

from config import MODEL_PATH, OUTPUT_ROOT


RUN_NAME = "calibration_forward_microbenchmark_v1"
RUN_ROOT = OUTPUT_ROOT / "microbenchmarks" / RUN_NAME

SKILL_TOKENS = 512
CALIBRATION_TOKENS = (16, 32, 128)
PREFIX_CONTEXT_TOKENS = 288
PROFILE_LAYER_IDS = (10, 20, 30)

REPETITIONS = 3
WARMUP_REQUESTS = 2
STABLE_LAYER_START = 5
STABLE_LAYER_STOP = 35
NUM_LAYERS = 40

CHUNK_SIZE_TOKENS = 256
STORAGE_LAYOUT = "packed_chunks_single_layer"
HOST_LAYOUT = "packed_chunks_single_layer"
EXECUTION_ORDER = "compute_first"

MAX_MODEL_LEN = 2048
GPU_MEMORY_UTILIZATION = 0.9
PROMPT_TOKEN_ID = 100
SUFFIX_TOKEN_ID = 101
MAX_TOKENS = 1

NATIVE_CASE_CONFIG_ENV = "CSKCACHE_NATIVE_FORWARD_CASE_CONFIG"
CSK_CASE_CONFIG_ENV = "CSKCACHE_PINNED_CASE_CONFIG"


def expected_csk_prefix_tokens(calibration_tokens: int) -> int:
    """Absolute calibration start for the fixed synthetic request shape."""

    if calibration_tokens % 16:
        raise ValueError("calibration tokens must preserve the fixed aligned prefix")
    return 256 + 32


for _calibration_tokens in CALIBRATION_TOKENS:
    if expected_csk_prefix_tokens(_calibration_tokens) != PREFIX_CONTEXT_TOKENS:
        raise ValueError("native and CSKCache attention contexts are not aligned")

if not Path(MODEL_PATH).is_dir():
    raise FileNotFoundError(MODEL_PATH)
