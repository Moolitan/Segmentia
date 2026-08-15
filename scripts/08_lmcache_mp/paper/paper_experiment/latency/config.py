"""Fixed configuration for the real-Agent CSKCache latency experiment."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[5]
MOTIVATION_DIR = ROOT / "scripts/08_lmcache_mp/paper/paper_motivation/3.1"
PROMPT_FILE = ROOT / "src/task_prompt/doc-coauthoring-retry-design-doc.txt"
MODEL_PATH = Path(
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
SKILL_SAVE_POOL_ROOT = Path("/mnt/990_pro/skill_save_pool")
RAW_POOL_DIR = SKILL_SAVE_POOL_ROOT / "Qwen3-14B/raw"
OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/08_lmcache_mp/"
    "end_to_end_latency"
)
RESULT_DIR = ROOT / "results/problem_exploration/cskcache_end_to_end_latency"

SKILL = "doc-coauthoring"
MODES = ("recompute", "cskcache")
REPLICAS = 3
WARMUP_CASES = 1
MEASURE_CASES = 5
MAX_AGENT_ITERATIONS = 2
PORT = 8120
SERVED_MODEL = "Qwen3"
API_KEY = "EMPTY"
GPU_MEMORY_UTILIZATION = 0.9
MAX_MODEL_LEN = 32768
