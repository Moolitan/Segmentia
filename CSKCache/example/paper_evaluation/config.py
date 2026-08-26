"""Shared editable configuration for every paper-evaluation subsection.

Every subsection is launched through its own ``run.sh`` without arguments.
Machine-specific paths belong here; experiment matrices belong in the local
``config.py`` next to each launcher.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
CSKCACHE_ROOT = ROOT / "CSKCache"
LMCACHE_ROOT = ROOT / "LMCache"
# This workspace copy contains the two server-side TTFT timeline hooks used by
# every latency experiment. Other examples may continue using /home/wsh/vllm.
VLLM_ROOT = ROOT / "vllm"
SKILLS_ROOT = ROOT / "skills"
TASK_PROMPT_ROOT = ROOT / "src/task_prompt"

OUTPUT_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/paper_evaluation"
)
RAW_POOL_ROOT = Path("/mnt/990_pro/skill_save_pool")
EXISTING_LAYOUT_IO_RUN = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/"
    "ssd_pinned_layout_io/20260824-130642"
)

API_KEY = "EMPTY"
HOST = "127.0.0.1"
BASE_PORT = 8300
SERVER_START_TIMEOUT_SECONDS = 900
REQUEST_TIMEOUT_SECONDS = 900
PREFETCH_TIMEOUT_SECONDS = 60


@dataclass(frozen=True)
class Platform:
    platform_id: str
    gpu_name: str
    model_id: str
    model_path: Path
    served_model: str
    gpu_ids: tuple[int, ...]
    tensor_parallel_size: int
    max_model_len: int
    gpu_memory_utilization: float
    dtype: str = "auto"
    quantization: str | None = None
    reasoning_parser: str | None = "qwen3"

    def as_record(self) -> dict[str, object]:
        record = asdict(self)
        record["model_path"] = str(self.model_path)
        record["gpu_ids"] = list(self.gpu_ids)
        return record


PLATFORMS = {
    "a6000_qwen3_14b": Platform(
        platform_id="a6000_qwen3_14b",
        gpu_name="NVIDIA RTX A6000",
        model_id="Qwen3-14B",
        model_path=Path(
            "/mnt/Large_Language_Model_Lab_1/llm_models/"
            "Qwen3-14B/Qwen/Qwen3-14B"
        ),
        served_model="Qwen3-14B",
        gpu_ids=(0,),
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    ),
    "a100_qwen3_32b": Platform(
        platform_id="a100_qwen3_32b",
        gpu_name="NVIDIA A100",
        model_id="Qwen3-32B",
        model_path=Path(
            "/mnt/Large_Language_Model_Lab_1/llm_models/"
            "Qwen3-32B/Qwen/Qwen3-32B"
        ),
        served_model="Qwen3-32B",
        gpu_ids=(0,),
        tensor_parallel_size=1,
        max_model_len=32768,
        gpu_memory_utilization=0.90,
    ),
    "2xa100_qwen3_70b": Platform(
        platform_id="2xa100_qwen3_70b",
        gpu_name="NVIDIA A100",
        model_id="Qwen3-70B",
        model_path=Path(
            "/mnt/Large_Language_Model_Lab_1/llm_models/"
            "Qwen3-70B/Qwen/Qwen3-70B"
        ),
        served_model="Qwen3-70B",
        gpu_ids=(0, 1),
        tensor_parallel_size=2,
        max_model_len=16384,
        gpu_memory_utilization=0.94,
    ),
}

# Enable only platforms physically present on the current machine. Results
# remain mergeable because every CSV row carries platform/model identity.
ACTIVE_PLATFORMS = ("a6000_qwen3_14b",)

# Root-level merge utilities scan these directories. Add result roots copied
# from other machines here; no command-line paths are required.
MERGE_INPUT_ROOTS = (OUTPUT_ROOT,)
MERGED_OUTPUT_DIR = OUTPUT_ROOT / "merged"
