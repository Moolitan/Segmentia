"""Frozen pilot configuration for the SkillsBench quality experiment."""

from pathlib import Path

from common.driver import SystemVariant
from paper_evaluation.config import ACTIVE_PLATFORMS


SKILLSBENCH_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench"
)
BENCH_EXECUTABLE = SKILLSBENCH_ROOT / ".venv/bin/bench"
DOCKER_COMPOSE_PLUGIN = Path("/home/wsh/.docker/cli-plugins/docker-compose")
WORKLOAD_FILE = Path(__file__).with_name("workloads.json")

PLATFORM_IDS = ACTIVE_PLATFORMS
SYSTEMS = (SystemVariant("Full", "full"),)
REPETITIONS = 1

AGENT = "openhands"
MODEL_PROVIDER = "vllm"
SANDBOX = "docker"
SKILL_MODE = "with-skill"
CONCURRENCY = 1
USAGE_TRACKING = "required"
AGENT_IDLE_TIMEOUT = "none"
SANDBOX_SETUP_TIMEOUT_SECONDS = 900
BENCHFLOW_WALL_TIMEOUT_SECONDS = 2400
RUN_ORACLE_PREFLIGHT = True
