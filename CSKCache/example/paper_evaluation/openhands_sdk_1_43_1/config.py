"""Frozen configuration for the OpenHands current-vs-SDK-1.43.1 comparison."""

from dataclasses import dataclass
from pathlib import Path

from paper_evaluation.config import ACTIVE_PLATFORMS


ROOT = Path(__file__).resolve().parent
SKILLSBENCH_ROOT = Path(
    "/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench"
)
BENCH_PYTHON = SKILLSBENCH_ROOT / ".venv/bin/python"
BENCH_ENTRY = ROOT / "bench_entry.py"
DOCKER_COMPOSE_PLUGIN = Path("/home/wsh/.docker/cli-plugins/docker-compose")
WORKLOAD_FILE = ROOT / "workloads.json"
IMAGE_LOCK_FILE = ROOT / "image.lock.json"

OPENHANDS_SDK_VERSION = "1.43.1"
OPENHANDS_TOOLS_VERSION = "1.43.1"
ACP_VERSION = "0.10.1"
LATEST_AGENT_NAME = "openhands-sdk-1.43.1"
LATEST_IMAGE_TAG = "cskcache/openhands-sdk-benchflow:1.43.1"
BASE_IMAGE_REFERENCE = (
    "python:3.12-slim@"
    "sha256:7a8b475003c4fe15a2cd4e55e5cfc2f3560bdc9333d624f24cdd6d4340fd7a17"
)
SDK_SOURCE_COMMIT = "ddac55697c5d15cf8a34495b5ed6d46c86db092a"

CURRENT_AGENT_NAME = "openhands"
CURRENT_CLI_COMMIT = "3ca17446c5d9c1e35e054803478a3501ec251ecf"
CURRENT_SDK_VERSION = "1.22.1"
CURRENT_TOOLS_VERSION = "1.22.1"


@dataclass(frozen=True)
class Harness:
    harness_id: str
    label: str
    agent: str
    sdk_version: str
    tools_version: str
    image_locked: bool


HARNESSES = (
    Harness(
        harness_id="current",
        label="Current OpenHands",
        agent=CURRENT_AGENT_NAME,
        sdk_version=CURRENT_SDK_VERSION,
        tools_version=CURRENT_TOOLS_VERSION,
        image_locked=False,
    ),
    Harness(
        harness_id="sdk_1_43_1",
        label="OpenHands SDK 1.43.1",
        agent=LATEST_AGENT_NAME,
        sdk_version=OPENHANDS_SDK_VERSION,
        tools_version=OPENHANDS_TOOLS_VERSION,
        image_locked=True,
    ),
)

PLATFORM_IDS = ACTIVE_PLATFORMS
REPETITIONS = 3

MODEL_PROVIDER = "vllm"
SANDBOX = "docker"
SKILL_MODE = "with-skill"
CONCURRENCY = 1
USAGE_TRACKING = "required"
AGENT_IDLE_TIMEOUT = "none"
SANDBOX_SETUP_TIMEOUT_SECONDS = 900
BENCHFLOW_WALL_TIMEOUT_SECONDS = 2400
RUN_ORACLE_PREFLIGHT = True
CONTINUE_AFTER_CASE_FAILURE = True
