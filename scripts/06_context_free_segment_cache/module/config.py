"""Module constants for context-free skill KV experiments.

This file intentionally duplicates the small amount of static configuration
needed from scripts/05_context_segment_agent_kv/core/config.py. The 06 harness
must not import 05 modules.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
TRACES_DIR = ROOT / "src" / "traces"
RESULTS_DIR = (
    ROOT
    / "results"
    / "problem_exploration"
)
SEGMENTIA_OUTPUT_DIR = Path(
    os.environ.get(
        "SEGMENTIA_OUTPUT_DIR",
        "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache",
    )
)
DEFAULT_KV_DIR = SEGMENTIA_OUTPUT_DIR / "offline_skill_kv"
DEFAULT_CONTEXTUAL_KV_DIR = SEGMENTIA_OUTPUT_DIR / "contextual_occ1_skill_kv"
DEFAULT_CKSIM_KV_DIR = SEGMENTIA_OUTPUT_DIR / "cksim_kv"
DEFAULT_OUTPUT_JSONL = (
    RESULTS_DIR / "headline_semantic_action_gap" / "data" / "decode_outputs.jsonl"
)
DEFAULT_METRICS_JSON = (
    RESULTS_DIR / "headline_semantic_action_gap" / "data" / "headline_summary.json"
)
DEFAULT_METRICS_CSV = (
    RESULTS_DIR / "headline_semantic_action_gap" / "tables" / "headline_metrics_rows.csv"
)
DEFAULT_STABILITY_CSV = (
    RESULTS_DIR
    / "headline_semantic_action_gap"
    / "tables"
    / "headline_stability_rows.csv"
)
DEFAULT_REPAIR_KV_DIR = SEGMENTIA_OUTPUT_DIR / "repair_arms_kv"

DEFAULT_MODEL_PATH = (
    "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
)
DEFAULT_SERVED_MODEL = "Qwen3"
DEFAULT_VLLM_PORT = 8000

DEFAULT_TASKS = [
    "internal_comms_incident_update",
    "doc_coauthoring_design_doc",
    "mcp_server_and_spec",
    "web_artifact_with_theme",
    "launch_poster_page_pack",
    "slack_launch_pack",
]

# Token spans are tied to Qwen3's chat template, src/traces/_system_prompt.txt,
# src/traces/_tools.json, and enable_thinking=True.
SKILL_TOKEN_LOCATIONS: dict[str, dict] = {
    "internal_comms_incident_update": {
        "skills": {
            "internal-comms": {
                "tokens": 338,
                "message_indices": [3, 19, 37],
                "token_spans": [[4145, 4483], [5774, 6112], [8115, 8453]],
                "invocation_indices": [2, 10, 19],
            },
        },
    },
    "doc_coauthoring_design_doc": {
        "skills": {
            "doc-coauthoring": {
                "tokens": 3313,
                "message_indices": [3, 21, 39],
                "token_spans": [[4168, 7481], [9196, 12509], [16981, 20294]],
                "invocation_indices": [2, 11, 20],
            },
        },
    },
    "mcp_server_and_spec": {
        "skills": {
            "mcp-builder": {
                "tokens": 1918,
                "message_indices": [3, 15, 41],
                "token_spans": [[4164, 6082], [7017, 8935], [19362, 21280]],
                "invocation_indices": [2, 8, 21],
            },
            "doc-coauthoring": {
                "tokens": 3313,
                "message_indices": [21, 27, 43],
                "token_spans": [[9559, 12872], [13643, 16956], [21353, 24666]],
                "invocation_indices": [11, 14, 22],
            },
        },
    },
    "web_artifact_with_theme": {
        "skills": {
            "web-artifacts-builder": {
                "tokens": 719,
                "message_indices": [3, 15, 41],
                "token_spans": [[4171, 4890], [6053, 6772], [14161, 14880]],
                "invocation_indices": [2, 8, 21],
            },
            "theme-factory": {
                "tokens": 669,
                "message_indices": [21, 27, 43],
                "token_spans": [[7604, 8273], [9568, 10237], [14950, 15619]],
                "invocation_indices": [11, 14, 22],
            },
        },
    },
    "launch_poster_page_pack": {
        "skills": {
            "canvas-design": {
                "tokens": 2356,
                "message_indices": [3, 29, 45],
                "token_spans": [[4143, 6499], [11179, 13535], [16538, 18894]],
                "invocation_indices": [2, 15, 23],
            },
            "web-artifacts-builder": {
                "tokens": 719,
                "message_indices": [11, 31, 47],
                "token_spans": [[7056, 7775], [13607, 14326], [18966, 19685]],
                "invocation_indices": [6, 16, 24],
            },
            "theme-factory": {
                "tokens": 669,
                "message_indices": [21, 33, 49],
                "token_spans": [[9212, 9881], [14396, 15065], [19755, 20424]],
                "invocation_indices": [11, 17, 25],
            },
        },
    },
    "slack_launch_pack": {
        "skills": {
            "internal-comms": {
                "tokens": 338,
                "message_indices": [3, 29, 47],
                "token_spans": [[4151, 4489], [8784, 9122], [13077, 13415]],
                "invocation_indices": [2, 15, 24],
            },
            "slack-gif-creator": {
                "tokens": 2090,
                "message_indices": [15, 31, 49],
                "token_spans": [[5196, 7286], [9197, 11287], [13490, 15580]],
                "invocation_indices": [8, 16, 25],
            },
            "brand-guidelines": {
                "tokens": 540,
                "message_indices": [21, 33, 51],
                "token_spans": [[7679, 8219], [11358, 11898], [15651, 16191]],
                "invocation_indices": [11, 17, 26],
            },
        },
    },
}


def parse_tasks(raw: str) -> list[str]:
    if raw == "all":
        return list(DEFAULT_TASKS)
    return [part.strip() for part in raw.split(",") if part.strip()]


def iter_task_skills(tasks: list[str]) -> list[tuple[str, str, dict]]:
    rows: list[tuple[str, str, dict]] = []
    for task in tasks:
        for skill, record in SKILL_TOKEN_LOCATIONS[task]["skills"].items():
            rows.append((task, skill, record))
    return rows


def cache_id_for_skill(skill: str) -> str:
    return f"context-free-skill-{skill}"


def get_skill_token_span(task: str, skill: str, occurrence: int) -> tuple[int, int]:
    spans = SKILL_TOKEN_LOCATIONS[task]["skills"][skill]["token_spans"]
    start, end = spans[occurrence - 1]
    return int(start), int(end)
