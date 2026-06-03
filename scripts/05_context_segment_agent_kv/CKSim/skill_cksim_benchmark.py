#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import shutil
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]

from vllm import LLMEngine, SamplingParams
from vllm.engine.arg_utils import EngineArgs
from vllm.inputs import TokensPrompt
try:
    from vllm.tokenizers import get_tokenizer
except ImportError:
    from vllm.transformers_utils.tokenizer import get_tokenizer


MODEL = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
TP = 1
MAX_MODEL_LEN = 16384
GPU_MEM_UTIL = 0.9

SKILLS_DIR = ROOT / "skills"
OUTPUT_ROOT = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
KV_SAVE_DIR = OUTPUT_ROOT / "kv_cache"
CSV_PATH = OUTPUT_ROOT / "skill_cksim_results.csv"
JSON_PATH = OUTPUT_ROOT / "skill_cksim_summary.json"

SKILLS = [
    "internal-comms",
    "brand-guidelines",
    "canvas-design",
    "web-artifacts-builder",
    "theme-factory",
    "slack-gif-creator",
]
MAX_SKILL_TOKENS = 2048
WARMUP_TOKENS = 256
KEEP_KV_CACHE = False

@dataclass
class SkillCase:
    skill_name: str
    skill_tokens: list[int]

    @property
    def token_count(self) -> int:
        return len(self.skill_tokens)


@dataclass
class CKSimResult:
    comparison: str
    skill_name: str
    history_tokens: int
    skill_tokens: int
    layer: str
    key_cksim: float
    value_cksim: float
    key_token_mean: float
    value_token_mean: float


def tokenize_text(tokenizer: Any, text: str) -> list[int]:
    encoded = tokenizer.encode(text, add_special_tokens=False)
    return [int(x) for x in encoded]


def render_agent_history_text(rounds: int = 24) -> str:
    """Concrete agent-like history before a skill block.

    The benchmark uses token prompts instead of the OpenAI chat template, so this
    text explicitly carries the role/tool structure that would normally appear in
    an OpenHands LLM request.
    """
    chunks = [
        "<system>\n"
        "You are OpenHands, a software engineering agent. You can inspect files, "
        "run terminal commands, edit code, and use skills when a task matches a "
        "known workflow. Keep reasoning grounded in the workspace state.\n"
    ]
    scenarios = [
        (
            "Create a launch poster page and keep the copy aligned with the brand guide.",
            "list_directory",
            '{"path": "/workspace/seed_files"}',
            "seed_files contains brief.md, assets/, and package.json.",
            "I found the product brief and will inspect the design constraints before editing.",
        ),
        (
            "Draft an internal Slack update for the rollout and include risks.",
            "read_file",
            '{"path": "/workspace/seed_files/brief.md"}',
            "The rollout targets workspace admins, emphasizes reliability, and needs a concise launch message.",
            "I will use the internal communications style and keep the status update action-oriented.",
        ),
        (
            "Build the frontend artifact and verify it renders locally.",
            "grep",
            '{"pattern": "vite|next|react", "path": "/workspace/package.json"}',
            "package.json uses Vite with React and has scripts for dev, build, and preview.",
            "The app is a Vite React project, so I will edit src files and run the build script.",
        ),
        (
            "Generate a small themed visual asset for the hero section.",
            "read_file",
            '{"path": "/workspace/seed_files/brand.md"}',
            "Brand guidance prefers crisp geometric composition, high contrast, and restrained accent colors.",
            "I will keep the visual asset consistent with the brand constraints and avoid decorative clutter.",
        ),
    ]
    for i in range(rounds):
        user, tool, args, result, assistant = scenarios[i % len(scenarios)]
        chunks.append(
            f"<user turn={i + 1}>\n"
            f"Working directory: /workspace\n\n{user}\n"
            f"<assistant turn={i + 1}>\n"
            f"I will inspect the relevant project files and then apply the matching skill if needed.\n"
            f"<tool_call name=\"{tool}\" turn=\"{i + 1}\">\n{args}\n"
            f"<tool_result name=\"{tool}\" turn=\"{i + 1}\">\n{result}\n"
            f"<assistant turn={i + 1} followup>\n{assistant}\n"
        )
    return "\n".join(chunks)


def make_agent_history_tokens(tokenizer: Any) -> list[int]:
    return tokenize_text(tokenizer, render_agent_history_text())


def render_skill_text(skill_dir: Path) -> str:
    skill_md = skill_dir / "SKILL.md"
    text = skill_md.read_text(encoding="utf-8")
    resources: list[str] = []
    for sub in ["scripts", "references", "assets", "examples", "core"]:
        sub_dir = skill_dir / sub
        if sub_dir.is_dir():
            files = sorted(p.name for p in sub_dir.iterdir() if p.is_file())
            if files:
                resources.append(f"  {sub}/: {', '.join(files)}")
    if resources:
        text += "\n\n--- Skill Resources ---\n" + "\n".join(resources)
    return text


def wrap_skill(skill_name: str, text: str) -> str:
    return f'<context_segment id="{skill_name}">\n{text}\n</context_segment>'


def load_skill_cases(
    tokenizer: Any,
    skills_dir: Path,
    skill_names: list[str],
    max_skill_tokens: int,
) -> list[SkillCase]:
    cases: list[SkillCase] = []
    for skill_name in skill_names:
        skill_dir = skills_dir / skill_name
        if not (skill_dir / "SKILL.md").is_file():
            raise FileNotFoundError(skill_dir / "SKILL.md")
        tokens = tokenize_text(tokenizer, wrap_skill(skill_name, render_skill_text(skill_dir)))
        if max_skill_tokens > 0 and len(tokens) > max_skill_tokens:
            tokens = tokens[:max_skill_tokens]
        if not tokens:
            raise ValueError(f"skill {skill_name!r} tokenized to an empty segment")
        cases.append(SkillCase(skill_name=skill_name, skill_tokens=tokens))
    return cases


def _drain(engine: LLMEngine) -> None:
    while engine.has_unfinished_requests():
        engine.step()


def run_request(
    engine: LLMEngine,
    request_id: str,
    tokens: list[int],
    sampling_params: SamplingParams,
) -> float:
    engine.add_request(
        request_id=request_id,
        prompt=TokensPrompt(prompt_token_ids=tokens),
        params=sampling_params,
    )
    start = time.perf_counter()
    while engine.has_unfinished_requests():
        engine.step()
    return time.perf_counter() - start


def save_source_span(
    engine: LLMEngine,
    request_id: str,
    tokens: list[int],
    cache_id: str,
    source_start: int,
    source_end: int,
) -> float:
    cfg = json.dumps(
        {
            "sources": [
                {
                    "cache_id": cache_id,
                    "source_start": source_start,
                    "source_end": source_end,
                }
            ]
        }
    )
    return run_request(
        engine,
        request_id,
        tokens,
        SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extra_args={"context_segment_cache": cfg},
        ),
    )


def save_reused_span(
    engine: LLMEngine,
    request_id: str,
    tokens: list[int],
    offline_cache_id: str,
    saved_cache_id: str,
    target_start: int,
    target_end: int,
    source_end: int,
) -> float:
    cfg = json.dumps(
        {
            "targets": [
                {
                    "cache_id": offline_cache_id,
                    "mode": "rope",
                    "target_start": target_start,
                    "target_end": target_end,
                }
            ],
            "sources": [
                {
                    "cache_id": saved_cache_id,
                    "source_start": target_start,
                    "source_end": source_end,
                }
            ],
        }
    )
    return run_request(
        engine,
        request_id,
        tokens,
        SamplingParams(
            max_tokens=1,
            temperature=0.0,
            extra_args={"context_segment_cache": cfg},
        ),
    )


def load_entry(cache_id: str) -> dict[str, Any]:
    path = KV_SAVE_DIR / f"{cache_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return torch.load(path, map_location="cpu", weights_only=False)


def as_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected saved KV tensor with 3 dims, got {tuple(x.shape)}")
    # ContextSegmentKV stores [tokens, kv_heads, head_dim].
    return x.permute(1, 0, 2).contiguous()


def cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> tuple[float, float]:
    # Reuse requests save [skill + first assistant token] so registration happens
    # after target injection. CKSim is only for the skill span.
    a_heads = as_heads(a[:tokens]).float()
    b_heads = as_heads(b[:tokens]).float()
    if a_heads.shape != b_heads.shape:
        raise ValueError(f"KV shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    head_scores = F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
    token_scores = F.cosine_similarity(a_heads, b_heads, dim=2)
    return float(head_scores.mean().item()), float(token_scores.mean().item())


def compare_entries(
    comparison: str,
    skill_name: str,
    history_tokens: int,
    skill_tokens: int,
    left_cache_id: str,
    right_cache_id: str,
) -> list[CKSimResult]:
    left = load_entry(left_cache_id)
    right = load_entry(right_cache_id)
    rows: list[CKSimResult] = []
    for layer in sorted(set(left["kv_by_layer"]) & set(right["kv_by_layer"])):
        left_k, left_v = left["kv_by_layer"][layer]
        right_k, right_v = right["kv_by_layer"][layer]
        key_score, key_token_mean = cksim(left_k, right_k, skill_tokens)
        value_score, value_token_mean = cksim(left_v, right_v, skill_tokens)
        rows.append(
            CKSimResult(
                comparison=comparison,
                skill_name=skill_name,
                history_tokens=history_tokens,
                skill_tokens=skill_tokens,
                layer=layer,
                key_cksim=key_score,
                value_cksim=value_score,
                key_token_mean=key_token_mean,
                value_token_mean=value_token_mean,
            )
        )
    return rows


def write_outputs(rows: list[CKSimResult], metadata: dict[str, Any]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    key_mean = sum(r.key_cksim for r in rows) / len(rows)
    value_mean = sum(r.value_cksim for r in rows) / len(rows)
    by_history: dict[str, dict[str, float]] = {}
    for h in sorted({r.history_tokens for r in rows}):
        subset = [r for r in rows if r.history_tokens == h]
        by_history[str(h)] = {
            "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
            "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
            "rows": len(subset),
        }
    by_comparison: dict[str, dict[str, float]] = {}
    for comparison in sorted({r.comparison for r in rows}):
        subset = [r for r in rows if r.comparison == comparison]
        by_comparison[comparison] = {
            "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
            "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
            "rows": len(subset),
        }

    summary = {
        **metadata,
        "rows": len(rows),
        "mean_key_cksim": key_mean,
        "mean_value_cksim": value_mean,
        "by_history_tokens": by_history,
        "by_comparison": by_comparison,
        "csv_path": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[done] rows={len(rows)}")
    print(f"[done] mean key CKSim={key_mean:.6f}")
    print(f"[done] mean value CKSim={value_mean:.6f}")
    print(f"[done] csv: {CSV_PATH}")
    print(f"[done] json: {JSON_PATH}")


def main() -> None:
    if KV_SAVE_DIR.exists() and not KEEP_KV_CACHE:
        shutil.rmtree(KV_SAVE_DIR)
    KV_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"] = str(KV_SAVE_DIR)
    os.environ["VLLM_CONTEXT_SEGMENT_KV_DIR"] = str(KV_SAVE_DIR)
    print(f"[kv dir] {KV_SAVE_DIR}")

    tokenizer = get_tokenizer(MODEL, trust_remote_code=True)
    assistant_tokens = tokenize_text(
        tokenizer,
        "\n\n<assistant>\nI will use the skill above to complete the task.",
    )
    if not assistant_tokens:
        assistant_tokens = [3]
    history = make_agent_history_tokens(tokenizer)
    print(f"[history] concrete OpenHands-like transcript: {len(history)} tokens")
    skill_cases = load_skill_cases(
        tokenizer,
        SKILLS_DIR,
        SKILLS,
        MAX_SKILL_TOKENS,
    )
    print("[skills]")
    for case in skill_cases:
        print(f"  {case.skill_name}: {case.token_count} tokens")

    engine_args = EngineArgs(
        model=MODEL,
        tensor_parallel_size=TP,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=GPU_MEM_UTIL,
        enable_prefix_caching=True,
        enforce_eager=False,
    )
    engine = LLMEngine.from_engine_args(engine_args)

    req_counter = [0]

    def next_id(tag: str) -> str:
        rid = f"{tag}-{req_counter[0]}"
        req_counter[0] += 1
        return rid

    if WARMUP_TOKENS > 0:
        print(f"\n[warmup] {WARMUP_TOKENS} tokens")
        run_request(
            engine,
            next_id("warmup"),
            history[:WARMUP_TOKENS],
            SamplingParams(max_tokens=1, temperature=0.0),
        )

    print("\nPhase 1: offline skill prefill")
    for case in skill_cases:
        cache_id = f"cksim-offline-{case.skill_name}"
        save_source_span(
            engine,
            next_id("offline"),
            case.skill_tokens,
            cache_id,
            0,
            case.token_count,
        )
        print(f"  [offline] {case.skill_name}: {case.token_count} tokens")

    print("\nPhase 2: agent-like CKSim cases")
    results: list[CKSimResult] = []
    case_records: list[dict[str, Any]] = []
    for case in skill_cases:
        offline_cache_id = f"cksim-offline-{case.skill_name}"
        full_tokens = history + case.skill_tokens + assistant_tokens
        skill_start = len(history)
        skill_end = skill_start + case.token_count
        collect_end = skill_end + 1
        if len(full_tokens) > MAX_MODEL_LEN:
            print(
                f"  [skip] {case.skill_name}: total={len(full_tokens)} "
                f"> max_model_len={MAX_MODEL_LEN}"
            )
            continue

        base_cache_id = f"cksim-base-{case.skill_name}"
        reuse_cache_id = f"cksim-reuse-{case.skill_name}"

        base_elapsed = save_source_span(
            engine,
            next_id("base"),
            full_tokens,
            base_cache_id,
            skill_start,
            collect_end,
        )
        reuse_elapsed = save_reused_span(
            engine,
            next_id("reuse"),
            full_tokens,
            offline_cache_id,
            reuse_cache_id,
            skill_start,
            skill_end,
            collect_end,
        )
        rows = compare_entries(
            "offline_vs_base",
            case.skill_name,
            len(history),
            case.token_count,
            offline_cache_id,
            base_cache_id,
        )
        results.extend(rows)
        reuse_rows = compare_entries(
            "reuse_vs_base",
            case.skill_name,
            len(history),
            case.token_count,
            reuse_cache_id,
            base_cache_id,
        )
        results.extend(reuse_rows)
        case_records.append(
            {
                "skill_name": case.skill_name,
                "history_tokens": len(history),
                "skill_tokens": case.token_count,
                "total_tokens": len(full_tokens),
                "baseline_elapsed_ms": round(base_elapsed * 1000, 2),
                "reuse_elapsed_ms": round(reuse_elapsed * 1000, 2),
                "layer_rows": len(rows) + len(reuse_rows),
            }
        )
        mean_key = sum(r.key_cksim for r in rows) / len(rows)
        mean_value = sum(r.value_cksim for r in rows) / len(rows)
        reuse_mean_key = sum(r.key_cksim for r in reuse_rows) / len(reuse_rows)
        reuse_mean_value = sum(r.value_cksim for r in reuse_rows) / len(reuse_rows)
        print(
            f"  {case.skill_name:24s} history={len(history):5d} "
            f"skill={case.token_count:5d} "
            f"offline/base key={mean_key:.6f} value={mean_value:.6f} "
            f"reuse/base key={reuse_mean_key:.6f} value={reuse_mean_value:.6f}"
        )

    if not results:
        raise RuntimeError("No CKSim rows produced; check history sizes and max_model_len")

    write_outputs(
        results,
        {
            "model": MODEL,
            "skills_dir": str(SKILLS_DIR),
            "skills": SKILLS,
            "history_kind": "concrete_openhands_like_transcript",
            "history_tokens": len(history),
            "max_skill_tokens": MAX_SKILL_TOKENS,
            "primary_comparison": "offline_vs_base",
            "kv_cache_dir": str(KV_SAVE_DIR),
            "cases": case_records,
        },
    )


if __name__ == "__main__":
    main()
