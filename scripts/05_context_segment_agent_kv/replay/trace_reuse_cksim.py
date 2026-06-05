from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Put the package root (.../05_context_segment_agent_kv) on sys.path so `core`
# resolves (distinct from the repo-root `core`).
PKG_ROOT = Path(__file__).resolve().parent.parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from core.config import DEFAULT_TASKS, ROOT, TRACES_DIR, get_skill_token_span  # noqa: E402
from core.message_convert import convert_messages, convert_tools  # noqa: E402
from core.segments import find_skill_segments  # noqa: E402
from core.trace_loader import load_invocations  # noqa: E402
from core.vllm_client import chat_completion  # noqa: E402

OUTPUT_ROOT = ROOT / "results" / "05_context_segment_agent_kv" / "CKSim"
DEFAULT_KV_SAVE_DIR = OUTPUT_ROOT / "kv_cache_trace"
CSV_PATH = OUTPUT_ROOT / "trace_reuse_cksim.csv"
JSON_PATH = OUTPUT_ROOT / "trace_reuse_cksim_summary.json"


@dataclass
class CKSimRow:
    comparison: str
    task: str
    skill_name: str
    occurrence: int
    skill_tokens: int
    layer: str
    key_cksim: float
    value_cksim: float
    key_token_mean: float
    value_token_mean: float


# --------------------------------------------------------------------------- #
# KV .pt loading + CKSim (same format/convention as skill_cksim_benchmark.py)
# --------------------------------------------------------------------------- #
def load_entry(kv_dir: Path, cache_id: str) -> dict[str, Any]:
    path = kv_dir / f"{cache_id}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def as_heads(x: torch.Tensor) -> torch.Tensor:
    if x.dim() != 3:
        raise ValueError(f"expected saved KV tensor with 3 dims, got {tuple(x.shape)}")
    # ContextSegmentKV stores [tokens, kv_heads, head_dim] -> [kv_heads, tokens, head_dim].
    return x.permute(1, 0, 2).contiguous()


def cksim(a: torch.Tensor, b: torch.Tensor, tokens: int) -> tuple[float, float]:
    import torch.nn.functional as F

    a_heads = as_heads(a[:tokens]).float()
    b_heads = as_heads(b[:tokens]).float()
    if a_heads.shape != b_heads.shape:
        raise ValueError(f"KV shape mismatch: {tuple(a.shape)} vs {tuple(b.shape)}")
    head_scores = F.cosine_similarity(a_heads.flatten(1), b_heads.flatten(1), dim=1)
    token_scores = F.cosine_similarity(a_heads, b_heads, dim=2)
    return float(head_scores.mean().item()), float(token_scores.mean().item())


def compare_kv_entries(
    kv_dir: Path,
    comparison: str,
    task: str,
    skill_name: str,
    occurrence: int,
    skill_tokens: int,
    recompute_cache_id: str,
    other_cache_id: str,
) -> list[CKSimRow]:
    recompute = load_entry(kv_dir, recompute_cache_id)
    other = load_entry(kv_dir, other_cache_id)
    rows: list[CKSimRow] = []
    for layer in sorted(set(recompute["kv_by_layer"]) & set(other["kv_by_layer"])):
        recompute_k, recompute_v = recompute["kv_by_layer"][layer]
        other_k, other_v = other["kv_by_layer"][layer]
        key_score, key_token_mean = cksim(recompute_k, other_k, skill_tokens)
        value_score, value_token_mean = cksim(recompute_v, other_v, skill_tokens)
        rows.append(
            CKSimRow(
                comparison=comparison,
                task=task,
                skill_name=skill_name,
                occurrence=occurrence,
                skill_tokens=skill_tokens,
                layer=str(layer),
                key_cksim=key_score,
                value_cksim=value_score,
                key_token_mean=key_token_mean,
                value_token_mean=value_token_mean,
            )
        )
    return rows


# --------------------------------------------------------------------------- #
# Request helpers (build context_segment_cache cfgs; send via core)
# --------------------------------------------------------------------------- #
def collect_source(base_url, model, msgs, tools, api_key, cache_id, start, end, req_id):
    cfg = {"sources": [{"cache_id": cache_id, "source_start": start, "source_end": end}]}
    chat_completion(base_url, model, msgs, tools, api_key,
                    max_tokens=1, request_id=req_id, context_segment_cache=cfg)


def dump_recompute(base_url, model, msgs, tools, api_key, cache_id, start, end_plus1, req_id):
    # Full-context recompute; dump the skill (+1) span KV.
    cfg = {"sources": [{"cache_id": cache_id, "source_start": start, "source_end": end_plus1}]}
    chat_completion(base_url, model, msgs, tools, api_key,
                    max_tokens=1, request_id=req_id, context_segment_cache=cfg)


def dump_reuse(base_url, model, msgs, tools, api_key,
               src_cache_id, dump_cache_id, start, end, end_plus1, req_id,
               mode="rope"):
    # Inject the saved source at [start,end), then dump the resulting injected
    # KV over [start,end_plus1). The +1 token forces registration to happen
    # after the injection has been applied.
    cfg = {
        "targets": [
            {"cache_id": src_cache_id, "mode": mode,
             "target_start": start, "target_end": end}
        ],
        "sources": [
            {"cache_id": dump_cache_id, "source_start": start, "source_end": end_plus1}
        ],
    }
    chat_completion(base_url, model, msgs, tools, api_key,
                    max_tokens=1, request_id=req_id, context_segment_cache=cfg)


# --------------------------------------------------------------------------- #
def write_outputs(rows: list[CKSimRow], metadata: dict[str, Any]) -> None:
    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(asdict(rows[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(r) for r in rows)

    by_comparison: dict[str, dict[str, Any]] = {}
    for cmp_name in sorted({r.comparison for r in rows}):
        cmp_rows = [r for r in rows if r.comparison == cmp_name]
        by_skill: dict[str, dict[str, float]] = {}
        for sk in sorted({r.skill_name for r in cmp_rows}):
            subset = [r for r in cmp_rows if r.skill_name == sk]
            by_skill[sk] = {
                "mean_key_cksim": sum(r.key_cksim for r in subset) / len(subset),
                "mean_value_cksim": sum(r.value_cksim for r in subset) / len(subset),
                "rows": len(subset),
            }
        by_comparison[cmp_name] = {
            "rows": len(cmp_rows),
            "mean_key_cksim": sum(r.key_cksim for r in cmp_rows) / len(cmp_rows),
            "mean_value_cksim": sum(r.value_cksim for r in cmp_rows) / len(cmp_rows),
            "by_skill": by_skill,
        }
    summary = {
        **metadata,
        "rows": len(rows),
        "by_comparison": by_comparison,
        "csv_path": str(CSV_PATH),
    }
    JSON_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] rows={len(rows)}")
    for cmp_name, stats in by_comparison.items():
        print(f"[done] {cmp_name} mean key   CKSim = {stats['mean_key_cksim']:.6f}")
        print(f"[done] {cmp_name} mean value CKSim = {stats['mean_value_cksim']:.6f}")
    print(f"[done] csv:  {CSV_PATH}")
    print(f"[done] json: {JSON_PATH}")


def parse_tasks(raw: str) -> list[str]:
    return DEFAULT_TASKS if raw == "all" else [t.strip() for t in raw.split(",") if t.strip()]


def iter_new_skill_occurrences(task: str, system_prompt: str):
    """Yield each first-seen (skill, occurrence) in trace order for a task."""
    seen: set[tuple[str, int]] = set()
    for inv in load_invocations(task):
        msgs, _ = convert_messages(inv["messages"], system_prompt)
        segments = find_skill_segments(msgs)
        local_count: dict[str, int] = {}
        for (name, _midx, _cs, _ce) in segments:
            local_count[name] = local_count.get(name, 0) + 1
            occ = local_count[name]
            key = (name, occ)
            if key in seen:
                continue
            seen.add(key)
            tstart, tend = get_skill_token_span(task, name, occ)
            yield inv, msgs, name, occ, tstart, tend


def recompute_cache_id(task: str, skill_name: str, occurrence: int) -> str:
    return f"cksim-recompute-{task}-{skill_name}-occ{occurrence}"


def reuse_source_cache_id(task: str, skill_name: str) -> str:
    return f"cksim-reuse-src-{task}-{skill_name}"


def reuse_cache_id(task: str, skill_name: str, occurrence: int) -> str:
    return f"cksim-reuse-{task}-{skill_name}-occ{occurrence}"


def reuse_no_rope_cache_id(task: str, skill_name: str, occurrence: int) -> str:
    return f"cksim-reuse-no-rope-{task}-{skill_name}-occ{occurrence}"


def run_recompute_phase(args, tasks: list[str], system_prompt: str, tools: list[dict]) -> None:
    base_url = f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    rid = 0

    def next_id() -> str:
        nonlocal rid
        rid += 1
        return f"cksim-recompute-{rid}"

    for task in tasks:
        print(f"\n--- recompute task={task} ---", flush=True)
        for _inv, msgs, name, occ, tstart, tend in iter_new_skill_occurrences(task, system_prompt):
            if occ == 1:
                continue
            cache_id = recompute_cache_id(task, name, occ)
            dump_recompute(
                base_url, args.model, msgs, tools, api_key,
                cache_id, tstart, tend + 1, next_id(),
            )
            print(f"  [dump recompute] {name:24s} occ{occ} span={tend - tstart} tok", flush=True)


def run_reuse_phase(
    args,
    tasks: list[str],
    system_prompt: str,
    tools: list[dict],
    *,
    mode: str,
    phase_label: str,
) -> None:
    base_url = f"http://127.0.0.1:{args.vllm_port}"
    api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    rid = 0

    def next_id(tag: str) -> str:
        nonlocal rid
        rid += 1
        return f"cksim-{phase_label}-{tag}-{rid}"

    for task in tasks:
        print(f"\n--- {phase_label} task={task} ---", flush=True)
        collected_src: dict[str, tuple[str, int]] = {}
        for _inv, msgs, name, occ, tstart, tend in iter_new_skill_occurrences(task, system_prompt):
            length = tend - tstart
            if occ == 1:
                src_cache = reuse_source_cache_id(task, name)
                collect_source(
                    base_url, args.model, msgs, tools, api_key,
                    src_cache, tstart, tend, next_id("src"),
                )
                collected_src[name] = (src_cache, length)
                print(f"  [src reuse]      {name:24s} occ1 span={length} tok", flush=True)
                continue

            if name not in collected_src:
                print(f"  [skip] {name} occ{occ}: no reuse source collected", flush=True)
                continue
            src_cache, src_len = collected_src[name]
            if src_len != length:
                print(f"  [skip] {name} occ{occ}: span len {length} != source {src_len}", flush=True)
                continue

            dump_cache = (
                reuse_no_rope_cache_id(task, name, occ)
                if mode == "direct"
                else reuse_cache_id(task, name, occ)
            )
            dump_reuse(
                base_url, args.model, msgs, tools, api_key,
                src_cache, dump_cache, tstart, tend, tend + 1, next_id("dump"),
                mode=mode,
            )
            print(
                f"  [dump {phase_label}] {name:24s} occ{occ} span={length} tok",
                flush=True,
            )


def summarize_phase(args, tasks: list[str]) -> None:
    kv_dir = Path(args.kv_dir)
    rows: list[CKSimRow] = []
    case_records: list[dict[str, Any]] = []
    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")

    for task in tasks:
        for _inv, _msgs, name, occ, tstart, tend in iter_new_skill_occurrences(task, system_prompt):
            if occ == 1:
                continue
            length = tend - tstart
            r_cache = recompute_cache_id(task, name, occ)
            comparisons = [
                ("recompute_vs_reuse", reuse_cache_id(task, name, occ)),
                ("recompute_vs_reuse_no_rope", reuse_no_rope_cache_id(task, name, occ)),
            ]
            case_record: dict[str, Any] = {
                "task": task,
                "skill_name": name,
                "occurrence": occ,
                "skill_tokens": length,
                "recompute_cache_id": r_cache,
                "comparisons": {},
            }
            for comparison, other_cache in comparisons:
                try:
                    case_rows = compare_kv_entries(
                        kv_dir, comparison, task, name, occ, length, r_cache, other_cache
                    )
                except FileNotFoundError as exc:
                    print(
                        f"  [skip] missing KV for {comparison} {task} {name} occ{occ}: {exc}",
                        flush=True,
                    )
                    continue
                rows.extend(case_rows)
                mk = sum(r.key_cksim for r in case_rows) / len(case_rows)
                mv = sum(r.value_cksim for r in case_rows) / len(case_rows)
                print(
                    f"  [cmp] {comparison:28s} {task:32s} {name:24s} occ{occ} "
                    f"span={length} tok key={mk:.6f} value={mv:.6f}",
                    flush=True,
                )
                case_record["comparisons"][comparison] = {
                    "layers": len(case_rows),
                    "mean_key_cksim": mk,
                    "mean_value_cksim": mv,
                    "cache_id": other_cache,
                }
            if case_record["comparisons"]:
                case_records.append(case_record)

    if not rows:
        raise RuntimeError(
            "No CKSim rows produced. Check that both recompute and reuse phases "
            "dumped .pt files to the same kv-dir."
        )
    write_outputs(
        rows,
        {
            "model": args.model,
            "tasks": tasks,
            "kv_cache_dir": str(kv_dir),
            "comparisons": ["recompute_vs_reuse", "recompute_vs_reuse_no_rope"],
            "cases": case_records,
        },
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trace-driven CKSim between independent recompute and reuse passes."
    )
    ap.add_argument(
        "--phase",
        choices=["recompute", "reuse", "reuse_no_rope", "summarize"],
        required=True,
        help="which phase to run; run_cksim.sh orchestrates all phases",
    )
    ap.add_argument("--tasks", default="all", help="comma list of task names, or 'all'")
    ap.add_argument("--vllm-port", type=int, default=int(os.environ.get("VLLM_PORT", "8000")))
    ap.add_argument("--model", default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"))
    ap.add_argument(
        "--kv-dir",
        default=os.environ.get("VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR", str(DEFAULT_KV_SAVE_DIR)),
        help="dir where the vLLM server dumps .pt KV (must match server env)",
    )
    args = ap.parse_args()

    tasks = parse_tasks(args.tasks)
    kv_dir = Path(args.kv_dir)
    if args.phase != "summarize" and not os.environ.get("VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR"):
        print(
            "[warn] VLLM_CONTEXT_SEGMENT_KV_SAVE_DIR not set in this process. The "
            "vLLM SERVER must have been started with it so .pt files are written "
            f"to {kv_dir}. Use run_cksim.sh if unsure.",
            flush=True,
        )

    if args.phase == "summarize":
        summarize_phase(args, tasks)
        return

    system_prompt = (TRACES_DIR / "_system_prompt.txt").read_text(encoding="utf-8")
    tools = convert_tools(json.loads((TRACES_DIR / "_tools.json").read_text(encoding="utf-8")))
    if args.phase == "recompute":
        run_recompute_phase(args, tasks, system_prompt, tools)
    elif args.phase == "reuse":
        run_reuse_phase(
            args, tasks, system_prompt, tools, mode="rope", phase_label="reuse"
        )
    elif args.phase == "reuse_no_rope":
        run_reuse_phase(
            args, tasks, system_prompt, tools, mode="direct", phase_label="reuse-no-rope"
        )


if __name__ == "__main__":
    main()
