"""
方法 1+2: Token-level logit divergence trajectory + Branch point analysis.

分两阶段：
  阶段 1 (collect): 按 mode 运行，每个 case 保存 top-k logprobs 到 JSON。
                     需要分 mode 重启 vLLM（recompute 不加载 KV，rope 加载 KV）。
  阶段 2 (analyze): 加载两个 mode 的 logprobs JSON，计算逐 token JSD 和分叉点分析。

用法：
  # 阶段 1：分别收集（由 shell 脚本调度，分 mode 重启 vLLM）
  python run_logit_divergence.py collect --task <task> --mode recompute
  python run_logit_divergence.py collect --task <task> --mode rope

  # 阶段 2：分析（不需要 vLLM）
  python run_logit_divergence.py analyze
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PACKAGE_DIR = SCRIPT_DIR.parents[1]
MODULE_DIR = PACKAGE_DIR / "module"
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

DEFAULT_OUTPUT_DIR = "/mnt/Large_Language_Model_Lab_1/wsh/Segmentia/output/06_context_free_segment_cache/cross_occurrence_function_vector/logit_divergence"

SUPPORTED_MODES = ("recompute", "rope")
OCCURRENCES = (3,)
TOP_LOGPROBS_K = 20


def safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def case_label(case: dict[str, Any]) -> str:
    return (
        f"inv{int(case['invocation_index']):03d}--"
        f"{safe_component(str(case['task']))}--"
        f"{safe_component(str(case['skill']))}--"
        f"occ{int(case['occurrence'])}"
    )


# ──────────────────────────────────────────────────────────
# 阶段 1: Collect logprobs
# ──────────────────────────────────────────────────────────

def extract_token_logprobs(response: dict[str, Any]) -> list[dict[str, Any]]:
    choices = response.get("choices") or []
    if len(choices) != 1:
        raise ValueError(f"Expected exactly one choice, got {len(choices)}")
    logprobs = choices[0].get("logprobs") or {}
    entries = logprobs.get("content")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Response is missing non-empty choices[0].logprobs.content")

    result = []
    for entry in entries:
        token = str(entry["token"])
        logprob = float(entry.get("logprob", 0.0))
        top = []
        for t in (entry.get("top_logprobs") or []):
            top.append({"token": str(t["token"]), "logprob": float(t["logprob"])})
        result.append({"token": token, "logprob": logprob, "top_logprobs": top})
    return result


def cmd_collect(args: argparse.Namespace) -> None:
    from config import (
        DEFAULT_KV_DIR,
        DEFAULT_SERVED_MODEL,
        DEFAULT_TASKS,
        DEFAULT_VLLM_PORT,
    )
    from replay import context_config_for_case, selected_cases
    from trace_utils import convert_messages, load_invocations, load_system_prompt, load_tools
    from vllm_client import chat_completion

    cases = selected_cases(
        [args.task],
        list(OCCURRENCES),
        include_first_occurrence=True,
    )
    if not cases:
        raise ValueError(f"No cases found for task={args.task}")

    system_prompt = load_system_prompt()
    tools = load_tools()
    invocations = load_invocations(args.task)
    base_url = args.base_url or f"http://127.0.0.1:{args.vllm_port}"
    # run_name 让同一 mode 的多次采样运行（如 recompute_run1 / recompute_run2）
    # 各自落到独立子目录，互不覆盖；默认回退到 mode，保持旧行为。
    run_name = args.run_name or args.mode
    logprobs_dir = args.output_dir / "logprobs" / run_name
    logprobs_dir.mkdir(parents=True, exist_ok=True)

    for case in cases:
        invocation_index = int(case["invocation_index"])
        invocation = invocations[invocation_index - 1]
        messages, _ = convert_messages(invocation["messages"], system_prompt)
        label = case_label(case)

        output_path = logprobs_dir / f"{label}.json"
        if output_path.exists() and not args.overwrite:
            print(f"[skip] {args.mode} {label} (already exists)", flush=True)
            continue

        cfg = context_config_for_case(args.mode, case, dump_kv_for_cksim=False)
        request_id = (
            f"cf-logit-div-{args.mode}-{case['task']}-{case['skill']}"
            f"-occ{case['occurrence']}-inv{invocation_index}"
        )

        response, elapsed = chat_completion(
            base_url,
            args.model,
            messages,
            tools,
            args.api_key,
            max_tokens=args.max_tokens,
            request_id=request_id,
            context_segment_cache=cfg,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            min_p=args.min_p,
            seed=args.seed,
            logprobs=True,
            top_logprobs=TOP_LOGPROBS_K,
        )
        token_data = extract_token_logprobs(response)

        with output_path.open("w", encoding="utf-8") as f:
            json.dump(token_data, f, ensure_ascii=False)

        print(
            f"[saved] {args.mode} {label} tokens={len(token_data)} "
            f"elapsed={elapsed:.1f}s path={output_path}",
            flush=True,
        )


# ──────────────────────────────────────────────────────────
# 阶段 2: Analyze pairs
# ──────────────────────────────────────────────────────────

def top_k_to_prob_dict(top_logprobs: list[dict]) -> dict[str, float]:
    d = {}
    for entry in top_logprobs:
        d[entry["token"]] = math.exp(entry["logprob"])
    return d


def jensen_shannon_divergence(
    p_top: list[dict], q_top: list[dict], epsilon: float = 1e-10
) -> float:
    p_probs = top_k_to_prob_dict(p_top)
    q_probs = top_k_to_prob_dict(q_top)
    all_tokens = set(p_probs.keys()) | set(q_probs.keys())
    if not all_tokens:
        return 0.0

    m_probs = {}
    for tok in all_tokens:
        m_probs[tok] = 0.5 * p_probs.get(tok, epsilon) + 0.5 * q_probs.get(tok, epsilon)

    kl_pm = 0.0
    kl_qm = 0.0
    for tok in all_tokens:
        p = p_probs.get(tok, epsilon)
        q = q_probs.get(tok, epsilon)
        m = m_probs[tok]
        if p > epsilon:
            kl_pm += p * math.log(p / m)
        if q > epsilon:
            kl_qm += q * math.log(q / m)
    return max(0.5 * kl_pm + 0.5 * kl_qm, 0.0)


def find_think_end(tokens: list[dict[str, Any]]) -> int | None:
    for i, t in enumerate(tokens):
        if "</think>" in t["token"]:
            return i
    return None


def find_first_divergence(
    rc_tokens: list[dict[str, Any]], rp_tokens: list[dict[str, Any]]
) -> int | None:
    min_len = min(len(rc_tokens), len(rp_tokens))
    for i in range(min_len):
        if rc_tokens[i]["token"] != rp_tokens[i]["token"]:
            return i
    if len(rc_tokens) != len(rp_tokens):
        return min_len
    return None


def analyze_pair(
    label: str,
    rc_tokens: list[dict[str, Any]],
    rp_tokens: list[dict[str, Any]],
    per_token_dir: Path,
) -> dict[str, Any]:
    min_len = min(len(rc_tokens), len(rp_tokens))

    jsd_per_token = []
    for i in range(min_len):
        jsd = jensen_shannon_divergence(
            rc_tokens[i]["top_logprobs"],
            rp_tokens[i]["top_logprobs"],
        )
        jsd_per_token.append({
            "index": i,
            "rc_token": rc_tokens[i]["token"],
            "rp_token": rp_tokens[i]["token"],
            "jsd": round(jsd, 6),
            "tokens_match": rc_tokens[i]["token"] == rp_tokens[i]["token"],
        })

    rc_think_end = find_think_end(rc_tokens)
    rp_think_end = find_think_end(rp_tokens)
    first_div = find_first_divergence(rc_tokens, rp_tokens)

    think_boundary = min(
        rc_think_end if rc_think_end is not None else min_len,
        rp_think_end if rp_think_end is not None else min_len,
        min_len,
    )
    think_jsds = [e["jsd"] for e in jsd_per_token[:think_boundary]]
    action_jsds = [e["jsd"] for e in jsd_per_token[think_boundary:]]

    branch_info = None
    if first_div is not None and first_div < min_len:
        rc_top = rc_tokens[first_div]["top_logprobs"]
        rp_top = rp_tokens[first_div]["top_logprobs"]
        rc_probs = top_k_to_prob_dict(rc_top)
        rp_probs = top_k_to_prob_dict(rp_top)

        rc_chosen = rc_tokens[first_div]["token"]
        rp_chosen = rp_tokens[first_div]["token"]
        branch_info = {
            "index": first_div,
            "in_thinking": first_div < think_boundary,
            "rc_chosen_token": rc_chosen,
            "rp_chosen_token": rp_chosen,
            "rc_chosen_prob_in_rc": round(rc_probs.get(rc_chosen, 0.0), 6),
            "rp_chosen_prob_in_rp": round(rp_probs.get(rp_chosen, 0.0), 6),
            "rc_chosen_prob_in_rp": round(rp_probs.get(rc_chosen, 0.0), 6),
            "rp_chosen_prob_in_rc": round(rc_probs.get(rp_chosen, 0.0), 6),
            "rc_top5": [
                {"token": t["token"], "prob": round(math.exp(t["logprob"]), 6)}
                for t in rc_top[:5]
            ],
            "rp_top5": [
                {"token": t["token"], "prob": round(math.exp(t["logprob"]), 6)}
                for t in rp_top[:5]
            ],
            "jsd_at_branch": round(jsd_per_token[first_div]["jsd"], 6),
        }

    summary = {
        "label": label,
        "rc_total_tokens": len(rc_tokens),
        "rp_total_tokens": len(rp_tokens),
        "rc_think_end": rc_think_end,
        "rp_think_end": rp_think_end,
        "first_divergence_index": first_div,
        "think_jsd_mean": round(float(np.mean(think_jsds)), 6) if think_jsds else None,
        "think_jsd_max": round(float(np.max(think_jsds)), 6) if think_jsds else None,
        "action_jsd_mean": round(float(np.mean(action_jsds)), 6) if action_jsds else None,
        "action_jsd_max": round(float(np.max(action_jsds)), 6) if action_jsds else None,
        "overall_jsd_mean": round(float(np.mean([e["jsd"] for e in jsd_per_token])), 6),
        "branch_point": branch_info,
    }

    detail_path = per_token_dir / f"{label}.json"
    detail_path.parent.mkdir(parents=True, exist_ok=True)
    with detail_path.open("w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "summary": summary,
            "per_token_jsd": jsd_per_token,
        }, f, ensure_ascii=False, indent=2)

    return summary


def cmd_analyze(args: argparse.Namespace) -> None:
    tag = args.tag or f"{args.rc_name}__vs__{args.rp_name}"
    rc_dir = args.output_dir / "logprobs" / args.rc_name
    rp_dir = args.output_dir / "logprobs" / args.rp_name
    per_token_dir = args.output_dir / f"per_token_{tag}"

    if not rc_dir.exists() or not rp_dir.exists():
        raise FileNotFoundError(
            f"Need both {rc_dir} and {rp_dir}. Run 'collect' for both runs first."
        )

    rc_files = {f.stem: f for f in sorted(rc_dir.glob("*.json"))}
    rp_files = {f.stem: f for f in sorted(rp_dir.glob("*.json"))}
    common = sorted(set(rc_files.keys()) & set(rp_files.keys()))

    if not common:
        raise FileNotFoundError("No matching pairs found between recompute and rope.")

    print(f"Found {len(common)} pairs to analyze\n")

    all_summaries = []

    for label in common:
        with rc_files[label].open(encoding="utf-8") as f:
            rc_tokens = json.load(f)
        with rp_files[label].open(encoding="utf-8") as f:
            rp_tokens = json.load(f)

        print(f"{'='*80}")
        print(f"[pair] {label}")
        print(f"  recompute: {len(rc_tokens)} tokens, rope: {len(rp_tokens)} tokens")

        summary = analyze_pair(label, rc_tokens, rp_tokens, per_token_dir)
        all_summaries.append(summary)

        bp = summary.get("branch_point")
        if bp:
            phase = "THINKING" if bp["in_thinking"] else "ACTION"
            print(f"  First divergence at token {bp['index']} ({phase})")
            print(f"    RC chose: {repr(bp['rc_chosen_token'])} (prob={bp['rc_chosen_prob_in_rc']:.4f})")
            print(f"    RP chose: {repr(bp['rp_chosen_token'])} (prob={bp['rp_chosen_prob_in_rp']:.4f})")
            print(f"    RC's token prob in RP: {bp['rc_chosen_prob_in_rp']:.4f}")
            print(f"    RP's token prob in RC: {bp['rp_chosen_prob_in_rc']:.4f}")
            print(f"    JSD at branch: {bp['jsd_at_branch']:.6f}")
            print(f"    RC top5: {bp['rc_top5']}")
            print(f"    RP top5: {bp['rp_top5']}")
        else:
            print("  No divergence (identical sequences)")

        print(f"  Think JSD: mean={summary['think_jsd_mean']}, max={summary['think_jsd_max']}")
        print(f"  Action JSD: mean={summary['action_jsd_mean']}, max={summary['action_jsd_max']}")

    # 汇总 CSV（按 pair tag 命名，避免不同对照互相覆盖）
    csv_path = args.output_dir / f"divergence_summary_{tag}.csv"
    with csv_path.open("w", encoding="utf-8") as f:
        cols = [
            "label", "rc_tokens", "rp_tokens",
            "first_div_idx", "div_in_thinking",
            "think_jsd_mean", "think_jsd_max",
            "action_jsd_mean", "action_jsd_max",
            "overall_jsd_mean",
            "rc_chosen", "rp_chosen",
            "rc_prob", "rp_prob",
            "rc_in_rp", "rp_in_rc",
            "jsd_at_branch",
        ]
        f.write(",".join(cols) + "\n")
        for s in all_summaries:
            bp = s.get("branch_point") or {}
            row = [
                s["label"],
                str(s["rc_total_tokens"]),
                str(s["rp_total_tokens"]),
                str(s.get("first_divergence_index", "")),
                str(bp.get("in_thinking", "")),
                str(s.get("think_jsd_mean", "")),
                str(s.get("think_jsd_max", "")),
                str(s.get("action_jsd_mean", "")),
                str(s.get("action_jsd_max", "")),
                str(s.get("overall_jsd_mean", "")),
                repr(bp.get("rc_chosen_token", "")),
                repr(bp.get("rp_chosen_token", "")),
                str(bp.get("rc_chosen_prob_in_rc", "")),
                str(bp.get("rp_chosen_prob_in_rp", "")),
                str(bp.get("rc_chosen_prob_in_rp", "")),
                str(bp.get("rp_chosen_prob_in_rc", "")),
                str(bp.get("jsd_at_branch", "")),
            ]
            f.write(",".join(row) + "\n")

    # 汇总统计
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")
    think_means = [s["think_jsd_mean"] for s in all_summaries if s["think_jsd_mean"] is not None]
    action_means = [s["action_jsd_mean"] for s in all_summaries if s["action_jsd_mean"] is not None]
    div_in_think = sum(1 for s in all_summaries if s.get("branch_point", {}).get("in_thinking"))
    div_in_action = sum(1 for s in all_summaries if s.get("branch_point") and not s["branch_point"].get("in_thinking"))
    no_div = sum(1 for s in all_summaries if s.get("branch_point") is None)

    if think_means:
        arr = np.array(think_means)
        print(f"  Think JSD mean:   mean={arr.mean():.6f}  median={np.median(arr):.6f}  max={arr.max():.6f}")
    if action_means:
        arr = np.array(action_means)
        print(f"  Action JSD mean:  mean={arr.mean():.6f}  median={np.median(arr):.6f}  max={arr.max():.6f}")
    print(f"  First divergence: {div_in_think} in thinking, {div_in_action} in action, {no_div} identical")

    # 分叉点概率差距
    margins = []
    for s in all_summaries:
        bp = s.get("branch_point")
        if bp:
            margin = abs(bp["rc_chosen_prob_in_rc"] - bp["rc_chosen_prob_in_rp"])
            margins.append(margin)
    if margins:
        arr = np.array(margins)
        print(f"  Branch prob margin (|P_rc(chosen) - P_rp(chosen)|):")
        print(f"    mean={arr.mean():.4f}  median={np.median(arr):.4f}  min={arr.min():.4f}  max={arr.max():.4f}")
        narrow = sum(1 for m in margins if m < 0.05)
        print(f"    Narrow margin (<5%): {narrow}/{len(margins)}")

    print(f"\n[done] pair: {args.rc_name} (rc) vs {args.rp_name} (rp)")
    print(f"[done] per-token data: {per_token_dir}")
    print(f"[done] summary CSV:    {csv_path}")


# ──────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    # collect subcommand
    p_collect = subparsers.add_parser("collect", help="Collect top-k logprobs for one mode")
    p_collect.add_argument("--task", required=True)
    p_collect.add_argument("--mode", choices=sorted(SUPPORTED_MODES), required=True)
    p_collect.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_collect.add_argument("--base-url", default=None)
    p_collect.add_argument("--vllm-port", type=int, default=8000)
    p_collect.add_argument(
        "--model",
        default=os.environ.get("VLLM_SERVED_NAME", "Qwen3"),
    )
    p_collect.add_argument("--api-key", default=os.environ.get("VLLM_API_KEY", "EMPTY"))
    p_collect.add_argument("--max-tokens", type=int, default=4096)
    # 采样参数：默认保持贪心（temperature=0，top_p=1，top_k/min_p/seed 不下发），
    # 与旧的 temp=0 实验行为完全一致。
    p_collect.add_argument("--temperature", type=float, default=0.0)
    p_collect.add_argument("--top-p", type=float, default=1.0)
    p_collect.add_argument("--top-k", type=int, default=None)
    p_collect.add_argument("--min-p", type=float, default=None)
    p_collect.add_argument("--seed", type=int, default=None)
    p_collect.add_argument(
        "--run-name",
        default=None,
        help="logprobs 子目录名（默认=mode）；多次采样运行用它区分，如 recompute_run1。",
    )
    p_collect.add_argument("--overwrite", action="store_true")

    # analyze subcommand
    p_analyze = subparsers.add_parser("analyze", help="Analyze collected logprobs pairs")
    p_analyze.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p_analyze.add_argument(
        "--rc-name", default="recompute",
        help="作为 rc（基准）的 logprobs 子目录名。",
    )
    p_analyze.add_argument(
        "--rp-name", default="rope",
        help="作为 rp（对照）的 logprobs 子目录名。",
    )
    p_analyze.add_argument(
        "--tag", default=None,
        help="输出文件后缀（默认 <rc-name>__vs__<rp-name>）。",
    )

    args = parser.parse_args()

    if args.command == "collect":
        cmd_collect(args)
    elif args.command == "analyze":
        cmd_analyze(args)


if __name__ == "__main__":
    main()