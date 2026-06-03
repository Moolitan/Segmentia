from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

from core.agent import create_agent_and_llm
from core.benchmark import load_benchmark_sequence
from core.constants import ROOT
from core.runner import run_sequence
from core.skills import resolve_skills_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="多轮任务序列运行器(单序列执行)")
    parser.add_argument("--benchmark-repo", required=True, metavar="REPO")
    parser.add_argument("--bench-root", default=None, metavar="DIR")
    parser.add_argument(
        "--workspace",
        default=os.path.join(ROOT, "workspace", "03_14B_anthropic"),
    )
    parser.add_argument("--skills-dir", default=None)
    parser.add_argument(
        "--output",
        default=os.path.join(
            ROOT,
            "results",
            "03_14B_anthropic",
            "multurn_bench",
            "multiturn_sequence_traces.json",
        ),
    )
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument(
        "--log-dir", default=os.path.join(ROOT, "log", "03_14B_anthropic")
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    template = load_benchmark_sequence(args.benchmark_repo, bench_root=args.bench_root)

    seq_workspace = os.path.abspath(args.workspace)
    seq_log_path = os.path.join(args.log_dir, f"{args.benchmark_repo}.log")
    skills_dir = resolve_skills_dir(args.workspace, args.skills_dir)

    print(f"  序列: {args.benchmark_repo}")
    print(f"  turns:    {len(template.turns)}")
    print(f"  workspace: {seq_workspace}")
    print(f"  skills:    {skills_dir}")
    print(f"  log:       {seq_log_path}")
    print(f"  output:    {args.output}")

    if args.dry_run:
        print("\n[DRY-RUN] 仅预览,未实际运行。")
        return

    os.makedirs(args.log_dir, exist_ok=True)

    print(f"[INFO] 从 {skills_dir} 加载 Skills...")
    _, agent = create_agent_and_llm(skills_dir, args.vllm_port)

    print(f"\n{'=' * 60}")
    print(f"开始运行: {args.benchmark_repo}")
    print(f"{'=' * 60}")

    try:
        result = run_sequence(
            template=template,
            agent=agent,
            seq_workspace=seq_workspace,
            seq_log_path=seq_log_path,
        )

        output = {
            "benchmark_repo": args.benchmark_repo,
            "llm_calls": result["llm_calls"],
        }

        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        n_calls = len(result["llm_calls"])
        print(f"\n完成! 共 {n_calls} 次 LLM 调用")
        print(f"  结果: {args.output}")
        print(f"  日志: {seq_log_path}")

    except Exception as e:
        print(f"\n[ERROR] {args.benchmark_repo}): {e}")
        traceback.print_exc()
        sys.exit(1)
