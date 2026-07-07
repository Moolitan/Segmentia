"""用通用判官给现有 decode 结果打分，并对比不同 arm（如 recompute vs 复用）。

它把每个 case 的「任务要求」（从 trace 抽）+「产物」（从 decode TXT 抽）喂给判官，
得到 success 概率 + PASS/FAIL/PARTIAL + 问题标签，逐 case 写 CSV。若同一 case 有多个
arm（多个结果子目录），会并排打分，便于做"复用是否不劣于重算"的对比。

判官模型默认本地 vLLM（Qwen3），需要一个在跑的 OpenAI 兼容端点；用 JUDGE_BASE_URL /
JUDGE_MODEL / JUDGE_API_KEY 或命令行参数切换到 GPT / MiniMax 等。

示例：给 cross_task_contextual_occ1_reuse 的两个 arm 打分并对比
  python run_judge.py \
    --results-dir results/problem_exploration/cross_task_contextual_occ1_reuse \
    --arms recompute cross_contextual_rope \
    --out results/problem_exploration/cross_task_contextual_occ1_reuse/judge
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
MODULE_DIR = SCRIPT_DIR.parent
for p in (str(SCRIPT_DIR), str(MODULE_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from extract import agent_output_text, task_request_text  # noqa: E402
from judge import GeneralJudge  # noqa: E402
from trace_utils import load_invocations  # noqa: E402
from trace_utils import load_tools  # noqa: E402


def load_manifest(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def request_for_case(row: dict[str, Any]) -> str:
    """从 target_task 的 invocation_index 取任务要求。"""
    task = row["target_task"]
    inv_index = int(row["invocation_index"])
    invocation = load_invocations(task)[inv_index - 1]
    return task_request_text(invocation)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, required=True,
                        help="含各 arm 子目录（每个子目录有 manifest.jsonl + *.txt）")
    parser.add_argument("--arms", nargs="+", required=True,
                        help="要打分的 arm 子目录名，例如 recompute cross_contextual_rope")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--limit", type=int, default=0, help=">0 时只跑前 N 个 case（调试）")
    args = parser.parse_args()

    judge = GeneralJudge(base_url=args.base_url, model=args.model, api_key=args.api_key)
    tools = load_tools()
    args.out.mkdir(parents=True, exist_ok=True)

    # 以第一个 arm 的 manifest 为 case 基准，用 (skill, target_task) 对齐各 arm。
    base_manifest = load_manifest(args.results_dir / args.arms[0] / "manifest.jsonl")
    arm_index: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    for arm in args.arms:
        rows = load_manifest(args.results_dir / arm / "manifest.jsonl")
        arm_index[arm] = {(r["skill"], r["target_task"]): r for r in rows}

    summary_rows: list[dict[str, Any]] = []
    detail_path = args.out / "judge_details.jsonl"
    with detail_path.open("w", encoding="utf-8") as detail_fh:
        for n, base in enumerate(base_manifest):
            if args.limit and n >= args.limit:
                break
            key = (base["skill"], base["target_task"])
            request = request_for_case(base)
            row_out: dict[str, Any] = {"skill": base["skill"], "target_task": base["target_task"]}
            for arm in args.arms:
                rec = arm_index[arm].get(key)
                if rec is None:
                    continue
                raw = Path(rec["sequence_path"]).read_text(encoding="utf-8")
                output = agent_output_text(raw)
                result = judge.evaluate(request, output, tools=tools)
                row_out[f"{arm}__score"] = round(result.success_probability, 4)
                row_out[f"{arm}__verdict"] = result.verdict
                row_out[f"{arm}__issues"] = ";".join(i["label"] for i in result.issues)
                detail_fh.write(json.dumps({
                    "skill": base["skill"], "target_task": base["target_task"], "arm": arm,
                    "success_probability": result.success_probability, "verdict": result.verdict,
                    "derived_criteria": result.derived_criteria,
                    "checks": [c.__dict__ for c in result.checks],
                    "issues": result.issues, "rationale": result.rationale,
                }, ensure_ascii=False) + "\n")
                detail_fh.flush()
                print(f"[judged] {arm:22s} {base['skill']:22s} score={result.success_probability:.2f} {result.verdict}", flush=True)
            summary_rows.append(row_out)

    # 汇总 CSV
    fields = ["skill", "target_task"]
    for arm in args.arms:
        fields += [f"{arm}__score", f"{arm}__verdict", f"{arm}__issues"]
    csv_path = args.out / "judge_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in summary_rows:
            writer.writerow({k: r.get(k, "") for k in fields})
    print(f"[done] {csv_path}\n[done] {detail_path}", flush=True)


if __name__ == "__main__":
    main()
