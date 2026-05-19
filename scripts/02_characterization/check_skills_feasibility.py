"""
可行性检查：SWE-Bench traces 中 activated_skills 字段非空率。

用法：
    python scripts/check_skills_feasibility.py <eval-dir> [--sample N]

参数：
    eval-dir    解压后的评估结果目录，包含 conversations/ 子目录
    --sample N  只抽取前 N 个对话检查（默认 10，设为 0 表示全量）

输出：
    - MessageEvent 总数及 activated_skills 非空率
    - 出现过的所有 Skill 名称及频次
    - 结论：走路径A（直接离线分析）还是路径B（需构建新工作负载）
使用：
    全量检查
    python scripts/check_skills_feasibility.py  eval_outputs/eth-sri__SWT-bench_Verified_bm25_27k_zsp-test/litellm_proxy/jade-spark-2862_sdk_cfe52af_maxiter_500_N_SWT-litellm_proxy-jade-spark-2862 --sample 0 
"""

import argparse
import json
import os
import shutil
import sys
import tarfile
import tempfile
from collections import Counter


def extract_conversation(tarpath: str, dest: str) -> str | None:
    """解压单个对话 .tar.gz，返回其中 events/ 目录的路径。"""
    with tarfile.open(tarpath, "r:gz") as tf:
        tf.extractall(dest, filter="data")
    for root, _, _ in os.walk(dest):
        if os.path.basename(root) == "events":
            return root
    return None


def check_conversation(tarpath: str) -> dict:
    """检查单个对话中的 activated_skills 情况。"""
    tmpdir = tempfile.mkdtemp()
    try:
        events_dir = extract_conversation(tarpath, tmpdir)
        if events_dir is None:
            return {"ok": False, "error": "no events dir"}

        total_msg = 0
        non_empty_count = 0
        skill_names: list[str] = []

        for fname in sorted(os.listdir(events_dir)):
            if not fname.endswith(".json"):
                continue
            fpath = os.path.join(events_dir, fname)
            try:
                with open(fpath) as f:
                    obj = json.load(f)
            except Exception:
                continue

            if obj.get("kind") != "MessageEvent":
                continue

            total_msg += 1
            skills = obj.get("activated_skills", [])
            if skills:
                non_empty_count += 1
                skill_names.extend(skills)

        return {
            "ok": True,
            "instance_id": os.path.basename(tarpath).replace(".tar.gz", ""),
            "total_message_events": total_msg,
            "non_empty_count": non_empty_count,
            "skill_names": skill_names,
        }
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("eval_dir", help="解压后的评估结果目录")
    parser.add_argument(
        "--sample",
        type=int,
        default=10,
        help="抽取前 N 个对话（0 = 全量，默认 10）",
    )
    args = parser.parse_args()

    conv_dir = os.path.join(args.eval_dir, "conversations")
    if not os.path.isdir(conv_dir):
        print(f"[错误] 找不到 conversations/ 目录：{conv_dir}")
        sys.exit(1)

    all_tarballs = sorted(
        os.path.join(conv_dir, f)
        for f in os.listdir(conv_dir)
        if f.endswith(".tar.gz")
    )
    tarballs = all_tarballs[: args.sample] if args.sample > 0 else all_tarballs

    print(f"检查 {len(tarballs)} 个对话（共 {len(all_tarballs)} 个）...\n")

    total_msg = 0
    total_non_empty = 0
    all_skill_names: list[str] = []

    for i, tarpath in enumerate(tarballs, 1):
        result = check_conversation(tarpath)
        if not result["ok"]:
            print(f"  [{i:3d}] ✗ {os.path.basename(tarpath)} — {result['error']}")
            continue

        total_msg += result["total_message_events"]
        total_non_empty += result["non_empty_count"]
        all_skill_names.extend(result["skill_names"])

        status = "✓" if result["non_empty_count"] > 0 else "○"
        print(
            f"  [{i:3d}] {status}  {result['instance_id'][:56]:<56}"
            f"  msg={result['total_message_events']:3d}"
            f"  skills_activated={result['non_empty_count']:3d}"
        )

    # 汇总
    print("\n" + "=" * 70)
    print(f"MessageEvent 总数        : {total_msg}")
    print(f"activated_skills 非空数  : {total_non_empty}")
    rate = total_non_empty / total_msg * 100 if total_msg > 0 else 0.0
    print(f"非空率                   : {rate:.1f}%")

    if all_skill_names:
        print(f"\n出现的 Skill 名称（共 {len(set(all_skill_names))} 种）：")
        for name, count in Counter(all_skill_names).most_common():
            print(f"  {count:5d}x  {name}")
    else:
        print("\n未发现任何激活的 Skill。")

    # 结论
    print("\n" + "=" * 70)
    if rate > 20:
        print("✅ 路径A — activated_skills 数据充足，可直接离线分析。")
    else:
        print("🔧 路径B — activated_skills 基本为空，需构建 Skill-rich 工作负载。")
        print("   参考 docs/week1-2-characterization-plan.md 路径B 说明。")


if __name__ == "__main__":
    main()
