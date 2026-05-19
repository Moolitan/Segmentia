"""
离线模拟 Skill 激活：对每个 SWE-Bench 实例，基于 keyword trigger 匹配，
计算会触发哪些 Skills 以及对应的 <EXTRA_INFO> 注入 token 开销。

用于 SkillCache 论文的跨请求（across conversations）KV Cache 低效特征化：
不同实例触发不同 Skill 组合 → system prompt 不同 → prefix cache 无法共享。

用法：
    python scripts/characterization/simulate_skill_activation.py <output-jsonl> [选项]

参数：
    output-jsonl          评估结果的 output.jsonl 文件路径

选项：
    --skills-dir DIR      Skills 目录（默认 skills/）
    --output PATH         输出 JSON 路径（默认 results/skill_activation_stats.json）
    --sample N            只处理前 N 个实例（0 = 全部，默认 0）

示例：
    # 快速测试
    python scripts/characterization/simulate_skill_activation.py \\
        eval_outputs/.../output.jsonl --sample 10

    # 全量运行
    python scripts/characterization/simulate_skill_activation.py \\
        eval_outputs/.../output.jsonl \\
        --output results/skill_activation_stats.json
"""

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations

import tiktoken

# 确保能 import openhands SDK
SDK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "software-agent-sdk", "openhands-sdk"
)
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from openhands.sdk.context.skills import load_skills_from_dir  # noqa: E402


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def extract_issue_text(obj: dict) -> str:
    """从 output.jsonl 的一条记录中提取 issue 文本。

    复用自 scan_issue_keywords.py 的模式。
    """
    instruction = obj.get("instruction", "")
    match = re.search(
        r"<issue_description>(.*?)</issue_description>", instruction, re.DOTALL
    )
    if match:
        return match.group(1)
    text = re.sub(
        r"<uploaded_files>.*?</uploaded_files>", "", instruction, flags=re.DOTALL
    )
    return text.strip()


def load_instances(jsonl_path: str, sample: int = 0) -> list[dict]:
    """读取 output.jsonl，返回 [{instance_id, issue_text}]。"""
    instances = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            instance_id = obj.get("instance_id", "")
            issue_text = extract_issue_text(obj)
            if instance_id and issue_text:
                instances.append(
                    {"instance_id": instance_id, "issue_text": issue_text}
                )
    if sample > 0:
        instances = instances[:sample]
    return instances


# ---------------------------------------------------------------------------
# Skill 激活模拟
# ---------------------------------------------------------------------------

def simulate_activation(skills: list, issue_text: str) -> list[dict]:
    """对一个实例运行所有 skill 的 match_trigger。

    返回 [{"name": str, "trigger_keyword": str}]。
    """
    activated = []
    for skill in skills:
        trigger = skill.match_trigger(issue_text)
        if trigger:
            activated.append({"name": skill.name, "trigger_keyword": trigger})
    return activated


# ---------------------------------------------------------------------------
# Token 计数
# ---------------------------------------------------------------------------

def render_extra_info(skill, trigger_keyword: str) -> str:
    """按 skill_knowledge_info.j2 模板渲染单个 Skill 的 EXTRA_INFO 块。"""
    parts = [
        "<EXTRA_INFO>",
        f'The following information has been included based on a keyword match for "{trigger_keyword}".',
        "It may or may not be relevant to the user's request.",
    ]
    if skill.source:
        parts.append(f"Skill location: {skill.source}")
        parts.append(
            "(Use this path to resolve relative file references in the skill content below)"
        )
    parts.append("")
    parts.append(skill.content)
    parts.append("</EXTRA_INFO>")
    return "\n".join(parts)


def compute_tokens(text: str, encoder: tiktoken.Encoding) -> int:
    """计算文本的 token 数。"""
    return len(encoder.encode(text))


# ---------------------------------------------------------------------------
# 哈希 & 聚合
# ---------------------------------------------------------------------------

def skill_set_hash(skill_names: list[str]) -> str:
    """sorted skill names → MD5 前 12 位。"""
    canonical = "|".join(sorted(skill_names))
    return hashlib.md5(canonical.encode()).hexdigest()[:12]


def compute_aggregate(per_instance: list[dict], skill_names: list[str]) -> dict:
    """计算聚合统计。"""
    total = len(per_instance)

    # 每个 skill 的激活频率
    skill_counter: Counter = Counter()
    for inst in per_instance:
        for s in inst["activated_skills"]:
            skill_counter[s] += 1
    skill_frequency = {
        name: {"count": skill_counter[name], "pct": round(skill_counter[name] / total * 100, 1)}
        for name in sorted(skill_names)
    }

    # 激活数量分布
    act_counts = [inst["num_skills_activated"] for inst in per_instance]
    activation_count_dist = dict(sorted(Counter(act_counts).items()))

    # 唯一 skill 组合
    hash_counter: Counter = Counter()
    hash_to_set: dict[str, list[str]] = {}
    for inst in per_instance:
        h = inst["skill_set_hash"]
        hash_counter[h] += 1
        if h not in hash_to_set:
            hash_to_set[h] = sorted(inst["activated_skills"])

    skill_set_frequency = [
        {"skill_set": hash_to_set[h], "hash": h, "count": c}
        for h, c in hash_counter.most_common()
    ]

    # 共现矩阵
    co_occurrence: dict[str, dict[str, int]] = {n: {} for n in skill_names}
    for inst in per_instance:
        skills = inst["activated_skills"]
        for a, b in combinations(sorted(skills), 2):
            co_occurrence[a][b] = co_occurrence[a].get(b, 0) + 1
            co_occurrence[b][a] = co_occurrence[b].get(a, 0) + 1

    # Token 统计
    extra_tokens = [inst["extra_info_tokens"] for inst in per_instance]
    issue_tokens = [inst["issue_text_tokens"] for inst in per_instance]
    token_stats = {
        "extra_info_tokens_mean": round(sum(extra_tokens) / total, 1),
        "extra_info_tokens_median": round(sorted(extra_tokens)[total // 2], 1),
        "extra_info_tokens_std": round(
            (sum((x - sum(extra_tokens) / total) ** 2 for x in extra_tokens) / total) ** 0.5,
            1,
        ),
        "extra_info_tokens_min": min(extra_tokens),
        "extra_info_tokens_max": max(extra_tokens),
        "issue_text_tokens_mean": round(sum(issue_tokens) / total, 1),
    }

    # 多样性指标
    diversity = compute_diversity(per_instance)

    return {
        "skill_frequency": skill_frequency,
        "activation_count_distribution": {str(k): v for k, v in activation_count_dist.items()},
        "unique_skill_combinations": len(hash_counter),
        "skill_set_frequency": skill_set_frequency,
        "co_occurrence_matrix": co_occurrence,
        "token_stats": token_stats,
        "diversity_metrics": diversity,
    }


def compute_diversity(per_instance: list[dict]) -> dict:
    """计算跨实例 Jaccard 距离和归一化熵。"""
    n = len(per_instance)
    skill_sets = [set(inst["activated_skills"]) for inst in per_instance]

    # Jaccard 距离（采样以避免 O(n^2) 过慢，433 choose 2 ≈ 93k，可以全算）
    jaccard_dists = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = skill_sets[i], skill_sets[j]
            if not a and not b:
                jaccard_dists.append(0.0)
            else:
                jaccard_dists.append(1.0 - len(a & b) / len(a | b))

    jaccard_mean = sum(jaccard_dists) / len(jaccard_dists) if jaccard_dists else 0.0
    jaccard_std = (
        (sum((x - jaccard_mean) ** 2 for x in jaccard_dists) / len(jaccard_dists)) ** 0.5
        if jaccard_dists
        else 0.0
    )

    # 归一化熵（skill 组合分布的均匀程度）
    hash_counts = Counter(inst["skill_set_hash"] for inst in per_instance)
    total = sum(hash_counts.values())
    k = len(hash_counts)
    if k <= 1:
        normalized_entropy = 0.0
    else:
        entropy = -sum(
            (c / total) * math.log2(c / total) for c in hash_counts.values()
        )
        normalized_entropy = entropy / math.log2(k)

    return {
        "jaccard_mean": round(jaccard_mean, 4),
        "jaccard_std": round(jaccard_std, 4),
        "normalized_entropy": round(normalized_entropy, 4),
    }


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------

def print_summary(aggregate: dict, total_instances: int, total_skills: int) -> None:
    """打印人类可读摘要。"""
    print("\n" + "=" * 70)
    print(f"实例数: {total_instances}    Skills 数: {total_skills}")
    print(f"唯一 Skill 组合数: {aggregate['unique_skill_combinations']}")
    print()

    print("Skill 激活频率:")
    for name, info in sorted(
        aggregate["skill_frequency"].items(), key=lambda x: -x[1]["count"]
    ):
        bar = "█" * (info["count"] * 40 // total_instances)
        print(f"  {name:<25s} {info['count']:4d} ({info['pct']:5.1f}%)  {bar}")
    print()

    print("每实例激活 Skill 数分布:")
    for k, v in sorted(aggregate["activation_count_distribution"].items(), key=lambda x: int(x[0])):
        print(f"  {k:>2s} skills: {v:4d} 个实例")
    print()

    ts = aggregate["token_stats"]
    print(f"<EXTRA_INFO> 注入 token 统计:")
    print(f"  mean={ts['extra_info_tokens_mean']:.0f}  "
          f"median={ts['extra_info_tokens_median']:.0f}  "
          f"std={ts['extra_info_tokens_std']:.0f}  "
          f"min={ts['extra_info_tokens_min']}  "
          f"max={ts['extra_info_tokens_max']}")
    print(f"  issue 文本 token 均值: {ts['issue_text_tokens_mean']:.0f}")
    print()

    dm = aggregate["diversity_metrics"]
    print(f"跨请求多样性指标:")
    print(f"  Jaccard 距离均值: {dm['jaccard_mean']:.4f}  (std={dm['jaccard_std']:.4f})")
    print(f"  归一化熵:         {dm['normalized_entropy']:.4f}")
    print()

    print(f"Top 10 最常见 Skill 组合:")
    for i, combo in enumerate(aggregate["skill_set_frequency"][:10], 1):
        skills_str = ", ".join(combo["skill_set"]) if combo["skill_set"] else "(无)"
        print(f"  {i:2d}. [{combo['count']:3d}x] {skills_str}")

    print("=" * 70)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_jsonl", help="output.jsonl 文件路径")
    parser.add_argument(
        "--skills-dir", default="skills/", help="Skills 目录（默认 skills/）"
    )
    parser.add_argument(
        "--output",
        default="results/characterization/skill_activation_stats.json",
        help="输出 JSON 路径（默认 results/characterization/skill_activation_stats.json）",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="只处理前 N 个实例（0 = 全部，默认 0）",
    )
    args = parser.parse_args()

    # 加载 Skills
    skills_dir = os.path.abspath(args.skills_dir)
    print(f"加载 Skills: {skills_dir}")
    _, _, agent_skills = load_skills_from_dir(skills_dir)
    all_skills = list(agent_skills.values())
    skill_names = sorted(s.name for s in all_skills)
    print(f"  已加载 {len(all_skills)} 个 Skills: {skill_names}")

    if not all_skills:
        print("[错误] 未加载到任何 Skill，请检查 --skills-dir 路径。")
        sys.exit(1)

    # 加载实例
    print(f"\n加载实例: {args.output_jsonl}")
    instances = load_instances(args.output_jsonl, args.sample)
    print(f"  共 {len(instances)} 个实例")

    if not instances:
        print("[错误] 未读取到任何实例。")
        sys.exit(1)

    # 初始化 tokenizer
    enc = tiktoken.get_encoding("cl100k_base")

    # 模拟激活
    print(f"\n模拟 Skill 激活...")
    per_instance = []
    for i, inst in enumerate(instances):
        activated = simulate_activation(all_skills, inst["issue_text"])
        activated_names = [a["name"] for a in activated]

        # 渲染 EXTRA_INFO 并计算 token
        extra_info_text = ""
        for a in activated:
            skill_obj = next(s for s in all_skills if s.name == a["name"])
            extra_info_text += render_extra_info(skill_obj, a["trigger_keyword"]) + "\n"

        extra_info_tokens = compute_tokens(extra_info_text, enc) if extra_info_text else 0
        issue_tokens = compute_tokens(inst["issue_text"], enc)

        per_instance.append({
            "instance_id": inst["instance_id"],
            "activated_skills": sorted(activated_names),
            "trigger_details": [
                {"skill": a["name"], "matched_keyword": a["trigger_keyword"]}
                for a in activated
            ],
            "num_skills_activated": len(activated),
            "skill_set_hash": skill_set_hash(activated_names),
            "extra_info_tokens": extra_info_tokens,
            "issue_text_tokens": issue_tokens,
        })

        if (i + 1) % 50 == 0:
            print(f"  已处理 {i + 1}/{len(instances)} 个实例...")

    print(f"  完成，共 {len(per_instance)} 个实例。")

    # 聚合统计
    aggregate = compute_aggregate(per_instance, skill_names)

    # 组装输出
    output = {
        "metadata": {
            "total_instances": len(per_instance),
            "total_skills": len(all_skills),
            "skill_names": skill_names,
            "skills_dir": skills_dir,
            "output_jsonl": os.path.abspath(args.output_jsonl),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tokenizer": "cl100k_base",
        },
        "per_instance": per_instance,
        "aggregate": aggregate,
    }

    # 写入
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n结果已写入: {args.output}")

    # 打印摘要
    print_summary(aggregate, len(per_instance), len(all_skills))


if __name__ == "__main__":
    main()
