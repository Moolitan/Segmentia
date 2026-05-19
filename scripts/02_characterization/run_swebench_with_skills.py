import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timezone
import pprint

# 确保能 import openhands SDK
SDK_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "software-agent-sdk", "openhands-sdk"
)
if os.path.isdir(SDK_PATH):
    sys.path.insert(0, SDK_PATH)

from openhands.sdk import LLM, Agent, Conversation, Tool, AgentContext  # noqa: E402
from openhands.sdk.context.skills import load_skills_from_dir  # noqa: E402
from openhands.sdk.event.base import Event  # noqa: E402
from openhands.sdk.event.llm_convertible.message import MessageEvent  # noqa: E402
from openhands.sdk.event.llm_convertible.action import ActionEvent  # noqa: E402
from openhands.sdk.event.llm_convertible.observation import ObservationEvent  # noqa: E402

# 必须显式 import 工具模块,触发工具注册
from openhands.tools.terminal import TerminalTool  # noqa: E402, F401
from openhands.tools.file_editor import FileEditorTool  # noqa: E402, F401
from openhands.tools.glob import GlobTool  # noqa: E402, F401
from openhands.tools.grep import GrepTool  # noqa: E402, F401
from openhands.tools.apply_patch import ApplyPatchTool  # noqa: E402, F401


# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------

def extract_issue_text(obj: dict) -> str:
    """从 output.jsonl 的一条记录中提取 issue 文本。"""
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


def load_instances(jsonl_path: str) -> dict[str, dict]:
    """读取 output.jsonl,返回 {instance_id: {instruction, issue_text}} 字典。"""
    instances = {}
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
            if instance_id:
                instances[instance_id] = {
                    "instruction": obj.get("instruction", ""),
                    "issue_text": extract_issue_text(obj),
                }
    return instances


# ---------------------------------------------------------------------------
# 事件收集回调
# ---------------------------------------------------------------------------

class EventCollector:
    """收集会话中所有事件,提取 per-turn skill 激活与实际使用信息。"""

    def __init__(self):
        self.events: list[dict] = []
        self.turn = 0

    def __call__(self, event: Event) -> None:
        """回调函数：每个事件触发时调用。"""
        try:
            entry = {
                "turn": self.turn,
                "event_type": type(event).__name__,
                "event_id": str(event.id) if hasattr(event, "id") else None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

            if isinstance(event, MessageEvent):
                entry["source"] = event.source
                entry["activated_skills"] = list(event.activated_skills) if event.activated_skills else []
                entry["has_extended_content"] = bool(event.extended_content)
                # llm_message 是 Message 对象(content 为 Sequence[TextContent|ImageContent])
                if event.llm_message and hasattr(event.llm_message, "content"):
                    entry["message_length"] = sum(
                        len(c.text) if hasattr(c, "text") else 0
                        for c in event.llm_message.content
                    )
                else:
                    entry["message_length"] = 0
                if event.source == "user":
                    self.turn += 1

            elif isinstance(event, ActionEvent):
                entry["tool_name"] = event.tool_call.name if hasattr(event, "tool_call") and event.tool_call else None
                # 尽量捕获 tool 参数（用于判断是否出现过读取 SKILL.md 等行为）
                tool_call = getattr(event, "tool_call", None)
                tool_args = None
                for key in ("arguments", "args", "input", "params"):
                    if tool_call is not None and hasattr(tool_call, key):
                        tool_args = getattr(tool_call, key)
                        break
                if tool_args is not None:
                    try:
                        s = json.dumps(tool_args, ensure_ascii=False)
                    except Exception:
                        s = str(tool_args)
                    entry["tool_args_preview"] = s[:8000]

            elif isinstance(event, ObservationEvent):
                entry["observation_length"] = len(str(event.content)) if hasattr(event, "content") else 0

            self.events.append(entry)
        except Exception as e:
            # 回调不应阻断会话运行
            self.events.append({
                "turn": self.turn,
                "event_type": type(event).__name__,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

    def get_skill_trace(self) -> list[dict]:
        """提取按 turn 组织的 skill 激活 trace。"""
        trace = []
        for e in self.events:
            if e["event_type"] == "MessageEvent" and e.get("activated_skills"):
                trace.append({
                    "turn": e["turn"],
                    "source": e["source"],
                    "activated_skills": e["activated_skills"],
                    "has_extended_content": e["has_extended_content"],
                })
        return trace

    def get_used_skills(self) -> list[dict]:
        """从 tool_args_preview 中检测 agent 是否实际读取了 SKILL.md 文件。

        返回列表，每项包含 turn、skill_name、tool_args_preview。
        """
        used = []
        for e in self.events:
            if e["event_type"] != "ActionEvent":
                continue
            args_preview = e.get("tool_args_preview", "")
            if not args_preview:
                continue
            m = _SKILL_PATH_RE.search(args_preview)
            if m:
                used.append({
                    "turn": e["turn"],
                    "skill_name": m.group(1),
                    "tool_args_preview": args_preview[:500],
                })
        return used


# ---------------------------------------------------------------------------
# 日志文件事后分析：检测 agent 是否实际读取了 SKILL.md
# ---------------------------------------------------------------------------

_SKILL_PATH_RE = re.compile(r"skills/([^/]+)/SKILL\.md")

_LOG_SKILL_READ_RE = re.compile(
    r'path:\s*"?([^"\n]*skills/([^/]+)/SKILL\.md)"?',
)


def analyze_log_for_skill_usage(log_path: str) -> list[dict]:
    """从日志文件中检测 file_editor view SKILL.md 的操作。

    这是对 EventCollector.get_used_skills() 的补充：
    当 tool_args_preview 为空时（如早期运行的 trace），
    可以从日志文件中解析出实际的 SKILL.md 读取行为。

    返回列表，每项包含 skill_name 和 matched_line。
    """
    if not os.path.isfile(log_path):
        return []
    results = []
    seen = set()
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _LOG_SKILL_READ_RE.search(line)
                if m:
                    skill_name = m.group(2)
                    if skill_name not in seen:
                        seen.add(skill_name)
                        results.append({
                            "skill_name": skill_name,
                            "matched_line": line.strip()[:200],
                        })
    except Exception:
        pass
    return results


# ---------------------------------------------------------------------------
# LLM 配置
# ---------------------------------------------------------------------------

def create_llm(args) -> "LLM":
    """根据 CLI 参数创建 LLM 实例。"""
    return LLM(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
        temperature=0.0,
        max_output_tokens=4096,  # 限制单次生成长度,防止模型无限输出
        stream=False,
        native_tool_calling=True,
        caching_prompt=True,
        drop_params=True,
        modify_params=True,
        disable_vision=True,
        # log_completions=True,
        # log_completions_folder=args.log_dir,
    )


# ---------------------------------------------------------------------------
# 单实例运行
# ---------------------------------------------------------------------------

def run_single_instance(
    instance_id: str,
    instruction: str,
    skills: list,
    llm: "LLM",
    workspace: str,
    max_iterations: int,
    log_dir: str,
) -> dict:
    """运行单个 SWE-Bench 实例,返回事件 trace。

    SDK 的终端输出(Agent Action / Observation)会重定向到
    {log_dir}/{instance_id}.log 文件。
    """

    collector = EventCollector()

    agent = Agent(
        llm=llm,
        tools=[
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=GlobTool.name),
            Tool(name=GrepTool.name),
            Tool(name=ApplyPatchTool.name),
        ],
        include_default_tools=["FinishTool", "ThinkTool"],
        tool_concurrency_limit=1,
        agent_context=AgentContext(
            skills=skills,
        ),
    )

    conversation = Conversation(
        agent=agent,
        workspace=workspace,
        callbacks=[collector],
        max_iteration_per_run=max_iterations,
        stuck_detection=True,
        delete_on_close=True,
    )

    start_time = time.time()

    # 将 SDK 终端输出(stdout + stderr)重定向到日志文件,去除 ANSI 颜色转义码
    log_path = os.path.join(log_dir, f"{instance_id}.log")
    saved_stdout = sys.stdout
    saved_stderr = sys.stderr

    _ansi_re = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

    class _StripAnsiWriter:
        """写入文件前去除 ANSI 转义码的包装器。"""
        def __init__(self, f):
            self._f = f
        def write(self, s):
            return self._f.write(_ansi_re.sub("", s))
        def flush(self):
            self._f.flush()

    try:
        log_file = open(log_path, "w", encoding="utf-8")
        writer = _StripAnsiWriter(log_file)
        sys.stdout = writer
        sys.stderr = writer

        # 发送 SWE-Bench 指令(包含 <uploaded_files> + <issue_description>)
        conversation.send_message(instruction)
        conversation.run()
    except Exception as e:
        # 恢复 stdout 以便主脚本能打印
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        print(f"  [错误] {instance_id}: {e}")
    finally:
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        elapsed = time.time() - start_time
        try:
            log_file.close()
        except Exception:
            pass
        try:
            conversation.close()
        except Exception:
            pass

    # 组装结果
    skill_trace = collector.get_skill_trace()
    all_activated = set()
    for t in skill_trace:
        all_activated.update(t["activated_skills"])

    # 检测 agent 是否实际读取了 SKILL.md（从事件 + 日志两个渠道）
    used_from_events = collector.get_used_skills()
    used_from_log = analyze_log_for_skill_usage(log_path)

    # 合并去重
    all_used_names = set()
    for u in used_from_events:
        all_used_names.add(u["skill_name"])
    for u in used_from_log:
        all_used_names.add(u["skill_name"])

    return {
        "instance_id": instance_id,
        "total_events": len(collector.events),
        "total_turns": collector.turn,
        "elapsed_seconds": round(elapsed, 1),
        "skill_trace": skill_trace,
        "all_activated_skills": sorted(all_activated),
        "num_unique_skills": len(all_activated),
        # 实际使用（agent 主动读取 SKILL.md）
        "all_used_skills": sorted(all_used_names),
        "num_used_skills": len(all_used_names),
        "skill_usage_from_events": used_from_events,
        "skill_usage_from_log": used_from_log,
        "events": collector.events,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("output_jsonl", help="output.jsonl 文件路径")
    parser.add_argument(
        "--instance-id",
        default=None,
        help="要运行的实例 ID(逗号分隔多个)",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        help="随机抽样 N 个实例运行(0 = 不抽样,默认 0)",
    )
    parser.add_argument(
        "--workspace",
        required=True,
        help="预初始化的 workspace 目录路径",
    )
    parser.add_argument(
        "--skills-dir",
        default=None,
        help="Skills 目录(默认 {workspace}/.agents/skills/)",
    )
    parser.add_argument(
        "--output",
        default="results/characterization/swebench_skill_traces",
        help="输出目录(默认 results/characterization/swebench_skill_traces)",
    )
    parser.add_argument(
        "--emit-activation-stats",
        default=None,
        help="额外输出一个用于可视化的统计 JSON（默认 <output>/skill_activation_stats.json）",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=20,
        help="每个实例最大迭代数(默认 30,节省时间)",
    )
    parser.add_argument(
        "--model",
        default="openai/Qwen3",
        help="LLM 模型名(默认 openai/Qwen3)",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="LLM API base URL(默认 http://localhost:$VLLM_PORT/v1)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="LLM API key(默认 $VLLM_API_KEY 或 EMPTY)",
    )
    parser.add_argument(
        "--log-dir",
        default="log",
        help="LLM 日志目录",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子(用于 --sample 抽样)",
    )
    args = parser.parse_args()

    # 默认值处理
    if args.base_url is None:
        vllm_port = os.environ.get("VLLM_PORT", "8000")
        args.base_url = f"http://localhost:{vllm_port}/v1"
    if args.api_key is None:
        args.api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

    # 加载 Skills(默认从 workspace/.agents/skills/ 加载,和 01_quick_start.py 一致)
    if args.skills_dir is None:
        args.skills_dir = os.path.join(args.workspace, ".agents", "skills")
    skills_dir = os.path.abspath(args.skills_dir)
    print(f"加载 Skills: {skills_dir}")
    _, _, agent_skills = load_skills_from_dir(skills_dir)
    all_skills = list(agent_skills.values())
    # pprint.pprint(all_skills)
    skill_names = sorted(s.name for s in all_skills)
    print(f"已加载 {len(all_skills)} 个 Skills: {skill_names}")

    if not all_skills:
        print("[错误] 未加载到任何 Skill。")
        sys.exit(1)

    # 加载实例
    print(f"\n加载实例: {args.output_jsonl}")
    all_instances = load_instances(args.output_jsonl)
    print(f"  共 {len(all_instances)} 个实例")

    # 确定要运行的实例
    if args.instance_id:
        target_ids = [x.strip() for x in args.instance_id.split(",")]
        missing = [x for x in target_ids if x not in all_instances]
        if missing:
            print(f"[警告] 以下实例未找到: {missing}")
        target_ids = [x for x in target_ids if x in all_instances]
    elif args.sample > 0:
        random.seed(args.seed)
        available = list(all_instances.keys())
        target_ids = random.sample(available, min(args.sample, len(available)))
    else:
        # 无筛选条件 → 运行全部实例
        target_ids = list(all_instances.keys())
        print(f"[提示] 未指定 --instance-id 或 --sample,将运行全部 {len(target_ids)} 个实例。")

    print(f"\n将运行 {len(target_ids)} 个实例:")
    for iid in target_ids:
        print(f"  - {iid}")

    # 创建 LLM
    llm = create_llm(args)

    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    if args.emit_activation_stats is None:
        args.emit_activation_stats = os.path.join(args.output, "skill_activation_stats.json")

    # 逐个运行
    results = []
    for idx, instance_id in enumerate(target_ids, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(target_ids)}] 运行实例: {instance_id}")
        print(f"{'='*60}")

        inst = all_instances[instance_id]
        # print(inst["instruction"])
        # print(f"\n{'='*60}")
        result = run_single_instance(
            instance_id=instance_id,
            instruction=inst["instruction"],
            skills=all_skills,
            llm=llm,
            workspace=args.workspace,
            max_iterations=args.max_iterations,
            log_dir=args.log_dir,
        )
        results.append(result)

        # 每个实例单独保存
        trace_path = os.path.join(args.output, f"{instance_id}.json")
        with open(trace_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"  保存: {trace_path}")

        # 打印摘要
        print(f"  事件数: {result['total_events']}")
        print(f"  总 turn 数: {result['total_turns']}")
        print(f"  激活 Skills (关键词匹配): {result['all_activated_skills']}")
        print(f"  使用 Skills (读取SKILL.md): {result['all_used_skills']}")
        print(f"  耗时: {result['elapsed_seconds']}s")

    # 汇总保存
    summary_path = os.path.join(args.output, "_summary.json")
    summary = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_instances_run": len(results),
            "skills_dir": skills_dir,
            "skill_names": skill_names,
            "model": args.model,
            "max_iterations": args.max_iterations,
            "source": "swebench_real_run",
            "sample": args.sample,
            "seed": args.seed,
        },
        "per_instance": [
            {
                "instance_id": r["instance_id"],
                "total_turns": r["total_turns"],
                "total_events": r["total_events"],
                "elapsed_seconds": r["elapsed_seconds"],
                "all_activated_skills": r["all_activated_skills"],
                "num_unique_skills": r["num_unique_skills"],
                "skill_trace": r["skill_trace"],
                "all_used_skills": r["all_used_skills"],
                "num_used_skills": r["num_used_skills"],
            }
            for r in results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 额外生成一个轻量统计文件，便于直接画激活和使用的图
    activated_any = [r for r in results if r["num_unique_skills"] > 0]
    activated_none = [r for r in results if r["num_unique_skills"] == 0]
    used_any = [r for r in results if r["num_used_skills"] > 0]
    used_none = [r for r in results if r["num_used_skills"] == 0]

    # 激活频率（关键词触发）
    skill_freq = {name: 0 for name in skill_names}
    for r in results:
        for s in r["all_activated_skills"]:
            if s in skill_freq:
                skill_freq[s] += 1
            else:
                skill_freq[s] = skill_freq.get(s, 0) + 1
    activation_count_dist: dict[int, int] = {}
    for r in results:
        k = int(r["num_unique_skills"])
        activation_count_dist[k] = activation_count_dist.get(k, 0) + 1

    # 使用频率（agent 实际读取 SKILL.md）
    skill_usage_freq = {name: 0 for name in skill_names}
    for r in results:
        for s in r["all_used_skills"]:
            if s in skill_usage_freq:
                skill_usage_freq[s] += 1
            else:
                skill_usage_freq[s] = skill_usage_freq.get(s, 0) + 1
    usage_count_dist: dict[int, int] = {}
    for r in results:
        k = int(r["num_used_skills"])
        usage_count_dist[k] = usage_count_dist.get(k, 0) + 1

    n_total = len(results) if results else 0
    stats = {
        "metadata": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": "swebench_real_run",
            "model": args.model,
            "total_instances": n_total,
            "total_skills": len(skill_names),
            "skill_names": skill_names,
            "skills_dir": skills_dir,
            "max_iterations": args.max_iterations,
            "sample": args.sample,
            "seed": args.seed,
            "output_jsonl": os.path.abspath(args.output_jsonl),
        },
        "aggregate": {
            # --- 激活（关键词匹配触发，注入到 system prompt） ---
            "any_skill_activation": {
                "count": len(activated_any),
                "pct": (len(activated_any) / n_total * 100.0) if n_total else 0.0,
            },
            "no_skill_activation": {
                "count": len(activated_none),
                "pct": (len(activated_none) / n_total * 100.0) if n_total else 0.0,
            },
            "activation_count_distribution": {str(k): v for k, v in sorted(activation_count_dist.items())},
            "skill_frequency": {
                name: {
                    "count": int(skill_freq.get(name, 0)),
                    "pct": (float(skill_freq.get(name, 0)) / n_total * 100.0) if n_total else 0.0,
                }
                for name in skill_names
            },
            # --- 使用（agent 主动读取 SKILL.md） ---
            "any_skill_usage": {
                "count": len(used_any),
                "pct": (len(used_any) / n_total * 100.0) if n_total else 0.0,
            },
            "no_skill_usage": {
                "count": len(used_none),
                "pct": (len(used_none) / n_total * 100.0) if n_total else 0.0,
            },
            "usage_count_distribution": {str(k): v for k, v in sorted(usage_count_dist.items())},
            "skill_usage_frequency": {
                name: {
                    "count": int(skill_usage_freq.get(name, 0)),
                    "pct": (float(skill_usage_freq.get(name, 0)) / n_total * 100.0) if n_total else 0.0,
                }
                for name in skill_names
            },
        },
        "per_instance": [
            {
                "instance_id": r["instance_id"],
                "activated_skills": r["all_activated_skills"],
                "num_unique_skills": r["num_unique_skills"],
                "used_skills": r["all_used_skills"],
                "num_used_skills": r["num_used_skills"],
                "total_turns": r["total_turns"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in results
        ],
        "instances_with_skill_activation": [
            {
                "instance_id": r["instance_id"],
                "activated_skills": r["all_activated_skills"],
                "num_unique_skills": r["num_unique_skills"],
                "skill_trace": r["skill_trace"],
            }
            for r in activated_any
        ],
        "instances_with_skill_usage": [
            {
                "instance_id": r["instance_id"],
                "used_skills": r["all_used_skills"],
                "num_used_skills": r["num_used_skills"],
                "skill_usage_from_events": r["skill_usage_from_events"],
                "skill_usage_from_log": r["skill_usage_from_log"],
            }
            for r in used_any
        ],
    }
    with open(args.emit_activation_stats, "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"统计 JSON: {args.emit_activation_stats}")

    print(f"\n{'='*60}")
    print(f"完成！共运行 {len(results)} 个实例")
    print(f"详细 trace: {args.output}/<instance_id>.json")
    print(f"汇总: {summary_path}")

    # 打印 per-turn skill 激活统计
    print(f"\nPer-turn Skill 激活统计 (关键词匹配):")
    for r in results:
        trace = r["skill_trace"]
        print(f"  {r['instance_id']}:")
        if not trace:
            print(f"    (无 skill 激活)")
        for t in trace:
            print(f"    turn {t['turn']}: {t['activated_skills']}")

    # 打印 skill 实际使用统计
    print(f"\nSkill 实际使用统计 (agent 读取 SKILL.md):")
    for r in results:
        used = r["all_used_skills"]
        print(f"  {r['instance_id']}:")
        if not used:
            print(f"    (未读取任何 SKILL.md)")
        else:
            print(f"    使用: {used}")

    # 汇总
    n_activated = sum(1 for r in results if r["num_unique_skills"] > 0)
    n_used = sum(1 for r in results if r["num_used_skills"] > 0)
    print(f"\n汇总: {len(results)} 个实例中:")
    print(f"  激活 (关键词匹配): {n_activated}/{len(results)} ({n_activated/len(results)*100:.1f}%)")
    print(f"  使用 (读取SKILL.md): {n_used}/{len(results)} ({n_used/len(results)*100:.1f}%)")


if __name__ == "__main__":
    main()

