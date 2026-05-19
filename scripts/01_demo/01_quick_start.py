import os
import re
import sys
from openhands.sdk import LLM, Agent, Conversation, Tool, AgentContext, LLMSummarizingCondenser
from openhands.sdk.context.skills import load_skills_from_dir

# 必须显式 import 工具模块，触发工具注册
from openhands.tools.terminal import TerminalTool       #shell命令
from openhands.tools.file_editor import FileEditorTool  #文件编辑
from openhands.tools.glob import GlobTool               #文件匹配
from openhands.tools.grep import GrepTool               #内容搜索   
from openhands.tools.apply_patch import ApplyPatchTool  #应用代码补丁


# ---------------------------------------------------------------------------
# 日志工具：将 SDK 终端输出重定向到文件，去除 ANSI 颜色转义码
# ---------------------------------------------------------------------------
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]")

class StripAnsiWriter:
    """写入文件前去除 ANSI 转义码的包装器。"""
    def __init__(self, f):
        self._f = f
    def write(self, s):
        return self._f.write(_ANSI_RE.sub("", s))
    def flush(self):
        self._f.flush()     #缓冲区写入文件

# 1. 配置 LLM（端口/密钥与 vllm_start.sh 保持一致，优先读环境变量）
vllm_port = os.environ.get("VLLM_PORT", "8000")
vllm_api_key = os.environ.get("VLLM_API_KEY", "EMPTY")

llm = LLM(
    model="openai/Qwen2.5",
    api_key=vllm_api_key,
    base_url=f"http://localhost:{vllm_port}/v1",
    temperature=0.0,                   # 采样温度 (0.0=确定性, >0.7=创造性)
    top_p=None,                         # 核采样 (0.0-1.0)
    top_k=None,                         # Top-k 采样
    seed=None,                          # 随机种子（可复现）
    max_message_chars=30000,            # 单条消息最大字符数
    max_input_tokens=None,              # 最大输入 token 数
    max_output_tokens=None,             # 最大输出 token 数
    custom_tokenizer=None,              # 自定义 tokenizer
    # num_retries=5,                      # 重试次数
    # retry_multiplier=8.0,               # 退避乘数
    # retry_min_wait=8,                   # 最小重试等待（秒）
    # retry_max_wait=64,                  # 最大重试等待（秒）
    # timeout=300,                        # HTTP 超时（秒）
    stream=False,                       # 流式响应
    native_tool_calling=True,           # 原生工具调用,?输出调用工具的结构化结果
    caching_prompt=True,                # 提示缓存
    prompt_cache_retention="24h",       # 缓存保留时间
    drop_params=True,                   # 自动丢弃不支持的参数
    modify_params=True,                 # 允许 LiteLLM 修改参数
    reasoning_effort="high",            # 推理力度: low/medium/high/xhigh/none
    extended_thinking_budget=200000,    # 思考 token 预算
    reasoning_summary=None,             # auto/concise/detailed
    litellm_extra_body={},              # 传递给 LiteLLM 的额外参数
    extra_headers=None,                 # 自定义 HTTP 头
    input_cost_per_token=None,          # 输入 token 单价
    output_cost_per_token=None,         # 输出 token 单价
    log_completions=True,              # 日志记录
    log_completions_folder="/home/wsh/openhands_code_research/log",          # 日志目录
    fallback_strategy=None,             # 回退策略, FallbackStrategy 对象
    disable_vision=True,                # 禁用视觉功能
    disable_stop_word=False,            # 禁用停止词
    model_canonical_name=None,          # 代理场景下的规范模型名
    force_string_serializer=None,       # 强制字符串序列化
    enable_encrypted_reasoning=True,    # 加密推理内容
)

# 2. 从工作区加载项目级 Skills（SKILL.md 文件）
#    工作区路径下 .agents/skills/ 包含：
#    - code-style-guide/SKILL.md   — 编码规范（无 trigger，渐进式披露）
#    - rot13-encryption/SKILL.md   — ROT13 加解密（有 trigger: encrypt/decrypt/cipher）
workspace = "/home/wsh/openhands_code_research/workspace/01"
skills_dir = os.path.join(workspace, ".agents", "skills")

_, _, agent_skills = load_skills_from_dir(skills_dir)
print(f"已加载 {len(agent_skills)} 个项目级 Skills: {list(agent_skills.keys())}")

# 3. 创建 Agent，指定可用工具 + 项目级 Skills
agent = Agent(
    llm=llm,                                # LLM 实例
    tools=[                                  # 可用工具列表，仅仅声明工具的名称，实际调用时是从注册系统解耦的
        Tool(name=TerminalTool.name),
        Tool(name=FileEditorTool.name),
        Tool(name=GlobTool.name),
        Tool(name=GrepTool.name),
        Tool(name=ApplyPatchTool.name),
    ],
    include_default_tools=["FinishTool", "ThinkTool"],  # 默认包含的工具
    tool_concurrency_limit=1,                # 并行工具执行上限（1=串行）
    filter_tools_regex=None,                 # 正则过滤工具名
    mcp_config={},                           # MCP 服务器配置
    system_prompt_filename="system_prompt.j2",
    system_prompt_kwargs={},                    # 模板变量
    security_policy_filename="security_policy.j2",  # 安全策略模板
    agent_context=AgentContext(
        # 使用从 .agents/skills/ 加载的 SKILL.md Skills
        # - code-style-guide: 无 trigger，列在 <available_skills>，Agent 按需读取
        # - rot13-encryption: 有 trigger，用户消息含 encrypt/decrypt 时自动注入
        skills=list(agent_skills.values()),
        system_message_suffix="始终保持严谨。",
    ),
    condenser=LLMSummarizingCondenser(         # 上下文压缩器（防止超出上下文窗口）
        llm=llm,
        max_size=240,
        keep_first=2,
    ),
    critic=None,                             # 动作评估器（实验性）
)

# 4. 创建会话并执行
conversation = Conversation(
    # === 必填 ===
    agent=agent,                              # Agent 实例
    workspace=workspace,                      # 工作目录（str 或 LocalWorkspace）

    # === 可选 ===
    plugins=None,                             # 插件列表
    persistence_dir=None,                     # 持久化目录
    conversation_id=None,                     # 会话 ID（用于恢复）
    callbacks=None,                           # 事件回调列表
    token_callbacks=None,                     # 流式 token 回调
    hook_config=None,                         # 钩子配置
    max_iteration_per_run=500,                # 单次 run 最大迭代
    stuck_detection=True,                     # 是否检测卡死
    stuck_detection_thresholds=None,          # 卡死检测阈值
    visualizer=None,                          # 可视化器（默认终端输出）
    secrets=None,                             # 密钥字典
    delete_on_close=True,                     # 关闭时是否删除临时数据
)

# 发送包含 "encrypt" 关键词的任务，触发 rot13-encryption skill
# Agent 会：
#   1. 通过 KeywordTrigger 匹配到 "encrypt"，自动获得 rot13-encryption skill 的完整内容
#   2. 收到的 <EXTRA_INFO> 中包含 Skill location 绝对路径，据此解析脚本位置
#   3. 使用 .agents/skills/rot13-encryption/scripts/encrypt.sh 脚本执行加密
#   4. 将加密结果写入文件
#
# 注意：SDK 注入的 <EXTRA_INFO> 会包含：
#   Skill location: /path/to/.agents/skills/rot13-encryption/SKILL.md
#   (Use this path to resolve relative file references in the skill content below)
# 但较弱的模型可能忽略这个提示，直接用相对路径 ./scripts/encrypt.sh（在 workspace 根目录下不存在）。
# 因此对于能力较弱的模型，建议在消息中显式提醒使用 skill location 来定位脚本。

# 5. 将 SDK 终端输出重定向到日志文件（终端保持安静）
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "log", "01_quick_start.log")
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

saved_stdout = sys.stdout
saved_stderr = sys.stderr

try:
    log_file = open(LOG_PATH, "w", encoding="utf-8")
    writer = StripAnsiWriter(log_file)
    sys.stdout = writer
    sys.stderr = writer

    conversation.send_message(
        "请 encrypt 消息 'Hello OpenHands SDK'。"
        "注意：请根据 <EXTRA_INFO> 中的 Skill location 路径找到加密脚本的绝对路径并执行，"
        "将加密结果保存到当前目录的 encrypted.txt 文件中。"
    )
    conversation.run()
finally:
    sys.stdout = saved_stdout
    sys.stderr = saved_stderr
    try:
        log_file.close()
    except Exception:
        pass

conversation.close()
print(f"完成！Agent 运行日志: {LOG_PATH}")