#!/usr/bin/env python3
"""
run_scenario_ABC.py
====================
场景 A / B / C:量化 skill 文档 KV cache 的位置依赖性。

  A (基线):   skill + query
  B (短前置): ~100 token 对话 + skill + query
  C (长前置): ~1000 token 对话 + skill + query

核心问题:skill 前面的上下文越长,其 KV 偏离基线越大吗?
评估指标:relative L2 偏差 + 余弦相似度,分层 × 分 token 双维度切片。

使用方法:
  python scripts/04_kv_cache_research/position_context_analysis/run_scenario_ABC.py

输出:
  results/position_context_analysis/abc/
    B_vs_A_summary.json
    C_vs_A_summary.json
    B_vs_A_matrices.npz
    C_vs_A_matrices.npz
    figures/
      ABC_per_layer.png       # 分层偏差曲线
      ABC_per_token.png       # skill 内 token 位置偏差
      ABC_heatmaps.png        # layer × token 热力图
      ABC_summary_bar.png     # 汇总对比条形图
"""

import os
import sys
import json
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore

# ============================================================
# 0. 全局配置
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
RESULTS_ROOT = BASE_DIR / "results"
MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
DUMP_ROOT  = Path("/tmp/kv_dump_abc")
MAX_MODEL_LEN = 32768
OUT_DIR  = RESULTS_ROOT / "position_context_analysis" / "abc"
FIG_DIR  = OUT_DIR / "figures"

# 必须在 import vllm 前设置,EngineCore 子进程会继承这些环境变量
COMBINED_DUMP_DIR = DUMP_ROOT / "scenario_abc_combined"
if COMBINED_DUMP_DIR.exists():
    shutil.rmtree(COMBINED_DUMP_DIR)
os.environ["DUMP_KV"]   = "1"
os.environ["SCENARIO"]  = "abc_combined"
os.environ["DUMP_DIR"]  = str(DUMP_ROOT)

# ============================================================
# 1. 构造 Prompt 素材
# ============================================================

# ── 固定 skill 文档 ──────────────────────────────────────────
# 选择一个工具调用型 skill,覆盖参数、返回值、示例,约 120 token 量级。
# 三个场景共享完全相同的字节序列,确保 tokenization 结果可对齐。
SKILL_TEXT = """\
---
name: send-email
description: |
  Send an email to one or more recipients via the corporate mail gateway.
  Supports plain text and HTML bodies, CC/BCC fields, and file attachments.
  Rate limit: 50 emails per hour per user.
parameters:
  to: string (required) - recipient email address(es), comma-separated
  subject: string (required) - email subject line
  body: string (required) - email body content
  cc: string (optional) - CC recipients
  bcc: string (optional) - BCC recipients
  attachments: list[string] (optional) - file paths to attach
returns:
  message_id: string - unique identifier for the sent message
  status: "sent" | "queued" | "failed"
examples:
  - to: "alice@example.com"
    subject: "Meeting tomorrow"
    body: "Hi Alice, can we meet at 10am?"
keywords: email send message mail attachment communication
---
"""

# ── 固定 query ───────────────────────────────────────────────
QUERY_TEXT = (
    "User: Please send a follow-up email to bob@example.com "
    "with subject 'Project Update' and body 'Hi Bob, checking in on the project status.'\n"
)

# ── 短历史前置 (~100 token) ──────────────────────────────────
SHORT_HISTORY = """\
User: Hello, what tools do you have available today?
Assistant: I have access to several tools including email sending, calendar management,
file search, and task creation. How can I help you today?
User: I need to communicate with a colleague about an ongoing project.
Assistant: Of course! I can help you send emails or messages. What would you like to say?
"""

# ── 长历史前置 (~1000 token) ─────────────────────────────────
# 内容刻意与 skill 无关,避免语义干扰;使用多轮对话模拟真实 agent 上下文。
LONG_HISTORY = """\
User: Good morning! I need to organize my tasks for today.
Assistant: Good morning! I am happy to help you plan your day. What tasks are on your plate?

User: I have a quarterly review to prepare, a team meeting at 2pm, and several client follow-ups.
Assistant: That sounds like a full day. Let us tackle them one by one. Shall we start with
the quarterly review, since it likely needs the most preparation time?

User: Yes, let us start there. Can you help me outline the key sections?
Assistant: A standard quarterly review typically covers: Executive Summary with headline
metrics, Financial Performance with revenue and margin trends, Operational Highlights
listing completed milestones, Challenges and Risks with mitigation plans, and Q4 Outlook
with strategic priorities. Shall I expand on any of these sections?

User: Please expand on the Financial Performance section.
Assistant: For Financial Performance you will want to include: Revenue breakdown by product
line or geography, Gross margin trends versus prior quarters, Operating expense variance
against budget, Cash flow and working capital position, and any material one-time items.
Show both absolute numbers and percentage changes versus prior quarter and prior year. Do
you have the data ready, or should we structure the narrative first?

User: I have the data. Revenue was 2.4M, up 12 percent quarter over quarter. Margins
improved by 2 points. Opex was flat.
Assistant: Great results. Here is a narrative draft: Revenue of 2.4M represented a 12
percent quarter-over-quarter improvement driven by strong demand. Gross margin expanded
2 percentage points reflecting improved pricing discipline and favorable product mix.
Operating expenses remained flat sequentially, demonstrating cost discipline as the
business scales. Would you like commentary on what drove the revenue growth, or shall
we move to the next section?

User: Move on. What about operational highlights?
Assistant: For Operational Highlights, feature major projects completed on time and on
budget, new customer wins or significant expansions, product launches or feature releases,
team milestones like headcount growth, and process improvements that drove efficiency. Lead
with outcomes rather than activities. Instead of saying we ran a marketing campaign, say the
campaign generated 200 qualified leads. Do you have specific highlights to feature?

User: We launched a new analytics module, won 3 enterprise clients, and grew the team by
5 engineers.
Assistant: Excellent. Here is the polished version: The quarter saw the successful launch
of the analytics module, enhancing platform data capabilities and opening new upsell
opportunities. The sales team closed three enterprise agreements, expanding the company
footprint in the mid-market segment. Headcount grew by five engineers, strengthening
product development capacity ahead of the Q4 roadmap. Shall I add quantitative impact
for the analytics module, such as adoption rate or revenue contribution?

User: Good. Now let us think about the 2pm team meeting. What should the agenda cover?
Assistant: For a post-quarterly team meeting, a productive agenda could include: Review of
Q3 results and key takeaways in ten minutes, Recognition of team wins in five minutes, Q4
priorities and how individual work maps to them in fifteen minutes, Blockers and cross-team
dependencies in ten minutes, and open Q and A in ten minutes. Keeping it under sixty
minutes with clear owners for each section tends to work best. Shall I draft this as a
formatted agenda to share with the team?

User: Yes please draft it formally.
Assistant: Team Q3 Review and Q4 Kick-off. Date: Today. Time: 2:00 PM to 3:00 PM.
Item 1, 2:00 to 2:10, Q3 Results Overview, Owner: Team Lead, summary of financial and
operational highlights. Item 2, 2:10 to 2:15, Recognition, Owner: Team Lead, celebrate
individual and team contributions. Item 3, 2:15 to 2:30, Q4 Priorities, Owner: All,
alignment on key objectives and individual workstreams. Item 4, 2:30 to 2:40, Blockers
and Dependencies, Owner: All, surface and assign resolution owners. Item 5, 2:40 to 2:50,
Open Q and A, Owner: All, address questions and close. Please come prepared with your Q4
goals and any blockers to discuss. Shall I send this to the team?

User: Not yet. Now I need to handle some client follow-ups.
Assistant: Client follow-ups are important for maintaining momentum and trust. What type
are these: post-meeting notes, outstanding proposal responses, or check-ins on active
projects? Knowing the context will help me draft the right tone and content for each one.

"""

# ── 三个完整 prompt ──────────────────────────────────────────
PROMPTS: dict[str, str] = {
    "A": SKILL_TEXT + QUERY_TEXT,
    "B": SHORT_HISTORY + SKILL_TEXT + QUERY_TEXT,
    "C": LONG_HISTORY  + SKILL_TEXT + QUERY_TEXT,
}

SCENARIO_LABELS = {
    "B": "B vs A (short prefix, ~100 tok)",
    "C": "C vs A (long prefix, ~1000 tok)",
}

# ============================================================
# 2. Tokenize & 定位 skill token 区间
# ============================================================
from transformers import AutoTokenizer  # type: ignore[import-not-found]

print(f"加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
assert tokenizer.is_fast, "需要 fast tokenizer 以获得 offset_mapping"

def locate_skill_tokens(prompt_text: str, skill_text: str) -> tuple[list[int], int, int]:
    """
    整体 tokenize prompt,用 offset_mapping 精确定位 skill 文本对应的 token 区间。

    返回 (token_ids, tok_start, tok_end);skill 区间为左闭右开 [tok_start, tok_end)。
    只保留"完全落在字符窗口内"的 token,对 BPE 边界鲁棒。
    """
    char_start = prompt_text.find(skill_text)
    assert char_start != -1, "SKILL_TEXT 在 prompt 中未找到,请检查字符串是否完全一致"
    char_end = char_start + len(skill_text)

    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = enc["input_ids"]
    offsets: list[tuple[int, int]] = enc["offset_mapping"]

    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s >= char_start and tok_start is None:
            tok_start = i
        if e <= char_end:
            tok_end = i + 1  # 左闭右开
    assert tok_start is not None and tok_end is not None and tok_end > tok_start, \
        f"skill token 区间定位失败: tok_start={tok_start}, tok_end={tok_end}"
    return ids, tok_start, tok_end


print("\n各场景 tokenization 结果:")
print("-" * 60)
scenario_info: dict[str, dict] = {}
for name, prompt in PROMPTS.items():
    ids, ts, te = locate_skill_tokens(prompt, SKILL_TEXT)
    scenario_info[name] = {
        "ids": ids,
        "N": len(ids),
        "skill_start": ts,
        "skill_end":   te,
        "skill_len":   te - ts,
    }
    print(f"  场景 {name}: 总 token={len(ids):5d}, "
          f"skill=[{ts},{te}), len={te-ts}")

# 验证 skill token 长度一致性
skill_lens = [v["skill_len"] for v in scenario_info.values()]
L_skill = min(skill_lens)
if len(set(skill_lens)) > 1:
    print(f"\n⚠ skill token 长度在不同场景中不一致: {skill_lens}")
    print(f"  可能是 BPE 边界漂移。对比时截到最小长度 {L_skill}。")
else:
    print(f"\n✓ skill token 长度在所有场景中一致: {L_skill}")

# 确认 prompt 不超过模型上下文窗口
for name, info in scenario_info.items():
    assert info["N"] < MAX_MODEL_LEN, \
        f"场景 {name} 的 token 数 {info['N']} 超过 MAX_MODEL_LEN={MAX_MODEL_LEN}"

# ============================================================
# 3. 加载 vLLM
# ============================================================
import torch
from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore[import-not-found]

print(f"\n加载模型 {MODEL_PATH} ...")
llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=MAX_MODEL_LEN,
    enable_prefix_caching=False,
    max_num_batched_tokens=MAX_MODEL_LEN,
)
print("✓ 模型加载完成\n")

# ============================================================
# 4. 逐场景 Prefill & Dump
# ============================================================
_SP = SamplingParams(max_tokens=1, temperature=0)


def run_prefill(scenario_name: str) -> None:
    """对单个场景跑一次 prefill,K/V 由 qwen3.py 内置 hook 自动 dump 到 COMBINED_DUMP_DIR。"""
    info = scenario_info[scenario_name]
    print(f"Prefill 场景 {scenario_name}  (N={info['N']} tokens) ...")
    llm.generate([TokensPrompt(prompt_token_ids=info["ids"])], _SP)
    n_files = len(list(COMBINED_DUMP_DIR.glob("*.pt"))) if COMBINED_DUMP_DIR.exists() else 0
    print(f"  → 当前 dump 目录共 {n_files} 个文件")


for name in ["A", "B", "C"]:
    run_prefill(name)

# ============================================================
# 5. 读取 KV Dump
# ============================================================
import numpy as np

LayerKV = dict[int, dict]  # layer_idx -> {"k": Tensor, "v": Tensor}


def load_kv(scenario_name: str) -> LayerKV:
    """从统一 dump 目录按 num_tokens 筛选出指定场景的所有层 KV。
    三个场景 token 数各不相同(239/315/1249),无需分目录。
    """
    N = scenario_info[scenario_name]["N"]
    result: LayerKV = {}
    for f in sorted(COMBINED_DUMP_DIR.glob("*.pt")):
        data = torch.load(f, weights_only=False)
        if int(data["num_tokens"]) != N:
            continue
        li = int(data["layer_idx"])
        if li not in result:
            result[li] = {
                "k": data["k"].float().cpu(),
                "v": data["v"].float().cpu(),
            }
    return result

# ============================================================
# 7. 计算 KV Deviation 指标
# ============================================================

def cos_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """逐行余弦相似度。"""
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-9
    return num / den


def compare_to_baseline(name_ref: str, name_tgt: str) -> dict:
    """
    将目标场景(B 或 C)的 skill KV 与基线场景(A)逐层对比。

    返回包含以下字段的 dict:
      label, per_layer, layers,
      kmat (layer × L_skill), vmat,
      per_pos_k, per_pos_v,
      k_rel_arr, v_rel_arr, k_cos_arr, v_cos_arr,
      summary
    """
    kv_ref = load_kv(name_ref)
    kv_tgt = load_kv(name_tgt)

    ref_info = scenario_info[name_ref]
    tgt_info = scenario_info[name_tgt]

    ref_s = ref_info["skill_start"]
    tgt_s = tgt_info["skill_start"]

    layers = sorted(set(kv_ref.keys()) & set(kv_tgt.keys()))
    if not layers:
        raise ValueError(f"场景 {name_ref} 和 {name_tgt} 没有共同的层索引,请检查 dump。")
    print(f"\n对比 {name_tgt} vs {name_ref}: {len(layers)} 层可用")

    per_layer    = []
    per_token_k_rel: list[list[float]] = []
    per_token_v_rel: list[list[float]] = []

    for li in layers:
        # 切出 skill 区间,截到 L_skill
        k_ref = kv_ref[li]["k"][ref_s : ref_s + L_skill]
        v_ref = kv_ref[li]["v"][ref_s : ref_s + L_skill]
        k_tgt = kv_tgt[li]["k"][tgt_s : tgt_s + L_skill]
        v_tgt = kv_tgt[li]["v"][tgt_s : tgt_s + L_skill]

        k_rel = (k_ref - k_tgt).norm(dim=-1) / (k_ref.norm(dim=-1) + 1e-9)
        k_cos = cos_per_row(k_ref, k_tgt)
        v_rel = (v_ref - v_tgt).norm(dim=-1) / (v_ref.norm(dim=-1) + 1e-9)
        v_cos = cos_per_row(v_ref, v_tgt)

        per_token_k_rel.append(k_rel.tolist())
        per_token_v_rel.append(v_rel.tolist())
        per_layer.append({
            "layer":          li,
            "k_rel_l2_mean":  float(k_rel.mean()),
            "k_rel_l2_max":   float(k_rel.max()),
            "k_cos_mean":     float(k_cos.mean()),
            "v_rel_l2_mean":  float(v_rel.mean()),
            "v_rel_l2_max":   float(v_rel.max()),
            "v_cos_mean":     float(v_cos.mean()),
        })

    kmat = np.array(per_token_k_rel)   # (num_layers, L_skill)
    vmat = np.array(per_token_v_rel)

    k_rel_arr = np.array([r["k_rel_l2_mean"] for r in per_layer])
    v_rel_arr = np.array([r["v_rel_l2_mean"] for r in per_layer])
    k_cos_arr = np.array([r["k_cos_mean"]    for r in per_layer])
    v_cos_arr = np.array([r["v_cos_mean"]    for r in per_layer])

    return {
        "label":       f"{name_tgt} vs {name_ref}",
        "verbose_label": SCENARIO_LABELS.get(name_tgt, f"{name_tgt} vs {name_ref}"),
        "per_layer":   per_layer,
        "layers":      layers,
        "kmat":        kmat,
        "vmat":        vmat,
        "per_pos_k":   kmat.mean(axis=0),   # 跨层平均,每个 skill token 一个值
        "per_pos_v":   vmat.mean(axis=0),
        "k_rel_arr":   k_rel_arr,
        "v_rel_arr":   v_rel_arr,
        "k_cos_arr":   k_cos_arr,
        "v_cos_arr":   v_cos_arr,
        "summary": {
            "k_rel_l2_mean": float(k_rel_arr.mean()),
            "v_rel_l2_mean": float(v_rel_arr.mean()),
            "k_cos_mean":    float(k_cos_arr.mean()),
            "v_cos_mean":    float(v_cos_arr.mean()),
            "k_rel_l2_range": [float(k_rel_arr.min()), float(k_rel_arr.max())],
            "v_rel_l2_range": [float(v_rel_arr.min()), float(v_rel_arr.max())],
        },
    }


result_BA = compare_to_baseline("A", "B")
result_CA = compare_to_baseline("A", "C")
results   = [result_BA, result_CA]

# ============================================================
# 8. 打印分层结果 + 汇总
# ============================================================
HDR = f"{'layer':>5} | {'K相对L2均值':>12} {'K相对L2最大':>12} {'K余弦':>8} | {'V相对L2均值':>12} {'V相对L2最大':>12} {'V余弦':>8}"
SEP = "-" * 92

for res in results:
    print(f"\n{'='*92}")
    print(f"  {res['verbose_label']}")
    print(f"{'='*92}")
    print(HDR)
    print(SEP)
    for r in res["per_layer"]:
        print(f"{r['layer']:>5} | "
              f"{r['k_rel_l2_mean']:>12.4f} {r['k_rel_l2_max']:>12.4f} {r['k_cos_mean']:>8.4f} | "
              f"{r['v_rel_l2_mean']:>12.4f} {r['v_rel_l2_max']:>12.4f} {r['v_cos_mean']:>8.4f}")
    s = res["summary"]
    print(f"\n全层汇总:")
    print(f"  K 相对 L2:    mean={s['k_rel_l2_mean']:.4f}  "
          f"range=[{s['k_rel_l2_range'][0]:.4f}, {s['k_rel_l2_range'][1]:.4f}]")
    print(f"  V 相对 L2:    mean={s['v_rel_l2_mean']:.4f}  "
          f"range=[{s['v_rel_l2_range'][0]:.4f}, {s['v_rel_l2_range'][1]:.4f}]")
    print(f"  K 余弦相似度:  mean={s['k_cos_mean']:.4f}")
    print(f"  V 余弦相似度:  mean={s['v_cos_mean']:.4f}")

    pk = res["per_pos_k"]
    pv = res["per_pos_v"]
    print(f"\nskill 内部 token 偏差分布 (跨层平均):")
    print("前 8 token K 相对L2: " + "  ".join(f"{x:.3f}" for x in pk[:8]))
    print("前 8 token V 相对L2: " + "  ".join(f"{x:.3f}" for x in pv[:8]))
    if len(pk) > 8:
        print(f"index ≥ 8 token K 均值: {pk[8:].mean():.3f},  V 均值: {pv[8:].mean():.3f}")

# 打印两个场景的对比摘要
print(f"\n{'='*60}")
print("两场景汇总对比")
print(f"{'='*60}")
print(f"{'':20} {'B vs A':>12} {'C vs A':>12}  {'变化':>8}")
for key, label in [("k_rel_l2_mean", "K 相对 L2"),
                   ("v_rel_l2_mean", "V 相对 L2"),
                   ("k_cos_mean",    "K 余弦"),
                   ("v_cos_mean",    "V 余弦")]:
    ba = result_BA["summary"][key]
    ca = result_CA["summary"][key]
    direction = "↑" if ca > ba else "↓"
    print(f"  {label:18} {ba:>12.4f} {ca:>12.4f}  {direction} {abs(ca-ba):.4f}")

# ============================================================
# 9. 保存 JSON & NPZ
# ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

for res in results:
    label = res["label"].replace(" ", "_")

    # JSON: 所有可序列化字段 + per_token 均值
    json_data = {
        "label":        res["label"],
        "model":        MODEL_PATH,
        "skill_len":    L_skill,
        "scenario_A":   {k: v for k, v in scenario_info["A"].items() if k != "ids"},
        "scenario_tgt": {k: v for k, v in
                         scenario_info[res["label"][0]].items() if k != "ids"},
        "per_layer":    res["per_layer"],
        "per_token_k_rel_l2_mean_over_layers": res["per_pos_k"].tolist(),
        "per_token_v_rel_l2_mean_over_layers": res["per_pos_v"].tolist(),
        "summary":      res["summary"],
    }
    json_path = OUT_DIR / f"{label}_summary.json"
    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=2)
    print(f"\nJSON: {json_path}")

    # NPZ: 完整 (layer × token) 矩阵,供后续深入分析
    npz_path = OUT_DIR / f"{label}_matrices.npz"
    np.savez(
        npz_path,
        k_rel_l2=res["kmat"],
        v_rel_l2=res["vmat"],
        layers=np.array(res["layers"]),
    )
    print(f"NPZ:  {npz_path}")

# ============================================================
# 10. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── 图 1: 分层 KV 偏差曲线 (2×2) ────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 8))

for col_i, res in enumerate(results):
    layers_arr = np.array(res["layers"])
    ax_l2  = axes[0][col_i]
    ax_cos = axes[1][col_i]

    ax_l2.plot(layers_arr, res["k_rel_arr"], marker="o", ms=4,
               label="K", color="tab:blue")
    ax_l2.plot(layers_arr, res["v_rel_arr"], marker="s", ms=4,
               label="V", color="tab:red")
    ax_l2.set_title(f"{res['label']}  —  Relative L2")
    ax_l2.set_xlabel("Layer index")
    ax_l2.set_ylabel("Mean relative L2 (skill tokens)")
    ax_l2.grid(alpha=0.3); ax_l2.legend()

    ax_cos.plot(layers_arr, res["k_cos_arr"], marker="o", ms=4,
                label="K", color="tab:blue")
    ax_cos.plot(layers_arr, res["v_cos_arr"], marker="s", ms=4,
                label="V", color="tab:red")
    ax_cos.set_title(f"{res['label']}  —  Cosine Similarity")
    ax_cos.set_xlabel("Layer index")
    ax_cos.set_ylabel("Mean cosine similarity (skill tokens)")
    ax_cos.set_ylim(-0.05, 1.05)
    ax_cos.grid(alpha=0.3); ax_cos.legend()

fig.suptitle("Scenarios A/B/C: Per-Layer KV Deviation of Skill Tokens", fontsize=13)
fig.tight_layout()
fig1_path = FIG_DIR / "ABC_per_layer.png"
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print(f"\n图1 (分层曲线):   {fig1_path}")

# ── 图 2: skill 内 token 位置的跨层偏差 ─────────────────────
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
styles = [("-", "tab:blue", "tab:orange"),
          ("--", "tab:green", "tab:red")]

for (ls, ck, cv), res in zip(styles, results):
    pos_arr = np.arange(len(res["per_pos_k"]))
    ax_k.plot(pos_arr, res["per_pos_k"], ls=ls, color=ck,
              linewidth=1.6, label=f"K {res['label']}")
    ax_v.plot(pos_arr, res["per_pos_v"], ls=ls, color=cv,
              linewidth=1.6, label=f"V {res['label']}")

# 高亮前 4 个 token(可能的 attention sink 区域)
for ax in (ax_k, ax_v):
    ax.axvspan(0, min(4, L_skill) - 0.5, alpha=0.12, color="gold",
               label="First 4 tokens (sink region)")
    ax.set_xlabel("Token index within skill document")
    ax.set_ylabel("Relative L2 (mean over layers)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)

ax_k.set_title("K deviation per skill token")
ax_v.set_title("V deviation per skill token")
fig.suptitle("Per-Token KV Deviation: B vs A and C vs A", fontsize=13)
fig.tight_layout()
fig2_path = FIG_DIR / "ABC_per_token.png"
fig.savefig(fig2_path, dpi=150)
plt.close(fig)
print(f"图2 (token位置):  {fig2_path}")

# ── 图 3: (layer × skill token) 热力图 4 格 ──────────────────
fig, axes = plt.subplots(2, 2, figsize=(16, 8))
vmax_global = float(max(
    result_BA["kmat"].max(), result_BA["vmat"].max(),
    result_CA["kmat"].max(), result_CA["vmat"].max(),
))

for row_i, res in enumerate(results):
    for col_i, (mat, kv_label) in enumerate([(res["kmat"], "K"), (res["vmat"], "V")]):
        ax = axes[row_i][col_i]
        im = ax.imshow(mat, aspect="auto", cmap="viridis",
                       vmin=0, vmax=vmax_global, origin="lower")
        ax.set_xlabel("Token index within skill")
        ax.set_ylabel("Layer index")
        ax.set_title(f"{res['label']}  —  {kv_label} Relative L2")
        fig.colorbar(im, ax=ax, shrink=0.8, label="relative L2")

fig.suptitle("KV Deviation Heatmaps: layer × token  (B vs A  |  C vs A)", fontsize=13)
fig.tight_layout()
fig3_path = FIG_DIR / "ABC_heatmaps.png"
fig.savefig(fig3_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"图3 (热力图):     {fig3_path}")

# ── 图 4: 汇总条形图 ─────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
bar_labels = [r["label"] for r in results]
x = np.arange(len(bar_labels))
w = 0.35

k_means = [r["summary"]["k_rel_l2_mean"] for r in results]
v_means = [r["summary"]["v_rel_l2_mean"] for r in results]
k_cos   = [r["summary"]["k_cos_mean"]    for r in results]
v_cos   = [r["summary"]["v_cos_mean"]    for r in results]

axes[0].bar(x - w/2, k_means, w, label="K", color="tab:blue",  alpha=0.8)
axes[0].bar(x + w/2, v_means, w, label="V", color="tab:red",   alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(bar_labels, fontsize=9)
axes[0].set_ylabel("Mean Relative L2")
axes[0].set_title("KV Relative L2 Deviation (lower = more reusable)")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)

axes[1].bar(x - w/2, k_cos, w, label="K", color="tab:blue",   alpha=0.8)
axes[1].bar(x + w/2, v_cos, w, label="V", color="tab:red",    alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(bar_labels, fontsize=9)
axes[1].set_ylabel("Mean Cosine Similarity")
axes[1].set_title("KV Cosine Similarity (higher = more reusable)")
axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Summary: Effect of Prefix Length on Skill KV Cache Reusability", fontsize=12)
fig.tight_layout()
fig4_path = FIG_DIR / "ABC_summary_bar.png"
fig.savefig(fig4_path, dpi=150)
plt.close(fig)
print(f"图4 (汇总条形图): {fig4_path}")

print(f"\n{'='*60}")
print(f"✓ 实验完成!所有结果写入: {OUT_DIR}")
print(f"{'='*60}")
