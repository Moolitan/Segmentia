#!/usr/bin/env python3
"""
run_semantic_distance.py
========================
语义距离实验：前缀内容的语义相似度对 skill KV 偏差的影响。

Decomposition 实验（B vs D）已证明：位置相同时，前缀内容不同会产生
K=0.287、V=0.349 的纯上下文偏差。那时 SHORT_HISTORY（工具/邮件）与
ALT_HISTORY（数据库）的语义距离很大。

本实验的问题：如果前缀内容语义相近（同样关于工具/邮件，但措辞不同），
KV 偏差是否更小？若是，则说明偏差量与前缀的"语义距离"正相关。

实验设计：
  A : skill + query                                     (无前文，基线)
  B : SHORT_HISTORY(76 tok) + skill + query            (工具/邮件，措辞 v1)
  E : NEAR_HISTORY (76 tok) + skill + query            (工具/邮件，措辞 v2，语义相近)

三种对比：
  B vs A : 总偏差 = RoPE(0→76) + Context(空→SHORT)    [参考基线]
  E vs A : 总偏差 = RoPE(0→76) + Context(空→NEAR)
  B vs E : 纯上下文差异（语义相近，位置严格相同=76）  [核心结果]

并从 Decomposition 实验的 JSON 加载 B vs D（语义远）数据作为对照：
  预期：B vs E 偏差 < B vs D 偏差，验证语义距离假说。

Layer 0 验证：B vs E 的 Layer 0 K 差异 ≡ 0（位置相同，同一 skill）。
"""

import os
import sys
import json
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# 0. 全局配置
# ============================================================
SCRIPT_DIR    = Path(__file__).resolve().parent
BASE_DIR      = SCRIPT_DIR.parent
RESULTS_ROOT  = BASE_DIR / "results"
MODEL_PATH    = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
DUMP_ROOT     = Path("/tmp/kv_dump_semantic")
MAX_MODEL_LEN = 32768
OUT_DIR       = RESULTS_ROOT / "position_context_analysis" / "semantic_distance"
FIG_DIR       = OUT_DIR / "figures"
SIGNAL_FILE   = Path("/tmp/kv_current_scenario")

# Decomposition 实验的 B vs D 结果（语义远基线），用于对比图
DECOMPOSE_BVD_JSON = RESULTS_ROOT / "position_context_analysis" / "decomposition" / "decompose_BvsD_summary.json"

if DUMP_ROOT.exists():
    shutil.rmtree(DUMP_ROOT)
DUMP_ROOT.mkdir(parents=True)
if SIGNAL_FILE.exists():
    SIGNAL_FILE.unlink()

os.environ["DUMP_KV"]          = "1"
os.environ["DUMP_DIR"]         = str(DUMP_ROOT)
os.environ["DUMP_SKIP_WARMUP"] = "1"
os.environ["SCENARIO"]         = "init"

SCENARIO_DUMP_DIRS: dict[str, Path] = {
    name: DUMP_ROOT / f"scenario_{name}"
    for name in ["A", "B", "E"]
}

# ============================================================
# 1. Prompt 素材
# ============================================================

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

QUERY_TEXT = (
    "User: Please send a follow-up email to bob@example.com "
    "with subject 'Project Update' and body 'Hi Bob, checking in on the project status.'\n"
)

# 场景 B 的前缀（76 token）：工具/邮件话题，措辞 v1
SHORT_HISTORY = """\
User: Hello, what tools do you have available today?
Assistant: I have access to several tools including email sending, calendar management,
file search, and task creation. How can I help you today?
User: I need to communicate with a colleague about an ongoing project.
Assistant: Of course! I can help you send emails or messages. What would you like to say?
"""

# 场景 E 的候选前缀：同样是工具/邮件话题，但措辞完全不同（语义近）
# 关键点：与 SHORT_HISTORY 的语义结构相同——
#   用户询问工具/通信能力 → 助手列举邮件等功能 → 用户提到联系同事/项目 → 助手提出帮助
NEAR_HISTORY_CANDIDATE = """\
User: What messaging and communication capabilities do you offer?
Assistant: I can handle various communication tasks such as sending emails, scheduling meetings,
locating files, and creating reminders. What can I assist you with today?
User: I want to write to a coworker about progress on our current project.
Assistant: Happy to help! I am able to draft and deliver messages for you. What should the message say?
"""

# ============================================================
# 2. Tokenize & 定位 skill token 区间
# ============================================================
from transformers import AutoTokenizer

print(f"加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
assert tokenizer.is_fast, "需要 fast tokenizer"


def locate_skill_tokens(prompt_text: str, skill_text: str) -> tuple[list[int], int, int]:
    char_start = prompt_text.find(skill_text)
    assert char_start != -1, "SKILL_TEXT 在 prompt 中未找到"
    char_end   = char_start + len(skill_text)
    enc     = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    ids     = enc["input_ids"]
    offsets = enc["offset_mapping"]
    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s >= char_start and tok_start is None:
            tok_start = i
        if e <= char_end:
            tok_end = i + 1
    assert tok_start is not None and tok_end is not None and tok_end > tok_start
    return ids, tok_start, tok_end


# 把 NEAR_HISTORY_CANDIDATE 截到与 SHORT_HISTORY 完全相同的 token 数
short_ids = tokenizer(SHORT_HISTORY,            add_special_tokens=False)["input_ids"]
near_ids  = tokenizer(NEAR_HISTORY_CANDIDATE,   add_special_tokens=False)["input_ids"]
P = len(short_ids)
print(f"\nSHORT_HISTORY token 数: {P}")
print(f"NEAR_HISTORY_CANDIDATE token 数: {len(near_ids)} (将截到 {P})")

if len(near_ids) < P:
    raise ValueError(f"NEAR_HISTORY_CANDIDATE 不足 {P} token,请扩充文本")

near_ids_trimmed = near_ids[:P]
NEAR_HISTORY = tokenizer.decode(near_ids_trimmed)

near_verify = tokenizer(NEAR_HISTORY, add_special_tokens=False)["input_ids"]
assert len(near_verify) == P, f"截断后 token 数不匹配: {len(near_verify)} != {P}"
print(f"NEAR_HISTORY 截断后 token 数: {len(near_verify)}  ✓")
print(f"\n-- SHORT_HISTORY --\n{SHORT_HISTORY}")
print(f"-- NEAR_HISTORY (截断后) --\n{NEAR_HISTORY}\n{'--'*30}")

# 构造三个 prompt
PROMPTS: dict[str, str] = {
    "A": SKILL_TEXT + QUERY_TEXT,
    "B": SHORT_HISTORY + SKILL_TEXT + QUERY_TEXT,
    "E": NEAR_HISTORY  + SKILL_TEXT + QUERY_TEXT,
}

print("\n各场景 tokenization 结果:")
print("-" * 65)
scenario_info: dict[str, dict] = {}
for name, prompt in PROMPTS.items():
    ids, ts, te = locate_skill_tokens(prompt, SKILL_TEXT)
    scenario_info[name] = {
        "ids":         ids,
        "N":           len(ids),
        "skill_start": ts,
        "skill_end":   te,
        "skill_len":   te - ts,
    }
    print(f"  场景 {name}: 总 token={len(ids):5d},  skill=[{ts}, {te}),  prefix={ts}")

# 验证 B 和 E 的 skill 起始位置严格一致（共同前提：位置相同）
assert scenario_info["B"]["skill_start"] == scenario_info["E"]["skill_start"], (
    f"B 和 E 的 skill 起始位置不一致: "
    f"B={scenario_info['B']['skill_start']}, E={scenario_info['E']['skill_start']}"
)
P_skill = scenario_info["B"]["skill_start"]
print(f"\n✓ B 和 E 的 skill 起始位置严格一致: {P_skill}")
print(f"  → B vs E 中 RoPE 旋转角度完全相同,Layer 0 K 差异 ≡ 0")

skill_lens = [v["skill_len"] for v in scenario_info.values()]
L_skill = min(skill_lens)
if len(set(skill_lens)) > 1:
    print(f"\n⚠  skill token 长度不一致: {skill_lens},截到 {L_skill}")
else:
    print(f"✓  skill token 长度一致: {L_skill}")

for name, info in scenario_info.items():
    assert info["N"] < MAX_MODEL_LEN, f"场景 {name} 超出 MAX_MODEL_LEN"

# ============================================================
# 3. 加载 vLLM（必须在 env var 设置之后）
# ============================================================
import torch
from vllm import LLM, SamplingParams, TokensPrompt

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
# 4. 逐场景 Prefill — 通过信号文件切换 dump 目录
# ============================================================
_SP = SamplingParams(max_tokens=1, temperature=0)


def run_prefill(scenario_name: str) -> None:
    SIGNAL_FILE.write_text(scenario_name)
    info = scenario_info[scenario_name]
    print(f"Prefill 场景 {scenario_name}  (N={info['N']} tokens, skill@{info['skill_start']}) ...")
    llm.generate([TokensPrompt(prompt_token_ids=info["ids"])], _SP)
    dump_dir = SCENARIO_DUMP_DIRS[scenario_name]
    n_files  = len(list(dump_dir.glob("*.pt"))) if dump_dir.exists() else 0
    print(f"  → dump 到 {dump_dir.name}/  ({n_files} 个文件)")


for name in ["A", "B", "E"]:
    run_prefill(name)

SIGNAL_FILE.unlink(missing_ok=True)

for name, d in SCENARIO_DUMP_DIRS.items():
    n = len(list(d.glob("*.pt"))) if d.exists() else 0
    print(f"  scenario_{name}/: {n} 个文件")
    assert n > 0, f"场景 {name} 的 dump 目录为空,信号文件切换可能未生效"

# ============================================================
# 5. 读取 KV Dump
# ============================================================
import numpy as np

LayerKV = dict[int, dict]


def load_kv(scenario_name: str) -> LayerKV:
    dump_dir = SCENARIO_DUMP_DIRS[scenario_name]
    result: LayerKV = {}
    for f in sorted(dump_dir.glob("*.pt")):
        data = torch.load(f, weights_only=False)
        if data.get("is_warmup_like", False):
            continue
        li = int(data["layer_idx"])
        if li not in result:
            result[li] = {
                "k": data["k"].float().cpu(),
                "v": data["v"].float().cpu(),
            }
    assert result, f"场景 {scenario_name} 的 dump 目录中没有有效文件"
    return result

# ============================================================
# 6. KV 偏差计算
# ============================================================

def cos_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-9
    return num / den


def compare(name_ref: str, name_tgt: str, label: str) -> dict:
    kv_ref = load_kv(name_ref)
    kv_tgt = load_kv(name_tgt)
    ref_s  = scenario_info[name_ref]["skill_start"]
    tgt_s  = scenario_info[name_tgt]["skill_start"]

    layers = sorted(set(kv_ref.keys()) & set(kv_tgt.keys()))
    assert layers
    print(f"\n  对比 [{label}]: {len(layers)} 层可用")

    per_layer: list[dict] = []
    per_tok_k: list[list[float]] = []
    per_tok_v: list[list[float]] = []

    for li in layers:
        k_ref = kv_ref[li]["k"][ref_s : ref_s + L_skill]
        v_ref = kv_ref[li]["v"][ref_s : ref_s + L_skill]
        k_tgt = kv_tgt[li]["k"][tgt_s : tgt_s + L_skill]
        v_tgt = kv_tgt[li]["v"][tgt_s : tgt_s + L_skill]

        k_rel = (k_ref - k_tgt).norm(dim=-1) / (k_ref.norm(dim=-1) + 1e-9)
        k_cos = cos_per_row(k_ref, k_tgt)
        v_rel = (v_ref - v_tgt).norm(dim=-1) / (v_ref.norm(dim=-1) + 1e-9)
        v_cos = cos_per_row(v_ref, v_tgt)

        per_tok_k.append(k_rel.tolist())
        per_tok_v.append(v_rel.tolist())
        per_layer.append({
            "layer":         li,
            "k_rel_l2_mean": float(k_rel.mean()),
            "k_rel_l2_max":  float(k_rel.max()),
            "k_cos_mean":    float(k_cos.mean()),
            "v_rel_l2_mean": float(v_rel.mean()),
            "v_rel_l2_max":  float(v_rel.max()),
            "v_cos_mean":    float(v_cos.mean()),
        })

    kmat      = np.array(per_tok_k)
    vmat      = np.array(per_tok_v)
    k_rel_arr = np.array([r["k_rel_l2_mean"] for r in per_layer])
    v_rel_arr = np.array([r["v_rel_l2_mean"] for r in per_layer])
    k_cos_arr = np.array([r["k_cos_mean"]    for r in per_layer])
    v_cos_arr = np.array([r["v_cos_mean"]    for r in per_layer])

    return {
        "label":     label,
        "ref":       name_ref,
        "tgt":       name_tgt,
        "per_layer": per_layer,
        "layers":    layers,
        "kmat":      kmat,
        "vmat":      vmat,
        "per_pos_k": kmat.mean(axis=0),
        "per_pos_v": vmat.mean(axis=0),
        "k_rel_arr": k_rel_arr,
        "v_rel_arr": v_rel_arr,
        "k_cos_arr": k_cos_arr,
        "v_cos_arr": v_cos_arr,
        "summary": {
            "k_rel_l2_mean":  float(k_rel_arr.mean()),
            "v_rel_l2_mean":  float(v_rel_arr.mean()),
            "k_cos_mean":     float(k_cos_arr.mean()),
            "v_cos_mean":     float(v_cos_arr.mean()),
            "k_rel_l2_range": [float(k_rel_arr.min()), float(k_rel_arr.max())],
            "v_rel_l2_range": [float(v_rel_arr.min()), float(v_rel_arr.max())],
        },
    }


print("\n计算 KV 偏差 ...")
result_BA = compare("A", "B", "B vs A  [Total: RoPE(76) + Context_short]")
result_EA = compare("A", "E", "E vs A  [Total: RoPE(76) + Context_near ]")
result_BE = compare("E", "B", "B vs E  [Pure Context: near (same pos=76)]")
results = [result_BA, result_EA, result_BE]

# ============================================================
# 7. 加载 Decomposition 实验的 B vs D（语义远基线）
# ============================================================
bd_far_data: dict | None = None
if DECOMPOSE_BVD_JSON.exists():
    with open(DECOMPOSE_BVD_JSON) as f:
        bd_far_data = json.load(f)
    bd_far_layers   = [r["layer"]         for r in bd_far_data["per_layer"]]
    bd_far_k_arr    = np.array([r["k_rel_l2_mean"] for r in bd_far_data["per_layer"]])
    bd_far_v_arr    = np.array([r["v_rel_l2_mean"] for r in bd_far_data["per_layer"]])
    bd_far_summary  = bd_far_data["summary"]
    print(f"\n✓ 加载 B vs D（语义远）基线: {DECOMPOSE_BVD_JSON.name}")
    print(f"  K L2={bd_far_summary['k_rel_l2_mean']:.4f}  "
          f"V L2={bd_far_summary['v_rel_l2_mean']:.4f}")
else:
    print(f"\n⚠  未找到 {DECOMPOSE_BVD_JSON.name}，跳过语义远基线对比。"
          f"\n   请先运行 run_decomposition.py。")

# ============================================================
# 8. 打印结果
# ============================================================
HDR = f"{'layer':>5} | {'K相对L2均值':>12} {'K余弦':>8} | {'V相对L2均值':>12} {'V余弦':>8}"
SEP = "-" * 58

for res in results:
    print(f"\n{'='*70}")
    print(f"  {res['label']}")
    print(f"{'='*70}")
    print(HDR); print(SEP)
    for r in res["per_layer"]:
        print(f"{r['layer']:>5} | "
              f"{r['k_rel_l2_mean']:>12.4f} {r['k_cos_mean']:>8.4f} | "
              f"{r['v_rel_l2_mean']:>12.4f} {r['v_cos_mean']:>8.4f}")
    s = res["summary"]
    print(f"\n  全层汇总:  K L2={s['k_rel_l2_mean']:.4f}  V L2={s['v_rel_l2_mean']:.4f}"
          f"  K cos={s['k_cos_mean']:.4f}  V cos={s['v_cos_mean']:.4f}")

# --- Layer 0 验证 ---
l0_be = next(r for r in result_BE["per_layer"] if r["layer"] == 0)
l0_ba = next(r for r in result_BA["per_layer"] if r["layer"] == 0)
l0_ea = next(r for r in result_EA["per_layer"] if r["layer"] == 0)

print(f"\n{'='*70}")
print("Layer 0 精细对比 (RoPE 纯读数)")
print(f"{'='*70}")
print(f"  B vs A  K L2 = {l0_ba['k_rel_l2_mean']:.6f}  (RoPE: 0→{P_skill})")
print(f"  E vs A  K L2 = {l0_ea['k_rel_l2_mean']:.6f}  (RoPE: 0→{P_skill}，同位置)")
print(f"  B vs E  K L2 = {l0_be['k_rel_l2_mean']:.6f}  ← 期望严格 = 0")
print(f"  B vs E  V L2 = {l0_be['v_rel_l2_mean']:.6f}  ← 期望严格 = 0")

# --- 语义距离对比汇总 ---
ba_s = result_BA["summary"]
be_s = result_BE["summary"]
print(f"\n{'='*70}")
print("语义距离假说验证")
print(f"{'='*70}")
print(f"  {'对比':40} {'K 相对L2':>10} {'V 相对L2':>10} {'K 余弦':>8} {'V 余弦':>8}")
print("  " + "-" * 82)
print(f"  {'B vs A (总偏差，参考基线)':40} "
      f"{ba_s['k_rel_l2_mean']:>10.4f} {ba_s['v_rel_l2_mean']:>10.4f} "
      f"{ba_s['k_cos_mean']:>8.4f} {ba_s['v_cos_mean']:>8.4f}")
if bd_far_data:
    print(f"  {'B vs D (纯上下文，语义远：数据库)':40} "
          f"{bd_far_summary['k_rel_l2_mean']:>10.4f} {bd_far_summary['v_rel_l2_mean']:>10.4f} "
          f"{bd_far_summary['k_cos_mean']:>8.4f} {bd_far_summary['v_cos_mean']:>8.4f}")
print(f"  {'B vs E (纯上下文，语义近：工具/邮件)':40} "
      f"{be_s['k_rel_l2_mean']:>10.4f} {be_s['v_rel_l2_mean']:>10.4f} "
      f"{be_s['k_cos_mean']:>8.4f} {be_s['v_cos_mean']:>8.4f}")

if bd_far_data:
    k_reduction = (bd_far_summary["k_rel_l2_mean"] - be_s["k_rel_l2_mean"])
    v_reduction = (bd_far_summary["v_rel_l2_mean"] - be_s["v_rel_l2_mean"])
    print(f"\n  K 偏差减少: {k_reduction:+.4f} "
          f"({k_reduction/bd_far_summary['k_rel_l2_mean']*100:+.1f}% vs 语义远)")
    print(f"  V 偏差减少: {v_reduction:+.4f} "
          f"({v_reduction/bd_far_summary['v_rel_l2_mean']*100:+.1f}% vs 语义远)")
    if k_reduction > 0 and v_reduction > 0:
        print("\n  ✓ 假说成立：语义相近的前缀产生更小的纯上下文 KV 偏差")
    elif k_reduction > 0 or v_reduction > 0:
        print("\n  △ 假说部分成立：K 或 V 中有一项偏差更小")
    else:
        print("\n  ✗ 假说不成立：语义相近的前缀未产生更小的 KV 偏差")

# ============================================================
# 9. 保存 JSON & NPZ
# ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_STEM = {
    result_BA["label"]: "semantic_BvsA",
    result_EA["label"]: "semantic_EvsA",
    result_BE["label"]: "semantic_BvsE",
}

for res in results:
    stem = LABEL_TO_STEM[res["label"]]
    jp   = OUT_DIR / f"{stem}_summary.json"
    np_  = OUT_DIR / f"{stem}_matrices.npz"
    json_data = {
        "label":         res["label"],
        "model":         MODEL_PATH,
        "skill_len":     L_skill,
        "prefix_len":    P_skill,
        "near_history":  NEAR_HISTORY,
        "short_history": SHORT_HISTORY,
        "scenario_ref":  {k: v for k, v in scenario_info[res["ref"]].items() if k != "ids"},
        "scenario_tgt":  {k: v for k, v in scenario_info[res["tgt"]].items() if k != "ids"},
        "per_layer":     res["per_layer"],
        "per_token_k_rel_l2_mean_over_layers": res["per_pos_k"].tolist(),
        "per_token_v_rel_l2_mean_over_layers": res["per_pos_v"].tolist(),
        "summary":       res["summary"],
    }
    with open(jp, "w") as f:
        json.dump(json_data, f, indent=2)
    np.savez(np_, k_rel_l2=res["kmat"], v_rel_l2=res["vmat"],
             layers=np.array(res["layers"]))
    print(f"保存: {jp.name}")

# ============================================================
# 10. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 图 1: 分层曲线 — B vs E（语义近）与 B vs A（总偏差）
#        并叠加 B vs D（语义远）作为对照
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
x = np.array(result_BE["layers"])
ax_k.plot(x, result_BA["k_rel_arr"], color="tab:purple", ls="-",  lw=1.4, label="B vs A (Total)")
ax_k.plot(x, result_BE["k_rel_arr"], color="tab:green",  ls="-",  lw=2.0, label="B vs E (Pure: Semantic Near)")
ax_v.plot(x, result_BA["v_rel_arr"], color="tab:purple", ls="-",  lw=1.4, label="B vs A (Total)")
ax_v.plot(x, result_BE["v_rel_arr"], color="tab:green",  ls="-",  lw=2.0, label="B vs E (Pure: Semantic Near)")
if bd_far_data:
    xf = np.array(bd_far_layers)
    ax_k.plot(xf, bd_far_k_arr, color="tab:red", ls="--", lw=1.4, alpha=0.7,
              label="B vs D (Pure: Semantic Far)")
    ax_v.plot(xf, bd_far_v_arr, color="tab:red", ls="--", lw=1.4, alpha=0.7,
              label="B vs D (Pure: Semantic Far)")
for ax, title in [(ax_k, "K Relative L2 per Layer"), (ax_v, "V Relative L2 per Layer")]:
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean relative L2 (skill tokens)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
fig.suptitle("Semantic Distance: Per-Layer KV Deviation\n"
             "Green=Near (B vs E), Red dashed=Far (B vs D), Purple=Total (B vs A)",
             fontsize=11)
fig.tight_layout()
p = FIG_DIR / "semantic_per_layer.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"\n图1: {p}")

# 图 2: Layer 0 验证 — B vs E 应严格为 0
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
xlabels = ["B vs A\n(Total)", "E vs A\n(RoPE+Near)", "B vs E\n(Pure Near)"]
colors   = ["tab:purple", "tab:blue", "tab:green"]
k_l0 = [r["per_layer"][0]["k_rel_l2_mean"] for r in results]
v_l0 = [r["per_layer"][0]["v_rel_l2_mean"] for r in results]
axes[0].bar(xlabels, k_l0, color=colors, alpha=0.8)
axes[0].set_title(f"Layer 0 — K Relative L2\n(B vs E 位置相同,应 = 0)")
axes[0].set_ylabel("Relative L2"); axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(k_l0):
    axes[0].text(i, v + 0.002, f"{v:.6f}", ha="center", fontsize=9)
axes[1].bar(xlabels, v_l0, color=colors, alpha=0.8)
axes[1].set_title("Layer 0 — V Relative L2\n(所有对比应 ≈ 0)")
axes[1].set_ylabel("Relative L2"); axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(v_l0):
    axes[1].text(i, v + 0.0001, f"{v:.6f}", ha="center", fontsize=9)
fig.suptitle("Layer 0: RoPE Verification (B vs E should = 0)", fontsize=11)
fig.tight_layout()
p = FIG_DIR / "semantic_layer0.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图2: {p}")

# 图 3: 汇总条形图 — Near vs Far vs Total
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
bar_labels = ["B vs A\n(Total)"]
k_vals = [ba_s["k_rel_l2_mean"]]
v_vals = [ba_s["v_rel_l2_mean"]]
kc_vals = [ba_s["k_cos_mean"]]
vc_vals = [ba_s["v_cos_mean"]]
bar_colors = ["tab:purple"]
if bd_far_data:
    bar_labels.append("B vs D\n(Far)")
    k_vals.append(bd_far_summary["k_rel_l2_mean"])
    v_vals.append(bd_far_summary["v_rel_l2_mean"])
    kc_vals.append(bd_far_summary["k_cos_mean"])
    vc_vals.append(bd_far_summary["v_cos_mean"])
    bar_colors.append("tab:red")
bar_labels.append("B vs E\n(Near)")
k_vals.append(be_s["k_rel_l2_mean"])
v_vals.append(be_s["v_rel_l2_mean"])
kc_vals.append(be_s["k_cos_mean"])
vc_vals.append(be_s["v_cos_mean"])
bar_colors.append("tab:green")

x = np.arange(len(bar_labels)); w = 0.35
axes[0].bar(x - w/2, k_vals, w, label="K", color="steelblue",  alpha=0.8)
axes[0].bar(x + w/2, v_vals, w, label="V", color="salmon", alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(bar_labels)
axes[0].set_ylabel("Mean Relative L2")
axes[0].set_title("KV Relative L2: Semantic Near vs Far")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
for i, (km, vm) in enumerate(zip(k_vals, v_vals)):
    axes[0].text(i - w/2, km + 0.004, f"{km:.3f}", ha="center", fontsize=8)
    axes[0].text(i + w/2, vm + 0.004, f"{vm:.3f}", ha="center", fontsize=8)
axes[1].bar(x - w/2, kc_vals, w, label="K", color="steelblue",  alpha=0.8)
axes[1].bar(x + w/2, vc_vals, w, label="V", color="salmon", alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(bar_labels)
axes[1].set_ylabel("Mean Cosine Similarity")
axes[1].set_title("KV Cosine Similarity: Semantic Near vs Far")
axes[1].set_ylim(0.5, 1.05); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
for i, (km, vm) in enumerate(zip(kc_vals, vc_vals)):
    axes[1].text(i - w/2, km - 0.03, f"{km:.3f}", ha="center", fontsize=8)
    axes[1].text(i + w/2, vm - 0.03, f"{vm:.3f}", ha="center", fontsize=8)
fig.suptitle("Semantic Distance Effect: All-Layer Summary\n"
             "Far = database context (B vs D),  Near = similar email/tools context (B vs E)",
             fontsize=11)
fig.tight_layout()
p = FIG_DIR / "semantic_summary_bar.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图3: {p}")

# 图 4: per-token 曲线 — B vs E（近）与 B vs D（远）对比
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
pos = np.arange(L_skill)
ax_k.plot(pos, result_BE["per_pos_k"], color="tab:green",  lw=1.6,
          label="K B vs E (Near Context)")
ax_k.plot(pos, result_BA["per_pos_k"], color="tab:purple", lw=1.0, ls="--", alpha=0.5,
          label="K B vs A (Total)")
ax_v.plot(pos, result_BE["per_pos_v"], color="tab:green",  lw=1.6,
          label="V B vs E (Near Context)")
ax_v.plot(pos, result_BA["per_pos_v"], color="tab:purple", lw=1.0, ls="--", alpha=0.5,
          label="V B vs A (Total)")
if bd_far_data:
    bd_per_tok_k = np.array(bd_far_data["per_token_k_rel_l2_mean_over_layers"])
    bd_per_tok_v = np.array(bd_far_data["per_token_v_rel_l2_mean_over_layers"])
    common_len = min(len(bd_per_tok_k), L_skill)
    ax_k.plot(np.arange(common_len), bd_per_tok_k[:common_len], color="tab:red",
              lw=1.0, ls=":", alpha=0.6, label="K B vs D (Far Context)")
    ax_v.plot(np.arange(common_len), bd_per_tok_v[:common_len], color="tab:red",
              lw=1.0, ls=":", alpha=0.6, label="V B vs D (Far Context)")
for ax in (ax_k, ax_v):
    ax.axvspan(0, min(4, L_skill) - 0.5, alpha=0.12, color="gold", label="First 4 tokens")
    ax.set_xlabel("Token index within skill")
    ax.set_ylabel("Relative L2 (mean over layers)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax_k.set_title("K: Near vs Far Context, per Skill Token")
ax_v.set_title("V: Near vs Far Context, per Skill Token")
fig.suptitle("Per-Token: Semantic Near (B vs E) vs Far (B vs D) vs Total (B vs A)",
             fontsize=11)
fig.tight_layout()
p = FIG_DIR / "semantic_per_token.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图4: {p}")

print(f"\n{'='*60}")
print(f"✓ 实验完成!所有结果写入: {OUT_DIR}")
print(f"{'='*60}")
