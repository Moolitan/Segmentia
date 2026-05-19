#!/usr/bin/env python3
"""
run_decomposition.py
====================
RoPE 贡献 vs 上下文贡献分解实验。

在 Scenario B (76 tok 前缀) 观察到 K 相对 L2 = 0.555 的总偏差。
本实验把这个偏差拆成两部分:

  RoPE 贡献  : skill 位置从 0 变到 76,K 被旋转了不同角度
  上下文贡献  : skill 前面有不同内容的前文,attention 把上下文差异注入进来

方法:新增场景 D —— ALT_HISTORY (同样 76 tok,完全不同内容) + skill + query。
     D 的 skill 与 B 落在完全相同的绝对位置(76)。

跨进程 dump 目录切换
---------------------
vLLM EngineCore 是 subprocess,os.environ 在启动后无法从主进程动态更新。
解决方案:修改 qwen3.py hook,优先读取信号文件 /tmp/kv_current_scenario。
主进程在每次 prefill 前写入当前场景名,hook 运行时实时读取,从而把每个
场景的 KV dump 写到各自独立的目录。

三种对比:
  B vs A : 总偏差 = RoPE(0→76) + Context(空 → SHORT_HISTORY)   [参考]
  D vs A :  RoPE(0→76) + Context(空 → ALT_HISTORY)
  B vs D : 纯上下文差异 (位置严格相同=76,内容不同)
           Layer 0 K 差异 ≡ 0 (严格验证点)
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
DUMP_ROOT     = Path("/tmp/kv_dump_decompose")
MAX_MODEL_LEN = 32768
OUT_DIR       = RESULTS_ROOT / "position_context_analysis" / "decomposition"
FIG_DIR       = OUT_DIR / "figures"
SIGNAL_FILE   = Path("/tmp/kv_current_scenario")

# 清理旧 dump 和信号文件
if DUMP_ROOT.exists():
    shutil.rmtree(DUMP_ROOT)
DUMP_ROOT.mkdir(parents=True)
if SIGNAL_FILE.exists():
    SIGNAL_FILE.unlink()

# 必须在 import vllm 前设置,EngineCore subprocess 启动时继承
# DUMP_DIR 告诉 hook 根目录;具体子目录由信号文件动态决定
os.environ["DUMP_KV"]          = "1"
os.environ["DUMP_DIR"]         = str(DUMP_ROOT)
os.environ["DUMP_SKIP_WARMUP"] = "1"
os.environ["SCENARIO"]         = "init"   # 兜底值,信号文件优先

# 场景 dump 目录(hook 会自动创建 DUMP_ROOT/scenario_{name}/)
SCENARIO_DUMP_DIRS: dict[str, Path] = {
    name: DUMP_ROOT / f"scenario_{name}"
    for name in ["A", "B", "D"]
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

# 场景 B 的前缀 (76 token)
SHORT_HISTORY = """\
User: Hello, what tools do you have available today?
Assistant: I have access to several tools including email sending, calendar management,
file search, and task creation. How can I help you today?
User: I need to communicate with a colleague about an ongoing project.
Assistant: Of course! I can help you send emails or messages. What would you like to say?
"""

# 场景 D 的候选前缀:完全不同领域(数据库性能分析)
ALT_HISTORY_CANDIDATE = """\
User: I need to review our database query performance from this morning.
Assistant: Of course! Which database system are you using, and do you have the slow query logs available to review?
User: It is PostgreSQL. Several queries are taking over three seconds to return results.
Assistant: Three seconds is too slow for most use cases. We should start by examining indexes and execution plans.
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


# 把 ALT_HISTORY_CANDIDATE 截到与 SHORT_HISTORY 完全相同的 token 数
short_ids = tokenizer(SHORT_HISTORY,         add_special_tokens=False)["input_ids"]
alt_ids   = tokenizer(ALT_HISTORY_CANDIDATE, add_special_tokens=False)["input_ids"]
P = len(short_ids)
print(f"\nSHORT_HISTORY token 数: {P}")
print(f"ALT_HISTORY_CANDIDATE token 数: {len(alt_ids)} (将截到 {P})")

if len(alt_ids) < P:
    raise ValueError(f"ALT_HISTORY_CANDIDATE 不足 {P} token,请扩充文本")
alt_ids_trimmed = alt_ids[:P]
ALT_HISTORY = tokenizer.decode(alt_ids_trimmed)

alt_verify = tokenizer(ALT_HISTORY, add_special_tokens=False)["input_ids"]
assert len(alt_verify) == P, f"截断后 token 数不匹配: {len(alt_verify)} != {P}"
print(f"ALT_HISTORY 截断后 token 数: {len(alt_verify)}  ✓")
print(f"\n-- ALT_HISTORY (截断后) --\n{ALT_HISTORY}\n{'--'*30}")

# 构造三个 prompt
PROMPTS: dict[str, str] = {
    "A": SKILL_TEXT + QUERY_TEXT,
    "B": SHORT_HISTORY + SKILL_TEXT + QUERY_TEXT,
    "D": ALT_HISTORY   + SKILL_TEXT + QUERY_TEXT,
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

# 验证 B 和 D 的 skill 起始位置严格一致
assert scenario_info["B"]["skill_start"] == scenario_info["D"]["skill_start"], (
    f"B 和 D 的 skill 起始位置不一致: "
    f"B={scenario_info['B']['skill_start']}, D={scenario_info['D']['skill_start']}"
)
P_skill = scenario_info["B"]["skill_start"]
print(f"\n✓ B 和 D 的 skill 起始位置严格一致: {P_skill}")
print(f"  → B vs D 中 RoPE 旋转角度完全相同,Layer 0 K 差异 ≡ 0")

skill_lens = [v["skill_len"] for v in scenario_info.values()]
L_skill = min(skill_lens)
if len(set(skill_lens)) > 1:
    print(f"\n⚠  skill token 长度不一致: {skill_lens},截到 {L_skill}")
else:
    print(f"✓  skill token 长度一致: {L_skill}")

for name, info in scenario_info.items():
    assert info["N"] < MAX_MODEL_LEN, f"场景 {name} 超出 MAX_MODEL_LEN"

# ============================================================
# 3. 加载 vLLM (必须在 env var 设置之后)
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
    """写信号文件 → hook 运行时读取 → dump 写入对应子目录。"""
    SIGNAL_FILE.write_text(scenario_name)
    info = scenario_info[scenario_name]
    print(f"Prefill 场景 {scenario_name}  (N={info['N']} tokens, skill@{info['skill_start']}) ...")
    llm.generate([TokensPrompt(prompt_token_ids=info["ids"])], _SP)
    dump_dir = SCENARIO_DUMP_DIRS[scenario_name]
    n_files  = len(list(dump_dir.glob("*.pt"))) if dump_dir.exists() else 0
    print(f"  → dump 到 {dump_dir.name}/  ({n_files} 个文件)")


for name in ["A", "B", "D"]:
    run_prefill(name)

SIGNAL_FILE.unlink(missing_ok=True)   # 清理信号文件

# ============================================================
# 5. 验证 dump 目录
# ============================================================
for name, d in SCENARIO_DUMP_DIRS.items():
    n = len(list(d.glob("*.pt"))) if d.exists() else 0
    print(f"  scenario_{name}/: {n} 个文件")
    assert n > 0, f"场景 {name} 的 dump 目录为空,信号文件切换可能未生效"

# ============================================================
# 6. 读取 KV Dump (每个场景从自己的目录读)
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
# 7. KV 偏差计算
# ============================================================

def cos_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-9
    return num / den


def compare(name_ref: str, name_tgt: str, label: str) -> dict:
    kv_ref = load_kv(name_ref) # 参考场景的 KV
    kv_tgt = load_kv(name_tgt) # 目标场景的 KV
    ref_s  = scenario_info[name_ref]["skill_start"]
    tgt_s  = scenario_info[name_tgt]["skill_start"]

    layers = sorted(set(kv_ref.keys()) & set(kv_tgt.keys()))
    assert layers, f"场景 {name_ref} 和 {name_tgt} 没有共同层"
    print(f"\n  对比 [{label}]: {len(layers)} 层可用")

    per_layer: list[dict] = []
    per_tok_k: list[list[float]] = []
    per_tok_v: list[list[float]] = []

    for li in layers:
        k_ref = kv_ref[li]["k"][ref_s : ref_s + L_skill]
        v_ref = kv_ref[li]["v"][ref_s : ref_s + L_skill]
        k_tgt = kv_tgt[li]["k"][tgt_s : tgt_s + L_skill]
        v_tgt = kv_tgt[li]["v"][tgt_s : tgt_s + L_skill]

        # 相对 L2 变化率：Key 向量的变化幅度占参考向量模长的比例。值越大表示该 token 位置的 Key 变化越剧烈。
        k_rel = (k_ref - k_tgt).norm(dim=-1) / (k_ref.norm(dim=-1) + 1e-9)
        # 余弦相似度：只关注方向变化，忽略模长缩放。越接近 1 表示方向越一致。
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

    kmat      = np.array(per_tok_k) #每个 token 位置、每一层的相对 L2 变化值，可用于绘制热力图
    vmat      = np.array(per_tok_v) #每个 token 位置、每一层的相对 L2 变化值，可用于绘制热力图
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
result_DA = compare("A", "D", "D vs A  [~RoPE(76)  + Context_alt       ]")
result_BD = compare("D", "B", "B vs D  [Pure Context (same pos=76)     ]")
results = [result_BA, result_DA, result_BD]

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

# --- 关键验证:B vs D 在 Layer 0 ---
l0_bd = next(r for r in result_BD["per_layer"] if r["layer"] == 0)
l0_ba = next(r for r in result_BA["per_layer"] if r["layer"] == 0)
l0_da = next(r for r in result_DA["per_layer"] if r["layer"] == 0)

print(f"\n{'='*70}")
print("Layer 0 精细对比 (RoPE 纯读数)")
print(f"{'='*70}")
print(f"  B vs A  K L2 = {l0_ba['k_rel_l2_mean']:.6f}  (RoPE: 0→{P_skill})")
print(f"  D vs A  K L2 = {l0_da['k_rel_l2_mean']:.6f}  (RoPE: 0→{P_skill}，同位置)")
print(f"  B vs D  K L2 = {l0_bd['k_rel_l2_mean']:.6f}  ← 期望严格 = 0")
print(f"  B vs D  V L2 = {l0_bd['v_rel_l2_mean']:.6f}  ← 期望严格 = 0")

# --- 贡献分解 ---
ba, da, bd = result_BA["summary"], result_DA["summary"], result_BD["summary"]
print(f"\n{'='*70}")
print("贡献分解汇总")
print(f"{'='*70}")
print(f"  {'对比':36} {'K 相对L2':>10} {'V 相对L2':>10} {'K 余弦':>8} {'V 余弦':>8}")
print("  " + "-" * 74)
for label, s in [
    ("B vs A (总偏差)", ba),
    ("D vs A (RoPE + Context_alt)", da),
    ("B vs D (纯上下文差异)", bd),
]:
    print(f"  {label:36} {s['k_rel_l2_mean']:>10.4f} {s['v_rel_l2_mean']:>10.4f}"
          f" {s['k_cos_mean']:>8.4f} {s['v_cos_mean']:>8.4f}")

k_ctx = bd["k_rel_l2_mean"]; k_tot = ba["k_rel_l2_mean"]
v_ctx = bd["v_rel_l2_mean"]; v_tot = ba["v_rel_l2_mean"]
print(f"\n  K 分解:  总={k_tot:.4f}  上下文{k_ctx:.4f}({k_ctx/k_tot*100:.0f}%)"
      f"  剩余(RoPE){k_tot-k_ctx:.4f}({(k_tot-k_ctx)/k_tot*100:.0f}%)")
print(f"  V 分解:  总={v_tot:.4f}  上下文{v_ctx:.4f}({v_ctx/v_tot*100:.0f}%)"
      f"  剩余(RoPE){v_tot-v_ctx:.4f}({(v_tot-v_ctx)/v_tot*100:.0f}%)")

# ============================================================
# 9. 保存 JSON & NPZ
# ============================================================
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

LABEL_TO_STEM = {
    result_BA["label"]: "decompose_BvsA",
    result_DA["label"]: "decompose_DvsA",
    result_BD["label"]: "decompose_BvsD",
}

for res in results:
    stem = LABEL_TO_STEM[res["label"]]
    jp   = OUT_DIR / f"{stem}_summary.json"
    np_  = OUT_DIR / f"{stem}_matrices.npz"
    json_data = {
        "label":        res["label"],
        "model":        MODEL_PATH,
        "skill_len":    L_skill,
        "prefix_len_B": P_skill,
        "prefix_len_D": P_skill,
        "pos_diff_BD":  0,
        "scenario_ref": {k: v for k, v in scenario_info[res["ref"]].items() if k != "ids"},
        "scenario_tgt": {k: v for k, v in scenario_info[res["tgt"]].items() if k != "ids"},
        "per_layer":    res["per_layer"],
        "per_token_k_rel_l2_mean_over_layers": res["per_pos_k"].tolist(),
        "per_token_v_rel_l2_mean_over_layers": res["per_pos_v"].tolist(),
        "summary":      res["summary"],
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

STYLE = {
    result_BA["label"]: ("tab:purple", "-",  "B vs A (Total)"),
    result_DA["label"]: ("tab:blue",   "--", "D vs A (RoPE + Context_alt)"),
    result_BD["label"]: ("tab:red",    ":",  "B vs D (Pure Context)"),
}

# 图 1: 分层曲线
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
for res in results:
    col, ls, lbl = STYLE[res["label"]]
    x = np.array(res["layers"])
    ax_k.plot(x, res["k_rel_arr"], color=col, ls=ls, marker="o", ms=3, lw=1.6, label=lbl)
    ax_v.plot(x, res["v_rel_arr"], color=col, ls=ls, marker="s", ms=3, lw=1.6, label=lbl)
for ax, title in [(ax_k, "K Relative L2 per Layer"), (ax_v, "V Relative L2 per Layer")]:
    ax.set_xlabel("Layer index"); ax.set_ylabel("Mean relative L2 (skill tokens)")
    ax.set_title(title); ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("RoPE vs Context Decomposition — Per-Layer KV Deviation", fontsize=13)
fig.tight_layout()
p = FIG_DIR / "decompose_per_layer.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"\n图1: {p}")

# 图 2: Layer 0 放大
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
xlabels = ["B vs A\n(Total)", "D vs A\n(RoPE+CtxAlt)", "B vs D\n(PureCtx)"]
colors  = ["tab:purple", "tab:blue", "tab:red"]
k_l0 = [r["per_layer"][0]["k_rel_l2_mean"] for r in results]
v_l0 = [r["per_layer"][0]["v_rel_l2_mean"] for r in results]
axes[0].bar(xlabels, k_l0, color=colors, alpha=0.8)
axes[0].set_title(f"Layer 0 — K Relative L2\n(B vs D 位置严格相同,应 = 0)")
axes[0].set_ylabel("Relative L2"); axes[0].grid(axis="y", alpha=0.3)
for i, v in enumerate(k_l0):
    axes[0].text(i, v + 0.002, f"{v:.6f}", ha="center", fontsize=9)
axes[1].bar(xlabels, v_l0, color=colors, alpha=0.8)
axes[1].set_title("Layer 0 — V Relative L2\n(所有对比应  0)")
axes[1].set_ylabel("Relative L2"); axes[1].grid(axis="y", alpha=0.3)
for i, v in enumerate(v_l0):
    axes[1].text(i, v + 0.0001, f"{v:.6f}", ha="center", fontsize=9)
fig.suptitle("Layer 0: Pure RoPE Rotation Signal", fontsize=11)
fig.tight_layout()
p = FIG_DIR / "decompose_layer0.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图2: {p}")

# 图 3: 汇总条形图
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
x = np.arange(3); w = 0.35
all_lbl = ["B vs A\n(Total)", "D vs A\n(RoPE)", "B vs D\n(Context)"]
k_m = [r["summary"]["k_rel_l2_mean"] for r in results]
v_m = [r["summary"]["v_rel_l2_mean"] for r in results]
kc  = [r["summary"]["k_cos_mean"]    for r in results]
vc  = [r["summary"]["v_cos_mean"]    for r in results]
axes[0].bar(x-w/2, k_m, w, label="K", color="tab:blue",  alpha=0.8)
axes[0].bar(x+w/2, v_m, w, label="V", color="tab:red",   alpha=0.8)
axes[0].set_xticks(x); axes[0].set_xticklabels(all_lbl)
axes[0].set_ylabel("Mean Relative L2"); axes[0].set_title("All-Layer KV Relative L2")
axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
for i, (km, vm) in enumerate(zip(k_m, v_m)):
    axes[0].text(i-w/2, km+0.004, f"{km:.3f}", ha="center", fontsize=8)
    axes[0].text(i+w/2, vm+0.004, f"{vm:.3f}", ha="center", fontsize=8)
axes[1].bar(x-w/2, kc, w, label="K", color="tab:blue",  alpha=0.8)
axes[1].bar(x+w/2, vc, w, label="V", color="tab:red",   alpha=0.8)
axes[1].set_xticks(x); axes[1].set_xticklabels(all_lbl)
axes[1].set_ylabel("Mean Cosine Similarity"); axes[1].set_title("All-Layer KV Cosine Similarity")
axes[1].set_ylim(0.5, 1.05); axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
fig.suptitle("RoPE vs Context: All-Layer Summary", fontsize=12)
fig.tight_layout()
p = FIG_DIR / "decompose_summary_bar.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图3: {p}")

# 图 4: B vs D 分层曲线叠加
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
x = np.array(result_BD["layers"])
ax_k.plot(x, result_BD["k_rel_arr"], color="tab:red",    lw=2.0, label="B vs D (Pure Context)")
ax_k.plot(x, result_BA["k_rel_arr"], color="tab:purple", lw=1.2, ls="--", alpha=0.6, label="B vs A (Total)")
ax_k.plot(x, result_DA["k_rel_arr"], color="tab:blue",   lw=1.2, ls=":",  alpha=0.6, label="D vs A (RoPE+CtxAlt)")
ax_v.plot(x, result_BD["v_rel_arr"], color="tab:red",    lw=2.0, label="B vs D (Pure Context)")
ax_v.plot(x, result_BA["v_rel_arr"], color="tab:purple", lw=1.2, ls="--", alpha=0.6, label="B vs A (Total)")
ax_v.plot(x, result_DA["v_rel_arr"], color="tab:blue",   lw=1.2, ls=":",  alpha=0.6, label="D vs A (RoPE+CtxAlt)")
for ax, t in [(ax_k, "K Rel L2: Context vs Total"), (ax_v, "V Rel L2: Context vs Total")]:
    ax.set_xlabel("Layer index"); ax.set_ylabel("Mean relative L2"); ax.set_title(t)
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
fig.suptitle("Per-Layer: Pure Context (B vs D) vs Total (B vs A)", fontsize=12)
fig.tight_layout()
p = FIG_DIR / "decompose_context_vs_total.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图4: {p}")

# 图 5: per-token 曲线
fig, (ax_k, ax_v) = plt.subplots(1, 2, figsize=(14, 5))
pos = np.arange(L_skill)
ax_k.plot(pos, result_BD["per_pos_k"], color="tab:red",    lw=1.4, label="K B vs D (Pure Context)")
ax_k.plot(pos, result_BA["per_pos_k"], color="tab:purple", lw=1.0, ls="--", alpha=0.5, label="K B vs A (Total)")
ax_v.plot(pos, result_BD["per_pos_v"], color="tab:orange", lw=1.4, label="V B vs D (Pure Context)")
ax_v.plot(pos, result_BA["per_pos_v"], color="tab:gray",   lw=1.0, ls="--", alpha=0.5, label="V B vs A (Total)")
for ax in (ax_k, ax_v):
    ax.axvspan(0, min(4, L_skill)-0.5, alpha=0.12, color="gold", label="First 4 tokens")
    ax.set_xlabel("Token index within skill"); ax.set_ylabel("Relative L2 (mean over layers)")
    ax.grid(alpha=0.3); ax.legend(fontsize=8)
ax_k.set_title("K: Pure Context vs Total, per Skill Token")
ax_v.set_title("V: Pure Context vs Total, per Skill Token")
fig.suptitle("Per-Token: B vs D (Pure Context) vs B vs A (Total)", fontsize=12)
fig.tight_layout()
p = FIG_DIR / "decompose_per_token.png"
fig.savefig(p, dpi=150); plt.close(fig)
print(f"图5: {p}")

print(f"\n{'='*60}")
print(f"✓ 实验完成!所有结果写入: {OUT_DIR}")
print(f"{'='*60}")
