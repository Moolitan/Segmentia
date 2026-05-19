import os
import re
import sys
import json
import shutil
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# ===== 路径 =====
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ROOT = BASE_DIR.parent.parent
RESULTS_ROOT = BASE_DIR / "results"
print(f"root_directory: {ROOT}")
print(f"scripts_directory: {BASE_DIR}\n{'='*60}")
TRACE_DIR = BASE_DIR / "prompts" / "multiturn_sequence_traces"
SKILL_NAME = "internal-comms"


REF_FILE    = "003.txt"   # 仅用作"skill 被包装版原文"的锚点,不参与 prefill
TARGET_FILE = "026.txt"   # 唯一的 prefill 目标;skill 在其中出现 2 次

# ===== Step 1: 环境变量 (必须在 import vllm 之前) =====
SCENARIO = f"E_{SKILL_NAME}_026_first_vs_last"
DUMP_ROOT = "/tmp/kv_dump"
os.environ["DUMP_KV"] = "1"
os.environ["SCENARIO"] = SCENARIO
os.environ["DUMP_DIR"] = DUMP_ROOT

DUMP_DIR = Path(DUMP_ROOT) / f"scenario_{SCENARIO}"
if DUMP_DIR.exists():
    print(f"清空已存在的 dump 目录: {DUMP_DIR}")
    shutil.rmtree(DUMP_DIR)

MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"

# ===== Step 2: tokenize prompt,并定位 skill 的 token 区间 =====
from transformers import AutoTokenizer  # type: ignore[import-not-found]
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
assert tokenizer.is_fast, "需要 fast tokenizer 以获得 offset_mapping"

ref_text    = (TRACE_DIR / REF_FILE   ).read_text(encoding="utf-8")
target_text = (TRACE_DIR / TARGET_FILE).read_text(encoding="utf-8")

# 注意:trace 里的 skill 内容经过了终端宽度换行(SPACE→NEWLINE),和原始
# SKILL.md 不是字节级相等。这里直接从 003.txt (skill 首次出现的请求) 抽出
# "被包装版",作为在 026.txt 里 find/rfind 的查询串 —— 所有 trace 的包装
# 逻辑一致,可精确匹配。
_SKILL_HEAD_RE = re.compile(r"---\nname:\s*" + re.escape(SKILL_NAME))
_SKILL_TAIL = "internal comms\n"  # SKILL.md 最后一行(keywords) + 末尾换行

def _extract_skill_wrapped(text: str) -> str:
    m = _SKILL_HEAD_RE.search(text)
    assert m, f"未在 {REF_FILE} 找到 skill 起始标记 '---\\nname: {SKILL_NAME}'"
    start = m.start()
    end = text.find(_SKILL_TAIL, start)
    assert end != -1, "未找到 skill 结束标记"
    return text[start : end + len(_SKILL_TAIL)]

SKILL_TEXT = _extract_skill_wrapped(ref_text)
print(f"skill text: \n{SKILL_TEXT}")
print(f"[参考] 从 {REF_FILE} 抽出的 skill 文本长度: {len(SKILL_TEXT)} 字符 \n{'='*60}")


def tokenize_and_locate_skill(
    prompt_text: str, skill_text: str, occurrence: str = "first"
) -> tuple[list[int], int, int]:
    """整体 tokenize,再根据 skill 原文的字符偏移找到对应 token 区间。

    BPE 可能在 skill 两端"吃掉"边界字符,导致子串不是严格的 token-id 序列匹配;
    但字符偏移 + offset_mapping 方式对边界鲁棒 —— 我们只取"完全落在字符窗口
    内部"的 token 作为 skill 区间。

    返回 (prompt_token_ids, tok_start, tok_end);skill 区间是 [tok_start, tok_end)。
    """
    if occurrence == "first":
        char_start = prompt_text.find(skill_text)
    elif occurrence == "last":
        char_start = prompt_text.rfind(skill_text)
    else:
        raise ValueError(occurrence)
    assert char_start != -1, f"SKILL 原文在 prompt 里没找到 ({occurrence})"
    char_end = char_start + len(skill_text)

    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]  # list[(char_start, char_end)]

    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if s >= char_start and tok_start is None:
            tok_start = i
        if e <= char_end:
            tok_end = i + 1  # [tok_start, tok_end) 左闭右开
    assert tok_start is not None and tok_end is not None and tok_end > tok_start, \
        f"定位 skill token 区间失败: {tok_start}, {tok_end}"
    return ids, tok_start, tok_end


# 同一段 prompt 文本 tokenize 一次,两次定位,避免重复编码。
target_ids, skill_start_1, skill_end_1 = tokenize_and_locate_skill(
    target_text, SKILL_TEXT, occurrence="first")
_,          skill_start_2, skill_end_2 = tokenize_and_locate_skill(
    target_text, SKILL_TEXT, occurrence="last")

N = len(target_ids)
L1 = skill_end_1 - skill_start_1
L2 = skill_end_2 - skill_start_2

print(f"{TARGET_FILE}:")
print(f"  总 token 数: {N}")
print(f"  首次 skill 区间: [{skill_start_1}, {skill_end_1})  长度 {L1}")
print(f"  末次 skill 区间: [{skill_start_2}, {skill_end_2})  长度 {L2}")
assert skill_start_2 > skill_start_1, "末次 skill 必须在首次之后"

if L1 != L2:
    print(f"\n⚠ skill token 长度不一致 ({L1} vs {L2}),可能是 BPE 边界漂移。"
          f"  对比时将截到 min 长度 {min(L1, L2)}。")
L_skill = min(L1, L2)

MAX_MODEL_LEN = 32768
assert N < MAX_MODEL_LEN

# ===== Step 3: 加载 vLLM,prefill 两个目标文件 =====
from vllm import LLM, SamplingParams, TokensPrompt  # type: ignore[import-not-found]

print(f"\n加载 {MODEL_PATH} ...")
llm = LLM(
    model=MODEL_PATH,
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=MAX_MODEL_LEN,
    enable_prefix_caching=False,          # 禁用:强制每次 prefill 重新计算全量 KV
    max_num_batched_tokens=MAX_MODEL_LEN,  # 确保最长 prompt 一次 forward 完成,不被 chunked prefill 拆分
)
print("✓ 模型加载完成\n")

# prefix caching 已禁用,只需 prefill 唯一目标 026.txt 一次。
print(f"prefill {TARGET_FILE} ({N} tokens) ...")
llm.generate(
    [TokensPrompt(prompt_token_ids=target_ids)],
    SamplingParams(max_tokens=1, temperature=0),
)
print()

# ===== Step 4: 扫描 dump,收集 026 prefill 的每层 KV =====
import torch
import numpy as np

files = sorted(DUMP_DIR.glob("*.pt"))
print(f"总共 dump {len(files)} 个文件\n")

# layer_idx -> {"k": ..., "v": ...}  (只有一次 prefill,无需区分 first/second)
layer_to_kv: dict[int, dict] = {}
for f in files:
    data = torch.load(f, weights_only=False)
    n = int(data["num_tokens"])
    if n != N:
        continue  # 跳过 warmup (num_tokens=256) 和 decode (1)
    pos = data["positions"].tolist()
    if pos != list(range(n)):
        continue  # 保险起见再 check 一下 positions 是完整的 0..N-1
    layer_idx = int(data["layer_idx"])
    if layer_idx in layer_to_kv:
        continue  # 同层只取第一条(防御性)
    layer_to_kv[layer_idx] = {
        "k": data["k"].float().cpu(),   # (N, num_kv_heads*head_dim)
        "v": data["v"].float().cpu(),
    }

layers = sorted(layer_to_kv.keys())
print(f"命中的层数: {len(layers)}")
if not layers:
    sys.exit(f"✗ 没有任何一层在 {TARGET_FILE} prefill 里被 dump,无法对比")

# ===== Step 5: 对比 026 内部 首次 vs 末次 skill 区间的 K/V =====
def cos_per_row(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    num = (a * b).sum(dim=-1)
    den = a.norm(dim=-1) * b.norm(dim=-1) + 1e-9
    return num / den


per_layer = []
per_token_k_rel = []
per_token_v_rel = []

for layer_idx in layers:
    kv = layer_to_kv[layer_idx]

    # 同一次 prefill 的 K/V 张量,分别切出首次与末次 skill 区间,截到 min 长度
    k1 = kv["k"][skill_start_1 : skill_start_1 + L_skill]
    k2 = kv["k"][skill_start_2 : skill_start_2 + L_skill]
    v1 = kv["v"][skill_start_1 : skill_start_1 + L_skill]
    v2 = kv["v"][skill_start_2 : skill_start_2 + L_skill]

    k_abs = (k1 - k2).norm(dim=-1)
    k_rel = k_abs / (k1.norm(dim=-1) + 1e-9)
    k_cos = cos_per_row(k1, k2)

    v_abs = (v1 - v2).norm(dim=-1)
    v_rel = v_abs / (v1.norm(dim=-1) + 1e-9)
    v_cos = cos_per_row(v1, v2)

    per_token_k_rel.append(k_rel.tolist())
    per_token_v_rel.append(v_rel.tolist())

    per_layer.append({
        "layer": layer_idx,
        "k_rel_l2_mean": float(k_rel.mean()),
        "k_rel_l2_max":  float(k_rel.max()),
        "k_cos_mean":    float(k_cos.mean()),
        "v_rel_l2_mean": float(v_rel.mean()),
        "v_rel_l2_max":  float(v_rel.max()),
        "v_cos_mean":    float(v_cos.mean()),
    })

# ===== Step 6: 打印分层结果 + 汇总 =====
print(f"{'layer':>5} | {'K相对L2均值':>12} {'K相对L2最大':>12} {'K余弦':>8} | {'V相对L2均值':>12} {'V相对L2最大':>12} {'V余弦':>8}")
print("-" * 92)
for r in per_layer:
    print(f"{r['layer']:>5} | "
          f"{r['k_rel_l2_mean']:>12.4f} {r['k_rel_l2_max']:>12.4f} {r['k_cos_mean']:>8.4f} | "
          f"{r['v_rel_l2_mean']:>12.4f} {r['v_rel_l2_max']:>12.4f} {r['v_cos_mean']:>8.4f}")

k_rel_arr = np.array([r["k_rel_l2_mean"] for r in per_layer])
v_rel_arr = np.array([r["v_rel_l2_mean"] for r in per_layer])
k_cos_arr = np.array([r["k_cos_mean"]    for r in per_layer])
v_cos_arr = np.array([r["v_cos_mean"]    for r in per_layer])

print("\n" + "=" * 60)
print("全层汇总 (跨 layer 平均)")
print("=" * 60)
print(f"  K 相对 L2:    mean={k_rel_arr.mean():.4f}  range=[{k_rel_arr.min():.4f}, {k_rel_arr.max():.4f}]")
print(f"  V 相对 L2:    mean={v_rel_arr.mean():.4f}  range=[{v_rel_arr.min():.4f}, {v_rel_arr.max():.4f}]")
print(f"  K 余弦相似度:  mean={k_cos_arr.mean():.4f}  range=[{k_cos_arr.min():.4f}, {k_cos_arr.max():.4f}]")
print(f"  V 余弦相似度:  mean={v_cos_arr.mean():.4f}  range=[{v_cos_arr.min():.4f}, {v_cos_arr.max():.4f}]")

kmat = np.array(per_token_k_rel)
vmat = np.array(per_token_v_rel)
per_pos_k = kmat.mean(axis=0)
per_pos_v = vmat.mean(axis=0)
print("\n" + "=" * 60)
print("skill 内部 token 偏差分布 (跨层平均)")
print("=" * 60)
print("前 8 个 token 的 K 相对 L2:  " + "  ".join(f"{x:.3f}" for x in per_pos_k[:8]))
print("前 8 个 token 的 V 相对 L2:  " + "  ".join(f"{x:.3f}" for x in per_pos_v[:8]))
if len(per_pos_k) > 8:
    print(f"后续 token (index >= 8) 的 K 相对 L2 均值: {per_pos_k[8:].mean():.3f}")
    print(f"后续 token (index >= 8) 的 V 相对 L2 均值: {per_pos_v[8:].mean():.3f}")

# ===== Step 7: 存 JSON + NPZ =====
OUT_DIR = RESULTS_ROOT / "position_context_analysis" / "scenario_e"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out_path = OUT_DIR / f"{SCENARIO}_summary.json"
with open(out_path, "w") as f:
    json.dump({
        "scenario": SCENARIO,
        "model": MODEL_PATH,
        "skill_name": SKILL_NAME,
        "ref_file":    REF_FILE,       # skill 锚点来源 (未 prefill)
        "target_file": TARGET_FILE,    # 唯一 prefill 对象
        "total_tokens": N,
        "first_skill_range":  [skill_start_1, skill_end_1],
        "last_skill_range":   [skill_start_2, skill_end_2],
        "skill_compare_len": L_skill,
        "per_layer": per_layer,
        "per_token_k_rel_l2_mean_over_layers": per_pos_k.tolist(),
        "per_token_v_rel_l2_mean_over_layers": per_pos_v.tolist(),
        "summary": {
            "k_rel_l2_mean": float(k_rel_arr.mean()),
            "v_rel_l2_mean": float(v_rel_arr.mean()),
            "k_cos_mean":    float(k_cos_arr.mean()),
            "v_cos_mean":    float(v_cos_arr.mean()),
        },
    }, f, indent=2)
print(f"\n结果 JSON 写入: {out_path}")

npz_path = OUT_DIR / f"{SCENARIO}_matrices.npz"
np.savez(
    npz_path,
    k_rel_l2=kmat,
    v_rel_l2=vmat,
    layers=np.array(layers),
)
print(f"矩阵 NPZ 写入: {npz_path}")

# ===== Step 8: 画图 =====
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG_DIR = OUT_DIR / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

layers_arr = np.array([r["layer"] for r in per_layer])

# 图 1: per-layer 相对 L2 + 余弦
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(layers_arr, k_rel_arr, marker="o", ms=4, label="K", color="tab:blue")
ax1.plot(layers_arr, v_rel_arr, marker="s", ms=4, label="V", color="tab:red")
ax1.set_xlabel("Layer index")
ax1.set_ylabel("Relative L2 deviation (mean over skill tokens)")
ax1.set_title("Per-layer K/V relative L2")
ax1.grid(alpha=0.3); ax1.legend()

ax2.plot(layers_arr, k_cos_arr, marker="o", ms=4, label="K", color="tab:blue")
ax2.plot(layers_arr, v_cos_arr, marker="s", ms=4, label="V", color="tab:red")
ax2.set_xlabel("Layer index")
ax2.set_ylabel("Cosine similarity (mean over skill tokens)")
ax2.set_title("Per-layer K/V cosine similarity")
ax2.set_ylim(-0.05, 1.05)
ax2.grid(alpha=0.3); ax2.legend()

fig.suptitle(f"Scenario E: {SKILL_NAME} KV re-use within a single request  "
             f"({TARGET_FILE}: first vs last skill occurrence)")
fig.tight_layout()
fig1_path = FIG_DIR / f"{SCENARIO}_per_layer.png"
fig.savefig(fig1_path, dpi=150); plt.close(fig)
print(f"图 1 写入: {fig1_path}")

# 图 2: skill 内 token 位置的跨层平均偏差
fig, ax = plt.subplots(figsize=(10, 4))
pos_arr = np.arange(len(per_pos_k))
ax.plot(pos_arr, per_pos_k, label="K", color="tab:blue", linewidth=1.2)
ax.plot(pos_arr, per_pos_v, label="V", color="tab:red", linewidth=1.2)
ax.axvspan(0, min(4, len(pos_arr)), alpha=0.1, color="orange")
ax.set_xlabel("Token index within skill")
ax.set_ylabel("Relative L2 (mean over layers)")
ax.set_title(f"Per-token deviation within {SKILL_NAME} skill "
             f"(avg over {len(per_layer)} layers)")
ax.grid(alpha=0.3); ax.legend()
fig.tight_layout()
fig2_path = FIG_DIR / f"{SCENARIO}_per_token.png"
fig.savefig(fig2_path, dpi=150); plt.close(fig)
print(f"图 2 写入: {fig2_path}")

# 图 3: (layer × skill-token) 热力图
fig, (axk, axv) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
vmax = float(max(kmat.max(), vmat.max()))
axk.imshow(kmat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, origin="lower")
axk.set_xlabel("Token index within skill"); axk.set_ylabel("Layer index")
axk.set_title("K relative L2 deviation")
imv = axv.imshow(vmat, aspect="auto", cmap="viridis", vmin=0, vmax=vmax, origin="lower")
axv.set_xlabel("Token index within skill")
axv.set_title("V relative L2 deviation")
fig.colorbar(imv, ax=[axk, axv], shrink=0.8, label="relative L2")
fig.suptitle(f"Scenario E heatmap: {SKILL_NAME} ({TARGET_FILE} first vs last)")
fig3_path = FIG_DIR / f"{SCENARIO}_heatmap.png"
fig.savefig(fig3_path, dpi=150, bbox_inches="tight"); plt.close(fig)
print(f"图 3 写入: {fig3_path}")
