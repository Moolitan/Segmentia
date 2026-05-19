"""
Scenario E 的 HF 轨分析:在 CPU 上用 eager attention 跑 026.txt(和 vLLM 轨同一
条 trace),dump 指定 3 层的 attention 切片,做"同一 skill 在两个位置下被看向的
方式"的 attention 视角分析。

和 run_scenario_E.py(vLLM 轨)的对应关系:
- 同一模型 Qwen3-14B
- 同一条 trace 026.txt(17426 tokens,internal-comms skill 出现 2 次)
- 同一 skill 定位逻辑(从 003.txt 抽包装版 + offset_mapping 锚点)
- 互补角度:vLLM 轨看"KV 数值差多少",本脚本看"两个位置的 skill 被 attention
  怎么看"(从同一条 forward 得到,无跨请求差异)

关键工程决定
============
1. CPU 而非 GPU:N=17426 时一层 attn_weights (40 heads × N × N × bf16) ≈ 24GB,
   A6000 48GB 塞不下。125GB host RAM 绰绰有余。
2. eager 而非 SDPA:SDPA/FlashAttention 拿不到 attention 矩阵。
3. 行分块:默认 eager 一次性分配 (40, N, N) bf16,加上 softmax fp32 临时量峰值
   ~72GB/层。本脚本 monkey-patch 成分块版(chunk=1024 行),峰值降到 ~6GB/层。
   权重 28GB + 单层临时 ~6GB + 其他 ~5GB ≈ 40GB,稳。
4. dump 的是 post-softmax(attention 概率),不是 pre-softmax。理由:
   - 可视化本来就画 post-softmax
   - pre-softmax 有 causal mask 的 -inf,数值跨度大、图里不友好
   - 需要 pre-softmax 时脚本里也顺手计算了 row-wise 最大 logit 作为辅助统计
5. dump 范围:只存 3 层 (5/15/25) × 关心的 query 行 (首 skill + 末 skill + 末尾
   100 tokens) × 全部 17426 keys,数据量 <5GB。其他层和其他行不存。

运行
====
    cd /home/wsh/openhands_code_research
    conda activate opencode
    python scripts/04_kv_cache_research/position_context_analysis/run_scenario_E_hf.py

预期用时:几分钟到二十分钟(Sapphire Rapids 48 核 + AMX,BF16)。
"""

from __future__ import annotations

import os
import re
import sys
import json
import time
import shutil
from pathlib import Path
from typing import Optional, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]


# ============================================================
# Step 0: 路径 & 常量
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent
ROOT = BASE_DIR.parent.parent
RESULTS_ROOT = BASE_DIR / "results"
TRACE_DIR = BASE_DIR / "prompts" / "multiturn_sequence_traces"
OUT_DIR = RESULTS_ROOT / "position_context_analysis" / "scenario_e_hf_attention"
FIG_DIR = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
SCENARIO = "E_internal-comms_026_hf_attention"

SKILL_NAME = "internal-comms"
REF_FILE    = "003.txt"   # 仅用作"skill 被终端换行包装过的版本"的锚点
TARGET_FILE = "026.txt"   # 真正喂给模型的 prompt,skill 在里面出现 2 次

# 想观察的 3 层(和方案 4.3 节对齐)
TARGET_LAYERS: set[int] = {5, 15, 25}

# Q 行分块大小:决定单层 attn_weights 临时量。chunk=1024 时,
# scores 形状 (1, 40, 1024, 17426) bf16 ≈ 1.4GB,softmax fp32 临时 ≈ 2.8GB。
CHUNK = 1024

# 末尾"生成位"上下文:取末尾这么多 token 作为"user query / 即将生成"区域,
# 分析它们的 attention 分布
TAIL_ROWS_LEN = 200

print("=" * 70)
print(f"Scenario:    {SCENARIO}")
print(f"Model:       {MODEL_PATH}")
print(f"Trace:       {TARGET_FILE}")
print(f"Target layers for attention dump: {sorted(TARGET_LAYERS)}")
print(f"Q-chunk size: {CHUNK}")
print("=" * 70)


# ============================================================
# Step 1: tokenize prompt,定位 skill 首末两次出现的 token 区间
# (逻辑和 run_scenario_E.py 完全一致,保持可对齐)
# ============================================================
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
assert tokenizer.is_fast, "需要 fast tokenizer 拿 offset_mapping"

ref_text    = (TRACE_DIR / REF_FILE).read_text(encoding="utf-8")
target_text = (TRACE_DIR / TARGET_FILE).read_text(encoding="utf-8")

_SKILL_HEAD_RE = re.compile(r"---\nname:\s*" + re.escape(SKILL_NAME))
_SKILL_TAIL = "internal comms\n"


def _extract_skill_wrapped(text: str) -> str:
    """从 003.txt 里抽出 skill 被包装换行后的那个版本"""
    m = _SKILL_HEAD_RE.search(text)
    assert m, f"未找到 skill 起始标记"
    start = m.start()
    end = text.find(_SKILL_TAIL, start)
    assert end != -1
    return text[start : end + len(_SKILL_TAIL)]


SKILL_TEXT = _extract_skill_wrapped(ref_text)


def tokenize_and_locate_skill(prompt_text: str, skill_text: str, occurrence: str) -> tuple[list[int], int, int]:
    """整体 tokenize,然后用 offset_mapping 定位 skill 子串对应的 token 区间"""
    if occurrence == "first":
        char_start = prompt_text.find(skill_text)
    elif occurrence == "last":
        char_start = prompt_text.rfind(skill_text)
    else:
        raise ValueError(occurrence)
    assert char_start != -1
    char_end = char_start + len(skill_text)

    enc = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    ids = enc["input_ids"]
    offsets = enc["offset_mapping"]

    tok_start, tok_end = None, None
    for i, (s, e) in enumerate(offsets):
        if s >= char_start and tok_start is None:
            tok_start = i
        if e <= char_end:
            tok_end = i + 1
    assert tok_start is not None and tok_end is not None and tok_end > tok_start
    return ids, tok_start, tok_end


target_ids, skill_start_1, skill_end_1 = tokenize_and_locate_skill(target_text, SKILL_TEXT, "first")
_,          skill_start_2, skill_end_2 = tokenize_and_locate_skill(target_text, SKILL_TEXT, "last")

N = len(target_ids)
L1 = skill_end_1 - skill_start_1
L2 = skill_end_2 - skill_start_2
L_skill = min(L1, L2)

print(f"\n{TARGET_FILE}:")
print(f"  总 token 数: {N}")
print(f"  首次 skill 区间: [{skill_start_1}, {skill_end_1})  长度 {L1}")
print(f"  末次 skill 区间: [{skill_start_2}, {skill_end_2})  长度 {L2}")
if L1 != L2:
    print(f"  ⚠ skill token 长度不一致 ({L1} vs {L2}),对比时截到 min={L_skill}")


# ============================================================
# Step 2: 规划"要 dump 的 query 行"
# ============================================================
# - skill1_rows: 首次 skill 区间内每个 token
# - skill2_rows: 末次 skill 区间内每个 token(截到 L_skill 保证两组等长可对比)
# - tail_rows:  prompt 末尾 TAIL_ROWS_LEN 个 token(代表"即将生成"的位置)
# 拼成一个 sorted 的 row_indices 数组,patched attention 会按这些行 dump。

skill1_rows = np.arange(skill_start_1, skill_start_1 + L_skill, dtype=np.int64)
skill2_rows = np.arange(skill_start_2, skill_start_2 + L_skill, dtype=np.int64)
tail_rows   = np.arange(max(0, N - TAIL_ROWS_LEN), N, dtype=np.int64)

# 合并去重(tail 有可能和 skill2 末端重叠)
rows_to_dump_np = np.unique(np.concatenate([skill1_rows, skill2_rows, tail_rows]))
# row -> 它在 dumped 数组里的 index,便于 patched 函数 O(1) 写入
row_to_local_idx = {int(r): i for i, r in enumerate(rows_to_dump_np)}
rows_to_dump = torch.from_numpy(rows_to_dump_np)

print(f"\n要 dump 的 query 行数: {len(rows_to_dump_np)}")
print(f"  skill1 贡献: {len(skill1_rows)} 行")
print(f"  skill2 贡献: {len(skill2_rows)} 行")
print(f"  tail   贡献: {len(tail_rows)} 行")


# ============================================================
# Step 3: monkey-patch Qwen3 的 eager_attention_forward
# - 行分块版本,避免一次性分配 (H, N, N) 的大张量
# - 对 TARGET_LAYERS dump 指定行的 post-softmax attention
# - 其他层也走分块路径(只为省内存),不 dump
# ============================================================
from transformers.models.qwen3 import modeling_qwen3 as _qwen3

repeat_kv = _qwen3.repeat_kv  # GQA 的 KV 头复制

# 全局 dump 容器:layer_idx -> (num_heads, num_dump_rows, N) fp16
# 用 fp16 存,精度够画 heatmap 和算 region 汇总,比 fp32 省一半空间
dumped_attn: dict[int, np.ndarray] = {}
# 额外统计(每 target 层每 dump 行):max logit(pre-softmax 最大 logit,只在
# causal 有效范围内;EPIC 有时看这个指标确认 attention 是否集中)
dumped_maxlogit: dict[int, np.ndarray] = {}


def patched_eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Any,
):
    """
    行分块版 eager attention。和原版数学上等价,但不会一次性物化整张
    (B, H, N, N) 的 attention 矩阵 —— 因此能在 125GB RAM 上安全跑 17k 长度。

    对 layer_idx in TARGET_LAYERS 的层,把 rows_to_dump 对应 query 行的
    post-softmax attention 写到 dumped_attn 里。
    """
    layer_idx: Optional[int] = getattr(module, "layer_idx", None)
    dump_this_layer = layer_idx in TARGET_LAYERS

    # GQA:把 K/V 的 head 数复制到 Q 的 head 数
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    B, H, N_q, _ = query.shape
    _, _, N_k, D_v = value_states.shape  # prefill 时 N_q == N_k

    # 为 dump 层准备容器(只在本层首次进入时分配)
    if dump_this_layer and layer_idx not in dumped_attn:
        dumped_attn[layer_idx] = np.zeros((H, len(rows_to_dump_np), N_k), dtype=np.float16)
        dumped_maxlogit[layer_idx] = np.full((H, len(rows_to_dump_np)), -np.inf, dtype=np.float32)

    # 输出占位:和原函数返回的 attn_output 形状一致(后面 .transpose(1,2))
    attn_output = torch.empty((B, H, N_q, D_v), dtype=query.dtype)

    # 行分块
    for start in range(0, N_q, CHUNK):
        end = min(start + CHUNK, N_q)
        q_chunk = query[:, :, start:end, :]                                 # (B, H, m, D)
        scores = torch.matmul(q_chunk, key_states.transpose(2, 3)) * scaling  # (B, H, m, N_k) bf16

        if attention_mask is not None:
            mask_chunk = attention_mask[:, :, start:end, :N_k]
            scores = scores + mask_chunk

        # 如果这层要 dump,先记 pre-softmax 的 max logit(只在 dump 的行上)
        if dump_this_layer:
            # 这个 chunk 覆盖 rows_to_dump 里哪些
            in_chunk_mask = (rows_to_dump_np >= start) & (rows_to_dump_np < end)
            if in_chunk_mask.any():
                local_idx_in_chunk = rows_to_dump_np[in_chunk_mask] - start
                global_dump_idx    = np.where(in_chunk_mask)[0]
                # scores[0, :, local_idx_in_chunk, :] -> (H, #rows, N_k)
                sel = scores[0, :, torch.from_numpy(local_idx_in_chunk), :]
                # pre-softmax max(忽略 -inf)
                ml = sel.float().max(dim=-1).values.cpu().numpy()  # (H, #rows)
                dumped_maxlogit[layer_idx][:, global_dump_idx] = ml

        # softmax(和原函数一致:用 fp32 算后再 cast 回 bf16)
        attn_chunk = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)

        # dump post-softmax 切片
        if dump_this_layer:
            if in_chunk_mask.any():
                sel = attn_chunk[0, :, torch.from_numpy(local_idx_in_chunk), :]
                dumped_attn[layer_idx][:, global_dump_idx, :] = sel.float().cpu().numpy().astype(np.float16)

        # dropout(推理时 training=False 实际等同 no-op,保留以对齐原函数)
        attn_chunk = F.dropout(attn_chunk, p=dropout, training=module.training)

        # 本 chunk 的 attention 输出
        attn_output[:, :, start:end, :] = torch.matmul(attn_chunk, value_states)

        del scores, attn_chunk

    attn_output = attn_output.transpose(1, 2).contiguous()
    # 返回 attn_weights=None:外层只在 output_attentions=True 时会拿,这里不需要,
    # 返回 None 能省掉一份 24GB 的 bf16 张量
    return attn_output, None


# 替换模块级引用。Qwen3Attention.forward 里每次都是从本模块的 scope 拿
# eager_attention_forward 的引用(当 _attn_implementation == "eager" 时),
# 所以 monkey-patch 生效。
_qwen3.eager_attention_forward = patched_eager_attention_forward
print("\n✓ monkey-patch 已装上 Qwen3 eager_attention_forward")


# ============================================================
# Step 4: 加载 Qwen3-14B 到 CPU(bf16,eager)
# ============================================================
# 线程数:Sapphire Rapids 单机 48 物理核 + AMX,让 PyTorch 用所有物理核
# (不含超线程,通常比开到 96 线程更快)
PHYSICAL_CORES = int(os.environ.get("NUM_CPU_THREADS", "48"))
torch.set_num_threads(PHYSICAL_CORES)
print(f"torch.set_num_threads({PHYSICAL_CORES})")

t0 = time.time()
print(f"\n加载模型到 CPU(bf16, eager attention) ...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",       # 必须 eager 才进我们的 monkey-patch 路径
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✓ 模型加载完成  耗时 {time.time()-t0:.1f}s")
# Qwen3-14B 有 40 层,所有层的 layer_idx 都会经过我们的 patched 函数,
# 但只有 TARGET_LAYERS 对应的层会填 dumped_attn


# ============================================================
# Step 5: 前向(不生成,不需要 cache)
# ============================================================
input_ids = torch.tensor([target_ids], dtype=torch.long)
print(f"\nprompt shape: {input_ids.shape}   (N={N})")
print(f"开始前向 ...  (CPU eager 预计几分钟)")
t0 = time.time()
with torch.inference_mode():
    _ = model(
        input_ids=input_ids,
        attention_mask=None,            # HF 会自动建 causal mask
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
    )
fwd_sec = time.time() - t0
print(f"✓ 前向完成  耗时 {fwd_sec:.1f}s  ({fwd_sec/60:.2f} min)")

# 清理模型,腾内存做后处理
del model
import gc
gc.collect()

# 完整性检查
missing = TARGET_LAYERS - set(dumped_attn.keys())
assert not missing, f"target layer 没被 dump 到: {missing}"
for lid in TARGET_LAYERS:
    assert dumped_attn[lid].shape == (dumped_attn[lid].shape[0], len(rows_to_dump_np), N), \
        f"layer {lid} dump 形状异常: {dumped_attn[lid].shape}"
print(f"✓ 所有 target 层都 dump 到了:  layers={sorted(dumped_attn.keys())}")


# ============================================================
# Step 6: 划分 key 轴的区域 + 计算每行的 attention mass
# ============================================================
# Key 区域:
#   pre      : [0, skill_start_1)
#   skill1   : [skill_start_1, skill_end_1)
#   between  : [skill_end_1, skill_start_2)
#   skill2   : [skill_start_2, skill_end_2)
#   tail     : [skill_end_2, N)   ← 末尾,含最后要生成的那一段
REGIONS = [
    ("pre",     0,                 skill_start_1),
    ("skill1",  skill_start_1,     skill_end_1),
    ("between", skill_end_1,       skill_start_2),
    ("skill2",  skill_start_2,     skill_end_2),
    ("tail",    skill_end_2,       N),
]
print(f"\nKey 区域划分:")
for name, s, e in REGIONS:
    print(f"  {name:>8s}: [{s:>6d}, {e:>6d})   长度 {e-s}")


def compute_region_mass(
    attn: np.ndarray,          # (H, R, N) fp16,R=要 dump 的行数
    rows: np.ndarray,          # 本行组对应的 row indices(原始 token 位置)
) -> dict[str, np.ndarray]:
    """
    对 attn 的每一行,把 N 个 key 按 region 分段求和,返回
    {region_name: (H, R)} 的 dict。
    """
    # attn 形状 (H, R, N);rows 是 R 长度的 1D
    out = {}
    for name, s, e in REGIONS:
        out[name] = attn[:, :, s:e].astype(np.float32).sum(axis=-1)  # (H, R)
    return out


# 每个 target 层,把三类 query 行(skill1 / skill2 / tail)的区域 mass 统计出来
results: dict[str, Any] = {
    "scenario": SCENARIO,
    "model": MODEL_PATH,
    "trace": TARGET_FILE,
    "n_tokens": N,
    "skill_first_range": [int(skill_start_1), int(skill_end_1)],
    "skill_last_range":  [int(skill_start_2), int(skill_end_2)],
    "L_skill": int(L_skill),
    "target_layers": sorted(TARGET_LAYERS),
    "tail_rows_len": int(len(tail_rows)),
    "forward_sec": float(fwd_sec),
    "regions": [[n, int(s), int(e)] for n, s, e in REGIONS],
    "per_layer": {},
}


def global_idx_for(rows: np.ndarray) -> np.ndarray:
    """把一组原始 row indices 映射到 rows_to_dump_np 里的位置"""
    return np.array([row_to_local_idx[int(r)] for r in rows], dtype=np.int64)


idx_skill1 = global_idx_for(skill1_rows)
idx_skill2 = global_idx_for(skill2_rows)
idx_tail   = global_idx_for(tail_rows)


for lid in sorted(TARGET_LAYERS):
    attn = dumped_attn[lid]        # (H, R_all, N)
    mass = compute_region_mass(attn, rows_to_dump_np)  # {region: (H, R_all)}

    # 三类行的平均 region mass(跨 head 均值、跨行均值)
    layer_stats = {"layer": int(lid)}
    for group_name, group_idx in [("skill1_rows", idx_skill1),
                                   ("skill2_rows", idx_skill2),
                                   ("tail_rows",   idx_tail)]:
        group_mass = {rname: float(mass[rname][:, group_idx].mean())
                      for rname, _, _ in REGIONS}
        layer_stats[group_name] = group_mass

    # 额外:tail 行对 skill1 vs skill2 的 mass 比较(最关心的一个量)
    tail_s1 = mass["skill1"][:, idx_tail].mean()
    tail_s2 = mass["skill2"][:, idx_tail].mean()
    layer_stats["tail_skill1_vs_skill2"] = {
        "skill1_mass": float(tail_s1),
        "skill2_mass": float(tail_s2),
        "ratio_s1_over_s2": float(tail_s1 / max(tail_s2, 1e-9)),
    }

    # attention sink 指标:所有 dump 行对前 4 个 token 的 mass(跨 head 均值)
    sink_mass_all = attn[:, :, :4].astype(np.float32).sum(axis=-1).mean(axis=0)  # (R_all,)
    layer_stats["sink_mass_per_group"] = {
        "skill1_rows_mean": float(sink_mass_all[idx_skill1].mean()),
        "skill2_rows_mean": float(sink_mass_all[idx_skill2].mean()),
        "tail_rows_mean":   float(sink_mass_all[idx_tail].mean()),
    }

    results["per_layer"][str(lid)] = layer_stats

    print(f"\nLayer {lid}:")
    print(f"  tail 行对 skill1 的 mass = {tail_s1:.4f}")
    print(f"  tail 行对 skill2 的 mass = {tail_s2:.4f}")
    print(f"  比例 skill1/skill2       = {tail_s1/max(tail_s2,1e-9):.3f}")
    print(f"  attention sink (前4 tok): tail={sink_mass_all[idx_tail].mean():.3f}")


# ============================================================
# Step 7: 存 JSON + NPZ
# ============================================================
json_path = OUT_DIR / f"{SCENARIO}_summary.json"
with open(json_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nJSON 写入: {json_path}")

npz_path = OUT_DIR / f"{SCENARIO}_attn.npz"
save_kwargs = {"rows_to_dump": rows_to_dump_np}
for lid in sorted(TARGET_LAYERS):
    # head-averaged 版本,用于画图和常规分析
    save_kwargs[f"attn_layer{lid}_mean_head"] = dumped_attn[lid].astype(np.float32).mean(axis=0)
    # per-head 版本(fp16),供后续深挖
    save_kwargs[f"attn_layer{lid}_per_head_fp16"] = dumped_attn[lid]
    save_kwargs[f"maxlogit_layer{lid}"] = dumped_maxlogit[lid]
np.savez_compressed(npz_path, **save_kwargs)
print(f"NPZ  写入: {npz_path}")


# ============================================================
# Step 8: 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REGION_COLORS = {
    "pre":     "#cccccc",
    "skill1":  "tab:orange",
    "between": "#dddddd",
    "skill2":  "tab:red",
    "tail":    "#aaaaaa",
}


# ---------- 图 1: 每层 tail 行对各区域的 attention mass ----------
fig, ax = plt.subplots(figsize=(10, 5))
bar_w = 0.2
x_layers = np.arange(len(sorted(TARGET_LAYERS)))
region_names = ["pre", "skill1", "between", "skill2", "tail"]

for ri, rname in enumerate(region_names):
    vals = [results["per_layer"][str(lid)]["tail_rows"][rname]
            for lid in sorted(TARGET_LAYERS)]
    ax.bar(x_layers + (ri - 2) * bar_w, vals, bar_w, label=rname,
           color=REGION_COLORS[rname])

ax.set_xticks(x_layers)
ax.set_xticklabels([f"L{lid}" for lid in sorted(TARGET_LAYERS)])
ax.set_ylabel("Attention mass (per-row sum, mean over rows & heads)")
ax.set_title(f"Tail-rows attention allocation across regions  ({TARGET_FILE})")
ax.legend(loc="upper right")
ax.grid(alpha=0.3, axis="y")
fig.tight_layout()
fig1_path = FIG_DIR / f"{SCENARIO}_tail_region_mass.png"
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print(f"\n图 1 写入: {fig1_path}")


# ---------- 图 2: EPIC 风格 heatmap,每 target 层一张 ----------
# 行: skill1_rows + skill2_rows + tail_rows 顺序拼起来(跨 head 平均)
# 列: 全部 N 个 key(标注区域边界)
ordered_rows = np.concatenate([idx_skill1, idx_skill2, idx_tail])
row_separators = [len(idx_skill1), len(idx_skill1) + len(idx_skill2)]
row_labels = [("skill1 rows", 0, len(idx_skill1)),
              ("skill2 rows", row_separators[0], row_separators[1]),
              ("tail rows",   row_separators[1], len(ordered_rows))]

for lid in sorted(TARGET_LAYERS):
    attn_mean = dumped_attn[lid].astype(np.float32).mean(axis=0)  # (R_all, N)
    ordered_attn = attn_mean[ordered_rows]  # (R_ordered, N)

    fig, ax = plt.subplots(figsize=(14, 5))
    # 用 log 画更能看到低值结构(attention 值经常 <0.01)
    with np.errstate(divide="ignore"):
        im = ax.imshow(np.log10(ordered_attn + 1e-6), aspect="auto",
                       cmap="viridis", origin="lower")
    ax.set_xlabel("Key token position")
    ax.set_ylabel("Query row (grouped)")
    ax.set_title(f"Layer {lid} attention (log10, head-mean)  —  {TARGET_FILE}")

    # 列轴:用竖线标 region 边界
    for _, s, e in REGIONS:
        ax.axvline(s, color="white", lw=0.4, alpha=0.5)
    # region 名字放顶部
    for name, s, e in REGIONS:
        ax.text((s + e) / 2, len(ordered_rows) * 1.01, name,
                ha="center", va="bottom", fontsize=8, color="k")

    # 行轴:用横线标组边界,并标注
    for sep in row_separators:
        ax.axhline(sep - 0.5, color="white", lw=1.0, alpha=0.8)
    for name, s, e in row_labels:
        ax.text(-N * 0.01, (s + e) / 2, name, ha="right", va="center", fontsize=8)

    fig.colorbar(im, ax=ax, label="log10(attention)")
    fig.tight_layout()
    fig_path = FIG_DIR / f"{SCENARIO}_heatmap_layer{lid}.png"
    fig.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"图 2.{lid} 写入: {fig_path}")


# ---------- 图 3: 同一 skill 在两个位置的"行视角" attention 对比 ----------
# 思路:skill1 的第 i 行 和 skill2 的第 i 行 对应同一段文本的同一个 token,
# 它们的 attention 分布(对全部 key)差多少?这是和 vLLM 轨结论最直接呼应的
# attention 视角证据。
# 我们把 (skill2 row i 的 attention) 和 (skill1 row i 的 attention) 对应区域
# mass 做差,看 skill2 是不是把 attention 从 skill1 那边"挪"过来了自己身上。

for lid in sorted(TARGET_LAYERS):
    attn_mean = dumped_attn[lid].astype(np.float32).mean(axis=0)  # (R_all, N)

    # skill1 第 i 行  attention 在 pre 区的 mass
    s1_pre  = np.array([attn_mean[gi, :skill_start_1].sum() for gi in idx_skill1])
    # skill2 第 i 行  attention 在 pre 区的 mass
    s2_pre  = np.array([attn_mean[gi, :skill_start_1].sum() for gi in idx_skill2])
    # skill2 第 i 行  attention 在 skill1 区的 mass(skill1 行本身落在 skill1
    # 区间内,只能对"自己之前"的部分求 mass,所以不对等,这里只比较 pre)
    s2_s1   = np.array([attn_mean[gi, skill_start_1:skill_end_1].sum() for gi in idx_skill2])
    # skill2 第 i 行  attention 在自己(skill2)区的 mass(含自己之前的 skill2 tokens)
    s2_s2   = np.array([attn_mean[gi, skill_start_2:skill_end_2].sum() for gi in idx_skill2])

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(L_skill)
    ax.plot(x, s1_pre, label="skill1 row i → pre region", color="tab:blue", lw=1)
    ax.plot(x, s2_pre, label="skill2 row i → pre region", color="tab:cyan", lw=1)
    ax.plot(x, s2_s1,  label="skill2 row i → skill1 region", color="tab:orange", lw=1)
    ax.plot(x, s2_s2,  label="skill2 row i → skill2 region (incl. before self)",
            color="tab:red", lw=1)
    ax.set_xlabel("Token index within skill (i)")
    ax.set_ylabel("Attention mass")
    ax.set_title(f"Layer {lid}: same skill token, two positions  —  how attention splits")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig_path = FIG_DIR / f"{SCENARIO}_skill_row_compare_layer{lid}.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"图 3.{lid} 写入: {fig_path}")


print("\n" + "=" * 70)
print("Done.")
print(f"  JSON: {json_path}")
print(f"  NPZ:  {npz_path}")
print(f"  图:   {FIG_DIR}")
print("=" * 70)
