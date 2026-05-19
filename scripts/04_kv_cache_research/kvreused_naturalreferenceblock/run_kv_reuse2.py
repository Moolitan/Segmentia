"""
run_kv_reuse2.py
================
KV 复用实验:第二次重复参考文本之后不再追加第二个用户 QUERY,只保留 assistant 续写前缀,观察后续 decode 行为。

说明:
  变量名仍沿用 SKILL_TEXT / skill_v1 / skill_v2,方便和前几个脚本对齐。
  但这里的 SKILL_TEXT 语义上不再是 tool-returned skill,而是一段自然语言参考卡片。

序列骨架:
  [USER_CONTEXT] + [SKILL_v1] + [MIDDLE_HISTORY] + [SKILL_v2] + [ASSISTANT_PREFIX]

场景:
  A  (参考)      :正常 prefill skill_v2 + assistant 续写前缀,拿续写前缀末尾 logit(ground truth)
  B1 (朴素复用)  :复用 skill_v1 全部 L 个 KV(位置错误),再正常 prefill assistant 续写前缀
  B2 (RoPE 校正) :V 照搬,K 逆旋转再重旋转到 skill_v2 位置(全 L 个),再正常 prefill assistant 续写前缀

指标(teacher forcing,T_STEPS 步):
  KL_first, KL_mean, KL_max, TVD_first, TVD_mean
  argmax_match_rate, top5_overlap_mean
  逐步 KL/TVD 曲线(PNG)

运行:
  cd /home/wsh/openhands_code_research
  conda activate opencode
  python scripts/04_kv_cache_research/kvreused_naturalreferenceblock/run_kv_reuse2.py

工程注意(transformers 4.57+):
  DynamicCache 内部结构已改为 self.layers: list[DynamicLayer]
  不再有 self.key_cache / self.value_cache。
  每层的张量通过 layer.keys / layer.values 访问。
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]

# ============================================================
# 0. 路径 & 常量
# ============================================================
SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR   = SCRIPT_DIR.parent
RESULTS_ROOT = BASE_DIR / "results"
OUT_DIR    = RESULTS_ROOT / "kv_reuse_natural_reference_block" / "no_query_v1"
FIG_DIR    = OUT_DIR / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B"
SCENARIO   = "KVReuse_NaturalReferenceBlock_NoQuery_v1"
T_STEPS    = 128   # teacher forcing 步数(阶段 1)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE  = torch.bfloat16

print("=" * 70)
print(f"Scenario : {SCENARIO}")
print(f"Model    : {MODEL_PATH}")
print(f"Device   : {DEVICE}   dtype={DTYPE}")
print(f"T steps  : {T_STEPS}")
print("=" * 70)

# ============================================================
# 1. Prompt 素材
# 目标:验证"相同文本在不同位置是否可复用"这个猜想
# 设计原则:
#   1) 不模拟 agent / tool 调用过程
#   2) 两次出现的 SKILL_TEXT 完全相同
#   3) USER_CONTEXT / MIDDLE_HISTORY 之间要有自然语言层面的逻辑连续性
#   4) 第二个任务请求提前放到第二次 SKILL_TEXT 之前,不再追加第二个用户 query
#   5) skill_v2 后只保留一个 assistant 续写前缀,让模型自然开始回答
#
# 序列结构:
#   USER_CONTEXT  : 用户先贴一份"平台更新格式参考卡"
#   SKILL_TEXT    : 参考卡全文(精确出现两次)
#   MIDDLE_HISTORY: 模型先根据该参考卡完成一个数据迁移更新,再把第二个任务请求提前交代清楚
#   SKILL_TEXT    : (第二次,相同内容)
#   QUERY_TEXT    : 不是第二个用户 query,而只是 assistant 续写前缀
#   → skill_v2 结束后,立刻进入 assistant 继续写第二个 update 的位置
# ============================================================

# SKILL_TEXT: 被重复粘贴的参考卡文本
SKILL_TEXT = """\
Reference Card: Platform Update Format

1. Start with a one-line title in the form "Platform Update - <topic>".
2. Use exactly three sections: Progress, Plans, Problems.
3. Under each section, write 2 short bullet points.
4. Keep the tone operational, calm, and specific.
5. Prefer concrete facts over vague claims.
6. Mention rollout status or the next checkpoint when relevant.
7. Keep the total length between 120 and 180 words.
8. End with one sentence stating the immediate next checkpoint.
"""

# USER_CONTEXT: 用户先贴参考卡,要求后续写作遵循它
USER_CONTEXT = """\
[system]
You are a concise workplace writing assistant. When the user provides a reference block, follow it
closely. Do not mention the reference block unless the user asks.

[user]
I need help drafting internal updates for the platform engineering team. First, please read the
reference card below and keep it in mind for later requests.
"""

# MIDDLE_HISTORY: 先完成一个数据迁移更新,再把第二个任务连同“马上继续写”这一要求提前放到第二次 reference card 之前
MIDDLE_HISTORY = """
[assistant]
Understood. I will use the reference card for the next update request.

[user]
Using that format, write an update about the dataset sync migration.

Facts:
- Progress: the new sync pipeline is running in staging; checksum validation caught two schema mismatches; backfill for the last 30 days finished successfully.
- Plans: start a 10% production rollout on Monday; publish rollback steps; monitor data freshness every hour.
- Problems: one downstream dashboard still reads from the legacy table; the on-call runbook is incomplete.

[assistant]
Platform Update - Dataset Sync Migration

Progress
- The new sync pipeline is stable in staging, and checksum validation has already exposed two schema mismatches before rollout.
- Backfill for the last 30 days completed successfully, which gives us a clean comparison baseline.

Plans
- We will start a 10% production rollout on Monday and publish rollback steps before expanding further.
- The team will monitor hourly freshness checks during the first rollout window.

Problems
- One downstream dashboard still depends on the legacy table.
- The on-call runbook is not complete enough for a wider rollout.

Immediate next checkpoint: complete the rollout checklist and confirm readiness for the Monday production ramp.

[user]
That format works. I have another update request about failed-task retry support in the experiment
dashboard.

Use these facts when you continue immediately after the repeated reference card below:
- Progress: the retry endpoint is deployed; the UI retry button is live for 30% of users; failure logs now label retryable and non-retryable cases separately.
- Plans: expand to 100% next week; add retry success rate to the weekly ops review; publish a short usage note for researchers.
- Problems: older tasks still show generic failure states; retry-related alerts are noisier than expected.

Keep it concise and operational. I am pasting the same reference card again below so you can use
the exact same format. As soon as the reference card ends, continue directly with the new update
without waiting for another user turn.
"""

# ============================================================
# 2. Tokenize & 定位 skill 区间
# ============================================================
from transformers import AutoTokenizer

print(f"\n加载 tokenizer: {MODEL_PATH}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
assert tokenizer.is_fast, "需要 fast tokenizer 拿 offset_mapping"


def locate_skill_tokens(
    prompt_text: str,
    skill_text: str,
    occurrence: str = "first",
) -> tuple[list[int], int, int]:
    if occurrence == "first":
        char_start = prompt_text.find(skill_text)
    elif occurrence == "last":
        char_start = prompt_text.rfind(skill_text)
    else:
        raise ValueError(occurrence)
    assert char_start != -1, f"skill 在 prompt 中未找到 (occurrence={occurrence})"
    char_end = char_start + len(skill_text)

    enc     = tokenizer(prompt_text, add_special_tokens=False, return_offsets_mapping=True)
    ids     = enc["input_ids"]
    offsets = enc["offset_mapping"]

    tok_start = tok_end = None
    for i, (s, e) in enumerate(offsets):
        if s >= char_start and tok_start is None:
            tok_start = i
        if e <= char_end:
            tok_end = i + 1
    assert tok_start is not None and tok_end is not None and tok_end > tok_start, (
        f"skill token 区间定位失败: tok_start={tok_start}, tok_end={tok_end}"
    )
    return ids, tok_start, tok_end


# QUERY_TEXT: 不再追加第二个用户 query,这里只保留 assistant 续写前缀
QUERY_TEXT = """
[assistant]
"""

FULL_PROMPT = USER_CONTEXT + SKILL_TEXT + MIDDLE_HISTORY + SKILL_TEXT + QUERY_TEXT

full_ids, skill1_start, skill1_end = locate_skill_tokens(FULL_PROMPT, SKILL_TEXT, "first")
_,         skill2_start, skill2_end = locate_skill_tokens(FULL_PROMPT, SKILL_TEXT, "last")
N_full  = len(full_ids)
L_skill = skill1_end - skill1_start

assert skill1_end <= skill2_start, "skill_v1 和 skill_v2 区间重叠"
assert L_skill == skill2_end - skill2_start, (
    f"两次 skill token 长度不一致: {L_skill} vs {skill2_end - skill2_start}"
)
assert skill2_end < N_full, "run_kv_reuse2.py 需要在 skill_v2 后保留 assistant 续写前缀"

context_end  = skill1_start
middle_start = skill1_end
middle_end   = skill2_start

print(f"\n完整序列 tokenization:")
print(f"  总 token 数       : {N_full}")
print(f"  user_context      : [0, {context_end})")
print(f"  skill_v1          : [{skill1_start}, {skill1_end})  长度 {L_skill}")
print(f"  middle_history    : [{middle_start}, {middle_end})  长度 {middle_end - middle_start}")
print(f"  skill_v2          : [{skill2_start}, {skill2_end})  长度 {L_skill}")
print(f"  query             : [{skill2_end}, {N_full})  长度 {N_full - skill2_end}")
print(f"  skill 位置偏移量  : {skill2_start - skill1_start}")

# phase1: context + skill_v1 + middle (不含 skill_v2)
phase1_ids = full_ids[:middle_end]
# skill2_ids: 整个 skill_v2
skill2_ids = full_ids[middle_end:skill2_end]
# query_ids: skill_v2 后的 assistant 续写前缀
query_ids = full_ids[skill2_end:]

print(f"\n  phase1_ids 长度   : {len(phase1_ids)}")
print(f"  skill2_ids 长度   : {len(skill2_ids)}")
print(f"  query_ids  长度   : {len(query_ids)}")
assert query_ids, "run_kv_reuse2.py 需要 assistant 续写前缀"

# ============================================================
# 3. 加载模型(GPU, bf16, eager)
# ============================================================
from transformers import AutoModelForCausalLM

print(f"\n加载模型到 {DEVICE}  (bf16, eager attention) ...")
t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    attn_implementation="eager",
    dtype=DTYPE,
    device_map=str(DEVICE),
    low_cpu_mem_usage=True,
)
model.eval()
print(f"✓ 模型加载完成  耗时 {time.time()-t0:.1f}s")

rotary_emb = model.model.rotary_emb
head_dim   = model.config.hidden_size // model.config.num_attention_heads

# ============================================================
# 4. DynamicCache 工具函数(适配 transformers 4.57+ 新 API)
# ============================================================
from transformers.cache_utils import DynamicCache, DynamicLayer


def _make_layer(k: torch.Tensor, v: torch.Tensor) -> DynamicLayer:
    """创建一个已初始化的 DynamicLayer,内容为 k/v。"""
    layer = DynamicLayer()
    layer.dtype = k.dtype
    layer.device = k.device
    layer.keys   = k
    layer.values = v
    layer.is_initialized = True
    return layer


def slice_cache(cache: DynamicCache, tok_start: int, tok_end: int) -> DynamicCache:
    """从 cache 中切出 [tok_start, tok_end) 的 KV 片段。"""
    new_cache = DynamicCache()
    for layer in cache.layers:
        k = layer.keys[:, :, tok_start:tok_end, :].clone()
        v = layer.values[:, :, tok_start:tok_end, :].clone()
        new_cache.layers.append(_make_layer(k, v))
    return new_cache


def concat_cache(cache1: DynamicCache, cache2: DynamicCache) -> DynamicCache:
    """沿 seq 维拼接两个 cache(cache1 在前)。"""
    assert len(cache1.layers) == len(cache2.layers), (
        f"层数不一致: {len(cache1.layers)} vs {len(cache2.layers)}"
    )
    new_cache = DynamicCache()
    for l1, l2 in zip(cache1.layers, cache2.layers):
        k = torch.cat([l1.keys, l2.keys], dim=-2).contiguous()
        v = torch.cat([l1.values, l2.values], dim=-2).contiguous()
        new_cache.layers.append(_make_layer(k, v))
    return new_cache


def clone_cache(cache: DynamicCache) -> DynamicCache:
    """深拷贝 cache(teacher forcing 时各场景独立进化)。"""
    new_cache = DynamicCache()
    for layer in cache.layers:
        new_cache.layers.append(_make_layer(layer.keys.clone(), layer.values.clone()))
    return new_cache


# ============================================================
# 5. RoPE 重旋转(B2)
# ============================================================

def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    把张量最后一个维度(x.shape[-1])分成前后两半,然后做一次“二维旋转式重排”
    本质上把每两个部分看成一种“复数旋转”里的变换
    这个函数非常常见于 RoPE(Rotary Positional Embedding,旋转位置编码) 的实现里
    前一半 x1(x.shape[-1] // 2 就是最后一维的一半)
    后一半 x2(x.shape[-1] // 2 就是最后一维的一半)
    输出变成 [-x2, x1]
    ... 表示前面的维度都保持不动,只在最后一维上切片
    """

    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    这就是 RoPE 在 x 上的核心应用步骤,把 x 旋转 cos/sin 定义的角度。
    原向量部分:x = (a,b)
    旋转辅助部分:rotate_half(x)
    变成(acosθ - bsinθ,asinθ + bcosθ) = x * cos + rotate_half(x) * sin

    x  : [batch, heads, seq, head_dim]
    cos/sin: [1, seq, head_dim]
    """
    cos = cos.unsqueeze(1)   # [1, 1, seq, head_dim]
    sin = sin.unsqueeze(1)   # 在第 1 维插入一个长度为 1 的新维度,变成 [1, 1, seq, head_dim],以便广播到 x 的形状
    return x * cos + _rotate_half(x) * sin


def get_cos_sin(pos_start: int, length: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    取出某一段位置区间对应的 RoPE 的 cos / sin 系数,供后面 _apply_rope() 去旋转 q 或 k。
    比如: 把第 10、11、12、13 这 4 个位置各自需要用的 RoPE 旋转参数拿出来(这串 token 在上下文里占据了哪些位置)。
    模型里通常是一次处理一串 token,而不是只处理一个,所以要“按一段”来取,而不是一个一个
    返回 [pos_start, pos_start+length) 的 cos/sin,shape [1, L, head_dim]。
    """
    position_ids = torch.arange(pos_start, pos_start + length, device=DEVICE).unsqueeze(0)
    dummy_x = torch.zeros(1, 1, length, head_dim, device=DEVICE, dtype=DTYPE)
    with torch.inference_mode():
        cos, sin = rotary_emb(dummy_x, position_ids)  # [1, L, head_dim],返回这一段位置对应的 RoPE 旋转参数
    return cos.to(DTYPE), sin.to(DTYPE)


def rerotate_k(k: torch.Tensor, pos_from: int, pos_to: int, length: int) -> torch.Tensor:
    """
    把某段 Key 向量 k,从“原来所在位置 pos_from 的 RoPE 表示”,转换成“放到新位置 pos_to 时对应的 RoPE 表示”。
    同一段文本如果原来在位置 100,现在想“假装它其实在位置 300”,
    那么因为 RoPE 和位置有关,这个 k 不能直接拿过去用,必须重新按新位置旋转一次。
    这个函数:
    1.先把旧位置的旋转去掉
    2.再加上新位置的旋转
    """
    orig_dtype = k.dtype # 记住原始数据类型
    k = k.float() # 转成 float32,临时转成 float32 做更稳定的计算，最后再转回去
    cos_from, sin_from = get_cos_sin(pos_from, length) # 取出旧位置的 RoPE 旋转参数
    cos_to,   sin_to   = get_cos_sin(pos_to,   length) # 取出新位置的 RoPE 旋转参数
    k_unrot = _apply_rope(k, cos_from.float(), -sin_from.float()) # 先逆旋转回“无 RoPE”状态(用旧位置的 cos/sin,sin 取负就是逆旋转)
    k_rerot = _apply_rope(k_unrot, cos_to.float(), sin_to.float()) # 再按新位置旋转到“有 RoPE”状态(用新位置的 cos/sin)
    return k_rerot.to(orig_dtype)


def make_b2_skill_cache(
    skill_v1_cache: DynamicCache,
    pos_from: int,
    pos_to: int,
    length: int,
) -> DynamicCache:
    """B2:V 直接照搬,K 从 pos_from 逆旋转后重旋转到 pos_to。"""
    new_cache = DynamicCache()
    for layer in skill_v1_cache.layers:
        k_new = rerotate_k(layer.keys.clone(), pos_from, pos_to, length)
        new_cache.layers.append(_make_layer(k_new.contiguous(), layer.values.clone()))
    return new_cache


# ============================================================
# 6. 单次 forward(自动推算 position_ids)
# ============================================================

def model_forward(
    token_ids: list[int] | torch.Tensor,
    cache: DynamicCache,
) -> tuple[torch.Tensor, DynamicCache]:
    """
    单次 forward。position_ids 从 cache.get_seq_length() 自动推算。
    返回 (logits_last [vocab_size], updated_cache)。
    """
    if isinstance(token_ids, list):
        input_ids = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    else:
        input_ids = token_ids if token_ids.ndim == 2 else token_ids.unsqueeze(0)
        input_ids = input_ids.to(DEVICE)

    past_len = cache.get_seq_length()
    seq_len  = input_ids.shape[1]
    position_ids = torch.arange(
        past_len, past_len + seq_len, dtype=torch.long, device=DEVICE
    ).unsqueeze(0)

    with torch.inference_mode():
        out = model(
            input_ids=input_ids,
            position_ids=position_ids,
            past_key_values=cache, # ← 把当前 cache (已算好的kv cache)传进去     
            use_cache=True,   # ← 让模型把新算出的 KV 写回 cache
            output_attentions=False,
            output_hidden_states=False,
        )
    return out.logits[0, -1, :], out.past_key_values


# ============================================================
# 7. Phase 1:Prefill [context + skill_v1 + middle]
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 1: Prefill [context + skill_v1 + middle]  ({len(phase1_ids)} tok)")
t0 = time.time()

cache_phase1 = DynamicCache() # 创建一个空的 KV Cache 容器,里面什么都没
_, cache_phase1 = model_forward(phase1_ids, cache_phase1) #phase1_ids:1082个token

print(f"✓ Phase 1 完成  {time.time()-t0:.1f}s  cache_len={cache_phase1.get_seq_length()}")

# 切出 skill_v1 的 KV(绝对位置与 full_prompt 一致)
skill_v1_cache = slice_cache(cache_phase1, skill1_start, skill1_end)
print(f"  skill_v1 KV 切出: [{skill1_start}, {skill1_end})  len={skill_v1_cache.get_seq_length()}")

# ============================================================
# 8. Phase 3A:Scenario A(正常 prefill skill_v2 + assistant 续写前缀)
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 3A: Scenario A -- prefill [skill_v2]({len(skill2_ids)} tok) + [assistant_prefix]({len(query_ids)} tok)")
t0 = time.time()

cache_A = clone_cache(cache_phase1)
_, cache_A = model_forward(skill2_ids, cache_A)
logits_A_first, cache_A = model_forward(query_ids, cache_A)

print(f"✓ Scenario A  {time.time()-t0:.1f}s  cache_len={cache_A.get_seq_length()}")
print(f"  top1 token: {int(logits_A_first.argmax())}  "
      f"= {tokenizer.decode([int(logits_A_first.argmax())])!r}")

# ============================================================
# 9. Phase 3B1:Scenario B1(朴素复用,全部 L 个 KV)
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 3B1: Scenario B1 -- naive reuse (all {L_skill} KV), prefill assistant prefix")

cache_B1_pre = concat_cache(cache_phase1, skill_v1_cache)
expect_len = skill2_end
print(f"  B1 concat cache_len = {cache_B1_pre.get_seq_length()}  (expect {expect_len})")
assert cache_B1_pre.get_seq_length() == expect_len, (
    f"B1 cache 长度错误: {cache_B1_pre.get_seq_length()} != {expect_len}"
)

t0 = time.time()
logits_B1_first, cache_B1 = model_forward(query_ids, cache_B1_pre)
print(f"✓ Scenario B1  {time.time()-t0:.1f}s  cache_len={cache_B1.get_seq_length()}")
print(f"  top1 token: {int(logits_B1_first.argmax())}  "
      f"= {tokenizer.decode([int(logits_B1_first.argmax())])!r}")

# ============================================================
# 10. Phase 3B2:Scenario B2(RoPE 校正复用,全部 L 个 KV)
# ============================================================
print("\n" + "=" * 60)
print(f"Phase 3B2: Scenario B2 -- RoPE-corrected reuse (all {L_skill} KV), prefill assistant prefix")
print(f"  重旋转 K: pos_from={skill1_start} -> pos_to={skill2_start}  length={L_skill}")

t0 = time.time()
skill_v1_rerotated = make_b2_skill_cache(
    skill_v1_cache,
    skill1_start, skill2_start, L_skill,
)
print(f"  RoPE 重旋转完成  {time.time()-t0:.1f}s")

cache_B2_pre = concat_cache(cache_phase1, skill_v1_rerotated)
assert cache_B2_pre.get_seq_length() == expect_len

t0 = time.time()
logits_B2_first, cache_B2 = model_forward(query_ids, cache_B2_pre)
print(f"✓ Scenario B2  {time.time()-t0:.1f}s  cache_len={cache_B2.get_seq_length()}")
print(f"  top1 token: {int(logits_B2_first.argmax())}  "
      f"= {tokenizer.decode([int(logits_B2_first.argmax())])!r}")

assert cache_A.get_seq_length() == cache_B1.get_seq_length() == cache_B2.get_seq_length(), (
    f"三场景 cache 长度不对齐: "
    f"A={cache_A.get_seq_length()} B1={cache_B1.get_seq_length()} B2={cache_B2.get_seq_length()}"
)
assert cache_A.get_seq_length() == N_full, (
    f"cache 长度应等于 N_full={N_full}, 实际={cache_A.get_seq_length()}"
)
print(f"\n✓ 三场景 cache 对齐: seq_len = {cache_A.get_seq_length()}")

# ============================================================
# 11. Teacher Forcing Decode(T_STEPS 步)
# ============================================================
print(f"\n{'=' * 60}")
print(f"Teacher Forcing Decode: T={T_STEPS} 步")
print("(每步喂 A 的 greedy token;三场景共享输入,只有 KV cache 不同)")

# step 0:来自 assistant 续写前缀最后一个 token 的 logits
p_A_steps  = [F.softmax(logits_A_first.float(),  dim=-1).cpu()]
p_B1_steps = [F.softmax(logits_B1_first.float(), dim=-1).cpu()]
p_B2_steps = [F.softmax(logits_B2_first.float(), dim=-1).cpu()]

y_A = [int(logits_A_first.argmax().item())]
print(f"  step 0: y_A[0] = {y_A[0]}  ({tokenizer.decode([y_A[0]])!r})")

dec_cache_A  = clone_cache(cache_A)
dec_cache_B1 = clone_cache(cache_B1)
dec_cache_B2 = clone_cache(cache_B2)

for t in range(1, T_STEPS):
    teacher = torch.tensor([[y_A[-1]]], dtype=torch.long, device=DEVICE)

    logits_A_t,  dec_cache_A  = model_forward(teacher, dec_cache_A)
    logits_B1_t, dec_cache_B1 = model_forward(teacher, dec_cache_B1)
    logits_B2_t, dec_cache_B2 = model_forward(teacher, dec_cache_B2)

    p_A_steps.append(F.softmax(logits_A_t.float(),  dim=-1).cpu())
    p_B1_steps.append(F.softmax(logits_B1_t.float(), dim=-1).cpu())
    p_B2_steps.append(F.softmax(logits_B2_t.float(), dim=-1).cpu())

    next_tok = int(logits_A_t.argmax().item())
    y_A.append(next_tok)

    # print(f"  step {t:2d}: y_A[{t}] = {next_tok}  ({tokenizer.decode([next_tok])!r})")

print(f"\n✓ Teacher forcing 完成")
print(f"  y_A = {tokenizer.decode(y_A)!r}")

# ============================================================
# 12. 计算指标
# ============================================================
print(f"\n{'=' * 60}")
print("计算指标 ...")


def compute_metrics(
    p_A_list:  list[torch.Tensor],
    p_B_list:  list[torch.Tensor],
    label: str,
) -> dict:
    """
    两组逐步生成的概率分布 p_A_list 和 p_B_list 到底有多像似? 计算一系列指标来量化它们的差异和相似度。
    p_A_list[t]:第 t 个 decode step 时，实验 A 的下一个 token 概率分布
    p_B_list[t]:第 t 个 decode step 时，实验 B 的下一个 token 概率分布
    从四个角度去衡量它们的差异:
    1. KL 散度(KL divergence): 衡量两个概率分布之间的差异程度。KL 越小,分布越相似。KL 不只是看“最大概率 token 一不一样”,而是看整个分布的形状差异。
    2. TVD(总变差距离,Total Variation Distance): 衡量两个概率分布之间的距离。TVD 越小，分布越相似。
    3. Argmax 匹配率: 直接比较两个分布的最高概率 token 是否一致。匹配率越高，分布越相似。两边最终最想选的 token 是否一致。
       如果更关心“decode 出来的字是不是一样”，这个指标就很重要。
    4. Top-5 Jaccard 相似度: 比较两个分布的 top-5 token 集合的重叠程度。Jaccard 越高，分布越相似。
       这个指标在问:如果第一名可能不一样，但高概率候选集合是否差不多？
       比如:A 的 top5 = {1,2,3,4,5},B 的 top5 = {1,2,3,8,9},它们的 Jaccard 相似度就是 3/7 ≈ 0.429,因为它们共有 3 个元素(1,2,3),
       总共有 7 个不同的元素(1,2,3,4,5,8,9)。
    """
    kl_steps      = []
    tvd_steps     = []
    match_steps   = []
    jaccard_steps = []

    for pA, pB in zip(p_A_list, p_B_list):
        kl = float(F.kl_div(pB.clamp(min=1e-9).log(), pA, reduction="sum").item())
        kl_steps.append(max(kl, 0.0))

        tvd = float(0.5 * (pA - pB).abs().sum().item())
        tvd_steps.append(tvd)

        match_steps.append(int(pA.argmax().item()) == int(pB.argmax().item()))

        top5A = set(pA.topk(5).indices.tolist())
        top5B = set(pB.topk(5).indices.tolist())
        jaccard_steps.append(len(top5A & top5B) / len(top5A | top5B))

    kl_arr  = np.array(kl_steps)
    tvd_arr = np.array(tvd_steps)
    match_arr   = np.array(match_steps, dtype=float)
    jac_arr = np.array(jaccard_steps)

    T = len(kl_arr)
    m = {
        "label":                 label,
        "T":                     T,
        "KL_first":              float(kl_arr[0]),
        "KL_mean":               float(kl_arr.mean()),
        "KL_max":                float(kl_arr.max()),
        "TVD_first":             float(tvd_arr[0]),
        "TVD_mean":              float(tvd_arr.mean()),
        "argmax_match_rate":     float(match_arr.mean()),
        "argmax_match_count":    int(match_arr.sum()),
        "top5_overlap_mean":     float(jac_arr.mean()),
        "kl_per_step":           kl_arr.tolist(),
        "tvd_per_step":          tvd_arr.tolist(),
        "argmax_match_per_step": match_arr.tolist(),
        "top5_jaccard_per_step": jac_arr.tolist(),
    }

    print(f"\n  [{label}]")
    print(f"    KL_first          = {m['KL_first']:.5f}")
    print(f"    KL_mean({T})       = {m['KL_mean']:.5f}")
    print(f"    KL_max({T})        = {m['KL_max']:.5f}")
    print(f"    TVD_first         = {m['TVD_first']:.5f}")
    print(f"    TVD_mean          = {m['TVD_mean']:.5f}")
    print(f"    argmax_match_rate = {m['argmax_match_rate']:.3f}  ({m['argmax_match_count']}/{T})")
    print(f"    top5_overlap_mean = {m['top5_overlap_mean']:.3f}")
    return m


metrics_B1 = compute_metrics(p_A_steps, p_B1_steps, "B1_naive_reuse")
metrics_B2 = compute_metrics(p_A_steps, p_B2_steps, "B2_rope_corrected")

print(f"\n  [B2 vs B1 RoPE 校正净收益]")
print(f"    KL_first  B1={metrics_B1['KL_first']:.5f}  B2={metrics_B2['KL_first']:.5f}"
      f"  delta={metrics_B1['KL_first']-metrics_B2['KL_first']:+.5f}")
print(f"    argmax    B1={metrics_B1['argmax_match_rate']:.3f}  "
      f"B2={metrics_B2['argmax_match_rate']:.3f}")

# ============================================================
# 13. Free-run 生成(greedy,max 256 tokens)
# ============================================================
print(f"\n{'=' * 60}")
print("Free-run 生成: 各场景独立 greedy decode,不使用 teacher forcing")
FREE_MAX_TOKENS = 256


def free_run_decode(
    start_cache: DynamicCache,
    first_logits: torch.Tensor,
    max_new_tokens: int = FREE_MAX_TOKENS,
) -> list[int]:
    """
    从 start_cache 出发做 greedy 自回归生成。
    first_logits 是 prefill 结束时最后一个 query token 的 logit(已算好)。
    每步用自己生成的 token 作为下一步输入(不做 teacher forcing)。
    """
    tokens: list[int] = []
    curr_cache = clone_cache(start_cache)
    next_logits = first_logits

    for _ in range(max_new_tokens):
        next_tok = int(next_logits.argmax().item())
        tokens.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break
        inp = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        next_logits, curr_cache = model_forward(inp, curr_cache)

    return tokens


t0 = time.time()
free_tokens_A  = free_run_decode(cache_A,  logits_A_first)
free_tokens_B1 = free_run_decode(cache_B1, logits_B1_first)
free_tokens_B2 = free_run_decode(cache_B2, logits_B2_first)
print(f"✓ Free-run 完成  {time.time()-t0:.1f}s")

free_text_A  = tokenizer.decode(free_tokens_A,  skip_special_tokens=True)
free_text_B1 = tokenizer.decode(free_tokens_B1, skip_special_tokens=True)
free_text_B2 = tokenizer.decode(free_tokens_B2, skip_special_tokens=True)

print(f"\n  [A  生成 {len(free_tokens_A)} tok] {free_text_A[:FREE_MAX_TOKENS]!r}")
print(f"  [B1 生成 {len(free_tokens_B1)} tok] {free_text_B1[:FREE_MAX_TOKENS]!r}")
print(f"  [B2 生成 {len(free_tokens_B2)} tok] {free_text_B2[:FREE_MAX_TOKENS]!r}")

# ============================================================
# 14. 文本层面指标(无需外部 NLP 包)
# ============================================================
print(f"\n{'=' * 60}")
print("文本层面指标: BLEU-4 / ROUGE-L / token-embedding cosine")

import math
from collections import Counter


# ---------- BLEU-4 ----------
def _ngrams(tokens: list[int], n: int) -> list[tuple]:
    return [tuple(tokens[i: i + n]) for i in range(len(tokens) - n + 1)]


def bleu4(ref: list[int], hyp: list[int]) -> float:
    """
    BLEU-4(token ID 级别)。看局部 n-gram 重合程度
    参考 ref,假设 hyp,计算 1-4 gram 精度的几何均值 × brevity penalty。
    BLEU-4 是一种衡量生成文本和参考文本有多接近的指标。
    如果生成结果和参考文本在局部短语上很像，比如很多连续的 token 片段都对得上，那 BLEU-4 就会高。
    如果只是偶尔有几个词一样，但整体短语结构不一样，那 BLEU-4 就会低。
    它更偏向衡量:生成文本在局部短语层面上和参考文本的重合程度，而不是整体句子结构或语义的相似度。
    一般来说:越接近 1:越像参考文本;越接近 0:越不像参考文本。
    """
    if not hyp:
        return 0.0
    bp = min(1.0, len(hyp) / max(len(ref), 1))
    log_sum = 0.0
    for n in range(1, 5):
        ref_ng = Counter(_ngrams(ref, n))
        hyp_ng = Counter(_ngrams(hyp, n))
        clipped = sum(min(cnt, ref_ng[g]) for g, cnt in hyp_ng.items())
        total   = max(sum(hyp_ng.values()), 1)
        prec    = clipped / total if total > 0 else 0.0
        log_sum += math.log(prec + 1e-9)
    return bp * math.exp(log_sum / 4)


# ---------- ROUGE-L ----------
def _lcs_len(a: list[int], b: list[int]) -> int:
    """动态规划求最长公共子序列长度,内存 O(min(m,n))。"""
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        curr = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            curr[j] = prev[j - 1] + 1 if x == y else max(prev[j], curr[j - 1])
        prev = curr
    return prev[len(b)]


def rouge_l(ref: list[int], hyp: list[int]) -> float:
    """
    ROUGE-L F1(token ID 级别)。看两段文本的 最长公共子序列 长度占比。
    “子序列”要求:
    1. 顺序一致:比如 ref=[A B C D E], hyp=[A X B Y C Z] 的 LCS 是 [A B C],长度 3;但 hyp=[C B A] 的 LCS 是 [C] 或 [B] 或 [A],长度 1。
    2. 不要求连续:比如 ref=[A B C D E], hyp=[A X Y B Z C] 的 LCS 也是 [A B C],长度 3;但 hyp=[A B X C] 的 LCS 是 [A B] 或 [A C],长度 2。
    本质上是在看两段文本是否保留了相似的整体生成骨架和顺序结构,所以它比 BLEU-4 更关注整体顺序有没有保持,主要内容轨迹像不像，而不是局部短语重合。
    """
    if not ref or not hyp:
        return 0.0
    lcs = _lcs_len(ref, hyp)
    p   = lcs / len(hyp)
    r   = lcs / len(ref)
    return 2 * p * r / (p + r + 1e-9)


# ---------- Token-embedding cosine ----------
def embed_cosine(tokens_a: list[int], tokens_b: list[int]) -> float:
    """
    用 Qwen3 的 embed_tokens 层(输入嵌入)做 mean-pooling,
    计算两段文字的余弦相似度。
    不需要额外加载模型,是轻量级的语义相似度代理指标。
    衡量两段文本整体“语义方向”是不是相近的。
    """
    with torch.inference_mode():
        ea = model.model.embed_tokens(
            torch.tensor(tokens_a, device=DEVICE)
        ).mean(dim=0)
        eb = model.model.embed_tokens(
            torch.tensor(tokens_b, device=DEVICE)
        ).mean(dim=0)
    return float(F.cosine_similarity(ea.unsqueeze(0), eb.unsqueeze(0)).item())


def text_metrics(ref_tok: list[int], hyp_tok: list[int], label: str) -> dict:
    b4  = bleu4(ref_tok, hyp_tok)
    rl  = rouge_l(ref_tok, hyp_tok)
    cos = embed_cosine(ref_tok, hyp_tok)
    m = {
        "label":        label,
        "ref_len":      len(ref_tok),
        "hyp_len":      len(hyp_tok),
        "bleu4":        b4,
        "rouge_l":      rl,
        "embed_cosine": cos,
    }
    print(f"\n  [{label}]")
    print(f"    ref_len / hyp_len = {len(ref_tok)} / {len(hyp_tok)}")
    print(f"    BLEU-4            = {b4:.4f}")
    print(f"    ROUGE-L F1        = {rl:.4f}")
    print(f"    embed cosine      = {cos:.4f}")
    return m


text_m_B1 = text_metrics(free_tokens_A, free_tokens_B1, "A_vs_B1")
text_m_B2 = text_metrics(free_tokens_A, free_tokens_B2, "A_vs_B2")

# ============================================================
# 15. 存 JSON
# ============================================================
results = {
    "scenario":  SCENARIO,
    "model":     MODEL_PATH,
    "T_steps":   T_STEPS,
    "sequence_info": {
        "N_full":             N_full,
        "context":            [0, context_end],
        "skill_v1":           [skill1_start, skill1_end],
        "middle":             [middle_start, middle_end],
        "skill_v2":           [skill2_start, skill2_end],
        "L_skill":            L_skill,
        "skill_position_gap": skill2_start - skill1_start,
        "note": (
            "no second user query; decode starts after skill_v2 plus the assistant continuation "
            "prefix, and first logits are measured at the end of that prefix"
        ),
    },
    "y_A":      y_A,
    "y_A_text": tokenizer.decode(y_A),
    "free_run": {
        "A_text": free_text_A,
        "B1_text": free_text_B1,
        "B2_text": free_text_B2,
    },
    "text_metrics": {
        "A_vs_B1": text_m_B1,
        "A_vs_B2": text_m_B2,
    },
    "B1": {k: v for k, v in metrics_B1.items() if not k.endswith("_per_step")},
    "B2": {k: v for k, v in metrics_B2.items() if not k.endswith("_per_step")},
    "B1_per_step": {
        k: metrics_B1[k]
        for k in ("kl_per_step", "tvd_per_step", "argmax_match_per_step", "top5_jaccard_per_step")
    },
    "B2_per_step": {
        k: metrics_B2[k]
        for k in ("kl_per_step", "tvd_per_step", "argmax_match_per_step", "top5_jaccard_per_step")
    },
}

json_path = OUT_DIR / f"{SCENARIO}_summary.json"
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"\nJSON 写入: {json_path}")

# ============================================================
# 14. 画图
# ============================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

steps = np.arange(T_STEPS)

# 图 1:KL + TVD 逐步曲线
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
ax.plot(steps, metrics_B1["kl_per_step"], label="B1 naive reuse",
        color="tab:red", lw=1.5, marker="o", ms=3)
ax.plot(steps, metrics_B2["kl_per_step"], label="B2 RoPE-corrected",
        color="tab:orange", lw=1.5, marker="s", ms=3)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("Decode step (teacher forced)")
ax.set_ylabel("KL(P_A || P_B)")
ax.set_title(f"KL divergence per step  (T={T_STEPS})")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[1]
ax.plot(steps, metrics_B1["tvd_per_step"], label="B1 naive reuse",
        color="tab:red", lw=1.5, marker="o", ms=3)
ax.plot(steps, metrics_B2["tvd_per_step"], label="B2 RoPE-corrected",
        color="tab:orange", lw=1.5, marker="s", ms=3)
ax.set_xlabel("Decode step (teacher forced)")
ax.set_ylabel("TVD(P_A, P_B)")
ax.set_title(f"Total Variation Distance per step  (T={T_STEPS})")
ax.legend()
ax.grid(alpha=0.3)

fig.tight_layout()
fig1_path = FIG_DIR / f"{SCENARIO}_kl_tvd_curve.png"
fig.savefig(fig1_path, dpi=150)
plt.close(fig)
print(f"图 1 写入: {fig1_path}")

# 图 2:argmax match + top-5 Jaccard
fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(steps, metrics_B1["argmax_match_per_step"], label="B1 argmax match",
        color="tab:red", lw=1, alpha=0.8)
ax.plot(steps, metrics_B2["argmax_match_per_step"], label="B2 argmax match",
        color="tab:orange", lw=1, alpha=0.8)
ax.plot(steps, metrics_B1["top5_jaccard_per_step"], label="B1 top-5 Jaccard",
        color="tab:red", lw=1.5, ls="--")
ax.plot(steps, metrics_B2["top5_jaccard_per_step"], label="B2 top-5 Jaccard",
        color="tab:orange", lw=1.5, ls="--")
ax.set_xlabel("Decode step")
ax.set_ylabel("Match / Overlap  (0-1)")
ax.set_title(f"Argmax match & Top-5 Jaccard  (T={T_STEPS})")
ax.set_ylim(-0.05, 1.05)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig2_path = FIG_DIR / f"{SCENARIO}_match_overlap.png"
fig.savefig(fig2_path, dpi=150)
plt.close(fig)
print(f"图 2 写入: {fig2_path}")

print("\n" + "=" * 70)
print("Done.")
print(f"  JSON : {json_path}")
print(f"  图   : {FIG_DIR}")
print("=" * 70)
