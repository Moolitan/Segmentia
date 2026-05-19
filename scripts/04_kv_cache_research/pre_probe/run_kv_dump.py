"""
KV Dump 最小验证脚本
跑一个简单 prompt,触发 dump,验证文件生成。
"""
import os
import sys
import shutil
from pathlib import Path

sys.stdout.reconfigure(line_buffering=True)

# ===== Step 1: 设置环境变量(必须在 import vllm 前) =====
os.environ["DUMP_KV"] = "1"
os.environ["SCENARIO"] = "hello_world_test"
os.environ["DUMP_DIR"] = "/tmp/kv_dump"

# 先清空目标目录
dump_dir = Path("/tmp/kv_dump/scenario_hello_world_test")
if dump_dir.exists():
    print(f"清空已存在的 dump 目录: {dump_dir}")
    shutil.rmtree(dump_dir)

# ===== Step 2: 加载 vLLM =====
from vllm import LLM, SamplingParams

print("加载 Qwen3-14B (这会花 30 秒左右)...")
llm = LLM(
    model="/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B",
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=1024,
)
print("✓ 模型加载完成\n")

# ===== Step 3: 跑一个简单 prompt =====
prompt = "Hello, what is 1+1?"
print(f"Prompt: {prompt!r}")
print()

outputs = llm.generate(
    [prompt],
    SamplingParams(max_tokens=5, temperature=0),
)
print(f"Output: {outputs[0].outputs[0].text!r}\n")

# ===== Step 4: 检查 dump 文件 =====
print("=" * 60)
print("Dump 文件检查")
print("=" * 60)

if not dump_dir.exists():
    print(f"✗ 致命错误:{dump_dir} 不存在,dump 没有发生!")
    print("  可能原因:")
    print("  1. 环境变量没传到子进程")
    print("  2. 源码改动没生效")
    sys.exit(1)

files = sorted(dump_dir.glob("*.pt"))
print(f"生成了 {len(files)} 个文件\n")

if len(files) == 0:
    print("✗ 目录存在但没文件,说明 hook 没被执行")
    sys.exit(1)

# 打印前几个文件的详细信息
import torch
print("前 3 个文件的内容:")
print("-" * 60)
for f in files[:3]:
    print(f"  文件名: {f.name}")
    data = torch.load(f, weights_only=False)
    print(f"    layer_idx:   {data['layer_idx']}")
    print(f"    module_id:      {data['module_id']}")
    print(f"    call_idx:    {data['call_idx']}")
    print(f"    scenario:  {data['scenario']}")
    print(f"    q:     {tuple(data['q'].shape)}, dtype: {data['q'].dtype}")
    print(f"    k:     {tuple(data['k'].shape)}")
    print(f"    v:     {tuple(data['v'].shape)}")
    print(f"    positions:   {data['positions']}")
    print(f"    num_tokens:  {data['num_tokens']}")
    print(f"    is_warmup_like:  {data['is_warmup_like']}")
    print()

# 统计每层被调用了几次
from collections import Counter
layer_calls = Counter()
for f in files:
    try:
        data = torch.load(f, weights_only=False)
        layer_calls[data["layer_idx"]] += 1
    except Exception as e:
        print(f"  读取失败: {f.name}, {e}")

print(f"总共覆盖了 {len(layer_calls)} 个层")
print(f"每层被 dump 的次数(前 10 层):")
for layer in sorted(layer_calls.keys())[:10]:
    print(f"  layer {layer:2d}: {layer_calls[layer]} 次")

if max(layer_calls.values()) == 1:
    print("\n观察:每层只 dump 了 1 次 → 说明是一次 forward(prefill 阶段)")
elif max(layer_calls.values()) > 1:
    print(f"\n观察:每层最多 dump 了 {max(layer_calls.values())} 次")
    print("  可能是 prefill + decode 都触发了 dump")
    print("  或者是 chunked prefill 把一次 prefill 拆成了多次")