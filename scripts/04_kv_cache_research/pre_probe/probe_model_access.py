"""
定位 V1 engine 里的模型访问路径,并确认 Qwen3 模型结构。
"""
import os
import sys

# 不 dump,只探测
sys.stdout.reconfigure(line_buffering=True)

from vllm import LLM, SamplingParams

print("加载 Qwen3-14B (gpu_memory_utilization=0.85)...")

llm = LLM(
    model="/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B",
    enforce_eager=True,
    gpu_memory_utilization=0.85,
    max_model_len=1024,
)
print("✓ 模型加载完成\n")

# 尝试多种可能的路径
print("=" * 60)
print("探测模型对象的访问路径")
print("=" * 60)

engine = llm.llm_engine
print(f"engine: {type(engine).__name__}")

# 打印 engine 的所有非私有属性
print(f"engine 属性: {[a for a in dir(engine) if not a.startswith('_')]}\n")

# 尝试路径
paths_to_try = [
    "engine.engine_core",
    "engine.engine_core.model_executor",
    "engine.engine_core.model_executor.driver_worker",
    "engine.engine_core.model_executor.driver_worker.model_runner",
    "engine.engine_core.model_executor.driver_worker.model_runner.model",
    "engine.model_executor",
    "engine.model_executor.driver_worker",
    "engine.model_executor.driver_worker.model_runner",
    "engine.model_executor.driver_worker.model_runner.model",
]

for path in paths_to_try:
    try:
        obj = eval(path)
        print(f"✓ {path} = {type(obj).__name__}")
    except Exception as e:
        print(f"✗ {path}: {type(e).__name__}")

# 找到真实路径后,打印模型结构
print("\n" + "=" * 60)
print("如果找到了 model,打印它的子模块")
print("=" * 60)

for path in paths_to_try:
    try:
        obj = eval(path)
        if hasattr(obj, 'model') or 'Qwen3' in type(obj).__name__:
            print(f"\n找到模型对象: {path}")
            print(f"类型: {type(obj).__name__}")
            
            # 如果是 Qwen3ForCausalLM,它有 .model 属性指向 Qwen3Model
            if hasattr(obj, 'model'):
                inner = obj.model
                print(f"  .model: {type(inner).__name__}")
                if hasattr(inner, 'layers'):
                    print(f"  .model.layers 数量: {len(inner.layers)}")
                    first_layer = inner.layers[0]
                    print(f"  第一层类型: {type(first_layer).__name__}")
                    if hasattr(first_layer, 'self_attn'):
                        attn = first_layer.self_attn
                        print(f"  第一层.self_attn: {type(attn).__name__}")
                        # 看有没有 prefix 属性
                        if hasattr(attn, 'prefix'):
                            print(f"  第一层.self_attn.prefix: {attn.prefix}")
                        if hasattr(attn, 'attn'):
                            print(f"  第一层.self_attn.attn: {type(attn.attn).__name__}")
            break
    except Exception:
        continue

print("\n探测完成")