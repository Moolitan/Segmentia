"""
vLLM attention 路径探测脚本
目标:弄清楚 vLLM 0.17 (V1 engine) 的 attention 实际在哪里、怎么调用。
此脚本不修改任何 vLLM 代码,只是打印信息。

python scripts/04_kv_cache_research/pre_probe/probe_attention.py 2>&1 | tee scripts/04_kv_cache_research/pre_probe/probe_attention_output.txt
"""
import os
import sys
import inspect

# 确保输出不被缓冲,能实时看到
sys.stdout.reconfigure(line_buffering=True)

print("=" * 70)
print("Part 1: vLLM 基本信息")
print("=" * 70)

import vllm
print(f"vllm.__version__     = {vllm.__version__}")
print(f"vllm 安装路径         = {vllm.__file__}")
print()


print("=" * 70)
print("Part 2: V1 vs V0 engine 路径探测")
print("=" * 70)

# 探测 V1 engine
try:
    import vllm.v1
    print(f"✓ vllm.v1 exists, path = {vllm.v1.__file__}")
    
    import vllm.v1.attention.backends as v1_backends
    v1_dir = os.path.dirname(v1_backends.__file__)
    print(f"✓ V1 attention backends 目录: {v1_dir}")
    files = sorted(os.listdir(v1_dir))
    print(f"  backends 文件列表:")
    for f in files:
        if f.endswith('.py') and not f.startswith('__'):
            print(f"    - {f}")
except Exception as e:
    print(f"✗ V1 not found: {type(e).__name__}: {e}")

print()

# 探测 V0 attention layer
try:
    from vllm.attention.layer import Attention
    print(f"✓ V0 vllm.attention.layer.Attention exists")
    print(f"  class defined at: {inspect.getfile(Attention)}")
    print(f"  forward signature:")
    sig = inspect.signature(Attention.forward)
    for name, param in sig.parameters.items():
        print(f"    {name}: {param}")
except Exception as e:
    print(f"✗ V0 Attention not found: {type(e).__name__}: {e}")

print()


print("=" * 70)
print("Part 3: Qwen3 模型定义探测")
print("=" * 70)

try:
    from vllm.model_executor.models import qwen3
    print(f"✓ qwen3 module: {qwen3.__file__}")
    
    # 找所有定义的类
    members = inspect.getmembers(qwen3, inspect.isclass)
    qwen3_classes = [name for name, cls in members if cls.__module__ == qwen3.__name__]
    print(f"  qwen3.py 里定义的类:")
    for cls_name in qwen3_classes:
        print(f"    - {cls_name}")
    
    # 重点看 attention 相关的类
    attn_classes = [name for name in qwen3_classes if 'ttention' in name or 'ttn' in name]
    if attn_classes:
        print(f"\n  疑似 attention 相关的类: {attn_classes}")
        for cls_name in attn_classes:
            cls = getattr(qwen3, cls_name)
            print(f"\n  {cls_name} 的 MRO (继承链):")
            for base in cls.__mro__:
                print(f"    - {base.__module__}.{base.__name__}")
except Exception as e:
    print(f"✗ qwen3 module error: {type(e).__name__}: {e}")

print()


print("=" * 70)
print("Part 4: 默认 attention backend 探测")
print("=" * 70)

# 看看环境变量有没有设
attn_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "(未设置)")
print(f"VLLM_ATTENTION_BACKEND = {attn_backend}")

# vLLM 的 backend selector
try:
    from vllm.attention.selector import get_attn_backend
    print(f"✓ 找到 backend selector: get_attn_backend")
    print(f"  signature: {inspect.signature(get_attn_backend)}")
except Exception as e:
    print(f"✗ selector not found: {e}")

print()


print("=" * 70)
print("Part 5: 加载小模型做运行时探测")
print("=" * 70)

print("开始加载 Qwen3-14B")
print("(如果你本地没有这个模型,可以改成你有的任何小模型)")

# 用环境变量控制,如果你本地没有这个模型,请手动改下面这行
SMALL_MODEL = os.environ.get("PROBE_MODEL", "/mnt/Large_Language_Model_Lab_1/llm_models/Qwen3-14B/Qwen/Qwen3-14B")
print(f"使用模型: {SMALL_MODEL}")
print()

try:
    from vllm import LLM, SamplingParams
    
    llm = LLM(
        model=SMALL_MODEL,
        enforce_eager=True,          # 禁用 CUDA graph,方便 patch
        gpu_memory_utilization=0.95,  # 只用 95% 显存,快速加载
        max_model_len=512,
        dtype="auto",
    )
    print("✓ 模型加载成功")
    print()
    
    # 探测内部结构
    print("-" * 50)
    print("engine 和 executor 结构:")
    print("-" * 50)
    engine = llm.llm_engine
    print(f"engine type: {type(engine).__module__}.{type(engine).__name__}")
    
    # V1 的 engine 接口不同
    if hasattr(engine, 'engine_core'):
        print(f"  engine.engine_core: {type(engine.engine_core).__name__}")
    if hasattr(engine, 'model_executor'):
        print(f"  engine.model_executor: {type(engine.model_executor).__name__}")
    
    # 打印 engine 的顶层公开属性
    public_attrs = [a for a in dir(engine) if not a.startswith('_')]
    print(f"\n  engine 公开属性(前 20 个): {public_attrs[:20]}")
    print()
    
    # 跑一次 generation 确认能正常工作
    print("-" * 50)
    print("跑一个 generation 测试...")
    print("-" * 50)
    outputs = llm.generate(
        ["Hello! What is 1+1?"],
        SamplingParams(max_tokens=5, temperature=0),
    )
    print(f"  输出: {outputs[0].outputs[0].text!r}")
    print("✓ Generation 正常")
    
except Exception as e:
    print(f"✗ 加载或运行失败: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print()
print("=" * 70)
print("探测完成!")
print("=" * 70)
