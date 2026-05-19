"""
probe_v1_attention.py
精确定位 V1 engine 里的 Attention 层位置和 forward 签名。
不加载模型,非常快。
"""
import os
import inspect

print("=" * 70)
print("V1 Attention 层定位")
print("=" * 70)

# V1 里 Attention 层应该在这些位置之一
candidates = [
    "vllm.attention.layer",          # V0 路径(我们知道这个没有)
    "vllm.v1.attention.layer",       # V1 可能的路径
    "vllm.model_executor.layers.attention",  # 另一个可能
]

found = None
for path in candidates:
    try:
        mod = __import__(path, fromlist=[''])
        print(f"✓ {path} 存在")
        # 找 Attention 类
        if hasattr(mod, 'Attention'):
            cls = mod.Attention
            print(f"  找到 {path}.Attention")
            print(f"  定义位置: {inspect.getfile(cls)}")
            found = cls
            break
        else:
            classes = [name for name in dir(mod) if not name.startswith('_') and name[0].isupper()]
            print(f"  模块存在但没有 Attention 类,里面的类: {classes}")
    except ImportError:
        print(f"✗ {path} 不存在")

print()
print("=" * 70)
print("通过 Qwen3Attention 反查 self.attn 的类型")
print("=" * 70)

from vllm.model_executor.models.qwen3 import Qwen3Attention

# 看 __init__ 源码
init_src = inspect.getsource(Qwen3Attention.__init__)
print("Qwen3Attention.__init__ 源码片段:")
# 只打印包含 'attn' 的行
for line in init_src.split('\n'):
    if 'attn' in line.lower() or 'Attention' in line:
        print(f"  {line.strip()}")

print()
# 看 forward 源码的关键部分
print("Qwen3Attention.forward 源码:")
fwd_src = inspect.getsource(Qwen3Attention.forward)
print(fwd_src[:2000])  # 前 2000 字符够看清楚结构

print()
print("=" * 70)
print("FlashAttentionImpl.forward 签名")
print("=" * 70)

try:
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    print(f"✓ FlashAttentionImpl 定义于: {inspect.getfile(FlashAttentionImpl)}")
    print()
    print(f"FlashAttentionImpl.forward 签名:")
    sig = inspect.signature(FlashAttentionImpl.forward)
    print(f"  forward({sig})")
    print()
    print(f"FlashAttentionImpl.forward 源码 (前 2500 字符):")
    src = inspect.getsource(FlashAttentionImpl.forward)
    print(src[:2500])
except Exception as e:
    print(f"✗ 无法导入 FlashAttentionImpl: {e}")

print()
print("探测完成")