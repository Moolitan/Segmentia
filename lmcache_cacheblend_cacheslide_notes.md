# LMCache, CacheBlend, and CacheSlide Notes

## 1. 一句话结论

普通 LMCache 主要是 vLLM 的外部 KV cache connector，负责 KV 的查找、存储、搬运和注入。

LMCache CacheBlend 在普通 LMCache 之上增加了 model-aware correction：它会访问 vLLM model，在 GPU 上另跑一段 layerwise mini-forward，对命中的非前缀/segment KV 做部分重算和修正。

CacheSlide 则是把 cross-position KV reuse 和 WCA correction 直接写进 vLLM model/attention forward 路径里。

## 2. 普通 LMCache 是什么

普通 LMCache 可以理解为：

```text
外挂 connector + 外部存储/搬运 KV
```

它走 vLLM 的 `KVConnector` 机制：

```text
vLLM scheduler
  -> LMCacheConnector.get_num_new_matched_tokens()
  -> 判断外部 cache 还能命中多少 token
  -> vLLM 为这些外部 KV 分配 paged KV slots

vLLM worker
  -> LMCacheConnector.start_load_kv()
  -> 从 LMCache backend 取 KV
  -> 写入 vLLM paged KV cache

forward 后
  -> save_kv_layer() / wait_for_save()
  -> 把新 KV 存回 LMCache
```

普通 LMCache 的重点是系统能力：KV lookup、chunking、CPU/SSD/remote backend、异步加载、layerwise load/save、跨实例共享等。

它本身不一定处理“旧 KV 因为位置变化而不再完全等价”的问题。默认路径更像是：

```text
命中旧 KV -> 搬回来 -> 放进 vLLM paged KV cache -> 主 forward 使用
```

## 3. 为什么会有跨位置 KV 漂移问题

KV cache 不只由 token 内容决定，也受位置和上下文影响。

同一句文本：

```text
Paris is the capital of France.
```

如果第一次作为独立 chunk 编码，它可能在位置 `[0, 10)`。

如果第二次放在完整 prompt 中，它可能在位置 `[800, 810)`，前面还有 system prompt、history、query。

对 RoPE/CoPE/位置相关模型，K/V 近似依赖：

```text
K = f(token, previous context, layer state, position)
V = f(token, previous context, layer state)
```

因此同样 token 序列换了绝对位置或上下文后，旧 KV 不一定等价于当前 prompt 中应该产生的 KV。

这个问题就是我们讨论的跨位置 KV 表示漂移。

## 4. LMCache CacheBlend 是什么

LMCache CacheBlend 可以理解为：

```text
外挂 connector + 外部存储/搬运 KV + 访问 vLLM model 做部分重算修正
```

它仍然通过 vLLM connector 接入，但比普通 LMCache 更深。

关键代码位置：

```text
/home/wsh/LMCache/lmcache/integration/vllm/vllm_v1_adapter.py
/home/wsh/LMCache/lmcache/v1/compute/blend/blender.py
/home/wsh/LMCache/lmcache/v1/compute/models/base.py
```

入口大致是：

```text
LMCacheConnectorV1Impl.start_load_kv()
  -> if enable_blending:
       self.blender.blend(...)
     else:
       self.lmcache_engine.retrieve(...)
```

CacheBlend 的算法流程大致是：

```text
1. 从 LMCache 取出 old KV
2. 通过 gpu_connector 暴露给 blender
3. blender 访问 vLLM model/layers
4. 在 GPU 上重算部分 q/k/v
5. 比较当前 k 和 old_k 的差异
6. 选择差异大的 top-k token
7. 用当前重算的 k/v 覆盖或修正 old_k/old_v
8. 修正后的 KV 回到 vLLM paged KV cache
9. vLLM 主 forward 继续执行
```

本地代码中的典型逻辑：

```python
old_k, old_v = self.gpu_connector.get_kv(layer_id)
q, k = attn_layer.rotary_emb(self.metadata.positions, q, k)

diff_k = torch.sum((k.to(torch.float32) - old_k.to(torch.float32)) ** 2, dim=[1])
top_indices = torch.topk(diff_k, k=topk_num).indices

old_k[self.metadata.imp_indices] = k
old_v[self.metadata.imp_indices] = v
```

注意：这里的重算是在 GPU 上，不是在 CPU 上。

CPU backend / disk backend 指的是 KV cache 的存储或 offload 后端，不代表重算在 CPU 上。

## 5. CacheBlend 是不是在主 forward 过程里重算

这是我们重点澄清的问题。

结论：CacheBlend 的重算发生在 vLLM 主 model forward 之前的 connector `pre_forward/start_load_kv` 阶段。它不是直接插进 vLLM 主 `self.model(**model_inputs)` 的每一层 attention forward 里。

调用顺序更像：

```text
GPUModelRunner.execute_model()
  -> set_forward_context(...)
  -> kv_connector.pre_forward(scheduler_output)
       -> LMCacheConnector.start_load_kv(...)
          -> if enable_blending:
               blender.blend(...)
                 -> layerwise_model.compute_layer(tokens)
                 -> 每层重算 q/k/v
                 -> process_qkv() 比较 old_k 和当前 k
                 -> 修改 old_k/old_v 或中间 KV buffer
  -> self.model(**model_inputs)      # vLLM 主 forward
  -> kv_connector.post_forward(...)
```

也就是说，CacheBlend 是：

```text
旁路 mini-forward -> 修正 KV -> 回到主推理流程
```

它确实调用 vLLM 的真实 layer/weights，但这条计算路径由 LMCache 包装出来，不是直接写在 vLLM 模型 forward 函数体里。

## 6. CacheSlide 是什么

CacheSlide 可以理解为：

```text
直接改 vLLM model/attention forward，把 cross-position reuse/WCA 写进模型路径
```

本地 CacheSlide 仓库中相关位置：

```text
/home/wsh/CacheSlide/vllm/model_executor/models/llama.py
/home/wsh/CacheSlide/vllm/model_executor/models/mpt.py
/home/wsh/CacheSlide/examples/CacheSlide.py
```

CacheSlide 的核心思路是：

```text
1. 对 chunk 单独 prefill，收集 chunk KV
2. 构造完整 prompt
3. 建立 target token 到 source chunk KV 的映射
4. 在 attention forward 中同时看到：
     K_rec/V_rec     当前 prompt 位置下重算出来的 KV
     K_reuse/V_reuse 旧 chunk cache 中拿来的 KV
5. 比较两者差异
6. 选择重要 token
7. 用 WCA 做融合或动态重选
8. 当前 attention 直接使用 fused KV
```

CacheSlide 中的 WCA 逻辑大致是：

```text
diff = ||K_rec - K_reuse||^2

K_fused = alpha * K_rec + (1 - alpha) * K_reuse
V_fused = alpha * V_rec + (1 - alpha) * V_reuse
```

它还会用 CKSim 周期性判断哪些 token 的旧 KV 不可靠，再重选一批 token。

和 CacheBlend 最大的架构差异是：

```text
CacheBlend:
  主 forward 前旁路算一遍部分 token
  -> 修正 KV buffer
  -> 主 forward 使用修正后的 KV

CacheSlide:
  主 forward 的 attention 内部直接算 K_rec/K_reuse
  -> 当场融合
  -> 当前 attention 立即使用 fused KV
```

## 7. CacheBlend 和 CacheSlide 到底像不像

从算法目标看，它们很像。

两者都不是纯 KV 搬运，都属于：

```text
model-aware segment KV reuse correction
```

共同点：

```text
1. 都处理非前缀/segment KV reuse
2. 都承认旧 KV 可能和当前位置下的 KV 不一致
3. 都需要访问模型计算 Q/K/V
4. 重算都在 GPU 上
5. 都需要在性能收益和重算成本之间做 tradeoff
```

不同点：

```text
LMCache CacheBlend:
  connector-based
  旁路 mini-forward
  修正 KV buffer
  更工程化，和 LMCache storage/connector 体系绑定

CacheSlide:
  model/attention-inline
  在主 forward 的 attention 中融合
  更直接改变模型计算语义
  更像论文原型或模型内算法实现
```

因此，如果只说“用 GPU 重算一部分 KV 来修正非前缀 KV reuse”，CacheBlend 和 CacheSlide 的确很接近。

如果要证明 CacheSlide 明显强于 CacheBlend，不能只说“我们也重算”，需要强调更具体的差异，例如：

```text
1. cross-position mapping 更细
2. WCA 是融合而不是简单覆盖
3. token 选择策略不同
4. 多层动态重选不同
5. 对 RoPE/位置漂移有显式处理
6. 在相同 recompute ratio 下质量或 TTFT 更好
```

## 8. 三种方案的架构对比

```text
普通 LMCache:
  外挂 connector
  + 外部存储/搬运 KV
  + 低侵入
  + 工程成熟
  - 默认不做 model-aware correction

LMCache CacheBlend:
  外挂 connector
  + 外部存储/搬运 KV
  + 访问 vLLM model 做 GPU 侧部分重算修正
  + 更接近非前缀/segment KV reuse 需求
  + 工程边界比直接改模型更清楚
  - 仍需要 model tracker / layerwise mini-forward / GPU connector 等深度集成

CacheSlide:
  直接改 model/attention forward
  + correction/fusion 逻辑最直接
  + attention 内可以细粒度处理 K_rec/K_reuse
  + 更容易表达 WCA 这类模型内算法
  - 侵入性高
  - 每个模型可能都要适配
  - 和 vLLM scheduler、paged KV、prefix cache、connector 体系耦合更难维护
```

## 9. 实现路线判断

如果目标是工程可维护，我更倾向：

```text
把 CacheSlide 的 cross-position mapping / WCA 思路
做成 connector 体系下的 blender/corrector
而不是散落修改每个 model 的 attention forward
```

也就是走 CacheBlend-style 架构：

```text
worker / connector correction module
  -> load old segment KV
  -> 调 model/layer 做局部重算
  -> diff/topk/WCA correction
  -> scatter 修正后的 KV 到 paged KV cache
  -> 主 forward 正常使用
```

这样好处是：

```text
1. 保持 vLLM 主模型 forward 尽量干净
2. 更容易和 scheduler / paged KV / block ids 对齐
3. 更容易复用 KV slot gather/scatter 逻辑
4. 更容易做 ablation：no-correction、RoPE-only、overwrite、WCA-fusion
5. 更接近 LMCache CacheBlend，可作为强 baseline 对比
```

如果目标是做最强的模型内算法探索，则可以考虑 CacheSlide-style：

```text
直接在 attention forward 内传入 K_reuse/cache_idx
并融合 K_rec/K_reuse
```

但这条路需要付出更高的模型适配和维护成本。

## 10. 当前讨论中的核心澄清

1. LMCache CacheBlend 不是 CPU 重算。  
   重算和 diff/topk/修正都在 GPU 上。

2. “CPU backend” 指 KV storage/offload backend，和重算设备不是一回事。

3. CacheBlend 不是纯 KV 搬运。  
   开启 `enable_blending` 后，它会访问 vLLM model 做部分重算。

4. CacheBlend 的重算不在 vLLM 主 model forward 内联执行。  
   它是在 connector `pre_forward/start_load_kv` 阶段旁路执行一个 layerwise mini-forward。

5. CacheSlide 和 CacheBlend 都是 model-aware KV correction，但架构边界不同。  
   CacheBlend 是外置 correction module；CacheSlide 是 attention-inline correction。
