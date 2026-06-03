### position_only_cksim.py
    先证明：只改 position，K 就会漂移。
    这是“纯位置因素”。

### skill_cksim_benchmark.py 的 offline_vs_base
    再证明：放到真实 agent-like prompt 里，offline skill KV 和真实 full-context skill KV 也明显不同。
    这是“位置变化 + history 上下文变化”的综合效果。

### skill_cksim_benchmark.py 的 reuse_vs_base
    最后证明：当前 ContextSegmentKV 的 rope rerotate 可以把 K 漂移大幅修回来。


## 更直观地说:

### position_only_cksim.py:
    RoPE 位置错位会不会造成 K 变化？会。

### skill_cksim_benchmark.py offline_vs_base:
    真实 agent 场景里，直接复用单独缓存的 skill KV 是否可靠？K 不够可靠。

### skill_cksim_benchmark.py reuse_vs_base:
    加了 rope 修正后是否更接近 full recompute？是，K 明显更接近。

## 阶段2
综合来看，后面不应该继续纠结位置编码了（或者先暂时不考虑位置编码的设计）。我们现在已经有一个很强的事实：

raw offline/base key CKSim  ≈ 0.682
rope reuse/base key CKSim   ≈ 0.963

这说明：K 的 RoPE 位置错位基本已经被 rerotate 解决了。继续在位置编码上做文章，收益和新意都不大。

下一步应该研究的是：

在位置已修复的前提下，剩下的 gap 到底来自哪里，以及是否能用少量重算修掉。

### 先做 token-level reuse gap

目标不是看平均 CKSim，而是定位：

reuse_vs_base 下，哪些 skill token / 哪些 layer 还差？

现在 layer 平均是：

key   ≈ 0.963
value ≈ 0.951

但里面肯定有局部低点。先写脚本读取：

cksim-reuse-*.pt
cksim-base-*.pt

输出：

skill, layer, token_idx, token_text, key_cksim, value_cksim

这一步回答：

剩余误差是少数 token 问题，还是整段普遍问题？

如果只是少数 token 差，那才有必要做类似论文 4.3 的 selective correction。


### 再看 attention，但不要只看 attention

你说 cross-attention 应该看 attention score，这个方向对。但 attention score 不是最终指标。

原因是：

attention score 高：
说明 token 确实看了 history / dynamic segment

但它不必然说明：
这个 token 的 KV drift 大
或这个 token 对最终输出重要

所以 attention 应该用来解释和选 token，CKSim 用来衡量修复效果。

我们应该看：

对每个 skill token:
full-context 下 attention mass to history 有多大？
reuse_vs_base 的 KV drift 有多大？

然后做相关性：

attention_mass_to_history 高的 token，CKSim 是否更低？

如果相关性强，attention score 可以作为非-oracle token selector。


### 做 oracle top-k correction 上限实验

这一步很关键，而且不需要先改 vLLM。

已有：

reuse KV = rope 修正后的缓存 KV
base KV  = full-context recompute KV

离线模拟：

选择 drift 最大的 top-k token
把这些 token 的 reuse KV 替换成 base KV
其他 token 保持 reuse KV

扫：

top-k = 1%, 5%, 10%, 20%, 30%

看 corrected_vs_base CKSim 能恢复多少。

这一步回答：

如果只修少量 token，理论上能不能解决剩余 gap？

如果 oracle top-k 效果都不好，那 WCA 这条路不值得继续。


### 再从 oracle 过渡到可实现策略

oracle 用了 base KV，真实在线拿不到。所以如果 oracle 有效，再找可实现的 token selection：

候选策略：

attention_mass_to_history top-k
layer-1 drift top-k
boundary token top-k
section heading / XML tag token top-k
固定层的固定比例 top-k

然后比较它们和 oracle top-k 的重合度，以及 correction 后 CKSim。


### 最后才考虑实现 Weighted Correction Attention

如果前面证明：

top-k token correction 有明显收益

再做真正系统实现。否则不要急着改 vLLM。

实现方向可以是：

K 用 rope rerotate
对 selected token 做上下文重算
把 selected token 的 recompute KV 与 reuse KV 融合或替换

先 hard replace，再试 weighted fusion：

corrected = alpha * recompute + (1 - alpha) * reuse

看 alpha=1 是否已经够好。如果够，就不需要复杂 weighted fusion。

## 当前脚本

### trace replay CKSim

真实 trace 上的 recompute-vs-reuse CKSim 已移动到 `../replay/`：

```bash
bash scripts/05_context_segment_agent_kv/replay/run_cksim.sh
```

### token_level_reuse_gap.py

读取：

cksim-reuse-*.pt
cksim-base-*.pt

输出每个 skill / layer / token 的 K/V CKSim，用来定位 rope 修复后剩余 gap 集中在哪里。

运行：

python scripts/05_context_segment_agent_kv/CKSim/token_level_reuse_gap.py

输出：

results/05_context_segment_agent_kv/CKSim/token_level_reuse_gap.csv
results/05_context_segment_agent_kv/CKSim/token_level_reuse_gap_summary.json

### oracle_topk_correction.py

读取：

cksim-reuse-*.pt
cksim-base-*.pt

用 base KV 作为 oracle，选择 drift 最大的 top-k token，离线模拟替换/融合后的 corrected KV。

运行：

python scripts/05_context_segment_agent_kv/CKSim/oracle_topk_correction.py

输出：

results/05_context_segment_agent_kv/CKSim/oracle_topk_correction.csv
results/05_context_segment_agent_kv/CKSim/oracle_topk_correction_summary.json
