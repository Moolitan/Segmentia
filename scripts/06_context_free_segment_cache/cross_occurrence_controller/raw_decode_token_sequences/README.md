# Raw Decode Token Sequences

## 目的

从头 replay Segmentia 的全部 task-skill occurrence，只查看模型真实生成的
token序列。该实验不做行为正确性、boundary、attention或Function Vector分析。

覆盖范围：

```text
12 task-skill pairs
× occurrence 1/2/3
× recompute/rope
= 72 raw sequences
```

## “Raw”的定义

脚本只读取：

```text
choices[0].logprobs.content[*].token
```

每个TXT由这些token按生成顺序直接拼接。脚本不读取或保存OpenAI chat parser
生成的：

```text
message.reasoning_content
message.content
message.tool_calls
```

因此工具调用TXT中可以直接看到模型生成的：

```text
<tool_call>
{"name": "...", "arguments": ...}
</tool_call>
```

纯文本TXT中则不会因为API返回`tool_calls=[]`而虚构任何token。

## 输出

```text
results/problem_exploration/raw_decode_token_sequences/
  rendered_prompts/
    recompute/*.txt
    rope/*.txt
  sequences/
    recompute/*.txt
    rope/*.txt
  data/
    sequence_manifest.jsonl
```

文件名以`invocation_index`开头，方便按真实trace顺序浏览。Manifest只保存
task、skill、occurrence、mode、token数量、文件路径和完整性hash，不保存
API解析后的assistant message。

`rendered_prompts/`是离线调试输出。`run_send_request_message.py`读取全部36个
唯一case，使用本地Qwen3 tokenizer的`apply_chat_template()`生成
`tokenize=False`文本，并按recompute/rope分别写出共72个TXT。它传入与decode
相同的system prompt、tools、`add_generation_prompt=True`和
`enable_thinking=True`，但不启动vLLM、不发送HTTP请求、不执行推理。

当前recompute和rope使用完全相同的messages、tools和chat template；rope差异
只存在于tokenization后的KV注入配置。因此两组rendered prompt理论上逐字
相同，分别导出是为了显式检查这一边界。

## Replay边界

```text
服务重启边界 = (mode, task)
task内按invocation_index顺序
occurrence 1/2/3全部是需要保存的正式case
后续occurrence通过prefix cache继承历史注入
当前轮只显式注入当前skill span
```

## 运行

只导出Qwen3 chat-template文本：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
python scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/run_send_request_message.py
```

运行完整raw decode：

```bash
conda activate opencode
cd /home/wsh/openhands_code_research
bash scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/run_raw_decode_token_sequences.sh
```

这是长时间vLLM实验，默认由用户手动运行。默认`CLEAN_OUTPUT=1`，会清理这个新
结果目录中的旧TXT和manifest后从头执行，不影响其他实验。

断点续跑：

```bash
CLEAN_OUTPUT=0 RESUME=1 \
bash scripts/06_context_free_segment_cache/cross_occurrence_controller/raw_decode_token_sequences/run_raw_decode_token_sequences.sh
```

断点续跑仍会按顺序发出已有case请求以重建prefix-cache状态。若同一case的新
raw sequence与已保存TXT不同，脚本立即失败，不覆盖旧结果。
