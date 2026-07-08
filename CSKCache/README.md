# CSKCache

CSKCache is the package-side migration target for Segmentia context-skill KV
cache reuse. The first version is intentionally small: it provides a vLLM v1
KV connector, a registry-derived token matcher, and paged-KV slot helpers.

## Current Scope

Implemented:

- `CSKCacheConnectorV1` vLLM connector entrypoint.
- Exact token matching against loaded KV registry entries for segment occurrence
  discovery.
- Scheduler-side load planning when `num_computed_tokens == segment_start`.
- Worker-side KV scatter from loaded `.pt` entries into vLLM paged KV cache.
- Direct reuse path and RoPE key correction for reused spans whose target
  positions differ from their source positions.
- Experimental probe-gated reuse path. When enabled, CSKCache asks the vLLM
  scheduler to prefill a short probe prefix of the matched segment, compares
  the recomputed probe KV with RoPE-shifted offline KV, then either loads the
  remaining segment or computes an anchor prefix before loading the tail.

Deferred:

- Prompt-builder metadata path. TODO(B): upstream prompt metadata should become
  the primary segment discovery source, with token matching used as validation.
- Save path for collecting canonical segment KV through CSKCache.
- End-to-end parity validation on real vLLM model runs.
- Dedicated internal mini-forward and attention-inline fusion paths are not
  implemented. The current gate uses normal vLLM prefill for the probe/anchor
  tokens.

## Cache Entry Metadata

The first version discovers reusable segments from loaded cache entries. Each
`<cache_id>.pt` file carries both the cached KV tensors and the token sequence
that identifies the segment:

```python
{
    "cache_id": str,
    "source_start": int,
    "source_end": int,
    "token_ids": list[int],
    "kv_by_layer": {
        layer_name: (key_tensor, value_tensor),
    },
}
```

`SegmentCatalog` is derived in memory from these registry entries. There is no
separate external catalog file in the CSKCache core path.

## vLLM Loading

The connector is designed to be loaded dynamically, without registering it in
vLLM's connector factory:

```bash
PYTHONPATH=/home/wsh/openhands_code_research/CSKCache:/home/wsh/vllm:$PYTHONPATH \
vllm serve ... \
  --kv-transfer-config '{
    "kv_connector": "CSKCacheConnectorV1",
    "kv_connector_module_path": "cskcache.integration.vllm.v1_connector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "cskcache.kv_dir": "/path/to/kv_dir"
    }
  }'
```

Environment variable alternatives:

```text
CSKCACHE_KV_DIR=/path/to/kv_dir
```

Optional probe-gated reuse settings:

```json
{
  "cskcache.probe_enabled": true,
  "cskcache.probe_tokens": 4,
  "cskcache.anchor_tokens": 32,
  "cskcache.probe_tau": 0.15,
  "cskcache.gate_metric": "max"
}
```

Probe-gated reuse requires the local vLLM scheduler hook that exposes two
duck-typed connector methods:

```text
cap_num_new_tokens(request, base_num_computed_tokens, num_new_tokens)
get_inprocess_load_tokens(request, num_computed_tokens)
```

These hooks let CSKCache stop chunked prefill at the segment start/probe/anchor
boundaries and schedule an in-process KV splice for the reused tail.

## Old vLLM Patch Status

The old `/home/wsh/vllm/vllm/v1/context_segment_cache/` implementation is kept
as experimental reference code for registry, slot ops, token identity, replay
parity checks, and later CSKCache migration work.

The old vLLM main-path hooks have been commented out with `# context_segment_cache:`
markers in request parsing, scheduler, scheduler output metadata, worker model
runners, Qwen3 function-vector capture, flash-attn attention probe, and the
`kv_cache_manager.py` cache-hit truncation path. This keeps the archived module
available for reference while preventing normal vLLM execution from using the
old Segmentia patch path.

Migration direction:

```text
old context_segment_cache:
  archived reference implementation for mechanism experiments

CSKCache:
  long-term package boundary for segment discovery, cache entry selection,
  connector metadata, and KV load/save

vLLM:
  keep only low-level generic execution hooks needed by CSKCache
```
