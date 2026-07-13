# CSKCache

CSKCache is the package-side migration target for Segmentia context-skill KV
cache reuse. It provides a vLLM v1 KV connector, explicit request reuse/save
signals, durable local storage, and paged-KV slot helpers.

## Current Scope

Implemented:

- `CSKCacheConnectorV1` vLLM connector entrypoint.
- Explicit request-local reuse signals; production requests are not discovered
  by scanning the prompt.
- Single- and multi-entry scheduler-side load planning. Multiple non-overlapping
  spans in one request are loaded in target-token order, with normal prefill for
  the gaps between them.
- Worker-side KV scatter from loaded `.pt` entries into vLLM paged KV cache.
- Offline save signals that collect a canonical source span and persist its
  per-layer K/V tensors and sidecar metadata.
- Direct reuse path and RoPE key correction for reused spans whose target
  positions differ from their source positions.
- Experimental probe-gated reuse path. When enabled, CSKCache asks the vLLM
  scheduler to prefill a short probe prefix of the matched segment, compares
  the recomputed probe KV with RoPE-shifted offline KV, then either loads the
  remaining segment or computes an anchor prefix before loading the tail.

Deferred:

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

## Request Reuse Signals

The legacy single-entry request remains supported:

```json
{
  "kv_transfer_params": {
    "cskcache": {
      "operation": "reuse",
      "cache_id": "doc-coauthoring",
      "target_start": 17735,
      "target_end": 21035
    }
  }
}
```

One request can explicitly reuse multiple entries:

```json
{
  "kv_transfer_params": {
    "cskcache": {
      "operation": "reuse",
      "entries": [
        {
          "cache_id": "doc-coauthoring",
          "target_start": 17735,
          "target_end": 21035
        },
        {
          "cache_id": "theme-factory",
          "target_start": 21039,
          "target_end": 21696
        }
      ]
    }
  }
}
```

Entries are sorted by `target_start` and must be in bounds, non-overlapping,
and the same length as their offline cache entries. `cache_id` and `entries`
cannot be supplied together. The engine trusts explicit placement metadata and
does not compare the target prompt slice with cached token IDs.

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
cap_prefill_before_reuse(request, base_num_computed_tokens, num_new_tokens)
get_boundary_reuse_load_tokens(request, num_computed_tokens)
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
