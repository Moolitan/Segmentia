# Segmentia

Segmentia contains experiments and scripts for KV cache reuse, context segment
cache injection, and software-agent workflows.

This repository uses Git submodules for the larger codebases that the
experiments depend on:

- `software-agent-sdk` -> `https://github.com/Moolitan/software-agent-sdk.git`
- `vllm` -> `https://github.com/Moolitan/vllm.git`

## Clone

Clone this repository with submodules:

```bash
git clone --recursive https://github.com/Moolitan/Segmentia.git
```

If you already cloned the repository without `--recursive`, initialize and fetch
the submodules with:

```bash
git submodule update --init --recursive
```

To update submodules later to the commits recorded by this repository:

```bash
git submodule update --recursive
```

## Repository Layout

- `scripts/04_kv_cache_research/`: KV cache reuse experiments and analysis.
- `scripts/05_context_segment_agent_kv/`: ContextSegmentKV agent integration
  scripts.
- `software-agent-sdk/`: software agent SDK submodule.
- `vllm/`: vLLM submodule with ContextSegmentKV support.
