# Section 6.4: logical Chunk granularity

This experiment derives 64/128/256/512-token logical Catalogs that reference
the same packed KV payload. It measures whole-Skill versus longest-contiguous-
Chunk invalidation and then measures TTFT for the real CSKCache and native
CacheBlend paths. No KV payload is duplicated by the Catalog derivation.

```bash
bash run.sh
```
