# SkillsBench CSKCache correction sweep

This directory contains two stages over the frozen public SkillsBench checkout:

- `offline_cache/`: build and verify the 202 unique Skill-body KV objects.
- `quality_latency/`: compare Full-prefill thinking with 1%, 3%, 5%, 7%, and
  10% calibration while measuring fixed-layer correction compute time.

See each subdirectory README for its exact state boundaries and commands.
