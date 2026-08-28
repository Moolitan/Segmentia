# SkillsBench offline CSKCache pool

This launcher scans the frozen SkillsBench checkout, collapses repeated exact
Skill bodies, and exact-saves all 202 `(skill_name, skill_version)` objects into
one dedicated raw-block pool. It starts one local vLLM service, processes every
object sequentially, stops the service on normal exit or interruption, and then
verifies every Catalog entry against the source and tokenizer.

The dedicated pool is:

```text
/mnt/990_pro/skill_save_pool/Qwen3-14B-SkillsBench-9a1f4dd5-v2/
```

Inventory only (no GPU service and no writes):

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/offline_cache/run.sh --plan
```

Build and verify:

```bash
cd /home/wsh/openhands_code_research
conda activate opencode
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/offline_cache/run.sh \
  2>&1 | tee /tmp/skillsbench_cskcache_offline.log
```

Re-run verification without starting vLLM:

```bash
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/offline_cache/run.sh --verify
```

Successful completion prints `[verified] objects=202` and writes
`skillsbench_manifest.json` beside the raw Catalog. A failed or interrupted run
keeps completed persistent data; running the same build command again resumes
by skipping exact versions already present.

The earlier pool without the `-v2` suffix is a preserved failed attempt. Its
64-MiB mirrored metadata area could checkpoint only 4,160 of the required 8,080
layer keys. The v2 pool reserves 256 MiB for metadata and fails an exact-save
request immediately if a durable checkpoint cannot be written.
