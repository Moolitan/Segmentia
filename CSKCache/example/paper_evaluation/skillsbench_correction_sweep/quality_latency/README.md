# SkillsBench correction quality--latency sweep

This experiment measures how CSKCache calibration ratio changes the generated
thinking text and the correction compute time. It uses the verified v2
SkillsBench offline pool and does not run Docker, BenchFlow, or a complete
OpenHands trajectory.

## Frozen experiment

Each selected workload is a public single-Skill SkillsBench task. The core set
contains six unique Skills stratified by Context Segment length and task type;
the optional extension adds six more. Every task runs these six isolated arms:

```text
Full, Ratio-1%, Ratio-3%, Ratio-5%, Ratio-7%, Ratio-10%
```

The service lifecycle is exactly `(task, arm)`. Each arm starts a fresh vLLM,
sends exactly two requests, and then stops the server:

1. Request A sends the unmodified `task.md`, exposes only the target Skill, and
   is forced to call the suite's `skill` tool. CSKCache uses this call to
   prefetch the exact offline object. Thinking is disabled for this request.
2. Native vLLM prefix cache is cleared without clearing the CSK external state.
3. Request B contains the same task, the forced tool call, and the exact
   `SKILL.md` tool result. Full performs ordinary prefill; ratio arms reuse the
   offline KV with ratio-prefix calibration. Thinking is enabled, decoding is
   greedy with seed 0, and output is capped at 384 tokens.
4. The server is stopped even when the request fails.

The exact offline object is selected by
`(task_id, skill_name, skill_version)`. A one-object Catalog view is written for
every attempt, so same-name multi-version Skills cannot be selected
ambiguously. The online raw backend uses the v2 pool geometry: 40 MiB slots and
256 MiB durable metadata.

## Metrics

For task `t`, Full request-B thinking is the reference `R_t`; ratio request-B
thinking is `C_t,r`. The quality diagnostic is word-level, case-folded
ROUGE-L Recall:

```text
LCS(tokens(R_t), tokens(C_t,r)) / len(tokens(R_t))
```

Thinking is read from `reasoning_content`, then `reasoning`, then an explicit
`<think>` block. A missing thinking field is an invalid sample; final-answer
content is never silently treated as thinking. This metric measures textual
fidelity to Full prefill, not task correctness.

The primary latency is frozen to layer 8:

```text
calibration_forward_ms[8] + residual_correction_ms[8]
```

Layer 8 was selected before this sweep from nine existing calibration
microprofiles (16/32/128 calibration tokens, three repetitions each). Its
combined compute range was 4.18% of the mean, the smallest across the 40 model
layers. H2D and KV commit time are excluded from the primary value; the raw
components and whole-layer `gpu_ms` are retained.

## Validate and run

The plan command validates the frozen SkillsBench commit, manifest, Catalog
digest, exact source paths, one-Skill attribution, object identity, token
counts, and 40 layer extents. It does not start vLLM:

```bash
cd /home/wsh/openhands_code_research
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh --plan
```

Run the six-arm smoke task first. These completed cases remain in the active
run and are not repeated by the core command:

```bash
SKILLSBENCH_SWEEP_LIMIT=smoke \
  bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh
```

If the smoke gate passes, run the remaining core tasks. Normally the active run
is selected automatically. If derived parser code was repaired after the smoke,
name the recovered run explicitly; the launcher verifies that its scientific
configuration is unchanged and skips its completed cases:

```bash
SKILLSBENCH_SWEEP_RESUME_DIR=/absolute/path/to/recovered/run-id \
  SKILLSBENCH_SWEEP_LIMIT=core \
  bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh
```

If the six-task curve is inconclusive, resume the same active run with the six
extension tasks:

```bash
SKILLSBENCH_SWEEP_RESUME_DIR=/absolute/path/to/recovered/run-id \
  SKILLSBENCH_SWEEP_LIMIT=all \
  bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh
```

`smoke` and `core` leave the run active so the next command resumes it when the
code fingerprint is unchanged. After artifact recovery across a parser-code
change, keep passing the explicit resume directory. To stop permanently after
the core subset, add `SKILLSBENCH_SWEEP_FINALIZE=1`; a later extension would
then correctly be rejected for that finalized run.

Before a real run, the launcher rejects a configured GPU with at least 500 MiB
already allocated. Failed cases keep `attempt-NNN/error.json`; rerunning the
same command creates a new attempt and never overwrites the old one. An
executed request pair with no valid reuse is retained as `status=invalid` and
is not retried automatically.

## Outputs

Run artifacts are written below:

```text
/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/paper_evaluation/
  skillsbench_correction_quality_latency/<run-id>/
```

Each attempt keeps the exact Catalog view, vLLM log and timeline, CSK profile,
raw response, visible content, thinking text, and normalized sample. Run-level
outputs include:

```text
samples.csv / samples.jsonl
paired_samples.csv
summary.csv
analysis.md
ratio_quality_latency.png / .pdf
quality_latency_curve.png / .pdf
```

`ratio_quality_latency` is a dual-axis per-task plot. Calibration ratio is on
the x-axis, solid lines use the left Layer-8 latency axis, and dashed lines use
the right ROUGE-L Recall axis. Each task keeps one color and marker across both
metrics; the task and line-style legends are placed above the plotting area.

Rebuild the tables and figures without rerunning the model:

```bash
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh \
  --analyze /absolute/path/to/run-id
```

If a completed request pair was marked invalid by an older profile parser,
reparse and replace only the derived `sample.json`, tables, and figures:

```bash
bash CSKCache/example/paper_evaluation/skillsbench_correction_sweep/quality_latency/run.sh \
  --recover /absolute/path/to/run-id
```

Recovery preserves response JSON, thinking text, vLLM logs, timelines, CSK
profiles, and attempt directories, and writes an explicit `recovery.json`.
An explicit resumed execution additionally appends `resume_history.jsonl`,
including the original run fingerprint, current code fingerprint, and the
scientific-configuration equality check.
