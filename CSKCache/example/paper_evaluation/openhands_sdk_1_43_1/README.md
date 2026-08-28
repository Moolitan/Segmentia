# OpenHands current vs SDK 1.43.1 quality comparison

This experiment compares the current SkillsBench OpenHands harness against a
new adapter built directly from `openhands-sdk==1.43.1` and
`openhands-tools==1.43.1`. The latest-SDK arm does not install or import the
legacy OpenHands CLI.

Both arms receive the same public SkillsBench `task.md`, task Skills, verifier,
and local `vllm/Qwen3-14B` endpoint. BenchFlow remains the owner of the raw ACP
trajectory, usage record, sandbox, and SkillsBench verifier. Task reward is the
primary quality metric. Required-Skill invocation is reported separately and is
not treated as proof that the task was solved.

## Reproducibility boundary

The latest arm is frozen by `image.lock.json`:

- SDK and Tools are both exactly `1.43.1`;
- `agent-client-protocol` is exactly `0.10.1`;
- the base is `python:3.12-slim` at an immutable registry digest;
- the final local image ID and adapter-source SHA-256 must match the lock;
- staged latest-SDK task Dockerfiles use the final image ID as their `FROM`.

The current control retains BenchFlow 0.6.3's existing pinned OpenHands CLI
commit and SDK/Tools 1.22.1. The original SkillsBench checkout is never edited.

## Gates and execution

Before any model trial, the runner requires all three smoke gates:

1. exact package versions and image fingerprint;
2. the real `InvokeSkillExecutor` returns the exact complete rendered Skill;
3. the real `TerminalExecutor` writes an artifact that a separate verifier
   process reads and validates.

The configured loop is `repetition -> workload -> harness`. Every
`(harness, task, repetition)` starts a new vLLM process and a new BenchFlow
container. A case never inherits prefix cache, workspace, or container state
from another case. Re-running a failed case creates the next `attempt-NNN`
directory and leaves the old attempt untouched.

Run:

```bash
bash run.sh
```

The frozen pilot runs the same `llm-prefix-cache-replay` workload three times
per harness. Outputs are written below
`/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/paper_evaluation/openhands_sdk_1_43_1/`.
Each attempt preserves BenchFlow's original result, ACP/LLM trajectories,
usage, verifier output, vLLM log, and the derived parser record.
