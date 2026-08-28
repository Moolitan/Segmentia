# Section 6.2: SkillsBench end-to-end quality

This directory evaluates whether OpenHands can use a public Skill and complete
the corresponding task. It does not modify the older keyword-based
`section_6_2_correction_quality` experiment.

The frozen pilot runs the public SkillsBench task
`llm-prefix-cache-replay` with OpenHands and the `with-skill` posture. The
primary quality value is the task's deterministic `rewards.reward`; only
`reward == 1.0` counts as task success. Thinking-text similarity is not used as
a correctness metric.

The initial `Full` arm routes `vllm/Qwen3-14B` to the local OpenAI-compatible
server and uses ordinary prefill with no CSKCache or LMCache KV connector.
Native vLLM prefix caching remains enabled so normal multi-turn OpenHands
prefixes behave as they do in the rest of the paper suite.

## One-time BenchFlow setup

The SkillsBench lockfile pins the compatible BenchFlow environment:

```bash
uv sync \
  --project /mnt/Large_Language_Model_Lab_1/wsh/CSKCache/workload/skillsbench \
  --locked --no-dev
```

Docker Compose v2 or newer must be available. On this machine the verified
user-level plugin is `/home/wsh/.docker/cli-plugins/docker-compose` (v5.4.0).
The first Docker rollout also downloads and installs BenchFlow's pinned
OpenHands CLI in the sandbox.

The host proxy is bound to loopback, as is the local vLLM endpoint. The runner
therefore copies each immutable public task into the run directory and adds
only a `network_mode: host` Compose override. A run-scoped Docker config carries
the HTTP/HTTPS proxy; the global Docker config and SkillsBench checkout are not
modified.

## Run

```bash
bash run.sh
```

The launcher accepts no arguments. Edit `config.py` to change the frozen task
set or platform. Results are written below:

```text
/mnt/Large_Language_Model_Lab_1/wsh/CSKCache/output/paper_evaluation/
  section_6_2_skillsbench_quality/<run-id>/
```

Each case keeps the unmodified BenchFlow rollout, ACP/LLM trajectories,
verifier output, vLLM timeline, parsed quality record, and logs. Failed reruns
use a new `attempt-NNN` directory rather than overwriting evidence.

The runner accepts a case only when model traffic and evaluation are real:

- `rewards.reward` is not null;
- `n_tool_calls > 0`;
- `n_skill_invocations > 0`;
- `include_task_skills` is true;
- the structured ACP trajectory records an invocation of the required task
  Skill, rather than only a distractor or generic built-in Skill;
- `agent_result.total_tokens > 0`;
- the local vLLM timeline contains a completed tagged request, or the vLLM
  access log contains a successful chat-completion request;
- neither the agent nor verifier reports an error.

A healthy run with reward zero is a valid failed task, not a broken pipeline.

If a completed rollout was rejected only by an older derived-metric parser,
reparse it without rerunning the model:

```bash
python recover.py /absolute/path/to/run
```

The recovery command never changes the canonical BenchFlow result, ACP/LLM
trajectory, verifier output, or vLLM log. It rebuilds normalized samples and
run-level analysis from those raw artifacts and records `recovery.json`.

## Full-prefill pilot result

The completed pilot is `20260826-171924`. Its pipeline was healthy, but the task
failed:

- oracle reward: `1.0`;
- OpenHands reward: `0.0` (`task_success=false`);
- tool calls: `5`;
- required `prefix-cache-replay` Skill invocations: `1`;
- provider-accounted tokens: `114017`;
- successful local vLLM chat requests: `7`;
- agent/verifier errors: none.

The trajectory shows that the model invoked the correct Skill, but then assumed
the Skill shipped an executable `/skills/prefix-cache-replay/replay.py`. The
public Skill contains guidance in `SKILL.md`, not that script, so the agent did
not implement the replay and never produced the required `report.json`. This is
a genuine end-to-end quality failure, not a harness failure.
