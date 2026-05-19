# Experiment Platform Background

The platform is used to run batches of agent benchmark tasks.
Each task is launched by a scheduler and writes outputs to a results directory when finished.

Current failure handling:
- If a task crashes or a tool call fails, the task is marked as failed
- A user must manually relaunch the task
- There is no automatic retry mechanism
- A rerun usually creates a new run record, and linkage to the original failed run is weak

Known issues:
1. Temporary network hiccups can cause otherwise recoverable tasks to fail
2. Some tool calls occasionally time out, but succeed when rerun manually
3. Fully automatic unlimited retries would waste resources and may hide real bugs
