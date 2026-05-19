# Experiment Logger

Experiment Logger is a lightweight internal tool used by a research group to record experiment runs.

Each record currently contains:
- experiment name
- model/version
- dataset
- parameter configuration
- result summary
- notes

Current problems:
1. Users must manually enter category tags such as "ablation", "debug", and "final-run"
2. Different people use inconsistent tagging styles, which makes later retrieval difficult
3. Many historical experiments have no tags at all, making review and comparison time-consuming

Feature idea:
Add an "auto-tagging" feature that recommends 1 to 3 tags based on the experiment title, notes,
parameters, and result summary, then lets the user confirm or edit them.
