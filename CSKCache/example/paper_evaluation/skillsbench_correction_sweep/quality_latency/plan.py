"""Print the frozen execution matrix without starting vLLM."""

from __future__ import annotations

import math

from . import config as local
from .run import _execution_limit, _selected_workloads
from .workload import load_workloads


def main() -> None:
    workloads, _ = load_workloads(
        workload_path=local.WORKLOAD_FILE,
        skillsbench_root=local.SKILLSBENCH_ROOT,
        manifest_path=local.MANIFEST_PATH,
        catalog_path=local.MASTER_CATALOG_PATH,
        expected_commit=local.SKILLSBENCH_COMMIT,
        expected_catalog_sha256=local.EXPECTED_CATALOG_SHA256,
    )
    execution_limit = _execution_limit()
    selected = _selected_workloads(workloads, execution_limit)
    ratios = [
        float(variant.calibration_ratio)
        for variant in local.SYSTEMS
        if variant.calibration_ratio is not None
    ]
    print(
        f"limit={execution_limit} tasks={len(selected)} systems={len(local.SYSTEMS)} "
        f"vllm_lifecycles={len(selected) * len(local.SYSTEMS)} "
        f"requests={2 * len(selected) * len(local.SYSTEMS)}"
    )
    print("task\ttier\tskill\tskill_tokens\tcalibration_tokens_1_3_5_7_10")
    for workload in selected:
        budgets = ",".join(
            str(max(1, math.ceil(workload.skill_tokens * ratio)))
            for ratio in ratios
        )
        print(
            f"{workload.task_id}\t{workload.tier}\t{workload.skill_name}\t"
            f"{workload.skill_tokens}\t{budgets}"
        )


if __name__ == "__main__":
    main()
