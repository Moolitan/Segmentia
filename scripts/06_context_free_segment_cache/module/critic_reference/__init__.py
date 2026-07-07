"""通用判官（general judge）：读「任务要求 + agent 产物」→ LLM → 成功概率 + 裁决。

不用一任务一张手写检查表：判官自己从任务要求里抽成功标准，对任意任务通用。
评判规则借鉴 Claude Code 的 verification agent（对抗心态 + 证据纪律 + PASS/FAIL/PARTIAL），
问题标签沿用 OpenHands critic 的 taxonomy 子集。详见 README.md。

主要入口：
    from judge import GeneralJudge
    from schema import JudgeResult
    r = GeneralJudge().evaluate(task_request, agent_output, tools=...)
"""
