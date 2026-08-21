# Harness Evaluation

VideoTrace 现在不仅评最终知识包，也评一次完整 Agent 运行。

## 评什么

- `plan_coverage`：计划中的关键工具有没有被执行。
- `tool_success_rate`：工具调用是否成功。
- `step_order_score`：工具调用顺序是否合理。
- `evidence_binding_score`：回答是否真正绑定了时间戳证据。
- `answer_structure_score`：回答是否有 `问题 / 结论 / 证据` 结构。
- `verification_score`：最终回答是否通过校验。
- `trace_completeness`：计划和实际 trace 是否一致。

## 为什么有用

这个分数能直接反映：

- Agent 不是只会“说”，而是会“找证据再说”。
- 整个执行过程是否稳定。
- 后续换更强的模型、记忆、或工具路由时，能不能公平比较。

## 命令入口

- `python scripts/eval_pack.py <video>`

输出里会同时有：

- `pack`：retrieval / timeline / evidence-window 维度
- `harness`：Agent 执行链路维度
