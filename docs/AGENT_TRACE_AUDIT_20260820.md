# Agent Trace 审计（2026-08-20 建档，2026-08-23 刷新）

这份审计把成功链路、durable job 持久化和失败降级放在同一套证据里。机器证据来自 canonical `metadata.agent_run`、`outputs/runs/latest/jobs/f6d582f2ef434a689845b23e8c671f07.json`、`outputs/reports/browser_e2e.json` 和 `outputs/reports/agent_failure_recovery.json`。

## 真实成功链路

- 任务：`f6d582f2ef434a689845b23e8c671f07`；真实上传指定视频后，问题要求概括开场、分国家试喝和最后盲测，并给出时间戳。
- 阶段：`queued → checking_resources → loading_models → analyzing → exporting → completed`，终态耗时 `29.2s`，实际执行跨度 `29.186s`。
- 工具顺序：`retrieve_segments → build_context → assess_evidence → search_memory → synthesize_answer → verify_answer`。
- 六次工具调用均记录 `call_id`、开始/结束时间、attempt、latency、error code、circuit state 和逐次重试明细。
- 上下文预算 `3200` 字符，实际使用 `714`，四个证据窗口为 `seg-0000/0010/0015/0020`。
- 回答时间窗全部绑定：`0-20`、`200-220`、`300-320`、`400-416.2`；claim-support coverage 为 `1.0`。
- calibrated verifier 概率 `0.92716`，阈值 `0.2`。它只在硬规则通过后做保守否决，不能补证据或覆盖硬失败。

canonical trace 中 `synthesize_answer` 为 `17.329s`，其他五个编排工具合计约 `0.006s`。这说明当前主要延迟在模型生成而非 Python 编排层；后续优化应优先验证生成缓存、批处理或 student 模型，而不是用微小的调度优化夸大收益。

## 持久化状态

E2E 完成时任务记录为 `durable=true, restored=false`，原子文件位于 `outputs/runs/latest/jobs`。代码与测试覆盖受管服务重启后的恢复路径，但本轮机器报告没有把该 job 包装成已发生 `restored=true` 的验收事实。

## 失败恢复

受控案例模拟 SigLIP 检索超时：两次工具调用各执行有限重试，随后熔断器打开；第三次调用以 `circuit_open` 在进入 handler 前被阻止。最终动作是返回用户可见的证据不足状态，不生成未经证据支持的答案。

这证明的是已定义工具故障具备可观测、有限、非幻觉式降级，不代表所有外部故障都已解决。

## 双端一致性

最终清单包含 60 个核心制品和 20 份不可变 adapter 准入历史；manifest hash mismatch 为 0，关键产物双端 SHA-256 一致；delivery `40/40`、interview `17/17`。
