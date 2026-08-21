# Agent Harness

`AgentHarness` 是 VideoTrace 的任务编排层，位置在 `src/videomemo/agent/harness.py`。

它解决的问题不是“模型会不会看视频”，而是把长视频问答拆成可控、可追踪、可验证的流程。

## 职责边界

- 注册工具：把检索、上下文压缩、记忆检索、回答生成、证据校验都接入统一接口。
- 执行计划：默认使用 Plan-Execute，保留 ReAct-like 执行模式。
- 控制上下文：只把压缩后的证据窗口交给生成模型，避免直接塞整段长视频。
- 收集轨迹：记录每个工具的输入、输出 envelope、唯一 `call_id`、开始/结束时间、attempt、耗时、错误码、熔断状态和逐次重试明细。
- 输出守卫：要求最终回答必须引用 `timestamp=start-end` 证据。

## 当前工具链

1. `retrieve_segments`
   - 输入用户问题。
   - 返回已经融合文本检索、片段 scorer、视觉相关性的候选片段。

2. `build_context`
   - 输入候选片段。
   - 输出带 `segment_id`、时间戳、分数、证据文本的上下文窗口。

3. `search_memory`
   - 从当前视频记忆和可选跨视频记忆中检索补充事实。
   - 记忆只能辅助，不能替代当前视频证据。

4. `synthesize_answer`
   - 调用模板模型、OpenAI-compatible 模型或 Qwen 多模态 API。
   - 输出中文结构化回答。

5. `verify_answer`
   - 检查回答是否包含时间戳证据。
   - 输出 `verified`、`coverage`、`matched_evidence` 和失败原因。

## 面试表达

可以这样讲：

> 单个 VLM 可以直接回答短视频问题，但长视频场景需要先检索证据、压缩上下文、调用生成模型、再做证据校验。我的项目把这些能力抽成 Agent Harness，每一步都有 trace 和可验证输出，所以它不是简单调用大模型，而是一个可调试、可评测、可扩展的长视频理解系统。

## 当前可验证证据

- harness-level evaluation 已进入 `agent_metrics.py` 与交付校验，覆盖回答验证、证据引用、上下文保留、记忆命中和工具成功率。
- 工具层已实现有限重试、线性 backoff、schema 输入/输出校验和按工具熔断；失败案例会返回证据不足状态，不伪造回答。
- Web 技术详情会展示每步状态、attempt、延迟、重试摘要和 verifier 结果；完整成功/失败双轨证据见 `docs/AGENT_TRACE_AUDIT_20260820.md`。
- 默认仍是固定 Plan-Execute。当前没有把开放式 router 或无限 ReAct 包装成已完成能力。

## 由真实需求触发的扩展

- 有足够多类问题和多轨迹监督后，再训练或评估 tool router。
- 只有 verifier 能区分“可修复生成错误”和“证据本身不足”后，才增加自动 rewrite，避免循环重写同一份坏证据。
- skill abstraction 仅在出现多个稳定复用任务时引入，避免为面试关键词增加空壳层。
