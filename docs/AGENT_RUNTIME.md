# AGENT_RUNTIME

## Runtime flow

1. `retrieve_segments`：读取稀疏、Qwen、SigLIP、scorer、reranker 融合候选。
2. `build_context`：保留 segment ID、时间戳和证据文本，按预算压缩。
3. `assess_evidence`：判断是否足以回答，必要时拒答。
4. `search_memory`：读取持久化片段记忆。
5. `synthesize_answer`：使用服务端固定 Qwen3.5 和已准入 adapter 生成回答。
6. `verify_answer`：先做时间戳与 claim-support 硬规则，再运行可选 calibrated veto。

每个工具调用记录 schema 输入、输出 envelope、耗时、attempt、错误码和 circuit state。默认稳定路径是受限 Plan-Execute，不是无限 ReAct。

## Safety and failure handling

- 硬时间戳/claim-support 失败不可被学习式 verifier 覆盖。
- calibrated verifier 只消费可审计标量特征，只能否决硬规则已通过的答案。
- 工具超时进入有限重试；达到阈值后打开 circuit，并返回用户可见的证据不足状态。
- GPU 请求由 Web 单 worker 串行执行；任务阶段和结果写入项目内 job store。

## Current limits

- memory 仍是 JSONL + 词法检索，不是向量数据库服务。
- 工具 registry 是本地 Python schema，不是 MCP server。
- speech ASR 未部署；只有字幕 sidecar 对齐接口。
- verifier 不是视觉 NLI；ReAct 仍不是默认演示路径。
