# VideoTrace Interview Readiness Report

VideoTrace 已形成一条可现场演示、可远端复现、可用 hash 交叉验证的长视频多模态 Agent 产品链路。动态权威证据以 `artifact_manifest.json` 和 `delivery_readiness.json` 为准。

## 已实现证据

- Qwen3.5-9B 片段理解与回答、SigLIP2 冻结视觉检索、神经 reranker、时序覆盖和原视频窗口回放。
- Plan-Execute 六步 trace、schema 工具、上下文预算、memory、重试、熔断和 controlled fallback。
- 不可绕过的时间戳/claim-support 硬规则，以及只可否决的 task-local calibrated answer verifier。
- 上传、即时预览、服务端 VLM 白名单、串行 GPU 队列、持久化任务、轮询阶段、Range `206`、桌面/移动 Web。
- 12 条 SFT、12 组偏好对、真实 BF16 LoRA SFT/DPO、冻结 cola test、DPO 默认与 SFT hash-validated fallback。
- 5/5 冻结错误回归、Agent 失败恢复、冷/热延迟、峰值显存和缓存复用。

## 最终机器验收

- 本地 pytest：`121 passed, 1 skipped`；iboy 标准测试入口：`122 passed`。
- compileall、4 个 JS syntax check、3 个 JS behavior test：通过。
- interview package：`17/17`；delivery readiness：`40/40`。
- 源码 SHA：`77091a151747aa189d28cf85ff29cc5b9b2d05de74cb4b5fe0da91b9f3ad363a`。
- 浏览器任务：`bba9b6fa535e45d596d2d31c7b9aadb4`，默认 URL `http://127.0.0.1:7860` 上 19 项 E2E 检查全部通过。
- Web 常驻模型任务终态耗时 `17.7s`，1.2 秒后复查仍为 `17.7s`，并与 `17.657s` 执行时间窗一致；任务记录 `durable=true, restored=false`。
- 浏览器报告 SHA-256：`b8bab40ed134c9abfbeaacc37283ade33ae9562c81fc482f84d7a19611dd0664`。
- canonical 加载 verifier checkpoint `2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`，实际 `safe_probability=0.92716`。
- 冷/热 profile 为 `27.559s/14.901s`，常驻与缓存复用加速 `1.849×`，正确性守恒。

## 面试定位

- 百度 J104536：多模态输入、规划、schema 工具、证据门、失败恢复、可序列化 trace 和复现。
- 百度 J99230：Qwen/SigLIP、多模态检索、真实 Web、LoRA 后训练、模型准入和工程交付。
- 百度 J97242：SFT→偏好数据→DPO、train/dev/frozen 隔离、reference 设计、训练恢复、泄漏与灾难性遗忘监控。
- 荣耀大模型算法：LLM/VLM/Agent、后训练、安全校验、评测闭环、显存/延迟/缓存优化。

不覆盖 VLA、Speech LLM、从零预训练、公开 benchmark 或已实现 PPO/RLHF。GRPO 是有条件的后续实验，不是当前成果。
