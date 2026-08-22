# VideoTrace Interview Readiness Report

VideoTrace 已形成一条可现场演示、可远端复现、可用 hash 交叉验证的长视频多模态 Agent 产品链路。动态权威证据以 `outputs/reports/artifact_manifest.json` 和 `outputs/reports/delivery_readiness.json` 为准。

## 已实现证据

- Qwen3.5-9B 片段理解与回答、SigLIP2 冻结视觉检索、神经 reranker、时序覆盖和原视频窗口回放。
- Plan-Execute 六步 trace、schema 工具、上下文预算、memory、重试、熔断和 controlled fallback。
- 不可绕过的时间戳/claim-support 硬规则，以及只可否决的 task-local calibrated answer verifier。
- 真实上传、即时预览、服务端 VLM 白名单、串行 GPU 队列、持久化任务、轮询阶段、Range `206`、桌面/移动 Web。
- 12 条 SFT、12 组偏好对、真实 BF16 LoRA SFT/DPO、冻结 cola test、DPO 默认与 SFT hash-validated fallback。
- 5/5 冻结错误回归、Agent 失败恢复、冷/热延迟、峰值显存和缓存复用。

## 当前机器验收

- interview package：`17/17`；delivery readiness：`40/40`。
- 本地完整 pytest：`147 passed, 1 skipped`；远端重验证预检 `145 passed`，compileall、4 个 JS syntax 和 3 个 JS behavior test 均通过。
- 源码 SHA：`7374516adfcea3f8f4505117c6d7c5362975f4ce0de45efad595f41226f195fc`。
- 浏览器任务：`f6d582f2ef434a689845b23e8c671f07`；真实视频上传和完整任务队列均执行，默认 URL `http://127.0.0.1:7860` 上 `19 项` E2E 检查全部通过。
- Web 任务终态耗时 `29.2s`，复查仍为 `29.2s`，与 `29.186s` 执行时间窗一致；任务记录 `durable=true, restored=false`。
- 浏览器报告 SHA-256：`0b05057627b7bee829455c73766f5b8861bccdaf25c07278966bbbc77331dd7c`。
- canonical 加载 verifier checkpoint `2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`，实际 `safe_probability=0.92716`。
- 冻结产品回归 `5/5`；冷/热 profile 为 `26.631s/15.151s`，缓存复用加速 `1.758×`，正确性守恒。
- 模型选型对照：同一 frozen pack 上 Qwen3.5-9B 为 `22.778s`，Qwen2.5-VL-7B 为 `15.52s`，两者均通过 grounding、claim-support 与 timestamp binding；面试可解释质量/延迟取舍。
- DPO 研究 sweep 已完成：10 个候选封存 frozen test 选择，`beta=0.05, step=2` 的 3-seed dev margin 为 `0.15501256±0.00265381`、绝对偏好准确率 `1.0`；一次性 frozen margin `0.16528702`，产品回归 `5/5`。该候选保留在研究目录，默认 Web adapter 未被静默替换。
- manifest 覆盖 60 个核心制品和 20 份不可变 adapter 准入历史；DPO/SFT evaluation 的精确 SHA 以 `outputs/models/best_adapter.json` 为准。

## 面试定位

- 百度 J104536：多模态输入、规划、schema 工具、证据门、失败恢复、可序列化 trace 和复现。
- 百度 J99230：Qwen/SigLIP、多模态检索、真实 Web、LoRA 后训练、模型准入和工程交付。
- 百度 J97242：SFT→偏好数据→DPO、train/dev/frozen 隔离、reference 设计、训练恢复、泄漏与灾难性遗忘监控。
- 荣耀大模型算法：LLM/VLM/Agent、后训练、安全校验、评测闭环、显存/延迟/缓存优化。

不覆盖 VLA、Speech LLM、从零预训练、公开 benchmark 或已实现 PPO/RLHF。GRPO 和蒸馏是有明确数据、reward 或 teacher/student 契约后才启动的后续实验，不是当前成果。
