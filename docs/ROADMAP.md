# ROADMAP

## 已完成

- 长视频切片、关键帧、字幕 sidecar/OCR 对齐接口与内容指纹缓存。
- Qwen3.5-9B 片段理解、SigLIP2 冻结检索、稀疏/传统 scorer/神经 reranker/时序覆盖融合。
- Plan-Execute Agent、schema 工具、上下文预算、记忆、重试、熔断、失败降级和完整 trace；成功/失败双轨 run-level 审计已记录唯一 call ID、逐次 attempt、延迟、错误码和持久化恢复。
- 原视频证据窗口、自动暂停后释放、继续播放、视频脉络持续播放。
- Web 上传、即时预览、远端队列、轮询进度、真实 VLM 白名单、持久化任务和 Range `206`。
- Playwright 桌面/移动 E2E；durable job store、受管服务重启机制和任务恢复单元测试。
- 12 条 SFT、12 组 DPO 偏好对、真实 BF16 LoRA 一步训练、冻结 cola test、DPO 默认/SFT 回退。
- 24 行 answer-verifier 数据、`14/8/2` split、纯 NumPy 可移植逻辑回归 checkpoint、canonical 实际加载验证。
- 5/5 冻结回归、冷/热 profile、峰值显存、BF16 和失败的 4bit preflight。
- artifact manifest、岗位矩阵、面试指南、开源 attribution。
- 本地完整 pytest `140 passed, 1 skipped`（本地无 Torch）；本轮远端重验证预检 `138 passed`（文档一致性按依赖顺序延后到最终阶段）。artifact manifest 包含 51 个核心制品和 18 份不可变准入历史，manifest hash mismatch 为 0。当前 interview `17/17`、delivery `40/40`，真实上传 Web E2E、5/5 冻结回归、adapter 准入与性能均绑定源码 `642cc023…`。

## 有意保留的边界

- 不做公开 benchmark、榜单复现、大规模 ablation 或大量输出目录。
- 不声称 VLA、Speech LLM、从零预训练、GRPO 或 PPO/RLHF 已实现。
- 远端没有 speech ASR 权重/运行时，当前只消费字幕 sidecar。
- calibrated verifier 是小规模 task-local safety veto，不是通用 NLI 或完整视觉 entailment。

## 下一步（由真实数据和产品需求触发）

1. 扩充许可的多视频证据和真实用户偏好，再评估多步 DPO/蒸馏及跨域稳定性；只有具备多轨迹数据、可分解 reward 和 reward-hacking 回归后，才在受限 Agent 工具/轨迹子任务上评估 GRPO。
2. 环境具备权重和依赖后部署 faster-whisper，并新增 speech-only 错误类别。
3. 在当前硬规则与 task-local veto 之上评估视觉 entailment，但必须防止 verifier/生成模型共错。
4. 真正多用户部署时再增加鉴权、配额、保留策略和多 worker 调度。
