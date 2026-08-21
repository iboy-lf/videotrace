# 岗位能力矩阵

| 岗位 | 已实现且有证据 | 面试中应诚实说明的边界 |
| --- | --- | --- |
| 百度 J104536 多模态 Agent | 视频/关键帧输入、Qwen+SigLIP 检索、Plan-Execute schema 工具、上下文预算、memory、硬时间戳/claim-support 门、calibrated veto、重试/熔断/controlled fallback、持久化任务和重启恢复 | 不是开放式无限 ReAct；calibrated verifier 是 task-local 标量模型，不是视觉 NLI；不覆盖 VLA/Speech LLM |
| 百度 J99230 大模型/多模态/微调 | 真实上传/排队/轮询/回放 Web、三种服务端视觉模式、Qwen3.5/SigLIP2、神经 reranker、BF16 LoRA SFT+DPO、hash-bound registry、浏览器 E2E | 一步 SFT/DPO 证明真实闭环和任务行为，不声称大规模泛化提升 |
| 百度 J97242 后训练/数据质量 | SFT 数据卡、显式偏好负例、video-group split、train-only gradient hash、冻结 reference DPO、loss/margin/吞吐/显存、checkpoint/resume、frozen cola、verifier `14/8/2` split | PPO/RLHF 未实现；偏好和 verifier 数据很小；灾难性遗忘通过冻结回归和 baseline non-regression 监控，不等于大规模研究结论 |
| 荣耀大模型算法 | LLM/VLM/Agent 统一链路、5/5 错误回归、DPO 准入与 SFT 回退、冷 27.559s/热 14.901s（1.849×）、21/21 缓存命中、40/40 交付校验 | ASR 未部署；不声称从零预训练、通用 entailment 或公开榜单领先 |

权威证据入口：`outputs/reports/delivery_readiness.json`、`outputs/reports/artifact_manifest.json`、`docs/INTERVIEW_GUIDE.md`、`docs/FINAL_ACCEPTANCE_20260820.md`。
