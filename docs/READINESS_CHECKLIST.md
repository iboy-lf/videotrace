# Interview Readiness Checklist

## Implemented

- [x] MP4 → 切片/关键帧 → Qwen3.5 理解 → SigLIP2 检索 → reranking → 带时间戳回答 → 原视频回放
- [x] caption/OCR/entities/actions/scene/confidence 缓存与字幕 sidecar ASR 对齐
- [x] 稀疏、传统 scorer、冻结 SigLIP2、神经 reranker、时序覆盖融合
- [x] 证据充分性、时间戳绑定和 claim-support 硬规则
- [x] `portable-numpy-logistic-v1` calibrated safety veto；只能否决，不能覆盖硬失败
- [x] Plan-Execute、schema、上下文预算、memory、重试、熔断、失败恢复和序列化 trace
- [x] 上传即时 `blob:` 预览、项目内远端媒体、问题输入和服务端 VLM 白名单
- [x] 串行 GPU worker、常驻模型、轮询阶段、持久化 job store 和重启恢复
- [x] 证据自动暂停后释放、继续播放、视频脉络完整播放
- [x] 桌面/移动 E2E、clean console、HTTP Range `206`
- [x] group-isolated SFT、真实 BF16 LoRA step、冻结 cola test
- [x] 显式偏好对、冻结 SFT reference、真实 BF16 DPO step、reward margin
- [x] DPO hash-bound 准入；SFT 是 fallback；浏览器不能选内部 adapter
- [x] 5 个冻结案例覆盖局部事实、全局流程、困难 OCR、盲测和拒答
- [x] 冷/热延迟、峰值显存、缓存复用、BF16 与失败的 4bit preflight
- [x] GPU 三次稳定探测、显式 `CUDA_VISIBLE_DEVICES`、不终止外部进程
- [x] 本地完整 pytest `147 passed, 1 skipped`；本轮远端重验证预检 `145 passed`（按依赖顺序显式延后文档一致性测试）；compileall、4 个 JS syntax、3 个 JS behavior test 通过，最终文档严格校验单独执行
- [x] delivery readiness 当前为 `40/40`，interview package 为 `17/17`；真实上传 Web E2E、5/5 冻结回归、性能、adapter 准入和当前源码指纹均已重跑
- [x] 研究 DPO sweep 已完成并通过 `validate_dpo_sweep.py`；研究 adapter 与默认 Web registry 分离
- [x] 同一 frozen pack 的 Qwen3.5/Qwen2.5 模型选型与延迟/grounding 对照已记录
- [x] source/video/data/checkpoint/adapter/report hash 可审计，且缺失产物由 manifest 记录而不是被隐藏
- [x] canonical、checkpoint、数据、metrics、模型卡和 registry 本地/远端 SHA-256 对齐

## Honest boundaries / future evidence

- [ ] 部署真实 speech ASR；当前远端只有字幕 sidecar 路径
- [ ] 扩大偏好与 verifier 数据后再声称跨域 DPO/校验泛化
- [ ] 评估视觉 entailment，且保留硬规则不可覆盖原则
- [ ] 多用户部署时再增加鉴权、配额和多 worker 调度

项目已可作为长视频 LLM/VLM/Agent/后训练/推理优化的核心面试项目；不声称 GRPO、PPO/RLHF、VLA、Speech LLM、从零预训练或公开 benchmark 领先。
