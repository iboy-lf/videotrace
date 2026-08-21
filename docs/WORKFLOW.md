# WORKFLOW

## 日常开发顺序

1. 视频切片、关键帧、可选 OCR/ASR 与视觉描述。
2. 稀疏检索、SigLIP2 特征、神经重排与时序覆盖选择。
3. Plan-Execute Agent 调用工具并记录完整 trace。
4. Qwen3.5 生成带时间戳回答，verifier 执行证据硬门控与学习式否决。
5. 在冻结任务回归上做错误分析、正确性检查和冷/热性能 profile。

这里的评估是小规模任务内回归，不做公开 benchmark、排行榜复现或大规模 ablation。

## 完成标准

- Web 能上传真实视频、选择服务端白名单视觉模式、显示任务阶段并返回回答/证据/时间线。
- 引用证据只在指定时间窗内播放；结束后解除限制；视频脉络从节点起点连续播放。
- canonical 知识包、adapter 准入、浏览器 E2E、性能报告和交付 manifest 绑定当前产品源码。
- SFT/DPO checkpoint、数据隔离、恢复训练和训练源码哈希可复核；冻结可乐视频不进入梯度。
- 本地与 iboy 的 pytest、compileall、JS 检查和交付校验全绿。

## 当前可运行入口

- Windows 一键入口：`start.ps1`（或 `start.bat`）。
- 本地 CPU/已有结果 Web：`python scripts/run_web.py --host 127.0.0.1 --port 7860`。
- iboy 完整视频链路：`bash scripts/remote/run_qwen35_demo.sh`。
- iboy Web 服务：`bash scripts/remote/start_web_service.sh`，本地通过 SSH 端口转发访问。
- 回归与交付：`python -m pytest -q`、`python scripts/validate_delivery_package.py`。
