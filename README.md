# VideoTrace

[![ci](https://github.com/iboy-lf/videotrace/actions/workflows/ci.yml/badge.svg)](https://github.com/iboy-lf/videotrace/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

VideoTrace 是一个面向长视频的多模态证据问答 Agent：它把视频切成可检索的时间窗口，用 Qwen3.5-9B 做片段理解与回答生成，用冻结的 SigLIP2 做视觉检索增强，再由可训练神经重排器和 Plan-Execute Agent 组织证据、上下文、回答与校验。最终产品返回可回看的原视频时间窗，而不是只给一段无法核验的文字。

这个项目的目标不是堆模型名，而是把一条真实可复现的 LLM/VLM/Agent/后训练/推理优化链路做完整：有真实视频、有远端训练、有失败恢复、有错误分类、有 hash-bound 产物准入，也明确记录没有实现的能力边界。

## 用户入口

普通用户只需要：选择视频、输入问题、选择一个视觉理解模式并点击“分析视频”。上传、问题输入和分析按钮始终可见；算力不可用时页面显示连接状态，不会把入口伪装成只读页面。上传视频会立即预览，并由项目内的定长流式 multipart 解析器写入远端 `data/uploads`；服务端同时限制请求大小、扩展名、单文件字段和最终落盘路径，不依赖 Python 3.13 已移除的 `cgi` 模块。

证据播放和视频脉络是两种不同的行为：

- 点击回答中的引用证据：跳到 `start_sec`，播放到 `end_sec` 后暂停并解除时间窗限制。
- 点击视频脉络：只跳到节点起点，之后继续播放完整视频，不在节点结尾强制暂停。
- 点击“从当前位置继续播放”：清除旧时间窗并立即继续播放。

## 算法链路

1. 视频切片、关键帧、可选 ASR/OCR 对齐，并为每个窗口保留时间戳。
2. Qwen3.5-9B 生成结构化描述、实体、动作与 OCR；按视频/时间窗/帧指纹缓存。
3. 融合稀疏检索、规则/传统 scorer、SigLIP2 图文相似度和时序覆盖选择。
4. 读取 `12 -> 16 -> 8 -> 1` 的神经 query-segment reranker checkpoint，改善实际多信号融合。
5. Plan-Execute Agent 依次执行检索、上下文压缩、证据判断、记忆搜索、生成和 verifier；工具 schema、重试、熔断、trace 都可审计。关键证据工具异常时任务 fail-closed 返回安全拒答，记忆工具异常时记录恢复事件并仅依赖当前视频证据继续，失败不会被伪装成成功。
6. verifier 先执行不可绕过的证据充分性、时间戳绑定和 claim-support 硬规则，再由小型 calibrated answer verifier 对硬规则通过的答案做保守否决；学习式层不能补证据、修时间戳或把硬失败改成通过。
7. 通过任务级回归后再导出知识包、Web/HTML、模型卡、性能报告和 hash 清单。

## 视觉模式

服务端扫描可信模型目录并只返回真实可执行的白名单模式：

- `自动最佳（Qwen3.5 + SigLIP2）`
- `Qwen3.5 视频理解`
- `SigLIP2 检索增强`

浏览器不能传任意模型路径、backend、dtype 或设备。回答模型、reranker、安全策略和已准入 adapter 均由服务端固定；用户选择的模式会写入知识包元数据。

## 一键启动

本地只查看最近一次已验证结果：

```powershell
cd D:\Agent\VideoTrace
powershell -ExecutionPolicy Bypass -File .\start.ps1
```

完整上传/分析服务运行在 `iboy`，通过 SSH 端口转发访问：

```powershell
cd D:\Agent\VideoTrace
powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1
```

成功后打开 `http://127.0.0.1:7860`。脚本会在远端检查连续稳定的空闲 GPU、启动常驻 Web 服务并建立隧道；没有安全空闲卡时会等待或明确报错，不会终止他人的 GPU 进程。

源视频是第三方内容，不随仓库分发。没有 `data/raw/cola_review.mp4` 时，`start.ps1` 不会失败，而是进入**证据模式**：回答、时间戳证据、视频脉络、Agent trace 和技术面板全部来自已提交的知识包，页面明确标注视频缺失、证据按钮禁用。它展示真实产物，也不假装证据可以回看。详见 `data/raw/README.md`。

## 真实后训练链路

`data/sft/grounded_qa.jsonl` 是 12 条人工核验的证据问答：train 7、dev 4、test 1；可乐视频只在 frozen test，绝不进入梯度更新。远端用 BF16 LoRA 对 Qwen3.5-9B 完成真实 SFT optimizer step，并保存 adapter、optimizer、RNG、trainer state 和原子 checkpoint manifest。交付验收还要求从正式 step 1 checkpoint 恢复并实际完成 step 2；仅有“支持 resume”的配置或静态文件不算通过。

`data/preference/grounded_dpo.jsonl` 进一步提供 12 组人工编写偏好对，覆盖错时间戳、缺时间戳、细节幻觉和无证据过度回答，同样保持 7/4/1 的视频级 split。DPO 以已验证 SFT adapter 为冻结 reference，在 optimizer 前预计算 chosen/rejected log-prob，避免同时驻留两个 9B 模型；正式一步训练和 step 1→2 恢复都必须通过 dev/frozen 正 reward margin 与哈希链校验。训练使用两张经连续探测确认空闲的 3090 做 model parallel，AdamW 动量在 forward 间隙卸载到 CPU、只在 optimizer step 前按参数所在设备短暂恢复，以覆盖完整 367-token 长样本并消除“新训练能跑、恢复却 OOM”的差异；这不改变 DPO 目标或数据。最新 loss、margin、吞吐和峰值显存以 `outputs/models/qwen35_dpo_metrics.json` 及 `outputs/reports/qwen35_dpo_resume_validation.json` 为准。4-bit QLoRA 因现有 bitsandbytes 为 CPU-only 而明确禁用，项目没有虚报 4bit 支持。

SFT 和 DPO 只有同时通过冻结可乐证据、时间戳、claim-support、覆盖不回退和 adapter/evaluation/pack/source SHA 校验后，才能写入服务端 `best_adapter.json`。DPO 任一门槛或文件哈希失效时自动回退到已验证 SFT；浏览器不能选择内部 adapter。PPO/RLHF 未实现，原因是当前任务不值得引入在线采样、reward hacking 和更高显存成本。

"你的 reward margin 是不是只是答案更长"这类追问用数字回答，不用措辞回答。`scripts/analyze_dpo_length_bias.py` 从已提交的 reference log-prob 缓存和训练指标重算，写入 `outputs/reports/dpo_length_bias.json`：token 长度差与 reference-relative reward margin 的 Pearson 相关为 `-0.1986`，12/12 pair 的 margin 为正，说明 margin 不是长度效应。同一份报告也给出反面的诚实结论：绝对偏好判断 `8/12` 基本继承自 reference，一步训练翻转了 `0/12`，而 reference 的 4 个错判中有 3 个在按 token 归一后消失——所以只能声称闭环真实且方向正确，不能声称模型已经学会判断时间戳错误。推导与读法见 `docs/POST_TRAINING_DECISION_GUIDE.md`。

同一批人工 DPO 偏好对还派生出 24 条 answer-verifier 样本（train 14、dev 8、frozen test 2）。模型是标准化标量特征上的逻辑回归，checkpoint 只保存可移植数值参数，运行时由 NumPy 执行，避免跨 sklearn 版本反序列化；阈值只在 dev 上选择，可乐 frozen test 不进入梯度。它是 task-local safety veto，不是通用 NLI、视觉 entailment 或 benchmark 结果。

## 关键证据与验证

```powershell
$env:PYTHONPATH = 'src'
python -m pytest -q
python -m compileall -q src scripts tests
node --check src/videomemo/web/static/app.js
node --check src/videomemo/web/static/playback.js
node --check src/videomemo/web/static/technical.js
node --check src/videomemo/web/static/job_status.js
node tests/js/playback.test.cjs
node tests/js/technical.test.cjs
node tests/js/job_status.test.cjs
python scripts/validate_outputs.py outputs/iboy_qwen35/cola_review/knowledge_pack.json --video data/raw/cola_review.mp4
python scripts/analyze_dpo_length_bias.py
python scripts/build_artifact_manifest.py
python scripts/validate_delivery_package.py
python scripts/validate_documentation_consistency.py --strict
python scripts/validate_documentation_links.py
```

其中 pytest、compileall、JS 检查、`validate_documentation_consistency.py` 和 `validate_documentation_links.py` 不依赖 GPU、模型权重或源视频，因此也是 `.github/workflows/ci.yml` 在每次推送时真实执行的内容。需要权重或视频的校验脚本不放进 CI，因为一个跑不到真实产物的绿灯没有意义。

仓库跟踪了约 1.1 MB 的小体积证据产物（`outputs/reports/*.json`、metrics、模型卡、canonical 知识包与 `demo.html`），所以上面 README 引用的每一个证据路径在 clone 之后都真实存在、可以独立核对。模型权重、索引和源视频不随仓库分发，它们的 SHA-256 记录在 manifest 里。

`validate_documentation_links.py` 会检查文档引用的每个仓库路径既存在、又对一个全新 clone 可见；`data/raw` 这类不分发的内容必须由文档明确说明来源，而不是留下死链接。

`outputs/reports/artifact_manifest.json` 是最终交付清单的权威入口，记录视频、canonical pack、reranker、SFT/DPO 数据、正式 checkpoint、实际恢复 checkpoint、reference log-prob、adapter registry/evaluation、错误分析、性能、失败恢复和 GPU 选择审计的 SHA-256。它不是公开 benchmark；5 个冻结回归案例和冷/热运行数据只用于任务内正确性与工程可信度。

`validate_documentation_consistency.py` 将 DPO/恢复指标、性能数据、最新浏览器 job 与 source/report SHA 同面试文档交叉校验，防止机器产物更新后 Markdown 仍保留旧数字。后训练方法选择与追问边界见 `docs/POST_TRAINING_DECISION_GUIDE.md`。

## 远端固定环境

- 项目：`/lavender/VideoTrace`
- Python：`/linyuanping/miniconda3/envs/guide2play-qwen35/bin/python`
- Qwen3.5-9B：`/lavender/models/Qwen3.5-9B`
- SigLIP2：`/lavender/models/siglip2-large-patch16-256`

## 明确边界

当前项目不声称覆盖 VLA、Speech LLM、从零预训练、PPO/RLHF 或公开榜单；远端尚未部署 speech ASR，当前只支持字幕 sidecar，并保留可选 faster-whisper 接口。claim-support 硬规则与 calibrated verifier 都不是通用 NLI 或完整视觉 entailment。小规模训练结果不是广泛泛化证明。

## 文档入口

- `docs/ARCHITECTURE.md`：端到端架构与数据流
- `docs/TRAINING.md`：reranker、Qwen SFT 与单策略 DPO
- `docs/DATA_CARD_SFT.md`、`docs/DATA_CARD_DPO.md`、`docs/MODEL_CARD_SFT.md`、`docs/MODEL_CARD_DPO.md`：数据和 adapter 契约
- `docs/ERROR_ANALYSIS.md`、`docs/PERFORMANCE.md`：真实任务内证据
- `docs/INTERVIEW_GUIDE.md`、`docs/JOB_READINESS_MATRIX.md`：面试叙事和岗位矩阵
- `docs/POST_TRAINING_DECISION_GUIDE.md`：SFT、DPO、GRPO、KTO/SimPO、蒸馏与 PPO/RLHF 的场景化选择，含 DPO↔RLHF 推导与长度偏差实测
- `docs/REVALIDATION.md`：源码指纹如何绑定证据链，以及改动代码后必须重跑什么
- `docs/FINAL_ACCEPTANCE_20260820.md`：指定视频、远端 Web、播放、移动端和交付包的最终验收快照
- `docs/REMOTE_IBOY.md`：远端启动、GPU 安全和端口转发
- `THIRD_PARTY_NOTICES.md`：开源参考与许可证 attribution
- `LICENSE`：MIT。不覆盖第三方模型权重与 `data/raw/` 下的源视频
