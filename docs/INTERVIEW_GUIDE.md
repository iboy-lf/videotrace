# Interview Guide

后训练方法的完整适用条件、公式、风险和追问回答见 `docs/POST_TRAINING_DECISION_GUIDE.md`。面试时以“优化对象 → 数据形态 → 是否需要探索 → reward 可信度 → 成本 → 验证”作为统一决策框架，不按算法热度罗列名词。

## 90 秒自我介绍

VideoTrace 把长视频问答拆成可验证的证据检索问题：先切成带时间戳的窗口，用 Qwen3.5-9B 做结构化片段理解和 OCR，用冻结 SigLIP2 处理视觉语义检索，再用可训练 query-segment reranker 融合多路信号。Plan-Execute Agent 在受限上下文中完成检索、证据判断、生成、硬时间戳/claim-support 校验和 calibrated safety veto，最终回答可回到原视频时间窗。项目还包含真实 BF16 LoRA SFT、以 SFT 为冻结 reference 的单策略 DPO、hash-bound adapter 准入、失败恢复和冷/热 profile。

## 五分钟演示顺序

1. 打开远端 Web，上传可乐视频，展示服务状态和真实 VLM 模式。
2. 提问整体流程，查看进度轮询、回答、证据、时间线和技术摘要。
3. 点击引用证据，观察到点暂停；点击“从当前位置继续播放”。
4. 点击视频脉络，说明它只 seek 后持续播放完整视频。
5. 展示 Agent trace：retrieve → context → evidence gate → memory → synthesize → verify。
6. 展示 5/5 回归、硬规则与 calibrated verifier 的不可覆盖关系、SFT/DPO/verifier split、registry 和性能报告。
7. 运行 `validate_delivery_package.py`，说明 40 个交叉检查如何绑定 source/video/data/checkpoint/adapter/browser/GPU。

## 核心技术问答

### 为什么不把整段视频直接交给 Qwen？

长视频 token/帧成本高、难定位、难做证据校验。窗口化后可以缓存片段理解和 SigLIP 特征，做可解释的检索指标、时序覆盖和局部错误分析。

### 为什么同时用 Qwen 和 SigLIP2？

Qwen 描述和 OCR 适合词法检索及回答，SigLIP 保留直接图文对齐，能补足 caption 漏写或措辞不同的视觉事实。两路分数在知识包中分开记录，便于定位 retrieval/visual error。

### Agent 是否只是“套壳”？

稳定路径是明确的 Plan-Execute，而不是无限 ReAct。价值在于 schema 工具、上下文预算、证据门、记忆、重试、熔断、可序列化 trace 和 verifier。失败恢复演示会让检索超时经过两次重试后打开 circuit，并进入不编造答案的 controlled fallback。

### 什么被训练了？

第一条是直接影响候选选择的 `12 -> 16 -> 8 -> 1` reranker；第二条是 12 条 group-isolated evidence QA 上的 Qwen3.5 BF16 LoRA SFT；第三条是 12 组显式偏好对上的 LoRA DPO；第四条是从偏好对派生的 24 行 answer verifier，导出纯 NumPy 逻辑回归参数。随后又做了封存 frozen test 的 step/beta/seed DPO sweep：10 个候选只按 dev 选择，选中 `beta=0.05, step=2` 后才一次性解封 frozen test，并用独立产品回归确认没有回退。可乐始终只在 frozen test。一步 SFT/DPO、sweep 和小 verifier 的指标是链路证据，不是大规模质量宣称。

### 为什么还需要 calibrated verifier？

硬规则擅长发现错时间戳和无支持 claim，但对多个弱信号组合缺少校准。小模型只读取 evidence count、匹配比例、claim coverage、拒答一致性等标量；先过硬规则才调用，低分只能否决。checkpoint 不 pickle sklearn pipeline，而是保存 mean/scale/coef/intercept，由 NumPy 执行，避免本地 sklearn 1.0.2 与远端 1.7.2 的反序列化风险。

### DPO 真的训完了吗？

要区分三个层次：第一，正式产品 DPO 已完成真实 BF16 LoRA step 1，且从 step 1 恢复到 step 2；第二，研究 sweep 又在封存 frozen test 的前提下跑了 10 个 step/beta/seed 候选，选定 `beta=0.05, step=2` 的 3 个 seed dev reward margin 为 `0.1550±0.0027`、绝对偏好准确率均为 `1.0`；第三，选中配置后才解封 frozen test，margin `0.16528702`、preference accuracy `1.0`，并通过独立 `5/5` 产品回归。它已经是“训练和实验闭环完成”，但仍不是大规模偏好训练或广泛泛化证明；研究候选也没有自动替换 Web 默认 adapter。

### 为什么做 DPO，但当前不做 GRPO/PPO/RLHF？

SFT 先建立了输出与证据契约，之后才能人工构造可审计的偏好对，所以做了小规模 DPO。选择预计算 reference 是为了适配 24 GiB 显存；选择一步真实更新是为了证明闭环并限制小数据过拟合。没有做 PPO/RLHF，因为缺少可靠在线 reward model，采样成本更高，还容易把启发式 verifier 的偏差优化进去。

GRPO 也属于在线策略优化，只是用同一 prompt 的多条采样结果做组内相对优势、通常不单独训练 value model；它仍然需要大量独立 prompt、稳定多轨迹采样和可信标量 reward。VideoTrace 当前的 12 组偏好对（`7/4/1`，可乐只在 frozen test）恰好是 DPO 的离线输入，而不是 GRPO 的 rollout 数据。现有 verifier 是从这些偏好对派生的 24 行、只可否决的 task-local safety veto，不是覆盖原始视频和 Agent 轨迹的 reward model。若现在直接用它做 GRPO，最容易优化出一律拒答、格式化时间戳或复制关键词等 reward hacking，而不是更准确的视觉证据绑定。因而准确的表述是：**当前阶段的答案后训练更适合 DPO；项目整体并非永远排斥 GRPO，未来可在有足够轨迹和可验证 reward 后，把 GRPO 限定用于 Agent 工具/轨迹子任务，并与 DPO 做冻结集对比。**

指标解释也要准确：`reward_preference_accuracy=1.0` 表示所有 pair 都获得了正的 **reference-relative** reward margin，即相对冻结 SFT reference 朝 chosen 方向移动；它不等同于当前 policy 已对每个 pair 绝对更偏好 chosen。正式一步的 absolute policy preference accuracy 为 train/dev/frozen `0.571/0.75/1.0`，因此只能说偏好方向和冻结产品门通过，不能说“DPO 已完全学会偏好”或证明广泛泛化。

### 如何证明没有数据泄漏？

SFT/DPO 校验器都检查 video group 单 split、可乐只在 test、`frozen_test` 不进入 optimizer，并记录 train-only gradient payload hash。DPO 还把 reference log-prob 绑定到 SFT 权重、数据和 max length。canonical 视频、adapter、evaluation、pack、registry 与 source SHA 在 admission 和 manifest 中再次交叉检查。

### 做了哪些推理优化？

模型常驻、GPU 请求串行、Qwen/SigLIP 特征缓存、持久 SigLIP index、热缓存复用和 BF16。真实可乐任务记录冷/热延迟、峰值显存、缓存命中和 correctness；4bit 只记录失败 preflight，不伪造成功结果。

## 岗位叙事

- 百度 J104536：多模态输入 → 规划 → schema 工具 → 证据 verifier → 超时恢复。
- 百度 J99230：真实产品、Qwen/SigLIP 融合、远端复现、LoRA 训练和模型卡。
- 百度 J97242：SFT→显式偏好数据→DPO、split isolation、reference 设计、训练排错和评测泄漏控制。
- 荣耀大模型算法：LLM/VLM/Agent 统一链路、后训练准入、缓存/显存/延迟优化和可解释验收。

## 不要夸大的话

不要称为公开 benchmark、通用 VLM、VLA、Speech LLM、从零预训练或已实现 PPO/RLHF；不要把 calibrated veto 说成视觉 NLI；不要把一步 DPO、2 行 frozen verifier test 或 tiny reranker holdout 当成广泛泛化证明。远端 speech ASR 仍未部署。
