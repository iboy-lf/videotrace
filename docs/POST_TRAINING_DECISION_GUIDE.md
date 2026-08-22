# VideoTrace 后训练方法决策指南

这份指南用于面试中的方法选择追问。核心原则不是先选一个热门算法，而是依次判断：优化对象、数据形态、是否需要探索、reward 是否可信、算力与方差、最终如何验证。

## 一页决策表

| 条件 | 优先方法 | 适合 VideoTrace 的位置 | 主要风险 |
|---|---|---|---|
| 有高质量输入输出示范，需要建立格式和基本能力 | SFT | 证据上下文到带时间戳回答 | 模仿数据错误；不会表达相对偏好 |
| 有同 prompt 的 chosen/rejected，训练必须离线稳定 | DPO | 时间戳正确、证据充分、不过度回答 | 偏好覆盖有限；reference/长度偏差；不能探索 |
| 只有未配对的 desirable/undesirable 标签 | KTO 类方法 | 将来真实用户只给赞/踩而没有成对排序时 | 标签校准和分布偏差；当前没有这种数据 |
| 想做更简单或 reference-free 的离线偏好优化 | SimPO/ORPO 等 | 数据规模扩大后作为 DPO 对照 | 超参数和长度偏差；不能仅因“更新”就替换 DPO |
| 同 prompt 可采样多条结果，reward 可程序验证，需要策略探索 | GRPO | 将来动态工具选择或受限 Agent 轨迹 | rollout 成本、低组内方差、reward hacking、credit assignment |
| 有可靠 reward model、在线交互和 critic 训练预算 | PPO/RLHF | 当前没有合适位置 | critic/采样成本高、稳定性复杂、放大奖励偏差 |
| teacher 强、student 有明确能力或部署目标 | 蒸馏 | 将来把昂贵 Qwen 片段理解蒸馏到小模型 | teacher 错误复制；若无 student/契约就只是改名 |

## 先判断到底要优化哪一层

面试中先拆优化对象，能避免把所有问题都回答成“再做一种 RL”：

1. **视觉感知层**：片段描述、OCR 或实体识别错。应先补视觉监督、数据或模型能力；回答 DPO/GRPO 不能凭空恢复视频里没有被编码的事实。
2. **检索与排序层**：正确片段没有进入上下文。VideoTrace 当前用标注窗口训练 neural reranker，并用时序覆盖规则修全局流程；答案偏好优化不能替代候选召回。
3. **证据回答层**：证据已经正确，但回答时间戳错、幻觉或过度回答。当前 SFT → DPO 正在优化这一层。
4. **Agent 轨迹层**：需要学习何时调用 OCR、扩大窗口、补检索、读取记忆或拒答。只有这一层存在真实可探索动作和状态变化时，GRPO 才可能比静态答案 DPO 更有价值。

因此 VideoTrace 的方法不是“整个项目选了 DPO 而没选 GRPO”，而是：感知层冻结并缓存、排序层监督学习、回答层使用 SFT+DPO、Agent 层先用可审计的受限 Plan-Execute。不同层使用与其监督信号匹配的方法。

## 当前为什么是 SFT → DPO

当前优化目标是：给定已检索证据，回答必须包含正确时间戳、避免幻觉与过度回答，并在证据不足时拒答。现有数据是 12 组人工审计的同 prompt chosen/rejected，错误类型包括 wrong timestamp、missing timestamp、hallucinated detail 和 unsupported overclaim。这正是离线 DPO 的输入契约，不是在线 rollout 数据。

实现上以已准入 SFT adapter 为冻结 reference，预计算全部 chosen/rejected sequence log-prob，训练时只常驻一个 9B policy。这使标准目标

`-log sigmoid(beta * ((logπ(y+|x)-logπ(y-|x))-(logπref(y+|x)-logπref(y-|x))))`

能在 24 GiB 级 GPU 上真实运行并恢复。`beta=0.1` 是隐式 reward 的尺度，也是底层 KL 正则系数：更大的 `beta` 对应更强的 KL 惩罚，同样的偏好 logit 只需要更小的 `log(π/πref)` 偏移；但它也会线性放大初始 DPO 梯度。更小的 `beta` 初始梯度更弱，却允许最优解离 reference 更远。因此不能把 `beta` 简化成“越大越激进”，必须同时看实际 KL、偏好门和冻结产品回归。

当前正式产品结果应表述为：step 1 后 train/dev/frozen mean reference-relative reward margin 为 `0.22440502/0.14014463/0.09733963`，absolute policy preference accuracy 为 `0.571429/0.75/1.0`，所以不能说每个 pair 已被完全解决。为回答“是不是只跑一步”，另有封存 frozen test 的 10 候选 sweep：选中 `beta=0.05, step=2` 的 3 个 seed dev margin 为 `0.1550±0.0027`，绝对偏好准确率均为 `1.0`；按 dev 选择后一次性解封 frozen test，margin `0.16528702`、accuracy `1.0`，并通过独立 `5/5` 产品回归。研究候选不自动替换 Web 默认 adapter；产品采用任一 adapter 仍必须经过冻结 grounding、时间戳、claim-support、coverage 和 source/hash 门。

## DPO 和 PPO/RLHF 是什么关系

这是最常见的一个追问：既然 RLHF 要训 reward model 和 critic，DPO 为什么可以都不要？答案不是"DPO 是另一个算法"，而是**DPO 是同一个 KL 约束优化问题的闭式解重参数化**。

RLHF 的目标是在不偏离 reference 太远的前提下最大化 reward：

`max_π E_{x,y~π}[r(x,y)] - beta * KL(π(·|x) || πref(·|x))`

这个问题有闭式最优解（标准的 Gibbs/指数倾斜形式）：

`π*(y|x) = (1/Z(x)) * πref(y|x) * exp(r(x,y)/beta)`

反解 reward，可得任何策略都隐式定义了一个 reward：

`r(x,y) = beta * log(π(y|x)/πref(y|x)) + beta * log Z(x)`

把它代入 Bradley-Terry 偏好似然 `P(y+ > y-) = sigmoid(r(x,y+) - r(x,y-))`，配分函数 `log Z(x)` 只依赖 `x`，在同 prompt 的**差**里被完全消掉，于是得到本项目实现的目标：

`-log sigmoid(beta * ((logπ(y+|x)-logπ(y-|x)) - (logπref(y+|x)-logπref(y-|x))))`

三个可被追问的推论：

1. **reward model 没有消失，它被参数化进了 policy 本身。** `beta * log(π/πref)` 就是隐式 reward——`scripts/analyze_dpo_length_bias.py` 报告的 `reward_margin` 正是这个量的 chosen/rejected 之差。
2. **KL 约束也没有消失。** 它就藏在 `beta` 里：在上面的 RL 目标中，更大的 `beta` 是更强的 KL 惩罚；在 DPO loss 中，它同时缩放 preference logit 与梯度。两种作用必须一起解释，所以超参选择应报告策略相对 reference 的实际偏移与产品回归，不能仅凭 loss 下降判断。
3. **消掉 `log Z(x)` 的前提是同一个 `x`。** 所以 DPO 必须用成对同 prompt 数据；未配对的赞/踩数据不满足这个条件，这才是要换 KTO 而不是"KTO 更方便"的理由。

代价也随之明确：DPO 只在偏好数据的支撑集上定义了 reward，off-distribution 的 `y` 没有任何约束；PPO 则通过在线采样持续在当前策略的分布上取样并打分。这是 DPO 会出现"在训练分布内变好、自由生成时漂移"的机理性原因。

### PPO 侧的目标长什么样

如果真的走在线 RLHF，优化的是 clipped surrogate 加 KL 惩罚：

`L = E[min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)] - beta * KL(π || πref)`，其中 `ratio = π(y|x)/π_old(y|x)`

需要额外具备而当前项目不具备的四样东西：一个在分布漂移下仍然可靠的 reward model；一个 value/critic 网络及其 GAE 估计；在线 rollout 预算；以及针对 reward hacking 的独立冻结评测。当前 12 组偏好对训不出一个能抵抗策略漂移的 reward model——策略会直接跑到 reward model 的外推区。因此不做 PPO 是数据规模决定的，不是实现难度决定的。

### 顺带回答 GRPO 的位置

GRPO 可以看成"去掉 critic 的 PPO"：用组内 reward 的均值代替 value baseline。它保留了 PPO 的在线采样、ratio clipping 和 KL 约束，只省掉 value 网络。所以它继承 PPO 的全部 reward 设计问题，只解决了显存和 critic 训练不稳定的那一部分——这也是不能把"不训 critic"当作"成本低"的原因。

## 这个 DPO 结果里的长度偏差有多大

`scripts/train_qwen35_dpo.py:_sequence_logp` 用的是答案 token 的**未归一化 log-prob 求和**，也就是标准 DPO 形式。长句天然 log-prob 更低，所以"你的 margin 会不会只是长度"是必须用数字回答的问题。`scripts/analyze_dpo_length_bias.py` 从两个已提交产物重算，结果写入 `outputs/reports/dpo_length_bias.json`：

| 诊断 | 值 | 含义 |
|---|---|---|
| chosen/rejected token 差 | mean `25.6667`，范围 `-23 ~ +162`，12 组中 7 组 chosen 更长 | 数据本身长度不平衡，两个方向都有 |
| `pearson(token 差, reward_margin)` | `-0.1986` | reference-relative reward 与长度**没有**正相关；长度偏好被 reference 项抵消了 |
| `reward_margin > 0` | `12/12` | 每一对都朝正确方向移动 |
| reference 按 **求和** log-prob 偏好 chosen | `8/12` | 未归一化时 reference 本身就判错 4 对 |
| reference 按 **每 token 平均** log-prob 偏好 chosen | `11/12` | 那 4 对的绝对判断错误主要是长度效应，不是语义判断错误 |
| policy 与 reference 的绝对偏好不同的对数 | `0/12` | 一步训练没有翻转任何一对的绝对偏好 |

诚实的读法有三条，面试中要主动说出来：

1. `reward_margin` 是**相对 reference** 的量。reference 和 policy 共享的长度偏好在减法中抵消，`-0.1986` 的相关系数说明抵消是有效的，所以用它作为一步训练是否真实生效的证据是成立的。
2. 但 `absolute policy preference accuracy` 的 `8/12` 基本继承自 reference，而 reference 的那 4 个错误在按 token 归一后消失了 `3` 个。所以**不能**说"模型已经能判断 4 组时间戳错误"；准确的说法是一步更新还不足以翻转任何一对的绝对偏好。
3. 数据集本身存在结构性混淆：`unsupported_overclaim` 负例普遍比 chosen 更长，`wrong_timestamp` 负例普遍更短。扩充偏好数据时必须先打破这个相关性，否则任何长度归一化变体（SimPO 的长度归一化奖励、ORPO）都会在这份数据上产生无法归因的差异。

这也是本项目不急于换成 SimPO/ORPO 的具体理由：在 12 组、且长度与错误类型相关的数据上，换目标函数得到的差异无法区分"目标更好"和"恰好利用了这个混淆"。


## DPO 的适用前提与问题

DPO 更适合：偏好数据已经离线收集；chosen/rejected 语义可审计；不需要通过在线环境发现新策略；希望训练稳定、实现简单并避免单独 reward model/critic。

DPO 的问题：

- 它只学习偏好数据覆盖到的行为，不能主动探索新的检索或工具策略。
- chosen 只是相对更好，不保证绝对正确；rejected 质量和难度会影响梯度。
- 对 reference policy、`beta`、长度归一方式和回答长度偏差敏感。
- sequence-level pair 不能精确定位多步 Agent 的哪一步造成错误。
- 小数据下容易过拟合格式、时间戳模板或标注者风格。
- 它优化回答策略，不会直接训练 SigLIP/Qwen 视觉编码能力。

## GRPO 什么时候有意义

GRPO 的关键价值是让同一 prompt 的多条采样结果产生组内相对优势，不需要单独训练 value model。它适合“探索能发现更好策略、结果可重复打分”的场景，而不是为了简历出现 RL 名字。

VideoTrace 将来满足以下条件时才值得做：

- 至少数百个按视频隔离的 prompt，并为每个 prompt 采样多条答案或轨迹；
- Agent 动作真实影响后续状态，例如是否调用 OCR、扩大时间窗、补检索、访问记忆或拒答；
- reward 能分解为时间窗 IoU、claim-support、证据充分性、工具 schema、错误拒答率、延迟/调用成本等；
- 有独立人工事实抽查和 anti-reward-hacking 冻结集；
- 能承担每个 prompt 多次视频/VLM 生成的成本。

GRPO 的问题：

- 一组 rollout reward 相同或近似时，归一化优势几乎没有学习信号。
- 视频多模态生成成本远高于纯文本，多 rollout 会显著增加延迟和 GPU 使用。
- 稀疏终局 reward 难以分配到检索、视觉理解、工具调用和生成各步。
- 策略可能通过一律拒答、堆时间戳、复制证据词或缩短答案作弊。
- 组内相对归一会弱化不同 prompt 的绝对难度差异。
- 不训练 value model 只减少一部分复杂度，并没有解决 reward 设计、KL、长度和策略漂移。

当前 calibrated verifier 只读取少量 task-local 标量，并与偏好数据同源。直接用它作为 GRPO reward 会形成训练—评测闭环，策略最可能钻 verifier 特征漏洞。因此现在不做 GRPO 是场景约束下的技术决策，不是不了解或排斥 GRPO。

从目标上看，GRPO 会对同一输入采样一组结果 `y_1...y_G`，再用组内 reward 的均值和标准差构造相对优势：

`A_i = (r_i - mean(r_1...r_G)) / (std(r_1...r_G) + eps)`

随后用带 clipping 和 KL 约束的策略梯度更新。它省掉 value model，不代表省掉 rollout、reward、reference/KL、长度控制或策略漂移问题。若一组结果都得 0 分、都因格式得到同分，或 reward 只会区分“拒答/不拒答”，`A_i` 就没有有效的事实学习信号。

### VideoTrace 真要做 GRPO 时的实验契约

只在一个受限 Agent 轨迹子任务上开始，例如“是否补一次 OCR/检索以及选择哪个时间窗”，而不是直接让 9B 模型对完整视频自由探索。每条轨迹至少记录动作、工具返回、最终证据、延迟和失败码。reward 需要来自相互独立的来源：

- 时间窗与人工 gold span 的覆盖或 IoU；
- 对应证据上的 claim-support 与时间戳绑定；
- 证据不足问题的正确拒答，以及对有证据问题的过度拒答惩罚；
- schema 合法、工具成功、调用次数和延迟成本；
- 独立人工抽查，且训练 reward verifier 与最终评测 verifier 不同源。

准入条件不是“训练 loss 下降”，而是独立视频 frozen set 上检索/轨迹成功率提高、事实与时间戳不回退、过度拒答率不升、reward-hacking 案例不过拟合，并保留与当前 DPO 产品路径相同的 hash-bound admission。若这些条件不成立，就不应把 GRPO adapter 接入 Web。

## VideoTrace 场景下的 DPO / GRPO 对照

| 追问 | 当前 DPO | 有条件的未来 GRPO |
|---|---|---|
| 优化对象 | 已给定证据后的答案行为 | 动态工具选择或多步轨迹策略 |
| 数据 | 同 prompt 的 chosen/rejected | 同 prompt 的多条在线 rollout |
| 探索 | 不需要，也做不到 | 必须存在动作导致的状态差异 |
| reward | 人工偏好对隐式定义 | 每条 rollout 的可信可分解标量 |
| 主要成本 | reference log-prob 与单 policy 更新 | 多次生成、工具/VLM 调用、打分、KL |
| 当前最大风险 | 小数据格式过拟合、reference/长度偏差 | 低组内方差、稀疏 credit、reward hacking、过度拒答 |
| 产品门 | frozen grounding/claim/coverage non-regression | 同一产品门，再加轨迹与 anti-hacking 门 |

面试中最重要的一句是：**DPO 是当前“静态证据回答偏好”问题的匹配解；GRPO 只有在问题升级为“可探索、可可靠打分的 Agent 轨迹优化”后才有新增价值。**

## 蒸馏和 PPO/RLHF

蒸馏适合存在明确 teacher/student 契约时，例如 teacher 输出片段描述、OCR、证据选择和拒答标签，student 学结构化结果以降低推理成本。需要独立验证 student 的事实保持、时间定位与错误传播；没有 student、部署目标和 teacher 契约时，不能把 SFT 改名为蒸馏。

### 蒸馏的目标函数选择才是重点

面试里问"你会怎么蒸馏"，答"用 teacher 的输出训 student"是不够的，因为它没有区分三种做法，而这三种在 VideoTrace 上的代价完全不同：

1. **序列级蒸馏（sequence-level KD）**：teacher 采样输出，student 做标准 SFT。实现等价于 SFT，唯一区别是标签来源。优点是不需要 teacher logits，可以离线跑一次就复用；缺点是丢掉了 teacher 的不确定性，且 student 会连 teacher 的错误一起学。
2. **logit 级蒸馏（forward KL）**：`KL(p_teacher || p_student)`，是 mode-covering 的——student 被迫在 teacher 有概率质量的地方都放概率。对 9B→小模型的容量差距，这会让 student 在 teacher 犹豫的位置输出被抹平的分布，在"证据不足要拒答"这种需要果断的行为上表现最差。
3. **反向 KL / on-policy 蒸馏（MiniLLM 一类）**：`KL(p_student || p_teacher)`，mode-seeking，student 只需覆盖 teacher 的一个高概率模式。它更适合本项目——回答格式是受约束的结构化输出，选定一种正确表达即可，不需要覆盖 teacher 的全部表达方式。代价是要在 student 自己的分布上采样，训练循环接近 RL，成本远高于第 1 种。

VideoTrace 若要做，正确的切入点不是蒸馏最终回答，而是蒸馏**片段理解**这一层：它是当前延迟的主要来源（每个时间窗一次 9B 前向），且输出是结构化的 caption/entities/actions/OCR，可以逐字段验证。

### 一个可验收的 teacher/student 契约

不满足下面全部条件就不应该叫蒸馏：

- **teacher**：已冻结的 Qwen3.5-9B 片段理解路径，输出已经按视频/时间窗/帧指纹缓存——也就是说训练数据可以零额外推理成本地从现有缓存产出，这是本项目做蒸馏的实际优势。
- **student**：明确的部署目标（例如可在单卡常驻、把当前冷运行 `26.631s` 中的昂贵前向压下去），而不是"换个小模型试试"。
- **契约**：逐字段的结构化输出 schema，而不是自由文本；否则无法逐项验证。
- **验收**：student 必须在**同一批冻结回归案例**上，保持时间戳绑定、claim-support 与覆盖不回退，并单独报告 OCR 这类困难字段的退化幅度——蒸馏的典型失败是整体指标持平但困难字段崩掉。
- **误差传播**：teacher 本身的错误率是 student 的性能上限。必须先报告 teacher 在冻结集上的错误分类，否则 student 的"接近 teacher"没有意义。

PPO/RLHF 适合有稳定 reward model、在线采样、critic 训练预算和成熟稳定性控制的场景。当前 12 对偏好和 task-local verifier 不足以支撑 reward model；引入 PPO 会增加 critic、rollout、KL 和显存复杂度，并放大奖励偏差，收益不成立。完整目标函数与"为什么 DPO 能省掉 reward model"的推导见上文《DPO 和 PPO/RLHF 是什么关系》。

## 追问速查

| 追问 | 一句话回答 | 支撑证据 |
|---|---|---|
| DPO 为什么不用 reward model？ | 它把 reward 重参数化成 `beta*log(π/πref)`，Bradley-Terry 的差消掉了配分函数 | 本文推导；`src/videomemo/training/dpo_objective.py` |
| `beta` 是什么？调大会怎样？ | 它是隐式 reward 尺度和底层 KL 系数；调大意味着更强的 KL 惩罚、同一偏好 logit 所需策略偏移更小，但 DPO 初始梯度也更大，必须联看实际 KL 与冻结回归 | `beta=0.1`，`outputs/models/qwen35_dpo_metrics.json` |
| 你的 margin 是不是长度造成的？ | 不是；`pearson(token 差, margin) = -0.1986`，但绝对偏好准确率确实受长度影响 | `outputs/reports/dpo_length_bias.json` |
| 一步训练能说明什么？ | 只能说明 12/12 相对 reference 方向正确、闭环真实可恢复；`0/12` 绝对偏好翻转，不能声称能力提升 | 同上 |
| DPO 是否只跑了一步？ | 产品正式 adapter 是 step 1，另有封存 frozen test 的 10 候选 step/beta/seed sweep；选中 `beta=0.05, step=2` 后 frozen margin `0.16528702`、5/5 回归通过，但研究候选未自动替换默认 adapter | `outputs/reports/dpo_sweep.json` |
| 为什么不用 GRPO？ | 当前是静态证据回答，没有动作导致的状态差异；且唯一可用 reward 与偏好数据同源，会形成训练—评测闭环 | 本文 GRPO 章节 |
| 为什么不用 PPO？ | 12 对数据训不出能抵抗策略漂移的 reward model，critic+rollout 成本无收益 | 本文 PPO 章节 |
| 蒸馏你会怎么做？ | 蒸片段理解而非最终回答，用反向 KL，逐字段 schema 验收，先报 teacher 错误率 | 本文蒸馏章节 |
| 灾难性遗忘怎么监控？ | 冻结回归案例 + baseline non-regression 门；DPO 任一门槛失效自动回退 SFT | `outputs/reports/qwen35_dpo_eval.json` |

## 面试回答模板

“我不会先按热度选择 DPO 或 GRPO，而是先看优化对象和数据。VideoTrace 当前优化的是基于既有证据的回答行为，数据是同 prompt 的人工 chosen/rejected，且没有可靠在线 reward 和需要探索的环境，所以 SFT 后接 reference-relative DPO 最匹配。为了适配 24 GiB GPU，我预计算冻结 SFT reference 的 log-prob，只训练一个 LoRA policy。结果说明所有 pair 相对 reference 朝正确方向移动并通过冻结产品门，不等于广泛泛化。GRPO 应用于同 prompt 多 rollout、探索有价值且 reward 可验证的场景；当前若直接把同源 verifier 当 reward，容易出现一律拒答和时间戳模板化等 reward hacking。未来 Agent 具备动态工具决策、多轨迹数据和分解 reward 后，我会在受限轨迹子任务上比较 GRPO 与离线偏好方法。”
