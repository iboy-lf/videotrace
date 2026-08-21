# Training and post-training

VideoTrace 有两条真实可运行的训练链路，均与产品输出相连。

## 1. Query-segment neural reranker

重排器读取 12 个可观测特征：词法覆盖、稀疏检索和 rank、传统 scorer 和 rank、SigLIP 和 rank、基础融合分数、Qwen 置信度、时间位置、时长比例、可检索文本长度。架构为 `12 -> 16 -> 8 -> 1`，输出在实际候选选择前与多信号基础融合分数混合。

监督数据来自 SafeDroid/Yoga 的开发标注；可乐视频不进入训练。当前任务内留出证据为 71 行、25 个正样本、11 个 query group、40 个 held-out pair：基础融合 pairwise accuracy `0.725`，部署融合 `0.775`，推荐 neural blend weight `0.45`。一个单独的 retrieval-rank 信号在这个微小 holdout 上为 `0.825`，所以正确表述是“改善实际多信号融合”，不是“胜过所有信号”。

    cd /lavender/VideoTrace
    VIDEOTRACE_GPU_WAIT_SECONDS=1800 bash scripts/remote/train_reranker.sh

## 2. Qwen3.5 evidence-grounded LoRA SFT

### 数据契约

`data/sft/grounded_qa.jsonl` 共 12 条：train 7、dev 4、test 1；SafeDroid 只进 train，Yoga 只进 dev，可乐只进 frozen test。校验器拒绝重复 record、跨 split video group、可乐泄漏、无证据 abstain 行带 evidence，以及非法时间范围。`gradient_payload_sha256` 只哈希 train split 中真正进入 token/optimizer 的 query/evidence/answer/record_id，不会被 dev/test 路径 provenance 干扰。

### 真实远端运行

    cd /lavender/VideoTrace
    python scripts/build_sft_dataset.py
    CUDA_VISIBLE_DEVICES=0 VIDEOTRACE_PHYSICAL_GPUS=0 /linyuanping/miniconda3/envs/guide2play-qwen35/bin/python scripts/train_qwen35_sft.py --config configs/qwen35_sft.yaml --max-steps 1

已完成的一步 BF16 LoRA 正式运行：trainable params `3,276,800`，total params `9,413,090,544`，train loss `1.4157356024`，dev loss `1.213108`，tokens/s `51.271`，峰值显存 `19130.73 MiB`。checkpoint 同时保存 adapter、tokenizer、`optimizer.pt`、`rng_state.pt`、`trainer_state.json` 和提交式 `checkpoint_manifest.json`。随后从该正式 step 1 checkpoint 真实恢复，仅新增一个 optimizer step：final step `2`、本次 loss `0.6611789465`、dev loss `1.072001`、tokens/s `50.775`、峰值 `19156.73 MiB`。正式产品 adapter 仍是准入的一步权重，恢复目录单独作为故障恢复证据；两者共享数据、源码和 checkpoint contract 哈希。该结果是一条链路可信性和产品准入证据，不被包装成大规模质量提升。

远端 bitsandbytes 为 CPU-only，4-bit QLoRA preflight 会失败并留下明确错误；因此当前真实路径是 BF16 LoRA，而不是虚报量化结果。

### SFT 产品门

`scripts/evaluate_qwen35_adapter.py` 在冻结可乐 pack 上分别运行 baseline/adapter。pack SHA 是排除动态 `metadata.llm_adapter` 的稳定 canonical JSON 哈希，避免把准入记录写回 pack 时形成自引用；原始文件 SHA 仍由 artifact manifest 校验。

## 3. Qwen3.5 evidence-preference LoRA DPO

### 偏好数据

`data/preference/preference_annotations.json` 逐条保存人工编写的 rejected answer、错误类型、理由和 provenance；builder 从 SFT 源复制 query、evidence、chosen、split 和 frozen flag，注释不能偷偷改 split。生成的 `grounded_dpo.jsonl` 为 train/dev/test `7/4/1`，错误类型计数为 wrong timestamp 4、missing timestamp 1、hallucinated detail 2、unsupported overclaim 5。可乐对仍然只在 frozen test，train-only `preference_gradient_payload_sha256` 排除 dev/test。

### 单策略、预计算 reference 的标准 DPO

    cd /lavender/VideoTrace
    /linyuanping/miniconda3/envs/guide2play-qwen35/bin/python scripts/remote/select_gpus.py --stable-checks 3 --audit-log outputs/reports/gpu_selection_dpo.json
    CUDA_VISIBLE_DEVICES=0,1 VIDEOTRACE_PHYSICAL_GPUS=0,1 /linyuanping/miniconda3/envs/guide2play-qwen35/bin/python scripts/train_qwen35_dpo.py --config configs/qwen35_dpo.yaml --force

reference policy 是已准入 SFT adapter。训练器在第一次 optimizer update 前为全部 chosen/rejected 预计算冻结 log-prob，并将 artifact 同数据、SFT 权重、base model 和 max length 绑定；随后只保留一个 9B policy，使用标准目标 `-log sigmoid(beta * ((π_c-π_r) - (ref_c-ref_r)))` 更新 LoRA。这一设计避免两个 9B 模型同时驻留 24 GiB GPU。

真实一步正式运行：DPO loss `0.69314718`，411 response tokens，`45.501 tokens/s`，双卡 model parallel 报告峰值显存 `9715.34 MiB`；更新后 train/dev/frozen-test mean reward margin 为 `0.22440502/0.14014463/0.09733963`，三者 reference-relative preference accuracy 均为 `1.0`。checkpoint 保存 adapter、tokenizer、optimizer、RNG 和绑定 reference/data/SFT/source/config hash 的 trainer state。随后从正式 step 1 checkpoint 真实恢复到 step 2，仅执行一个新 optimizer step：loss `0.59365082`、`61.506 tokens/s`、峰值 `11141.65 MiB`，train/dev/frozen margin 提升到 `0.46911376/0.29970732/0.37768097`。恢复路径仍保持双卡 model parallel，且 contract、reference log-prob、数据、初始 SFT 和源码哈希全部一致。正式产品 adapter 仍使用通过冻结产品门的一步权重；step 2 恢复目录用于证明 checkpoint 可恢复性。该结果证明可运行、可恢复和行为方向正确，不等于大规模泛化提升。

### 最佳 adapter registry

`scripts/select_best_qwen35_adapter.py` 只比较服务端白名单 `qwen35_sft` 和 `qwen35_dpo`。候选必须通过冻结 pack 的 grounding、时间戳、claim-support、coverage non-regression 与 source hash；DPO 还必须通过真实训练、reference chain、dev/frozen reward margin 门。全部通过才选择 DPO，并把 SFT 保留为 hash-validated fallback。`resolve_validated_adapter()` 不信任浏览器路径；权重、config、metrics、model card 或不可变 evaluation 任一哈希变化时，DPO 会回退 SFT。

## 4. Calibrated answer verifier

`scripts/train_answer_verifier.py` 从人工 DPO chosen/rejected 对派生 24 行监督：train 14、dev 8、frozen test 2。每行只包含可审计标量特征，例如时间戳匹配比例、claim-support coverage、拒答一致性和 unsupported-overclaim；可乐 frozen test 不进入梯度，阈值只在 dev 上选择。

训练器使用 `StandardScaler + LogisticRegression`，但 checkpoint 不保存 sklearn 对象，而导出 `mean/scale/coef/intercept/classes`，运行时由 NumPy 计算 sigmoid，格式为 `portable-numpy-logistic-v1`。这解决了本地 sklearn 1.0.2 与远端 1.7.2 的反序列化兼容风险。

当前 checkpoint SHA-256 为 `2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`，threshold `0.2`；dev/frozen accuracy、safe recall 和 pairwise accuracy 均为 `1.0`。canonical 上 `safe_probability=0.92716`。这些数字只说明小规模 task-local veto 契约成立，不是通用 NLI 结论。

硬规则始终先执行；模型只能否决硬规则通过的答案，不能修复错误时间戳、添加证据或把硬失败改成通过。

## 5. DPO、GRPO、蒸馏与 PPO/RLHF 的取舍

- 已实现小规模 DPO，因为任务已有明确正负契约，且它直接影响时间戳、拒答和过度回答偏好。
- 当前没有直接做 GRPO。GRPO 仍然是在线策略优化：需要对同一 prompt 采样多条回答，得到可信的标量 reward，再用组内相对优势更新策略。它不是“只要有 verifier 就能运行”的 DPO 替代品。
- VideoTrace 目前只有 12 组人工、可审计的 chosen/rejected 对（train/dev/frozen test 为 `7/4/1`），而不是足够多的独立 prompt、稳定的多轨迹采样和可靠 reward model。现有 calibrated verifier 只读取少量可审计特征，是硬规则之后的 task-local safety veto；它可以否决可疑回答，但不具备作为在线 RL dense reward 的覆盖面和鲁棒性。
- 直接把 verifier 分数喂给 GRPO 会有明显的 reward hacking 风险：策略可能学会一律拒答、复制时间戳格式、堆叠证据关键词或钻过 verifier 的特征漏洞，而不是提升视频事实和时间定位。多步 Agent 轨迹还会引入检索、视觉理解、工具调用和生成之间的 credit-assignment 问题。
- 当前 24 GiB 级别 GPU 约束下，DPO 可以预计算冻结 reference 的 chosen/rejected log-prob，并在小规模静态数据上复现；GRPO 还要承担每个 prompt 的多次生成、打分、log-prob 和 KL/长度控制，成本与方差都会显著增加。
- 因此，当前选择是 `SFT -> 显式偏好对 -> reference-relative DPO`：先建立证据、时间戳和拒答契约，再用可解释的对比样本把策略推向产品需要的行为。DPO 的结果仍只作为任务内闭环证据，不声称广泛泛化。
- 未实现 PPO/RLHF：同样因为当前没有可靠在线 reward model；引入 critic、在线采样和更复杂的稳定性控制，会放大 verifier 偏差、增加采样成本和显存，并不适合这个小规模产品闭环。
- 未把 DPO 包装成通用对齐：偏好对是任务内人工 contrast，不是线上用户排序。
- 如做蒸馏，必须定义 teacher/student 契约（teacher 产出候选证据、回答与拒答标签，student 学结构化目标），不能把 SFT 改名。

### 什么时候才值得做 GRPO？

只有在补齐以下条件后，才把 GRPO 作为 Agent 轨迹或答案策略的独立实验：至少数百个按视频隔离的 prompt、每个 prompt 的多轨迹采样日志、可分解且不依赖单一 verifier 的 reward（claim-support、时间窗 IoU、证据充分性、拒答校准、工具成本）、reward-hacking 回归集，以及完全独立的视频 frozen test。届时优先在受限的工具选择/轨迹决策子任务上做小规模 GRPO，与当前 DPO 在同一冻结集上比较；在这些证据出现前，不把 GRPO 写成已实现能力。

## 6. 失败恢复和排错

训练失败时优先检查数据 split/gradient hash、模型本地路径、dtype/quantization preflight、显存和 checkpoint 可写性。远端 GPU 选择器必须连续稳定探测并记录物理卡；不满足条件就等待，不杀进程。所有训练/评测产物最终进入 `artifact_manifest.json`。
