# VideoTrace Calibrated Answer Verifier Model Card

## Intended use

这是 VideoTrace 回答生成后的保守安全否决层。它只在确定性时间戳和 claim-support 规则已经通过时执行，用于拒绝多个弱风险信号组合下的低置信答案。

## Model and data

- 模型：标准化特征上的二分类逻辑回归。
- 运行格式：`portable-numpy-logistic-v1`，只保存数值参数，不反序列化 sklearn estimator。
- 数据：从 12 组人工 DPO 偏好对派生 24 行，train/dev/frozen test 为 `14/8/2`。
- 可乐视频：仅 frozen test，不进入梯度或阈值选择。
- checkpoint SHA-256：`2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`。

## Task-local results

- threshold：`0.2`，只由 dev 选择。
- dev accuracy / safe recall / pairwise accuracy：`1.0 / 1.0 / 1.0`。
- frozen test accuracy / safe recall / pairwise accuracy：`1.0 / 1.0 / 1.0`。
- canonical safe probability：`0.92716`。

## Safety contract

- 硬时间戳、证据充分性或 claim-support 失败永远保持失败。
- 模型不能添加证据、改写时间戳或修复回答。
- 加载失败是否 fail-open 由服务端配置固定，浏览器不能控制。

## Limitations

- 数据规模极小，且 frozen cola 只有一对 chosen/rejected。
- 输入是文本/结构标量，不直接进行视觉 entailment。
- 不应作为通用 NLI、reward model 或公开 benchmark 结果使用。
