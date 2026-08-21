# SFT 数据卡

- schema: `videotrace-grounded-sft-v1`
- records: 12（train 7 / dev 4 / frozen test 1）
- behavior: answer 7 / abstain 5
- groups: SafeDroid train、Yoga dev、cola review frozen test
- provenance: 人工核验证据标注 + frozen canonical pack；不是公开 benchmark

## 防泄漏契约

每个 video group 只能属于一个 split；可乐不能进入 train/dev；frozen test 不进入 optimizer；abstain 行不允许携带 evidence；时间范围必须满足 `end_sec > start_sec`。构建和训练前都会运行 `validate_sft_records()`。

## 梯度载荷

`gradient_payload_sha256` 只覆盖 train split 的 `record_id/query/evidence/answer`。Windows 与 iboy 的绝对路径 provenance 可以不同，但真正送入 token/梯度的字段必须同 hash。最终权威值以 `outputs/reports/artifact_manifest.json` 为准。

## 限制

数据规模小、人工标注集中于证据格式和拒答，不能证明通用视觉识别或跨域泛化；可乐只用于最终冻结回归和 adapter admission。

