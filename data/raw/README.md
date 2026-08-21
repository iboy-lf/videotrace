# data/raw

源视频是第三方内容，不随仓库分发。这个目录在 clone 后是空的，只有本文件。

## 需要哪些文件

`outputs/reports/artifact_manifest.json` 固定了每个视频的 SHA-256。canonical 演示只依赖一个文件：

| 路径 | 用途 | 说明 |
| --- | --- | --- |
| `data/raw/cola_review.mp4` | canonical 演示与冻结回归的唯一视频 | 全球可乐横评，约 416 秒 |
| `data/raw/yoga.mp4` | 开发期 reranker 监督 | 可选 |
| `data/raw/safedroid_demo.mp4` | 开发期 reranker 监督 | 可选 |

## 没有视频时会发生什么

不会伪造演示。行为是明确降级的：

- `scripts/start.py` 进入 evidence-only 模式：回答、证据、时间戳、视频脉络和技术面板全部来自已提交的 `outputs/iboy_qwen35/cola_review/knowledge_pack.json`，页面顶部明确标注视频文件缺失、证据不可回看。
- `scripts/validate_outputs.py` 会因视频 SHA-256 不匹配而失败，而不是通过。
- `scripts/validate_delivery_package.py` 和 `scripts/validate_interview_package.py` 依赖视频与模型权重，在没有这些文件时应当失败；CI 因此不运行它们（见 `.github/workflows/ci.yml`）。

替换成任意同名视频不会让演示"看起来能跑"：knowledge pack 记录的是原视频的哈希与时间窗，哈希校验会直接拒绝。
