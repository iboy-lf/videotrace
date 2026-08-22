# 推理性能报告

报告：`outputs/reports/performance_report.json`。范围是指定 416.2 秒可乐视频的一次真实任务 profile，不是公开 benchmark。

| 项目 | 实测 |
| --- | ---: |
| 冷 pipeline 构造+运行 | 26.531 s |
| 热缓存运行 | 15.228 s |
| speedup | 1.742× |
| Qwen peak allocated（冷/热） | 19487.53 / 19469.84 MiB |
| SigLIP peak allocated | 1691.63 MiB |
| 热 Qwen 片段缓存 | 21/21 命中 |
| 热 SigLIP 缓存 | 21/21 命中 |

产品使用常驻 pipeline、串行 GPU worker、片段理解缓存、SigLIP index 复用和 BF16。冷/热答案均 `verified=true`，时间戳绑定保持正确。4bit preflight 因现有 bitsandbytes 为 CPU-only 而明确失败，因此没有虚报 int8/4bit 成功。

profile 前连续三次确认物理 GPU `0,1` 无 compute PID，并显式绑定 `CUDA_VISIBLE_DEVICES=0,1`。这组数据只证明当前任务的缓存收益、显存和正确性守恒，不用于榜单比较。
