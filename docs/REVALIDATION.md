# 证据链重新验证（REVALIDATION）

## 为什么改一行代码就会让交付校验变红

`src/videomemo/eval/reproducibility.py:source_fingerprint` 对 `src/`、`scripts/`、`configs/`、`tests/`、`pyproject.toml` 和 `README.md` 的全部内容取 SHA-256，并把这个指纹写进每一个机器产物：canonical 知识包、模型卡、adapter 评估、性能报告、浏览器 E2E 和交付 manifest。

因此 `scripts/validate_delivery_package.py` 里的 `core::source_fingerprint_matches` 一类检查回答的不是"这些文件存在吗"，而是**"这些证据是不是由现在这份源码产生的"**。

这是有意的设计。它的代价是：任何产品源码改动——哪怕只是新增一个测试——都会让证据链失效，直到在 `iboy` 上重跑一遍。它的收益是：绿灯不可能是陈旧的，也不可能靠手改 JSON 得到。面试里这一点本身就是可讲的工程决策，不是缺陷。

## 什么可以在本地恢复，什么必须在远端重跑

| 产物 | 能否本地重建 | 说明 |
|---|---|---|
| `outputs/reports/artifact_manifest.json` | 是 | `python scripts/build_artifact_manifest.py`，只对本地文件取哈希 |
| `outputs/reports/documentation_consistency.json` | 是 | `python scripts/validate_documentation_consistency.py` |
| `outputs/reports/documentation_links.json` | 是 | `python scripts/validate_documentation_links.py` |
| `outputs/reports/dpo_length_bias.json` | 是 | `python scripts/analyze_dpo_length_bias.py`，只读已提交的训练产物 |
| canonical 知识包与模型卡 | 否 | 需要 Qwen3.5-9B + SigLIP2 前向 |
| adapter 评估与准入 | 否 | 需要加载 9B 权重 |
| `performance_report.json` | 否 | 冷/热延迟与峰值显存必须在目标 GPU 上测 |
| `browser_e2e.json` | 否 | 需要运行中的远端 Web 服务与真实任务队列 |

不要为了让校验变绿而手工编辑上表下半部分的任何文件：那样得到的绿灯正是这个项目明确拒绝的东西。

## 改动产品源码后的完整重跑顺序

### 一条命令

同步源码后，整条链路由 `scripts/remote/revalidate_all.sh` 按依赖顺序执行：

```bash
# 忙碌主机上先干跑，检查前置条件，不消耗任何 GPU
bash scripts/remote/revalidate_all.sh --dry-run

# 正式执行；最多等 1 小时直到出现一对安全空闲卡
bash scripts/remote/revalidate_all.sh --wait 3600
```

GPU 安全策略不重新实现，全部走 `scripts/remote/select_gpus.py`：要求连续稳定探测确认真正空闲、不碰他人进程；等待窗口内没有安全的卡就非零退出，不会挤占。跑完默认停掉常驻 Web 服务（`--keep-web` 可保留），因为这是共享机器，不该让一个 9B 模型在无人值守时占着卡。

### 或者逐条执行

在 `iboy` 上，项目目录 `/lavender/VideoTrace`：

```bash
# 0. 同步当前源码到远端（在本地 Windows 执行）
powershell -ExecutionPolicy Bypass -File .\scripts\remote\sync_to_iboy.ps1

# 1. 远端测试基线（首次预检会暂时跳过依赖旧机器报告的文档一致性测试；最终校验仍必须完整运行）
bash scripts/remote/run_tests.sh

# 2. 重新生成 canonical 知识包（Qwen3.5 + SigLIP2 全链路）
bash scripts/remote/run_qwen35_demo.sh

# 3. 在安全空闲卡上依次跑 baseline、SFT、DPO 和 compare，再重新准入；
#    DPO 未过门时自动回退 SFT
bash scripts/remote/run_adapter_evaluations.sh

# 4. 冷/热性能 profile
bash scripts/remote/run_profile_runtime.sh

# 5. 重新跑 5 个冻结回归案例（同样走 GPU 安全选择器）
bash scripts/remote/run_regression_suite.sh

# 6. 常驻 Web 服务 + 浏览器 E2E（传入真实视频路径）
bash scripts/remote/start_web_service.sh
bash scripts/remote/run_browser_e2e.sh /lavender/VideoTrace/data/raw/cola_review.mp4
```

回到本地后：

顺序有依赖，不能打乱：manifest 必须先于 delivery 重建，delivery 的结果又会被文档一致性检查读取。

```powershell
$env:PYTHONPATH = 'src'
python scripts/analyze_dpo_length_bias.py
python scripts/build_artifact_manifest.py
python scripts/validate_delivery_package.py          # 期望 40/40
python scripts/validate_documentation_consistency.py --strict # 期望全部通过
python scripts/validate_documentation_links.py
```

`validate_documentation_consistency.py` 会在这一步拦截"机器指标已更新但 Markdown 还写着旧数字"的情况，所以文档里的数字应当在它报错时修改，而不是反过来。

## 当前状态

2026-08-23 已按上述顺序完成全量重验证：真实视频上传 Web E2E、canonical、SFT/DPO candidate compare 与准入、5 个冻结回归、失败恢复、冷/热 profile、reranker model card 和最终 manifest 均绑定源码 `642cc023…`。当前 delivery 为 `40/40`，interview package 为 `17/17`；最新 job、哈希和性能数字已写入最终验收与面试文档。
