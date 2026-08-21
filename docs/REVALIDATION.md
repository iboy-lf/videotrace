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

在 `iboy` 上，项目目录 `/lavender/VideoTrace`：

```bash
# 0. 同步当前源码到远端（在本地 Windows 执行）
powershell -ExecutionPolicy Bypass -File .\scripts\remote\sync_to_iboy.ps1

# 1. 远端测试基线
bash scripts/remote/run_tests.sh

# 2. 重新生成 canonical 知识包（Qwen3.5 + SigLIP2 全链路）
bash scripts/remote/run_qwen35_demo.sh

# 3. 重新评估并重新准入 adapter；DPO 未过门时自动回退 SFT
python scripts/evaluate_qwen35_adapter.py
python scripts/select_best_qwen35_adapter.py

# 4. 冷/热性能 profile
python scripts/profile_runtime.py

# 5. 常驻 Web 服务 + 浏览器 E2E
bash scripts/remote/start_web_service.sh
bash scripts/remote/run_browser_e2e.sh
```

回到本地后：

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

本轮改动涉及 `scripts/start.py`、`src/videomemo/web/static/*`、新增的诊断脚本与测试，因此源码指纹已经变化。本地可重建的四个产物已经重跑并与当前源码一致；依赖 GPU 与远端 Web 的六项检查需要按上面的顺序在 `iboy` 上重跑一次才会恢复 40/40。README 与 `docs/FINAL_ACCEPTANCE_20260820.md` 记录的是上一次完整验收快照，重跑后应一并更新其中的 job id、哈希与性能数字。
