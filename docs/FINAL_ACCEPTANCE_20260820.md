# VideoTrace 最终验收快照（2026-08-20）

本文件记录指定可乐视频上的任务内最终验收，不是公开 benchmark。机器可读权威证据是 `outputs/reports/artifact_manifest.json`、`outputs/reports/delivery_readiness.json`、`outputs/reports/browser_e2e.json` 和 `outputs/runs/latest/jobs/bba9b6fa535e45d596d2d31c7b9aadb4.json`。

## 代码与交付包

- 本地 pytest：`121 passed, 1 skipped`；iboy 标准入口 `bash scripts/remote/run_tests.sh`：`122 passed`。远端测试复用现有 `wyf_vm` 解释器，模型服务、训练和推理固定使用 `guide2play-qwen35`。
- Python `compileall`、4 个 JS syntax check、3 个 JS behavior test 全部通过。
- interview package：`17/17`；delivery readiness：`40/40`，`failures=[]`。
- 当前产品源码 SHA-256：`77091a151747aa189d28cf85ff29cc5b9b2d05de74cb4b5fe0da91b9f3ad363a`。
- 不可变训练源码 SHA-256：`590521072edecaa85c8f4e9ddb591f7c6aa1e7b6bd8aeba96707daad87b4238e`；训练来源与当前产品重新准入来源分开记录，不把 UI/验收改动冒充重新训练。
- 指定视频 SHA-256：`04b0b3320cb1776069e056bee59095841f7d6d61490bcc0838835eeaf96f5781`。
- canonical pack 稳定 SHA-256：`460d2299530bbbfae4c20ea1db61c53db6880a9d48223719ce42e710bf44f4d8`；原始文件 SHA-256：`ea86abd823d535308c6885d8dfeefe9088888e9f9f9a630edb677b4f7423723a`。
- DPO adapter SHA-256：`11b5d8ea86227ad63450f658d06864c2086d7abcc532c78d95315432237b37c6`；SFT fallback SHA-256：`d224b1ea1066f67906deca440f9aa3c2e2099fec4589ec4d30d0c485fa703f98`。
- best-adapter registry SHA-256：`b36614bdf314866a94f57f4fd887b3277f9143e24cfc66a8d026aae04483a641`，选择 `qwen35_dpo`；DPO/SFT evaluation SHA 分别为 `d28ae971b5c80fe21e265999f2f2d24c2f7b2b1ce988e0bdc885b1d2117759ed` 和 `9b91e531d627425ecc496bfa497138f4253ebd3548197bde39beee427084b57a`。
- artifact manifest 覆盖 51 个核心制品和 14 份不可变 adapter 准入历史；manifest SHA-256 为 `ca2e7efdcf543c7826c447a38bf03129fa59088765c448b0880d715b2f06793c`。

本地旧产物没有删除；本轮同步前报告归档为 `D:\Agent\VideoTrace_archives\20260820_launcher_fix_pre_77091a15_local\evidence_before_sync.tar.gz`。远端源码和最终证据分别归档到 `/lavender/VideoTrace_history/20260820_launcher_fix_pre_0a236f90` 与 `/lavender/VideoTrace_history/20260820_launcher_fix_final_77091a15`。

## 真实 Web 链路

- 远端服务：`iboy:/lavender/VideoTrace`；受管 Web PID `2030289`，PID、CWD 和启动命令已核验为 VideoTrace 自有进程。
- 本地访问：`http://127.0.0.1:7860`；SSH tunnel PID `127800`。
- 一键启动：`powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1`。
- 服务端只返回真实白名单：自动最佳、Qwen3.5 视频理解、SigLIP2 检索增强；浏览器不能指定模型路径、backend、adapter、device 或 dtype。
- 上传先显示 `blob:` 即时预览，再切换到项目内 `data/uploads` 的 `/media` URL；GPU 请求进入单 worker 串行队列，模型实例常驻复用。
- 最终 E2E 任务 `bba9b6fa535e45d596d2d31c7b9aadb4` 经历 `queued → checking_resources → loading_models → analyzing → exporting → completed`，终态耗时 `17.7s`。
- 完成后等待 1.2 秒再次查询，`elapsed_sec` 仍为 `17.7s`；它与 `completed_at - started_at = 17.657s` 一致，证明终态耗时不再随服务存活增长。
- 任务记录 `durable=true, restored=false`。持久化恢复路径有单元测试，但本轮没有把“真实完成任务曾被重启恢复”包装成已发生事实。
- canonical 六步 trace 中 `synthesize_answer` 为 `17.136s`，其余编排工具合计远低于生成耗时；当前主要延迟在模型生成而非 Python 编排。

## 播放、布局和 Range

- 引用证据从 `start_sec` 播放，在 `end_sec` 自动暂停并释放 `activeWindow`。
- 点击“从当前位置继续播放”会清除窗口限制并立即播放。
- 点击视频脉络节点只 seek 到起点，越过节点结尾仍持续播放。
- HTTP Range：`206`，`Content-Range: bytes 0-1023/29935671`，长度 1024 字节。
- 桌面 1440×900 与移动端 390×844 均无横向溢出，核心控件在视口内；console warning/error 与 page error 均为 0。
- 浏览器报告 SHA-256：`b8bab40ed134c9abfbeaacc37283ade33ae9562c81fc482f84d7a19611dd0664`，19 项检查全部为 true。

## 算法与训练证据

- canonical 回答覆盖 `0-20`、`200-220`、`300-320`、`400-416.2` 四个窗口，硬时间戳规则、claim-support 和 Agent verifier 全通过。
- calibrated answer verifier 使用 24 行 `14/8/2` 数据和 `portable-numpy-logistic-v1` checkpoint；SHA-256 为 `2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`。
- canonical 上 `safe_probability=0.92716`、阈值 `0.2`；它只能否决硬规则已通过的可疑答案，不能补证据、修时间戳或覆盖硬失败。
- SFT 正式一步：train/dev/frozen test `7/4/1`，train loss `1.4157356024`，dev loss `1.213108`，`51.271 tokens/s`，峰值 `19130.73 MiB`。真实恢复到 step 2：loss `0.6611789465`，dev loss `1.072001`，`50.775 tokens/s`，峰值 `19156.73 MiB`。
- DPO 正式一步：loss `0.69314718`，`45.501 tokens/s`，峰值 `9715.34 MiB`，train/dev/frozen reward margin `0.22440502/0.14014463/0.09733963`。真实恢复到 step 2 后为 loss `0.59365082`、`61.506 tokens/s`、峰值 `11141.65 MiB`，margin `0.46911376/0.29970732/0.37768097`。
- 正式一步 DPO adapter 进入产品，SFT 是 hash-validated fallback；step 2 目录只作为恢复可信度证据。
- 冻结可乐回归：`5/5`，`error_category_counts={"none": 5}`，包含无证据拒答。
- 性能：冷 `27.559s`，热 `14.901s`，`1.849×`；Qwen peak allocated 冷/热为 `19501.21/19487.36 MiB`，SigLIP 为 `1692.13 MiB`；热运行 Qwen/SigLIP 均 `21/21` 命中且正确性不回退。

## GPU 非干扰证据

- 最终 canonical、adapter evaluation、冻结回归、性能和 Web 均显式绑定物理 GPU `0,1`；每次启动前至少连续三次检查 compute PID、显存和利用率。
- 历史正式 DPO 训练记录物理 GPU `0,1`，SFT 训练记录 GPU `0`；历史训练卡与最终产品卡是独立审计事实。
- 选择器只读，不 kill、不 signal、不抢占。旧 VideoTrace Web 只在队列空闲且 PID/CWD/命令全部匹配后使用 SIGTERM 优雅停止并重启为当前源码；未触碰其他项目进程。

## 明确边界

当前实现覆盖 LLM/VLM/多模态 Agent、reranker、SFT、标准 reference-relative DPO、task-local calibrated safety veto、真实 Web、失败恢复和推理 profile。当前没有部署 speech ASR 权重；GRPO、PPO/RLHF、VLA、Speech LLM、从零预训练和公开榜单没有实现，也不作为成果声称。GRPO 只有在具备足够多独立 prompt、多轨迹采样、可分解 reward、reward-hacking 回归和独立视频 frozen test 后，才值得用于受限工具/轨迹策略实验。
