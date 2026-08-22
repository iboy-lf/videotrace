# VideoTrace 最终验收记录（2026-08-20 建档，2026-08-23 刷新）

本文件记录指定 416.2 秒可乐视频上的任务内验收，不是公开 benchmark。当前机器证据以 `outputs/reports/artifact_manifest.json`、`outputs/reports/delivery_readiness.json`、`outputs/reports/browser_e2e.json` 和 `outputs/runs/latest/jobs/f6d582f2ef434a689845b23e8c671f07.json` 为准。

## 代码与交付闭环

- interview package：`17/17`；delivery readiness：`40/40`，`failures=[]`。
- 本地完整 pytest：`147 passed, 1 skipped`；唯一 skip 是本地环境没有 Torch，远端 GPU 环境已执行相应 DPO 契约。远端重验证预检 `145 passed`，并按依赖顺序把文档一致性留到最终阶段单独严格执行。
- 当前产品源码 SHA-256：`7374516adfcea3f8f4505117c6d7c5362975f4ce0de45efad595f41226f195fc`。
- 不可变训练源码 SHA-256：`590521072edecaa85c8f4e9ddb591f7c6aa1e7b6bd8aeba96707daad87b4238e`。产品重新准入来源与训练来源分开记录，没有把 UI 或验收改动冒充重新训练。
- 指定视频 SHA-256：`04b0b3320cb1776069e056bee59095841f7d6d61490bcc0838835eeaf96f5781`。
- canonical pack 稳定 SHA-256：`eb97e384b17df7a917627b1c97bd5072f3f77fc3975a385a1bba8ce0f05765c6`；原始文件 SHA-256：`483d72b8511e4d5f46cc1e7025ccbe35f6556762b25020d3d601334ac08cae25`。
- DPO adapter SHA-256：`11b5d8ea86227ad63450f658d06864c2086d7abcc532c78d95315432237b37c6`；SFT fallback SHA-256：`d224b1ea1066f67906deca440f9aa3c2e2099fec4589ec4d30d0c485fa703f98`。
- best-adapter registry SHA-256：`c3dc42bcd327485399eb436de23e853521f0f223513e7dd202f19dc790c05d3d`，选择 `qwen35_dpo`；DPO/SFT evaluation SHA 分别为 `d4b73a5eb91ad784dfa26f5831a3a013cb4766d4b7e1961cd3a7f34b5e95c55a` 和 `97060b2e52d89996f6b33c66ac3f51186bb4b5ee13878707dbcb4ea32ecc76de`。
- artifact manifest 覆盖 `60 个`核心制品和 `20 份`不可变 adapter 准入历史；当前 manifest SHA-256 以 `outputs/reports/artifact_manifest.json` 为准。

## 真实 Web 链路

- 远端项目为 `iboy:/lavender/VideoTrace`，本地通过 `http://127.0.0.1:7860` 访问；一键入口是 `powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1`。
- 服务端只暴露真实白名单模式：自动最佳、Qwen3.5 视频理解、SigLIP2 检索增强；浏览器不能指定模型路径、adapter、device、dtype 或任意 backend。
- E2E 使用真实视频上传：先显示 `blob:` 即时预览，再切换到项目内 `/media` URL，并进入单 worker GPU 队列。
- 最终任务 `f6d582f2ef434a689845b23e8c671f07` 经历 `queued → checking_resources → loading_models → analyzing → exporting → completed`，终态耗时 `29.2s`。
- 完成后再次查询，`elapsed_sec` 仍为 `29.2s`；实际 `completed_at - started_at = 29.186s`，终态耗时不会随服务存活继续增长。
- 任务记录 `durable=true, restored=false`。持久化恢复路径由单元测试覆盖，但本轮没有把未发生的真实重启恢复包装成事实。
- 浏览器报告 SHA-256：`0b05057627b7bee829455c73766f5b8861bccdaf25c07278966bbbc77331dd7c`，`19 项`桌面、移动、上传、轮询、播放、Range 与 console 检查全部通过。

## 播放、布局与证据

- 引用证据从 `start_sec` 播放，到 `end_sec` 自动暂停并释放时间窗；“从当前位置继续播放”会清除限制。
- 视频脉络节点只 seek 到起点，之后持续播放，不会在节点末尾强制暂停。
- HTTP Range 返回 `206`，`Content-Range: bytes 0-1023/29935671`，长度 1024 字节。
- 桌面与移动布局均无横向溢出；console warning/error 与 page error 均为 0。
- canonical 回答绑定 `0-20`、`200-220`、`300-320`、`400-416.2` 四个窗口，硬时间戳规则、claim-support 与 Agent verifier 全通过。

## 训练与回归证据

- calibrated answer verifier 使用 24 行 `14/8/2` 数据与 `portable-numpy-logistic-v1` checkpoint；SHA-256 为 `2bf74b976b366fb939368cb82c8077aeae32b703ee2ccf69da7bda394f2e7db1`。canonical 上 `safe_probability=0.92716`、阈值 `0.2`；它只能否决，不能覆盖硬失败。
- SFT 正式一步：train/dev/frozen `7/4/1`，train loss `1.4157356024`，dev loss `1.213108`，`51.271 tokens/s`，峰值 `19130.73 MiB`；真实恢复到 step 2 后 loss `0.6611789465`、dev loss `1.072001`、`50.775 tokens/s`、峰值 `19156.73 MiB`。
- DPO 正式一步：loss `0.69314718`，`45.501 tokens/s`，峰值 `9715.34 MiB`，train/dev/frozen reward margin `0.22440502/0.14014463/0.09733963`；真实恢复到 step 2 后 loss `0.59365082`、`61.506 tokens/s`、峰值 `11141.65 MiB`，margin `0.46911376/0.29970732/0.37768097`。
- 研究 DPO sweep（不替换默认 Web adapter）：封存 frozen test 的 10 个 step/beta/seed 候选仅按 dev 选择；最终选中 `beta=0.05, step=2`，3 个 seed 的 dev margin `0.15501256±0.00265381`、绝对偏好准确率均为 `1.0`；一次性 frozen test margin `0.16528702`、accuracy `1.0`，独立产品回归 `5/5`。
- 正式一步 DPO adapter 进入产品，SFT 是 hash-validated fallback；step 2 目录仅作为恢复可信度证据。
- 冻结回归为 `5/5`，`error_category_counts={"none": 5}`，覆盖全局流程、开场产品、困难 OCR、末尾盲测和无证据拒答。
- 性能：冷 `26.631s`，热 `15.151s`，`1.758×`；Qwen peak allocated 冷/热为 `19487.53/19469.84 MiB`，SigLIP 为 `1691.63 MiB`；热运行 Qwen/SigLIP 均 `21/21` 命中且正确性不回退。
- 同一 frozen pack 的模型选型对照：Qwen3.5-9B `22.778s`、Qwen2.5-VL-7B `15.52s`，两者均通过 grounding、claim-support 和 timestamp binding；因此保留 Qwen3.5 作为质量/产品路径，Qwen2.5 作为可复现实验中的速度对照，而不是只按参数量选模型。

## GPU 非干扰与边界

- canonical、adapter evaluation、冻结回归、性能和 Web 均在启动前连续三次检查 compute PID、显存与利用率，并绑定安全空闲卡；选择器只读，不 kill、不 signal、不抢占。
- 当前实现覆盖 LLM/VLM/Agent、reranker、SFT、标准 reference-relative DPO、task-local calibrated veto、真实 Web、失败恢复和推理 profile。
- 当前没有部署 speech ASR 权重；GRPO、PPO/RLHF、VLA、Speech LLM、从零预训练和公开榜单没有实现，也不作为成果声称。
