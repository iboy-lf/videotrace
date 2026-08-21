# iboy 远端运行

## 固定环境

- 项目：`/lavender/VideoTrace`
- Python：`/linyuanping/miniconda3/envs/guide2play-qwen35/bin/python`
- Qwen3.5-9B：`/lavender/models/Qwen3.5-9B`
- SigLIP2：`/lavender/models/siglip2-large-patch16-256`

复用现成环境，不创建新 conda 环境，不中断其他 GPU 进程。

## 同步代码

    powershell -ExecutionPolicy Bypass -File scripts\remote\sync_to_iboy.ps1

同步脚本不覆盖远端 `data/sft`、模型、缓存和 outputs；历史结果只移动到归档目录。

## GPU 安全协议

所有 GPU 脚本调用 `select_gpus.py`：检查 compute PID、显存、利用率，并要求至少 3 次稳定探测。选择结果写入 `outputs/reports/gpu_selection_canonical.json`（Web 服务写入 `outputs_runtime/web/gpu_selection_audit.json`），包括每次探测的物理卡、PID 和阈值。选择器只读，不 kill、不 signal、不抢占。

    cd /lavender/VideoTrace
    nvidia-smi
    nvidia-smi
    nvidia-smi

## 真实 canonical 运行

    VIDEOTRACE_GPU_WAIT_SECONDS=1800 bash scripts/remote/run_qwen35_demo.sh /lavender/VideoTrace/data/raw/cola_review.mp4 '这个视频的整体流程是什么？请概括开场、分国家试喝和最后盲测三个阶段并给出时间戳。'

## 启动 Web（推荐）

本地：

    powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1

或远端手动：

    VIDEOTRACE_GPU_WAIT_SECONDS=1800 bash scripts/remote/run_web.sh

本地访问 `http://127.0.0.1:7860`。服务使用常驻 pipeline、串行 GPU worker、上传目录约束和轮询任务状态。停止服务前必须先检查 PID 是否属于 VideoTrace，不得按端口盲杀。

## 验证与制品

    bash scripts/remote/run_tests.sh
    bash scripts/remote/validate_interview_package.sh

`run_tests.sh` 会在已有环境中自动选择同时具备 pytest、scikit-learn、OpenCV MP4 编码能力的测试解释器；本次实际选择 `/linyuanping/miniconda3/envs/wyf_vm/bin/python`。模型服务、训练和推理仍固定使用 `guide2play-qwen35`。不要用 `guide2play-qwen35/bin/python scripts/remote/run_pytest.py` 代替标准入口：该模型环境当前没有独立 pytest/OpenCV 测试栈，追加系统 site-packages 还会触发 NumPy 2 与旧 OpenCV ABI 冲突。标准脚本只复用现有环境，不创建或安装新环境。

远端 Web 已健康后，可执行真实浏览器 E2E。运行真实分析前仍须按 GPU 安全协议确认当前 GPU 进程只属于 VideoTrace：

    bash scripts/remote/run_browser_e2e.sh /lavender/VideoTrace/data/raw/cola_review.mp4

最终交付以 `outputs/reports/artifact_manifest.json` 和 `outputs/reports/delivery_readiness.json` 为准。
