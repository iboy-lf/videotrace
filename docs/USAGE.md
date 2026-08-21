# 使用说明

## 本地只读演示

    cd D:\Agent\VideoTrace
    powershell -ExecutionPolicy Bypass -File .\start.ps1

本地启动器优先读取 `outputs/iboy_qwen35/cola_review/knowledge_pack.json`。canonical 中保留的 `/lavender` 视频路径只作为来源记录；启动器会用知识包里的视频 SHA-256 在项目 `data/raw` 中寻找字节一致的本地视频，匹配后才启用原视频时间窗，不改写 canonical 产物，也不会误降级到旧 pack。若本机没有 `/lavender` 权重，页面仍显示上传、问题和视觉模式入口，并明确提示远端算力未连接；不会把入口隐藏成“产品不可操作”。

## 远端完整产品

    cd D:\Agent\VideoTrace
    powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1

默认 URL：`http://127.0.0.1:7860`。启动过程：远端三次稳定 GPU 探测 → 常驻 Web 服务 → SSH 隧道。远端工作目录为 `/lavender/VideoTrace`，上传文件落在项目内 `data/uploads`，服务端校验扩展名、大小和路径。

若本地 7860 已被其他服务占用，不要终止它，直接使用：

    powershell -ExecutionPolicy Bypass -File .\scripts\start_remote.ps1 -LocalPort 7861

然后访问 `http://127.0.0.1:7861`。

页面流程：

1. 选择视频后立即预览，并等待上传完成状态。
2. 从服务端返回的真实模式中选择“自动最佳”或其他视觉模式。
3. 输入问题，点击分析；页面轮询 `/api/jobs/{job_id}` 显示检查资源、加载模型、分析、导出等阶段。
4. 结果中查看回答、证据、时间线和技术摘要。

主要 API：`/api/health`、`/api/capabilities`、`/api/upload`、`/api/jobs`、`/api/jobs/{id}`、`/api/latest`。GPU 请求串行，模型实例和缓存复用。

## 播放语义

- 引用证据：从 `start_sec` 开始，在 `end_sec` 暂停，然后自动清除 `activeWindow`。
- 视频脉络：seek 到节点起点后播放完整视频，不在节点结尾暂停。
- “从当前位置继续播放”：清除时间窗并调用 `video.play()`。
- 手动拖动明显越出证据窗口、切换视频或重新分析，会清除旧窗口状态。

## 验收命令

    $env:PYTHONPATH = 'src'
    python -m pytest -q
    python -m compileall -q src scripts tests
    node --check src/videomemo/web/static/app.js
    node --check src/videomemo/web/static/playback.js
    node --check src/videomemo/web/static/technical.js
    node --check src/videomemo/web/static/job_status.js
    node tests/js/playback.test.cjs
    node tests/js/technical.test.cjs
    node tests/js/job_status.test.cjs
    python scripts/validate_outputs.py outputs/iboy_qwen35/cola_review/knowledge_pack.json --video data/raw/cola_review.mp4
    python scripts/build_artifact_manifest.py
    python scripts/validate_delivery_package.py
