const state = {
  data: null,
  capabilities: null,
  currentJobId: "",
  previewUrl: "",
  uploading: false,
  running: false,
  mediaReady: true,
};

const PRESET_QUERIES = [
  { label: "整体流程", value: "这个视频主要讲了什么？请给出带时间戳的证据。" },
  { label: "产品与口感", value: "视频里介绍了哪些产品？请结合时间戳概括口感或比较内容。" },
  { label: "最后环节", value: "视频最后的盲测环节是什么？请给出对应时间戳。" },
  { label: "无证据拒答", value: "视频里主持人有没有驾驶汽车穿越沙漠？请给出证据。" },
];

const PHASE_LABELS = {
  queued: "等待 GPU 队列",
  checking_resources: "复核 GPU 状态",
  loading_models: "加载或复用模型",
  analyzing: "视频理解与证据生成",
  exporting: "整理知识包",
  completed: "分析完成",
  failed: "分析失败",
};

const TEXT = {
  running: "任务已提交，正在分析视频。",
  done: "分析完成，点击引用证据可回看对应时间窗。",
  failed: "分析失败",
  loaded: "已载入最近一次分析结果",
  evidenceOnly: "已载入已验证知识包；视频文件不在本地，证据可查看但不能回看。",
  uploading: "正在上传到远端执行目录...",
  uploadDone: "视频已上传，可以开始分析",
  uploadFailed: "视频上传失败",
  noFile: "未选择视频",
  chooseVideoFirst: "请先选择一个视频并等待上传完成。",
  noAnswer: "暂时没有可展示的回答",
  verified: "证据已核验",
  unverified: "证据不足",
  conclusion: "核心结论",
  evidence: "引用证据",
  noEvidence: "没有找到足够的时间证据",
  playingEvidence: "正在回看证据",
  retry: "重新提交上一个问题",
};

const $ = (id) => document.getElementById(id);

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (ch) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;",
  }[ch]));
}

function clearActivePlaybackNodes() {
  document.querySelectorAll(".inlineCitation.active, .evidenceButton.active, .timelineItem.active").forEach((item) => item.classList.remove("active"));
}

const playback = VideoTracePlayback.createPlaybackController({
  video: $("mainVideo"),
  onEvidenceStarted(windowState) {
    clearActivePlaybackNodes();
    windowState.sourceNode?.classList.add("active");
    $("nowPlaying").hidden = false;
    $("clearWindowBtn").textContent = "从当前位置继续播放";
    $("nowPlayingTitle").textContent = `${TEXT.playingEvidence} · ${windowState.label} · ${formatRange(windowState.start, windowState.end)}`;
  },
  onEvidenceEnded(completed) {
    clearActivePlaybackNodes();
    $("nowPlaying").hidden = false;
    $("clearWindowBtn").textContent = "从当前位置继续播放";
    $("nowPlayingTitle").textContent = `证据播放完成 · ${formatRange(completed.start, completed.end)}`;
    setStatus("证据时间窗已结束并解除限制，可从当前位置继续播放完整视频。");
  },
  onTimelineStarted(info) {
    clearActivePlaybackNodes();
    info.sourceNode?.classList.add("active");
    $("nowPlaying").hidden = true;
    setStatus(`已跳转到${info.label}起点，将继续播放完整视频。`);
  },
  onCleared(info) {
    clearActivePlaybackNodes();
    $("nowPlaying").hidden = true;
    if (info.reason === "manual_seek") setStatus("已离开证据时间窗，恢复完整视频播放。 ");
  },
  onPlayBlocked() {
    setStatus("已定位到目标时间，请点击播放器继续播放。");
  },
});

function setStatus(text, isError = false) {
  const node = $("status");
  node.textContent = text;
  node.classList.toggle("error", isError);
}

function setRetryVisible(visible) {
  const button = $("retryBtn");
  if (!button) return;
  button.hidden = !visible;
  button.disabled = state.running || state.uploading;
}

function setVideoSource(url, options = {}) {
  const video = $("mainVideo");
  const next = String(url || "");
  playback.reset("source_change");
  if (state.previewUrl && state.previewUrl !== next) {
    URL.revokeObjectURL(state.previewUrl);
    state.previewUrl = "";
  }
  if (options.preview) state.previewUrl = next;
  if (!next) {
    video.removeAttribute("src");
    video.load();
    return;
  }
  if (video.getAttribute("src") !== next) {
    video.setAttribute("src", next);
    video.load();
  }
}

function formatTime(value) {
  const seconds = Math.max(0, Number(value || 0));
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(1).padStart(4, "0")}`;
}

function formatRange(start, end) {
  return `${formatTime(start)} - ${formatTime(end)}`;
}

function videoName(path) {
  const raw = String(path || "").split(/[\\/]/).pop() || "当前视频";
  const stem = raw.replace(/\.[^.]+$/, "") || "当前视频";
  const labels = { cola_review: "全球 17 款可乐横评" };
  return labels[stem] || stem.replaceAll("_", " ");
}

async function readJsonResponse(res) {
  const text = await res.text();
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    throw new Error(`服务返回了无法解析的结果（${res.status}）`);
  }
}

async function loadCapabilities() {
  try {
    const res = await fetch("/api/capabilities", { cache: "no-store" });
    const data = await readJsonResponse(res);
    if (!res.ok) throw new Error(data.message || "无法读取算力状态");
    state.capabilities = data;
    renderCapabilities(data);
  } catch (error) {
    state.capabilities = { analysis_available: false, vlm_modes: [], message: error.message };
    renderCapabilities(state.capabilities);
  }
}

function renderCapabilities(data) {
  const ready = Boolean(data.analysis_available);
  $("serviceNotice").classList.toggle("unavailable", !ready);
  $("serviceDot").classList.toggle("ready", ready);
  $("serviceTitle").textContent = ready ? "远端算力已连接" : "远端算力未连接";
  $("serviceMessage").textContent = data.message || "暂时无法读取远端状态。";
  $("serviceMode").textContent = ready ? "算力已连接" : "仅预览";

  const select = $("vlmMode");
  const previous = select.value;
  const modes = Array.isArray(data.vlm_modes) ? data.vlm_modes : [];
  select.innerHTML = "";
  if (!modes.length) {
    select.append(new Option("暂无可用视觉模式", ""));
    select.disabled = true;
  } else {
    for (const mode of modes) {
      const option = new Option(mode.label, mode.id);
      option.title = mode.description || mode.label;
      select.append(option);
    }
    const preferred = [previous, state.data?.metadata?.vlm_mode?.id, data.default_mode].find(
      (value) => value && modes.some((mode) => mode.id === value),
    );
    select.value = preferred || modes[0].id;
    select.disabled = false;
  }
  updateControls();
}

async function loadLatest() {
  try {
    const res = await fetch("/api/latest", { cache: "no-store" });
    const data = await readJsonResponse(res);
    if (!res.ok) throw new Error(data.message || "无法读取最近结果");
    state.data = data;
    render(data);
  } catch (error) {
    setStatus(`${TEXT.failed}：${error.message}`, true);
  }
}

async function runAnalysis() {
  const videoId = $("videoId").value.trim();
  const query = $("query").value.trim();
  const vlmMode = $("vlmMode").value;
  if (!videoId) {
    setStatus(TEXT.chooseVideoFirst, true);
    return;
  }
  if (!query) {
    setStatus("请先输入问题。", true);
    $("query").focus();
    return;
  }
  if (!state.capabilities?.analysis_available || !vlmMode) {
    setStatus("远端视觉算力尚未连接，当前不能提交分析任务。", true);
    return;
  }

  playback.reset("new_analysis");
  state.running = true;
  setRetryVisible(false);
  setStatus(TEXT.running);
  setJobProgress({ phase: "queued", progress: 0, message: TEXT.running });
  updateControls();
  try {
    const res = await fetch("/api/jobs", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ video_id: videoId, query, vlm_mode: vlmMode }),
    });
    const job = await readJsonResponse(res);
    if (!res.ok || !job.job_id) throw new Error(job.message || TEXT.failed);
    state.currentJobId = job.job_id;
    setJobProgress(job);
    await pollJob(job.job_id, job.poll_url || `/api/jobs/${job.job_id}`);
  } catch (error) {
    setStatus(`${TEXT.failed}：${error.message}`, true);
    setJobProgress({ phase: "failed", progress: 100, message: error.message });
    setRetryVisible(true);
  } finally {
    state.running = false;
    updateControls();
  }
}

async function pollJob(jobId, pollUrl) {
  while (state.currentJobId === jobId) {
    const res = await fetch(pollUrl, { cache: "no-store" });
    const job = await readJsonResponse(res);
    if (!res.ok) throw new Error(job.message || "无法读取任务状态");
    setJobProgress(job);
    setStatus(job.error ? `${job.message}：${job.error}` : job.message || PHASE_LABELS[job.phase] || TEXT.running, job.status === "failed");
    if (job.status === "completed") {
      state.data = job.result;
      render(job.result);
      setStatus(TEXT.done);
      state.currentJobId = "";
      return;
    }
    if (job.status === "failed") {
      state.currentJobId = "";
      throw new Error(job.error || job.message || TEXT.failed);
    }
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
}

function setJobProgress(job) {
  const display = VideoTraceJobStatus.presentation(job, PHASE_LABELS);
  $("jobProgress").hidden = false;
  $("progressBar").style.width = `${display.progress}%`;
  $("progressStage").textContent = display.stage;
  $("progressValue").textContent = display.value;
  const jobMeta = $("jobMeta");
  if (jobMeta) {
    const jobId = String(job?.job_id || state.currentJobId || "");
    const elapsed = Math.max(0, Number(job?.elapsed_sec || 0));
    jobMeta.hidden = !jobId;
    if (jobId) {
      $("jobIdLabel").textContent = "任务 " + jobId.slice(0, 12) + "…";
      $("jobElapsedLabel").textContent = "已用时 " + Math.round(elapsed) + "s";
    }
  }
}

async function uploadVideo(file) {
  if (!file) {
    $("uploadName").textContent = TEXT.noFile;
    return;
  }
  playback.reset("video_switch");
  state.data = null;
  state.uploading = true;
  $("videoId").value = "";
  $("uploadName").textContent = file.name;
  $("uploadState").textContent = "正在上传";
  $("videoTitle").textContent = videoName(file.name);
  $("videoMeta").textContent = "本地预览 · 等待远端上传";
  setVideoSource(URL.createObjectURL(file), { preview: true });
  renderAnswer({ answer: "" }, {});
  renderTimeline([]);
  setStatus(TEXT.uploading);
  updateControls();
  try {
    const body = new FormData();
    body.append("file", file, file.name);
    const res = await fetch("/api/upload", { method: "POST", body });
    const data = await readJsonResponse(res);
    if (!res.ok || !data.ok) throw new Error(data.message || TEXT.uploadFailed);
    $("videoId").value = data.video_id || "";
    setVideoSource(data.video_url);
    $("uploadState").textContent = TEXT.uploadDone;
    $("videoTitle").textContent = videoName(data.filename || file.name);
    setStatus(TEXT.uploadDone);
  } catch (error) {
    $("uploadState").textContent = "上传失败";
    setStatus(`${TEXT.uploadFailed}：${error.message}`, true);
  } finally {
    state.uploading = false;
    updateControls();
  }
}

function updateControls() {
  const hasMode = Boolean($("vlmMode").value);
  const hasVideo = Boolean($("videoId").value);
  const computeReady = Boolean(state.capabilities?.analysis_available);
  $("runBtn").disabled = state.uploading || state.running || !hasMode || !hasVideo || !computeReady;
  $("videoUpload").disabled = state.uploading || state.running;
  $("vlmMode").disabled = state.running || !(state.capabilities?.vlm_modes || []).length;
  const retry = $("retryBtn");
  if (retry) retry.disabled = state.running || state.uploading;
}

function render(data) {
  const ready = Boolean(data.ready);
  if (!ready) {
    $("videoId").value = data.sample_video_id || "";
    setVideoSource(data.sample_video_url || "");
    $("query").value = data.sample_query || PRESET_QUERIES[0].value;
    $("videoTitle").textContent = videoName(data.sample_video);
    $("videoMeta").textContent = data.sample_video_url ? "样例视频 · 等待分析" : "请选择视频开始分析";
    renderPresets($("query").value);
    renderAnswer({ answer: "" }, {});
    renderTimeline([]);
    renderTechnical({}, {});
    setStatus(data.message || "请选择一个视频开始分析");
    updateControls();
    return;
  }

  $("videoId").value = data.video_id || $("videoId").value;
  $("query").value = data.metadata?.query || $("query").value;
  // A pack can be fully valid while its source video is absent locally: the
  // video is third-party content and is not redistributed. Say so explicitly
  // instead of rendering a dead <video> whose evidence buttons do nothing.
  const mediaReady = data.media_ready !== false;
  applyMediaAvailability(mediaReady);
  setVideoSource(mediaReady ? data.video_url : "");
  $("videoTitle").textContent = videoName(data.video_path);
  $("videoMeta").textContent = mediaReady
    ? `${formatTime(data.duration_sec)} · ${data.timeline?.length || 0} 个关键节点`
    : `${formatTime(data.duration_sec)} · ${data.timeline?.length || 0} 个关键节点 · 证据模式`;

  $("uploadName").textContent = videoName(data.video_path);
  $("uploadState").textContent = state.capabilities?.analysis_available ? "可以重新提问" : "已载入验证结果";
  renderPresets($("query").value);

  const selectedMode = data.metadata?.vlm_mode?.id;
  if (selectedMode && Array.from($("vlmMode").options).some((option) => option.value === selectedMode)) {
    $("vlmMode").value = selectedMode;
  }
  const agent = data.agent || {};
  const verification = agent.verification || {};
  const evidenceCount = (verification.matched_evidence || []).length || (data.timeline || []).length;
  $("verifiedBadge").textContent = agent.verified ? TEXT.verified : TEXT.unverified;
  $("verifiedBadge").classList.toggle("unverified", !agent.verified);
  $("evidenceCount").textContent = `${evidenceCount} 条证据`;
  renderAnswer(data, agent);
  renderTimeline(data.timeline || []);
  renderTechnical(data, agent);
  setStatus(mediaReady ? TEXT.loaded : TEXT.evidenceOnly);
  setRetryVisible(false);
  updateControls();
}

// Evidence-only mode: keep every timestamp visible and auditable, but make it
// obvious that the buttons cannot seek, rather than failing silently on click.
function applyMediaAvailability(mediaReady) {
  state.mediaReady = mediaReady;
  $("mediaNotice").hidden = mediaReady;
  $("mainVideo").hidden = !mediaReady;
  document.body.classList.toggle("evidenceOnly", !mediaReady);
  if (!mediaReady) $("nowPlaying").hidden = true;
}

function markPlaybackUnavailable(nodes) {
  nodes.forEach((node) => {
    node.disabled = true;
    node.title = "视频文件不在本地，无法回看这一段证据";
  });
}

function renderPresets(currentQuery) {
  $("presetRow").innerHTML = PRESET_QUERIES.map((preset) => `
    <button class="preset${String(currentQuery || "") === preset.value ? " active" : ""}" data-query="${esc(preset.value)}" type="button">${esc(preset.label)}</button>
  `).join("");
  $("presetRow").querySelectorAll(".preset").forEach((button) => {
    button.addEventListener("click", () => {
      $("query").value = button.dataset.query || "";
      $("presetRow").querySelectorAll(".preset").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      setStatus("问题已切换，点击“分析视频”获取新回答。");
    });
  });
}

function parseAnswer(answer) {
  const lines = String(answer || "").split(/\n+/).map((line) => line.trim()).filter(Boolean);
  const result = { question: "", conclusion: "", evidence: [] };
  for (const line of lines) {
    if (line.startsWith("问题：") || line.startsWith("用户问题：")) {
      result.question = line.split("：", 2)[1].trim();
    } else if (line.startsWith("结论：") || line.startsWith("总体结论：") || line.startsWith("回答：")) {
      result.conclusion = line.split("：", 2)[1].trim();
    } else if (line.startsWith("- ")) {
      const item = parseBullet(line.slice(2));
      if (item.time) result.evidence.push(item);
    } else if (!result.conclusion) {
      result.conclusion = line;
    }
  }
  return { ...result, evidence: dedupeEvidence(result.evidence) };
}

function parseBullet(text) {
  const timestampMatch = text.match(/timestamp=([0-9.]+-[0-9.]+)/);
  const timePrefix = text.match(/^([0-9.]+s?-[0-9.]+s?)：/);
  const range = timePrefix?.[1] || timestampMatch?.[1] || "";
  const numbers = range.match(/([0-9.]+)s?-([0-9.]+)s?/);
  const body = text
    .replace(/^([0-9.]+s?-[0-9.]+s?)：/, "")
    .replace(/\s*\(timestamp=[^)]+\)\s*$/g, "")
    .trim();
  return {
    time: range,
    start: numbers ? Number(numbers[1]) : 0,
    end: numbers ? Number(numbers[2]) : 0,
    text: body,
  };
}

function dedupeEvidence(items) {
  const seen = new Set();
  return items.filter((item) => {
    const key = `${item.time}|${item.text}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

function timelineForAnswer(data, parsed) {
  const source = data.timeline || [];
  return parsed.evidence.map((evidence) => {
    const match = source.find((item) => Math.abs(Number(item.start_sec) - evidence.start) < 0.2);
    return match ? { ...match, text: evidence.text || cleanEvidenceText(match.text) } : {
      start_sec: evidence.start,
      end_sec: evidence.end,
      text: evidence.text,
      score: 0,
    };
  });
}

function cleanEvidenceText(value) {
  const text = String(value || "").trim();
  return text.split(/\s+场景：/, 1)[0].trim() || text;
}

function renderAnswer(data, agent) {
  const parsed = parseAnswer(data.answer || agent.answer || "");
  const verification = agent.verification || {};
  const evidence = timelineForAnswer(data, parsed);
  const conclusion = parsed.conclusion || data.summary || TEXT.noAnswer;
  const citationHtml = evidence.length
    ? "<div class=\"inlineCitationBlock\"><span class=\"inlineCitationLabel\">结论依据</span><div class=\"inlineCitationList\">" + evidence.map((item, index) => citationButton(item, index)).join("") + "</div></div>"
    : "";
  $("answer").innerHTML = `
    <section class="answerConclusion">
      <h3>${TEXT.conclusion}</h3>
      <p>${esc(conclusion)}</p>
      ${citationHtml}
    </section>
    <section>
      <div class="evidenceHeader">
        <h3>${TEXT.evidence}</h3>
        <span>${verification.coverage ? `覆盖 ${Math.round(Number(verification.coverage) * 100)}%` : "点击回看"}</span>
      </div>
      <div class="evidenceList">
        ${evidence.length ? evidence.map((item, index) => evidenceButton(item, index)).join("") : `<div class="emptyState">${TEXT.noEvidence}</div>`}
      </div>
    </section>
  `;
  $("answer").querySelectorAll("[data-playback='evidence']").forEach((node) => {
    node.addEventListener("click", () => playback.playEvidence(
      Number(node.dataset.start || 0),
      Number(node.dataset.end || 0),
      node.dataset.label || TEXT.playingEvidence,
      node,
      Number(state.data?.duration_sec || 0),
    ));
  });
  if (state.mediaReady === false) {
    markPlaybackUnavailable($("answer").querySelectorAll("[data-playback='evidence']"));
  }
}

function citationButton(item, index) {
  const start = Number(item.start_sec ?? item.start ?? 0);
  const end = Number(item.end_sec ?? item.end ?? start);
  return "<button class=\"inlineCitation\" type=\"button\" data-playback=\"evidence\" data-start=\"" + start + "\" data-end=\"" + end + "\" data-label=\"证据 " + (index + 1) + "\" aria-label=\"播放证据 " + (index + 1) + "，" + esc(formatRange(start, end)) + "\">证据 " + (index + 1) + " · " + esc(formatRange(start, end)) + "</button>";
}

function evidenceButton(item, index) {
  const start = Number(item.start_sec ?? item.start ?? 0);
  const end = Number(item.end_sec ?? item.end ?? start);
  return `
    <button class="evidenceButton" type="button" data-playback="evidence" data-start="${start}" data-end="${end}" data-label="证据 ${index + 1}">
      <span class="evidenceTime">${esc(formatRange(start, end))}</span>
      <span class="evidenceText">${esc(cleanEvidenceText(item.text) || TEXT.noEvidence)}</span>
      <span class="playSymbol" aria-hidden="true">▶</span>
    </button>
  `;
}

function renderTimeline(items) {
  $("timeline").innerHTML = items.length ? items.map((item, index) => `
    <button class="timelineItem" type="button" data-playback="timeline" data-start="${Number(item.start_sec || 0)}" data-label="节点 ${index + 1}">
      <time>${esc(formatRange(item.start_sec, item.end_sec))}</time>
      <p>${esc(cleanEvidenceText(item.text))}</p>
      <span class="playSymbol" aria-hidden="true">▶</span>
    </button>
  `).join("") : `<div class="emptyState">暂无视频脉络</div>`;
  $("timeline").querySelectorAll("[data-playback='timeline']").forEach((node) => {
    node.addEventListener("click", () => playback.playTimeline(
      Number(node.dataset.start || 0),
      node.dataset.label || "视频脉络",
      node,
      Number(state.data?.duration_sec || 0),
    ));
  });
  if (state.mediaReady === false) {
    markPlaybackUnavailable($("timeline").querySelectorAll("[data-playback='timeline']"));
  }
}

function renderTechnical(data, agent) {
  const metadata = data.metadata || {};
  const verification = agent.verification || {};
  const trace = agent.tool_trace || [];
  const environment = metadata.environment || {};
  const packages = environment.packages || {};
  const stack = VideoTraceTechnical.describeTechnicalStack(metadata, trace.length, agent);
  const traceRows = VideoTraceTechnical.describeAgentTrace(agent);
  const verificationRows = VideoTraceTechnical.describeVerification(agent);
  const safeguards = VideoTraceTechnical.describeSafeguards(agent);
  $("technicalContent").innerHTML = `
    <div class="techGrid">
      ${stack.map(([label, value]) => `<div class="techItem"><strong>${esc(label)}</strong><span>${esc(value)}</span></div>`).join("")}
    </div>
    <p class="traceLine">证据覆盖 ${Math.round(Number(verification.coverage || 0) * 100)}%；${agent.verified ? "回答已通过时间证据校验" : "当前回答需要更多证据"}。运行环境：Python ${esc(environment.python || "远端环境")}，Torch ${esc(packages.torch || "已配置")}。</p>
    <section class="verificationPanel" aria-label="回答校验明细">
      <div class="technicalSubhead">
        <h4>回答校验</h4>
        <span>硬规则优先，学习式模型只能否决</span>
      </div>
      <div class="verificationGrid">
        ${verificationRows.map((item) => `
          <div class="verificationItem ${item.ok ? "verificationPass" : "verificationFail"}">
            <span>${esc(item.label)}</span>
            <strong>${esc(item.value)}</strong>
          </div>
        `).join("")}
      </div>
    </section>
    <section class="agentTracePanel" aria-label="Agent 执行轨迹">
      <div class="technicalSubhead">
        <h4>Agent 执行轨迹</h4>
        <span>${traceRows.length} 步 · ${traceRows.every((item) => item.ok) ? "全部成功" : "已触发受控降级"}</span>
      </div>
      <ol class="agentTraceList">
        ${traceRows.map((item) => `
          <li class="agentTraceItem ${item.ok ? "traceSuccess" : "traceFailure"}">
            <span class="traceStep">${item.step}</span>
            <div class="traceBody">
              <strong>${esc(item.label)}</strong>
              <span>${esc(item.observation || item.name)}</span>
              ${item.attemptSummary ? `<span class="attemptDetail">${esc(item.attemptSummary)}</span>` : ""}
            </div>
            <div class="traceMeta">
              <strong>${esc(item.status)}</strong>
              <span>${item.attempts} 次 · ${esc(item.latency)}</span>
            </div>
          </li>
        `).join("") || `<li class="emptyTrace">当前结果未记录工具轨迹</li>`}
      </ol>
      <p class="safeguardLine">最多 ${safeguards.maxSteps} 步；单工具最多 ${safeguards.maxAttempts} 次；本次重试 ${safeguards.retryCount}，开放熔断器 ${safeguards.openCount}/${safeguards.breakerCount}。${esc(safeguards.contextPolicy)}</p>
    </section>
  `;
}

$("videoUpload").addEventListener("change", (event) => uploadVideo(event.target.files?.[0]));
$("runBtn").addEventListener("click", runAnalysis);
$("retryBtn").addEventListener("click", runAnalysis);
$("vlmMode").addEventListener("change", updateControls);
$("clearWindowBtn").addEventListener("click", () => playback.continueFromCurrent());
$("mainVideo").addEventListener("timeupdate", () => playback.handleTimeUpdate());
$("mainVideo").addEventListener("seeking", () => playback.handleSeeking());
$("mainVideo").addEventListener("emptied", () => playback.reset("video_emptied"));
$("mainVideo").addEventListener("loadedmetadata", () => {
  const duration = Number.isFinite($("mainVideo").duration) ? $("mainVideo").duration : 0;
  if (duration) $("videoMeta").textContent = `${formatTime(duration)} · ${state.data?.timeline?.length || 0} 个关键节点`;
});

window.VideoTraceDebug = Object.freeze({
  playback: () => playback.snapshot(),
  jobId: () => state.currentJobId,
  capabilities: () => state.capabilities,
});

Promise.all([loadCapabilities(), loadLatest()]).then(updateControls);
