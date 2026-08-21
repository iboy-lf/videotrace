(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.VideoTraceTechnical = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function normalized(value) {
    return String(value || "").trim().toLowerCase();
  }

  function segmentUnderstandingLabel(metadata) {
    const modeId = normalized(metadata?.vlm_mode?.id);
    const backend = normalized(metadata?.segment_understanding?.backend);
    if (modeId === "siglip_retrieval" || backend === "baseline") {
      return "基础切片描述（无 Qwen3.5 逐片段理解）";
    }
    if (backend === "qwen35_local" || modeId === "auto_best" || modeId === "qwen35_video") {
      return "Qwen3.5 结构化视频理解";
    }
    return backend ? `后端：${backend}` : "未记录";
  }

  function visualRetrievalLabel(metadata) {
    const modeId = normalized(metadata?.vlm_mode?.id);
    const backend = normalized(metadata?.vlm?.backend);
    if (modeId === "qwen35_video" || backend === "baseline") {
      return "稀疏检索 + neural reranker（无 SigLIP2）";
    }
    if (modeId === "auto_best" || modeId === "siglip_retrieval" || backend.includes("siglip")) {
      return "SigLIP2 图文检索";
    }
    return backend ? `后端：${backend}` : "未记录";
  }

  const TOOL_LABELS = Object.freeze({
    retrieve_segments: "检索候选片段",
    build_context: "压缩证据上下文",
    assess_evidence: "判断证据充分性",
    search_memory: "检索视频记忆",
    synthesize_answer: "生成证据回答",
    verify_answer: "校验时间戳与事实",
  });

  function postTrainingLabel(metadata) {
    const adapter = metadata?.llm_adapter || {};
    if (!adapter.enabled) return "基础模型（未加载 adapter）";
    const method = normalized(adapter.method || adapter.candidate_id);
    if (method.includes("dpo")) return "已准入 DPO，SFT 可回退";
    if (method.includes("sft")) return "已准入证据 SFT";
    return adapter.validated_for_web ? "已准入后训练 adapter" : "adapter 未通过产品准入";
  }

  function verifierLabel(agent) {
    const verification = agent?.verification || {};
    const calibrated = verification.calibrated_verifier || {};
    if (!calibrated.enabled) return "硬时间戳与 claim-support 规则";
    const probability = Number(calibrated.safe_probability);
    const threshold = Number(calibrated.threshold);
    if (Number.isFinite(probability) && Number.isFinite(threshold)) {
      return `硬规则 + 学习式否决器 ${probability.toFixed(3)} / ${threshold.toFixed(2)}`;
    }
    return "硬规则 + 学习式安全否决器";
  }

  function describeTechnicalStack(metadata, traceLength, agent) {
    const safeMetadata = metadata || {};
    return [
      ["视觉模式", safeMetadata.vlm_mode?.label || "服务端固定模式"],
      ["片段理解", segmentUnderstandingLabel(safeMetadata)],
      ["视觉检索", visualRetrievalLabel(safeMetadata)],
      ["Agent", `${Number(traceLength) || 0} 步证据流程`],
      ["后训练", postTrainingLabel(safeMetadata)],
      ["回答校验", verifierLabel(agent)],
    ];
  }

  function describeAgentTrace(agent) {
    const trace = Array.isArray(agent?.tool_trace) ? agent.tool_trace : [];
    const plan = Array.isArray(agent?.plan) ? agent.plan : [];
    return trace.map((item, index) => {
      const attempts = Math.max(0, Number(item?.attempts) || 0);
      const latencyMs = Math.max(0, Number(item?.latency_ms) || 0);
      const errorCode = String(item?.error_code || "");
      const circuitState = normalized(item?.circuit_state) || "closed";
      const ok = item?.ok !== false && !errorCode;
      let status = ok ? "成功" : "失败";
      if (errorCode === "circuit_open" || circuitState === "open") status = "熔断阻止";
      else if (attempts > 1 && !ok) status = "重试耗尽";
      else if (attempts > 1) status = "重试后成功";
      const planStep = plan.find((step) => step?.action === item?.name) || {};
      const attemptTrace = Array.isArray(item?.attempt_trace) ? item.attempt_trace : [];
      const attemptSummary = attemptTrace.map((attempt) => {
        const number = Math.max(0, Number(attempt?.attempt) || 0);
        if (attempt?.ok === true) return `第 ${number} 次成功`;
        const code = String(attempt?.error_code || "失败");
        return number > 0 ? `第 ${number} 次 ${code}` : code;
      }).join(" → ");
      return {
        step: index + 1,
        name: String(item?.name || "unknown_tool"),
        label: TOOL_LABELS[item?.name] || String(item?.name || "未知工具"),
        status,
        ok,
        attempts,
        latency: formatLatency(latencyMs),
        errorCode,
        circuitState,
        observation: String(planStep.observation || ""),
        attemptSummary,
      };
    });
  }

  function describeVerification(agent) {
    const verification = agent?.verification || {};
    const refs = Array.isArray(verification.timestamp_refs) ? verification.timestamp_refs.length : 0;
    const matched = Array.isArray(verification.matched_timestamp_refs)
      ? verification.matched_timestamp_refs.length
      : 0;
    const calibrated = verification.calibrated_verifier || {};
    const probability = Number(calibrated.safe_probability);
    const threshold = Number(calibrated.threshold);
    return [
      {
        label: "时间戳绑定",
        value: `${matched}/${refs || matched}`,
        ok: refs > 0 && matched === refs,
      },
      {
        label: "Claim support",
        value: `${Math.round(Number(verification.claim_support_coverage || 0) * 100)}%`,
        ok: verification.claim_support_ok === true,
      },
      {
        label: "学习式否决器",
        value: calibrated.enabled && Number.isFinite(probability) && Number.isFinite(threshold)
          ? `${probability.toFixed(3)} ≥ ${threshold.toFixed(2)}`
          : "未启用",
        ok: calibrated.enabled ? calibrated.passed === true : true,
      },
    ];
  }

  function describeSafeguards(agent) {
    const safeguards = agent?.safeguards || {};
    const runtime = safeguards.tool_runtime || {};
    const breakers = runtime.circuit_breakers || {};
    const breakerRows = Object.values(breakers);
    const openCount = breakerRows.filter((item) => normalized(item?.state) === "open").length;
    const retryCount = (Array.isArray(agent?.tool_trace) ? agent.tool_trace : [])
      .filter((item) => Number(item?.attempts) > 1).length;
    return {
      maxSteps: Math.max(0, Number(safeguards.max_steps) || 0),
      maxAttempts: Math.max(1, Number(runtime.default_max_attempts) || 1),
      retryCount,
      openCount,
      breakerCount: breakerRows.length,
      contextPolicy: String(safeguards.context_policy || "保留时间戳和证据字段"),
      oscillationPolicy: String(safeguards.path_oscillation_control || "固定计划并阻止重复路径"),
    };
  }

  function formatLatency(value) {
    const latency = Number(value) || 0;
    if (latency < 1) return "<1 ms";
    if (latency < 1000) return `${Math.round(latency)} ms`;
    return `${(latency / 1000).toFixed(2)} s`;
  }

  return {
    describeAgentTrace,
    describeSafeguards,
    describeTechnicalStack,
    describeVerification,
    formatLatency,
    postTrainingLabel,
    segmentUnderstandingLabel,
    verifierLabel,
    visualRetrievalLabel,
  };
}));
