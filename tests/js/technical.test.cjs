"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const technical = require(path.resolve(__dirname, "../../src/videomemo/web/static/technical.js"));

function asMap(metadata) {
  return Object.fromEntries(technical.describeTechnicalStack(metadata, 6));
}

const autoBest = asMap({
  vlm_mode: { id: "auto_best", label: "自动最佳（Qwen3.5 + SigLIP2）" },
  segment_understanding: { backend: "qwen35_local" },
  vlm: { backend: "frozen_siglip" },
  llm_adapter: { enabled: true, method: "dpo", validated_for_web: true },
});
assert.equal(autoBest["片段理解"], "Qwen3.5 结构化视频理解");
assert.equal(autoBest["视觉检索"], "SigLIP2 图文检索");
assert.equal(autoBest.Agent, "6 步证据流程");
assert.equal(autoBest["后训练"], "已准入 DPO，SFT 可回退");

const qwenOnly = asMap({
  vlm_mode: { id: "qwen35_video", label: "Qwen3.5 视频理解" },
  segment_understanding: { backend: "qwen35_local" },
  vlm: { backend: "baseline" },
});
assert.equal(qwenOnly["片段理解"], "Qwen3.5 结构化视频理解");
assert.equal(qwenOnly["视觉检索"], "稀疏检索 + neural reranker（无 SigLIP2）");

const siglipOnly = asMap({
  vlm_mode: { id: "siglip_retrieval", label: "SigLIP2 检索增强" },
  segment_understanding: { backend: "baseline" },
  vlm: { backend: "frozen_siglip" },
});
assert.equal(siglipOnly["片段理解"], "基础切片描述（无 Qwen3.5 逐片段理解）");
assert.equal(siglipOnly["视觉检索"], "SigLIP2 图文检索");

const agent = {
  plan: [
    { action: "retrieve_segments", observation: "取回候选片段" },
    { action: "verify_answer", observation: "校验回答" },
  ],
  tool_trace: [
    { name: "retrieve_segments", ok: true, attempts: 1, latency_ms: 0.4, circuit_state: "closed" },
    {
      name: "verify_answer",
      ok: false,
      attempts: 2,
      latency_ms: 1234,
      error_code: "execution_error",
      circuit_state: "closed",
      attempt_trace: [
        { attempt: 1, ok: false, error_code: "execution_error" },
        { attempt: 2, ok: false, error_code: "execution_error" },
      ],
    },
  ],
  verification: {
    timestamp_refs: ["0.0-1.0"],
    matched_timestamp_refs: ["0.0-1.0"],
    claim_support_ok: true,
    claim_support_coverage: 1,
    calibrated_verifier: { enabled: true, passed: true, safe_probability: 0.92716, threshold: 0.2 },
  },
  safeguards: {
    max_steps: 6,
    context_policy: "保留证据",
    tool_runtime: {
      default_max_attempts: 2,
      circuit_breakers: { retrieve_segments: { state: "closed" }, verify_answer: { state: "open" } },
    },
  },
};
const trace = technical.describeAgentTrace(agent);
assert.equal(trace[0].label, "检索候选片段");
assert.equal(trace[0].latency, "<1 ms");
assert.equal(trace[1].status, "重试耗尽");
assert.equal(trace[1].latency, "1.23 s");
assert.equal(trace[1].observation, "校验回答");
assert.equal(trace[1].attemptSummary, "第 1 次 execution_error → 第 2 次 execution_error");

const verification = Object.fromEntries(
  technical.describeVerification(agent).map((item) => [item.label, item.value]),
);
assert.equal(verification["时间戳绑定"], "1/1");
assert.equal(verification["Claim support"], "100%");
assert.equal(verification["学习式否决器"], "0.927 ≥ 0.20");

const safeguards = technical.describeSafeguards(agent);
assert.equal(safeguards.maxAttempts, 2);
assert.equal(safeguards.retryCount, 1);
assert.equal(safeguards.openCount, 1);

process.stdout.write("technical labels ok\n");
