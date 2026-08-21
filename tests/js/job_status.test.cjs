const assert = require("node:assert/strict");
const JobStatus = require("../../src/videomemo/web/static/job_status.js");

const queued = JobStatus.presentation(
  { status: "queued", phase: "queued", progress: 0, queue_position: 2, elapsed_sec: 12.4 },
  { queued: "等待 GPU 队列" },
);
assert.equal(queued.stage, "等待 GPU 队列（第 2 位）");
assert.equal(queued.value, "0% · 12s");

const running = JobStatus.presentation(
  { status: "running", phase: "analyzing", progress: 41, elapsed_sec: 20 },
  { analyzing: "视频理解与证据生成" },
);
assert.equal(running.stage, "视频理解与证据生成");
assert.equal(running.value, "41% · 20s");

console.log("job status behavior ok");
