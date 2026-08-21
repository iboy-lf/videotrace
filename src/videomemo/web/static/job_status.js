(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.VideoTraceJobStatus = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  function presentation(job, phaseLabels = {}) {
    const progress = Math.max(0, Math.min(100, Number(job?.progress || 0)));
    const queuePosition = Math.max(0, Number(job?.queue_position || 0));
    const elapsed = Math.max(0, Number(job?.elapsed_sec || 0));
    let stage = phaseLabels[job?.phase] || job?.message || "处理中";
    if (job?.status === "queued" && queuePosition > 0) {
      stage += queuePosition === 1 ? "（下一位）" : `（第 ${queuePosition} 位）`;
    }
    const value = elapsed > 0
      ? `${Math.round(progress)}% · ${Math.round(elapsed)}s`
      : `${Math.round(progress)}%`;
    return { progress, stage, value };
  }

  return Object.freeze({ presentation });
});
