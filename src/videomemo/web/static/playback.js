(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  root.VideoTracePlayback = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  function clampWindow(start, end, duration) {
    const finiteDuration = Number.isFinite(Number(duration)) && Number(duration) > 0
      ? Number(duration)
      : Math.max(Number(end) || 0, Number(start) + 20 || 20);
    const safeStart = Math.max(0, Math.min(Number(start) || 0, Math.max(0, finiteDuration - 0.1)));
    const requestedEnd = Number(end) > safeStart ? Number(end) : safeStart + 20;
    const safeEnd = Math.min(finiteDuration, Math.max(safeStart + 0.1, Math.min(requestedEnd, finiteDuration)));
    return { start: safeStart, end: safeEnd };
  }

  function createPlaybackController(options) {
    const video = options.video;
    const callbacks = {
      onEvidenceStarted: options.onEvidenceStarted || function () {},
      onEvidenceEnded: options.onEvidenceEnded || function () {},
      onTimelineStarted: options.onTimelineStarted || function () {},
      onCleared: options.onCleared || function () {},
      onPlayBlocked: options.onPlayBlocked || function () {},
    };
    let activeWindow = null;
    let mode = "full";

    function snapshot() {
      return {
        mode: mode,
        activeWindow: activeWindow ? {
          start: activeWindow.start,
          end: activeWindow.end,
          label: activeWindow.label,
        } : null,
      };
    }

    function safePlay() {
      let result;
      try {
        result = video.play();
      } catch (error) {
        callbacks.onPlayBlocked(error);
        return Promise.resolve(false);
      }
      if (result && typeof result.catch === "function") {
        return result.then(function () { return true; }).catch(function (error) {
          callbacks.onPlayBlocked(error);
          return false;
        });
      }
      return Promise.resolve(true);
    }

    function clear(reason, notify) {
      const previous = activeWindow;
      activeWindow = null;
      mode = "full";
      if (notify !== false) callbacks.onCleared({ reason: reason || "clear", previous: previous });
    }

    function playEvidence(start, end, label, sourceNode, durationHint) {
      const duration = Number.isFinite(video.duration) ? video.duration : durationHint;
      const window = clampWindow(start, end, duration);
      activeWindow = {
        start: window.start,
        end: window.end,
        label: label || "引用证据",
        sourceNode: sourceNode || null,
      };
      mode = "evidence";
      video.currentTime = window.start;
      callbacks.onEvidenceStarted(activeWindow);
      safePlay();
      if (typeof video.focus === "function") video.focus({ preventScroll: true });
      return snapshot();
    }

    function playTimeline(start, label, sourceNode, durationHint) {
      const duration = Number.isFinite(video.duration) ? video.duration : durationHint;
      const point = clampWindow(start, Number(start) + 0.1, duration).start;
      activeWindow = null;
      mode = "full";
      video.currentTime = point;
      callbacks.onTimelineStarted({ start: point, label: label || "视频脉络", sourceNode: sourceNode || null });
      safePlay();
      if (typeof video.focus === "function") video.focus({ preventScroll: true });
      return snapshot();
    }

    function continueFromCurrent() {
      clear("continue", true);
      return safePlay();
    }

    function handleTimeUpdate() {
      if (!activeWindow) return false;
      if (Number(video.currentTime) >= activeWindow.end - 0.05) {
        const completed = activeWindow;
        video.pause();
        activeWindow = null;
        mode = "full";
        callbacks.onEvidenceEnded(completed);
        return true;
      }
      return false;
    }

    function handleSeeking() {
      if (!activeWindow) return false;
      const current = Number(video.currentTime);
      if (current < activeWindow.start - 0.75 || current > activeWindow.end + 0.75) {
        clear("manual_seek", true);
        return true;
      }
      return false;
    }

    function reset(reason) {
      clear(reason || "reset", true);
    }

    return {
      playEvidence: playEvidence,
      playTimeline: playTimeline,
      continueFromCurrent: continueFromCurrent,
      handleTimeUpdate: handleTimeUpdate,
      handleSeeking: handleSeeking,
      reset: reset,
      snapshot: snapshot,
    };
  }

  return {
    clampWindow: clampWindow,
    createPlaybackController: createPlaybackController,
  };
}));
