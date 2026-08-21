"use strict";

const assert = require("node:assert/strict");
const path = require("node:path");
const playbackApi = require(path.resolve(__dirname, "../../src/videomemo/web/static/playback.js"));

function createVideo() {
  return {
    currentTime: 0,
    duration: 120,
    paused: true,
    playCalls: 0,
    pauseCalls: 0,
    play() {
      this.playCalls += 1;
      this.paused = false;
      return Promise.resolve();
    },
    pause() {
      this.pauseCalls += 1;
      this.paused = true;
    },
    focus() {},
  };
}

async function main() {
  const video = createVideo();
  const events = [];
  const controller = playbackApi.createPlaybackController({
    video,
    onEvidenceStarted: (event) => events.push(["evidence-start", event.start, event.end]),
    onEvidenceEnded: (event) => events.push(["evidence-end", event.start, event.end]),
    onTimelineStarted: (event) => events.push(["timeline", event.start]),
    onCleared: (event) => events.push(["cleared", event.reason]),
  });

  controller.playEvidence(10, 20, "证据 1", null, 120);
  assert.equal(video.currentTime, 10);
  assert.equal(controller.snapshot().mode, "evidence");
  assert.equal(controller.snapshot().activeWindow.end, 20);
  video.currentTime = 19.96;
  assert.equal(controller.handleTimeUpdate(), true);
  assert.equal(video.pauseCalls, 1);
  assert.equal(controller.snapshot().activeWindow, null, "evidence boundary must release the window");
  await controller.continueFromCurrent();
  assert.equal(video.playCalls, 2, "continue must immediately resume playback");
  assert.equal(controller.snapshot().mode, "full");

  controller.playTimeline(35, "节点 2", null, 120);
  assert.equal(video.currentTime, 35);
  assert.equal(controller.snapshot().activeWindow, null, "timeline playback must never install an end boundary");
  video.currentTime = 80;
  assert.equal(controller.handleTimeUpdate(), false);
  assert.equal(video.pauseCalls, 1, "timeline playback must continue beyond the node end");

  controller.playEvidence(40, 50, "证据 2", null, 120);
  video.currentTime = 63;
  assert.equal(controller.handleSeeking(), true, "manual seek outside the evidence window must clear it");
  assert.equal(controller.snapshot().activeWindow, null);

  controller.playEvidence(70, 80, "证据 3", null, 120);
  controller.reset("source_change");
  assert.equal(controller.snapshot().activeWindow, null, "video switches must clear old state");
  assert.deepEqual(events.map((event) => event[0]), [
    "evidence-start", "evidence-end", "cleared", "timeline", "evidence-start", "cleared", "evidence-start", "cleared",
  ]);

  const clamped = playbackApi.clampWindow(119.99, 200, 120);
  assert.ok(clamped.start < 120);
  assert.equal(clamped.end, 120);
  process.stdout.write("playback behavior ok\n");
}

main().catch((error) => {
  process.stderr.write(`${error.stack || error}\n`);
  process.exitCode = 1;
});
