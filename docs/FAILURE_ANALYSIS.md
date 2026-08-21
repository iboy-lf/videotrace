# Failure Analysis

## Case: overview query omitted the opening

### Input

- Video: `cola_review.mp4`, 416.2 seconds
- Query: asks for the opening, country-by-country tasting, and final blind tasting

### Previous behavior

Pure fused relevance returned approximately `100-120s`, `140-160s`, `300-320s`, and `400-416.2s`. The answer missed `0-20s`, even though the opening lineup was required.

The first temporal-coverage implementation fixed the global/local intent problem but used an opening target at normalized time `0.08`. On a 416-second video, the midpoint of `40-60s` was closer to that target than the midpoint of `0-20s`, so the true opening could still be omitted.

### Root cause

Sparse, scorer, and SigLIP signals initially ranked each segment independently. A global overview question was treated like a local evidence lookup, so four semantically strong middle/late windows consumed the entire top-k budget. The intermediate selector then modeled the opening as an interior point instead of a distance from the start boundary.

### Fix

1. Classify the query as `overview/distributed`.
2. Reserve evidence slots for opening, middle, and ending regions.
3. Measure opening distance from `start_sec=0`, ending distance from `end_sec=duration`, and middle distance from the temporal midpoint.
4. Add a 416.2-second regression case where both `0-20s` and `40-60s` are present.
5. Fill remaining slots with relevance plus temporal novelty.
6. Preserve `selection_reason` in the context, timeline, JSON artifacts, and Web UI.

This is not a hard-coded cola-video timestamp rule. The selector operates on normalized boundary distance and applies only to distributed query intents. Local comparison and lookup questions retain relevance order. The final run selects `0-20s`, `200-220s`, `300-320s`, and `400-416.2s`.

## Case: media preview stayed black

### Observation

Evidence was originally exported as OpenCV `mp4v/FMP4` files, which embedded Chromium could not decode reliably even though the user-provided source video was valid H.264.

### Fix

- The Web server now supports HTTP Range requests and client disconnects.
- Dynamic video source changes call `video.load()`.
- Every evidence item now stores the original video path plus `start_sec/end_sec`; clicking it seeks the single H.264 player and pauses at the evidence boundary.
- The static HTML report uses the same source-video-window interaction, so there are no duplicate clip files or codec drift.
- Final browser verification covers HTTP Range `206`, real decoding, the final `400.0-416.2s` boundary, and automatic pause at `416.198s`.

## Residual risks

- Speech-only evidence requires a subtitle sidecar or optional faster-whisper installation.
- The verifier checks evidence binding and timestamp validity; it is not a full learned entailment model.
- A tiny reranker can overfit if dev supervision is too small, so its checkpoint metadata and inference weight remain visible.
- The single-process Web server serializes GPU pipeline runs to avoid duplicate model loading; concurrent production serving would need a job queue and model worker pool.
