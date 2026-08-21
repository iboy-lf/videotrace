# ARCHITECTURE

## End-to-end data flow

    upload/video file
      -> ingest: timestamped windows + keyframes + optional ASR/OCR
      -> Qwen3.5 segment understanding (frame-hash cache)
      -> sparse lexical + scorer + SigLIP2 retrieval (persistent dense index)
      -> temporal coverage + neural reranker
      -> bounded context + memory search
      -> Plan-Execute tools
      -> fixed Qwen3.5 answer model / admitted LoRA
      -> deterministic timestamp/claim gate
      -> portable calibrated safety veto
      -> knowledge pack + source-video window playback + Web/HTML

## Product/runtime boundaries

- `web/server.py` owns upload validation, capability discovery, serial job queue, resident pipeline reuse and public status.
- `web/vlm_modes.py` maps three public visual modes to a server-side whitelist; browser input never becomes a model path or backend.
- `web/static/playback.js` owns a small state machine: evidence window vs full-video mode.
- `pipeline.py` creates the canonical pack and records selected mode, adapter metadata, environment, GPU mapping and cache statistics.

## Algorithm modules

### Ingest/index/VLM

Windows retain `segment_id`, `start_sec`, `end_sec`, ordered keyframes and content fingerprints. Qwen descriptions expose entities/actions/OCR for lexical retrieval; SigLIP2 adds direct visual-text alignment. Their scores remain separate before fusion and reranking.

### Reranker and temporal selection

The neural reranker consumes the 12-feature contract and is loaded only when its checkpoint contract matches. Overview questions trigger explicit opening/middle/ending coverage; stage hints such as “开场/最后” and time anchors such as “300 秒左右” constrain selection before generation.

### Agent

Six schema tools execute in a bounded Plan-Execute trace: retrieval, context construction, evidence assessment, memory search, synthesis and verification. Each call records inputs, output envelope, latency, attempts, error code and circuit state. Tool timeout, evidence insufficiency and model errors enter controlled fallback instead of fabricated answers.

### Context/memory/verifier

Context preserves segment IDs and timestamps while applying a character budget. Memory is searchable historical evidence, not an unbounded prompt. Verification has two layers: deterministic timestamp/claim checks are authoritative, while a small portable NumPy logistic model may only veto an otherwise valid answer. It is task-local safety calibration, not general visual entailment.

### Training/post-training

Reranker training is dev-only pairwise learning. Qwen SFT and DPO use separate 12-row/12-pair train/dev/frozen-test contracts. The answer verifier derives 24 labeled rows from the preference pairs with a separate `14/8/2` split. Adapter admission binds adapter/evaluation/pack/video/source hashes, and the Web resolves the adapter only through that gate.

### Evaluation and governance

`error_analysis.json` is a five-case frozen task regression, `performance_report.json` is a single real-video cold/warm profile, and `artifact_manifest.json` is a hash inventory. None is a public benchmark or ranking claim.
