# INTERVIEW_DEPTH_AUDIT

## Current Highlight-Level Modules

### Agent Runtime
- Tool registry has explicit schemas, structured responses, inputs, outputs, latency, attempts, error codes, and circuit state.
- Bounded retries and per-tool circuit breakers are active safeguards rather than documentation-only claims.
- Default path is Plan-Execute for stability.
- ReAct-like path exists as an optional research implementation and has action-input repeat control.
- Demo exports execution plan and tool trace.

### Context Management
- Context is separated from memory.
- Compression is query-aware sentence selection first, head-tail fallback second.
- The system preserves segment id, timestamp, score, and evidence text.
- Dropped segments are recorded for debugging and interview explanation.

### Memory
- Memory has episodic records for concrete segment facts.
- Memory also has semantic records for extracted topic keywords.
- Retrieval combines text overlap, keyword overlap, importance, and salience.

### Retrieval
- Retrieval is no longer plain TF-IDF.
- Current retriever combines Chinese-friendly tokenization, TF-IDF similarity, lexical overlap, visual statistics, temporal bias, MMR diversity, frozen SigLIP2, query intent, temporal coverage, and an optional trained reranker.
- Each result carries retrieval signals for debugging.
- Dense segment embeddings can be persisted as a NumPy cosine index without adding FAISS to the existing environment.

### Verification And Evaluation
- Verifier returns matched evidence, missing evidence, timestamp refs, and coverage.
- Agent evaluation measures verification, evidence coverage, context keep rate, memory hit rate, tool success rate, and tool calls.

## Still Baseline

### VLM Understanding
- The remote path now runs local Qwen3.5-9B on ordered keyframes for structured segment descriptions and OCR.
- A frozen local SigLIP2 adapter provides image-text segment embeddings on a separate GPU.
- Lightweight deterministic components remain available only for narrow unit tests.
- Sidecar ASR alignment is implemented; faster-whisper remains optional because the existing remote environment is reused without new packages.

### LLM Synthesis
- Template mode remains the deterministic test backend.
- The remote path can synthesize with the same local Qwen3.5-9B runtime used for visual segment understanding.
- The answer is canonicalized and checked for timestamp binding after generation.

### Memory Persistence
- Memory has JSONL persistence, cross-video search, and query-time upsert.

### Dataset Scale
- Development supervision covers multiple local videos, while the product pitch stays focused on the real end-to-end demo.
- The reranker uses dev-only supervision; the real cola video remains frozen for the final demo.

## Interview Positioning
Do not claim this is already a production VLM system.
The correct positioning is:

VideoTrace is a complete, explainable long-video Agent prototype.
Its strength is the evidence-first pipeline: local Qwen segment understanding, optional ASR, sparse+dense retrieval, temporal query planning, trained reranking, context control, memory, tool safeguards, evidence gating, verification, and replayable Web export.
The next justified upgrades are more licensed dev supervision and claim-level verification, not training a foundation model for appearance.
