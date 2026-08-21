# MODULES

## ingest / index / vlm

- timestamped video windows and ordered keyframes
- optional subtitle/ASR alignment and OCR fields
- Chinese-friendly sparse retrieval, lexical overlap, temporal bias and MMR
- frozen SigLIP2 embeddings with frame/content cache and persistent NumPy index

## scorer / reranker

- rule/scikit-learn feature scorer
- 12-feature neural query-segment reranker
- group-isolated dev training, checkpoint contract and recommended blend weight

## agent / context / memory

- schema-checked tool registry and serializable trace
- Plan-Execute orchestration with bounded context budget
- episodic and semantic memory records
- bounded retries, circuit breaker and controlled fallback

## llm / verifier

- template, OpenAI-compatible and local Qwen3.5 adapters
- hash-bound Qwen3.5 LoRA adapter admission
- evidence sufficiency, timestamp binding, coverage and abstention checks

## training

- `sft_data.py`: schema, split isolation, frozen test and train-only gradient payload hash
- `build_sft_dataset.py`: clean JSONL + data summary/data card inputs
- `train_qwen35_sft.py`: BF16 LoRA path, optional 4bit preflight, checkpoint/resume
- `evaluate_qwen35_adapter.py` / `admit_qwen35_adapter.py`: frozen product regression and Web gate

## web / export

- upload and project-scoped path checks
- server-returned executable VLM modes
- serial GPU job queue and polling progress
- original source-video window playback, static HTML, JSON/Markdown knowledge pack

## eval / governance

- task-local error taxonomy and five frozen regression cases
- cold/warm latency, cache, peak VRAM, precision and correctness profile
- GPU non-interference audit
- artifact manifest and delivery readiness validator

已补充任务内 LoRA DPO 和确定性 claim-to-evidence 支持检查。未实现且有意不冒充已实现：PPO/RLHF、公开 benchmark、VLA、Speech LLM、从零预训练和学习式 claim entailment。
