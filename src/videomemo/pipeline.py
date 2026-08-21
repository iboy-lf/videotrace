from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Optional
import os

from .agent import AgentHarness, ToolCircuitBreaker, ToolSpec
from .config import VideoMemoConfig
from .context import ContextBudget, ContextManager
from .export.clip_export import export_clip_files
from .export.demo_html import export_demo_html
from .export.simple_export import export_manifest, export_pack
from .index.simple_index import build_segment_index, index_statistics, rank_segments
from .ingest.asr import build_asr_backend, enrich_segments_with_asr
from .ingest.video import sample_segment_text, split_video_into_segments
from .llm import build_llm_client
from .llm.adapter_admission import adapter_admission_metadata, resolve_validated_adapter
from .memory import PersistentMemoryStore, VideoMemoryStore
from .models import KnowledgePack
from .planner.simple_planner import SimplePlanner
from .query import classify_query, select_ranked_segments
from .reranker import NeuralSegmentReranker
from .scorer import SegmentScorer, SklearnSegmentScorer, build_score_examples
from .verifier import CalibratedAnswerVerifier, assess_evidence_sufficiency, inspect_answer_grounding
from .eval.reproducibility import file_sha256, runtime_environment, source_fingerprint
from .vlm import build_segment_analyzer, build_vlm_embedder
from .vlm.scoring import attach_vlm_scores


class VideoMemoPipeline:
    def __init__(self, config: Optional[VideoMemoConfig] = None):
        self.config = config or VideoMemoConfig()
        self.project_root = Path(__file__).resolve().parents[2]
        if self.config.llm_backend == "qwen35_local" and not self.config.llm_adapter_path:
            self.config.llm_adapter_path = resolve_validated_adapter(self.project_root)
        self.llm_adapter_admission = adapter_admission_metadata(self.project_root)
        self.planner = SimplePlanner()
        self.scorer = SegmentScorer()
        self.llm_client = build_llm_client(
            self.config.llm_backend,
            self.config.llm_base_url,
            self.config.llm_model,
            self.config.llm_api_key,
            adapter_path=self.config.llm_adapter_path,
            device=self.config.llm_device,
            dtype=self.config.llm_dtype,
            max_new_tokens=self.config.llm_max_new_tokens,
            num_frames_per_segment=self.config.llm_num_frames_per_segment,
        )
        self.asr_backend = build_asr_backend(
            self.config.asr_backend,
            model=self.config.asr_model,
            device=self.config.asr_device,
            compute_type=self.config.asr_compute_type,
            language=self.config.asr_language,
        )
        self.segment_analyzer = build_segment_analyzer(
            self.config.segment_understanding_backend,
            self.config.segment_understanding_base_url or self.config.llm_base_url,
            self.config.segment_understanding_model or self.config.llm_model,
            self.config.segment_understanding_api_key or self.config.llm_api_key,
            self.config.segment_understanding_cache_dir,
            self.config.segment_understanding_frames,
            self.config.segment_understanding_timeout_sec,
            self.config.segment_understanding_fail_open,
            device=self.config.segment_understanding_device,
            dtype=self.config.segment_understanding_dtype,
            max_new_tokens=self.config.segment_understanding_max_new_tokens,
        )
        self.reranker = None
        if self.config.reranker_backend == "neural" and Path(self.config.reranker_model_path).exists():
            self.reranker = NeuralSegmentReranker(
                self.config.reranker_model_path, device=self.config.reranker_device
            )
        self.reranker_weight = float(self.config.reranker_weight)
        if self.reranker is not None and self.reranker_weight < 0:
            self.reranker_weight = self.reranker.recommended_blend_weight
        self.answer_verifier = None
        self.answer_verifier_load_error = ""
        verifier_path = Path(self.config.answer_verifier_model_path)
        if not verifier_path.is_absolute():
            verifier_path = self.project_root / verifier_path
        if self.config.answer_verifier_backend == "calibrated" and verifier_path.exists():
            try:
                threshold = (
                    self.config.answer_verifier_threshold
                    if self.config.answer_verifier_threshold >= 0
                    else None
                )
                self.answer_verifier = CalibratedAnswerVerifier(str(verifier_path), threshold=threshold)
            except Exception as exc:
                self.answer_verifier_load_error = f"{type(exc).__name__}: {exc}"
        self.scorer_model_path = Path(self.config.scorer_model_path)
        self.trained_scorer = None
        if self.scorer_model_path.exists():
            try:
                self.trained_scorer = SklearnSegmentScorer(str(self.scorer_model_path))
            except Exception:
                self.trained_scorer = None
        self.tool_circuit_breaker = ToolCircuitBreaker(
            failure_threshold=self.config.tool_circuit_failure_threshold,
            recovery_sec=self.config.tool_circuit_recovery_sec,
        )

    def run(self, video_path: str, query: str = "总结这个视频，并给出带时间戳的证据。") -> KnowledgePack:
        total_started = perf_counter()
        source_sha256 = source_fingerprint(Path(__file__).resolve().parents[2])
        stage_seconds: dict[str, float] = {}
        _cuda_memory_profile(reset_peak=True)

        stage_started = perf_counter()
        video_sha256 = file_sha256(Path(video_path))
        asset, segments = split_video_into_segments(
            video_path,
            self.config.segment_seconds,
            use_scene_cut=self.config.use_scene_cut,
        )
        segments = sample_segment_text(video_path, segments)
        stage_seconds["ingest"] = perf_counter() - stage_started

        stage_started = perf_counter()
        asr_report = enrich_segments_with_asr(
            video_path,
            segments,
            self.asr_backend,
            self.config.asr_cache_dir,
            fail_open=self.config.asr_fail_open,
        )
        stage_seconds["asr"] = perf_counter() - stage_started

        stage_started = perf_counter()
        understanding_report = self.segment_analyzer.enrich(video_path, segments)
        stage_seconds["segment_understanding"] = perf_counter() - stage_started

        stage_started = perf_counter()
        vlm_embedder = build_vlm_embedder(
            self.config.vlm_backend,
            self.config.vlm_model_name,
            self.config.vlm_cache_dir,
            self.config.vlm_num_frames,
            self.config.vlm_device or None,
        )
        vlm_report = attach_vlm_scores(
            video_path,
            query,
            segments,
            vlm_embedder,
            persist_index=self.config.persist_dense_index,
            index_dir=self.config.dense_index_dir,
        )
        stage_seconds["vlm_retrieval"] = perf_counter() - stage_started

        stage_started = perf_counter()
        index = build_segment_index(segments)
        full_retrieved = rank_segments(query, index, max(1, len(segments)))
        retrieved = full_retrieved[: self.config.top_k]
        retrieval_by_id = {item["segment_id"]: item for item in full_retrieved}

        score_examples = build_score_examples(query, segments)
        if self.trained_scorer is not None:
            segments = self.trained_scorer.rank(segments)
        else:
            segments = self.scorer.rank(query, segments)

        for seg in segments:
            retrieved_item = retrieval_by_id.get(seg.segment_id, {})
            seg.retrieval_score = float(retrieved_item.get("score", 0.0))
            seg.scorer_score = float(seg.score)
        self._attach_rank_scores(segments)
        for seg in segments:
            seg.score = self._fuse_scores(seg)
        if self.reranker is not None:
            for seg in segments:
                seg.reranker_score = self.reranker.score(query, seg, asset.duration_sec)
                seg.score = (1.0 - self.reranker_weight) * seg.score + self.reranker_weight * seg.reranker_score
        segments = sorted(segments, key=lambda seg: seg.score, reverse=True)

        query_intent = classify_query(query)
        fused_candidates = [
            {
                "segment_id": seg.segment_id,
                "start_sec": seg.start_sec,
                "end_sec": seg.end_sec,
                "text": seg.searchable_text(),
                "score": seg.score,
                "retrieval_score": seg.retrieval_score,
                "scorer_score": seg.scorer_score,
                "reranker_score": seg.reranker_score,
                "vlm_score": seg.vlm_score,
                "retrieval_rank_score": seg.retrieval_rank_score,
                "scorer_rank_score": seg.scorer_rank_score,
                "vlm_rank_score": seg.vlm_rank_score,
                "understanding_confidence": seg.understanding_confidence,
                "caption": seg.caption,
                "ocr_text": seg.ocr_text,
                "entities": list(seg.entities),
                "actions": list(seg.actions),
                "scene": seg.scene,
                "retrieval_signals": retrieval_by_id.get(seg.segment_id, {}).get("retrieval_signals", {}),
            }
            for seg in segments
        ]
        ranked = select_ranked_segments(
            fused_candidates,
            top_k=self.config.top_k,
            duration_sec=asset.duration_sec,
            intent=query_intent,
            enabled=self.config.temporal_coverage_enabled,
            min_segments=self.config.overview_min_segments,
        )
        stage_seconds["ranking"] = perf_counter() - stage_started

        stage_started = perf_counter()
        plan = self.planner.plan(query, ranked)
        video_id = Path(video_path).stem
        memory_store = VideoMemoryStore.from_segments(segments, video_id=video_id, video_path=video_path)
        persistent_memory_hits: list[dict] = []
        memory_records_written = 0
        if self.config.persistent_memory_enabled:
            persistent_store = PersistentMemoryStore(self.config.persistent_memory_path)
            persistent_memory_hits = persistent_store.search(query, top_k=3)
            memory_records_written = persistent_store.upsert(memory_store)
        context_manager = ContextManager(
            ContextBudget(
                max_chars=self.config.max_context_chars,
                per_segment_chars=self.config.per_segment_context_chars,
                min_segments=min(
                    self.config.top_k,
                    self.config.overview_min_segments if query_intent.coverage_mode == "distributed" else 2,
                ),
            )
        )
        agent_run = self._run_agent(
            query,
            ranked,
            context_manager,
            memory_store,
            video_path,
            persistent_memory_hits,
            query_intent=query_intent,
        )
        stage_seconds["agent"] = perf_counter() - stage_started

        timeline = [
            {
                "start_sec": r["start_sec"],
                "end_sec": r["end_sec"],
                "segment_id": r["segment_id"],
                "text": r["text"],
                "score": r["score"],
                "retrieval_score": r["retrieval_score"],
                "scorer_score": r["scorer_score"],
                "reranker_score": r.get("reranker_score", 0.0),
                "vlm_score": r["vlm_score"],
                "retrieval_rank_score": r["retrieval_rank_score"],
                "scorer_rank_score": r["scorer_rank_score"],
                "vlm_rank_score": r["vlm_rank_score"],
                "selection_reason": r.get("selection_reason", ""),
                "caption": r.get("caption", ""),
                "ocr_text": r.get("ocr_text", ""),
                "entities": list(r.get("entities", []) or []),
                "actions": list(r.get("actions", []) or []),
                "scene": r.get("scene", ""),
            }
            for r in ranked
        ]
        clips = [
            {
                "segment_id": r["segment_id"],
                "start_sec": max(0.0, r["start_sec"] - self.config.clip_margin_sec),
                "end_sec": r["end_sec"] + self.config.clip_margin_sec,
                "score": r["score"],
            }
            for r in ranked
        ]
        evidence = agent_run.context.get("evidence_tags", [])
        answer = agent_run.answer
        ok = bool(agent_run.verified)
        reason = str(agent_run.verification_reason)
        summary = (
            f"已处理 {len(segments)} 个视频片段；"
            f"Agent 模式={agent_run.mode}；"
            f"证据校验={reason}；通过={ok}"
        )
        stage_seconds["total"] = perf_counter() - total_started
        performance = {
            "stage_seconds": {key: round(value, 6) for key, value in stage_seconds.items()},
            "cuda_memory": _cuda_memory_profile(reset_peak=False),
            "cache": _cache_profile(understanding_report, vlm_report),
        }
        return KnowledgePack(
            video_path=video_path,
            duration_sec=asset.duration_sec,
            segments=segments,
            summary=summary,
            answer=answer,
            timeline=timeline,
            clips=clips,
            metadata={
                "query": query,
                "plan": [step.__dict__ for step in plan],
                "verified": ok,
                "index_stats": index_statistics(index),
                "retrieved_segments": retrieved,
                "ranked_segments": ranked,
                "score_examples": [ex.__dict__ for ex in score_examples],
                "scorer_mode": "sklearn" if self.trained_scorer is not None else "rule_based",
                "reranker": {
                    "backend": "neural" if self.reranker is not None else "none",
                    "checkpoint": self.config.reranker_model_path if self.reranker is not None else "",
                    "weight": self.reranker_weight if self.reranker is not None else 0.0,
                    "configured_weight": self.config.reranker_weight,
                    "checkpoint_recommended_weight": (
                        self.reranker.recommended_blend_weight if self.reranker is not None else 0.0
                    ),
                },
                "llm_backend": self.config.llm_backend,
                "llm_adapter": self.llm_adapter_admission,
                "vlm_mode": {
                    "id": self.config.selected_vlm_mode,
                    "label": self.config.selected_vlm_mode_label,
                },
                "asr": asr_report,
                "answer_verifier": self._answer_verifier_metadata(),
                "segment_understanding": understanding_report,
                "vlm": vlm_report,
                "query_intent": query_intent.dump(),
                "retrieval_selection": {
                    "strategy": "temporal_coverage" if query_intent.coverage_mode == "distributed" else "relevance_top_k",
                    "selected_segment_ids": [item["segment_id"] for item in ranked],
                    "selection_reasons": {item["segment_id"]: item.get("selection_reason", "") for item in ranked},
                },
                "resolved_config": self.config.dump(),
                "source_sha256": source_sha256,
                "video_sha256": video_sha256,
                "environment": runtime_environment(),
                "deployment": {
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
                    "physical_gpu_ids": os.environ.get("VIDEOTRACE_PHYSICAL_GPUS", ""),
                },
                "score_fusion": {
                    "retrieval_weight": self.config.retrieval_weight,
                    "scorer_weight": self.config.scorer_weight,
                    "vlm_weight": self.config.vlm_weight,
                    "normalization": "per_query_min_max",
                },
                "performance": performance,
                "agent_run": agent_run.dump(),
                "memory_records": memory_store.dump(),
                "persistent_memory": {
                    "enabled": self.config.persistent_memory_enabled,
                    "path": self.config.persistent_memory_path,
                    "records_written": memory_records_written,
                    "hits_before_upsert": persistent_memory_hits,
                },
            },
        )

    def run_and_export(self, video_path: str, query: str = "总结这个视频，并给出带时间戳的证据。") -> Path:
        pack = self.run(video_path, query=query)
        output_dir = Path(self.config.output_dir) / Path(video_path).stem
        if self.config.export_clips:
            export_clip_files(pack, str(output_dir))
        export_manifest(pack, str(output_dir))
        export_pack(pack, str(output_dir))
        if self.config.export_html:
            return export_demo_html(pack, str(output_dir))
        return output_dir / "knowledge_pack.json"

    def _fuse_scores(self, segment) -> float:
        return float(
            self.config.retrieval_weight * segment.retrieval_rank_score
            + self.config.scorer_weight * segment.scorer_rank_score
            + self.config.vlm_weight * segment.vlm_rank_score
        )

    @staticmethod
    def _attach_rank_scores(segments) -> None:
        retrieval = _min_max([float(seg.retrieval_score) for seg in segments])
        scorer = _min_max([float(seg.scorer_score) for seg in segments])
        vlm = _min_max([float(seg.vlm_score) for seg in segments])
        for seg, retrieval_score, scorer_score, vlm_score in zip(segments, retrieval, scorer, vlm):
            seg.retrieval_rank_score = retrieval_score
            seg.scorer_rank_score = scorer_score
            seg.vlm_rank_score = vlm_score

    def _run_agent(
        self,
        query: str,
        ranked: list[dict],
        context_manager: ContextManager,
        memory_store: VideoMemoryStore,
        video_path: str,
        persistent_memory_hits: list[dict] | None = None,
        query_intent=None,
    ):
        persistent_memory_hits = persistent_memory_hits or []
        harness = AgentHarness(
            max_steps=self.config.max_agent_steps,
            tool_max_attempts=1,
            tool_retry_backoff_sec=self.config.tool_retry_backoff_sec,
            circuit_breaker=self.tool_circuit_breaker,
        )
        self._register_agent_tools(
            harness=harness,
            ranked=ranked,
            context_manager=context_manager,
            memory_store=memory_store,
            video_path=video_path,
            persistent_memory_hits=persistent_memory_hits,
            query_intent=query_intent,
        )
        return harness.run_plan_execute(query)

    def _register_agent_tools(
        self,
        harness: AgentHarness,
        ranked: list[dict],
        context_manager: ContextManager,
        memory_store: VideoMemoryStore,
        video_path: str,
        persistent_memory_hits: list[dict],
        query_intent,
    ) -> None:
        harness.register_tool(
            ToolSpec(
                name="retrieve_segments",
                description="返回已经由检索器和 scorer 排好序的候选视频片段。",
                input_keys=["query"],
                output_keys=["segment_id", "start_sec", "end_sec", "text", "score"],
                input_schema={"query": "str"},
            ),
            lambda query: ranked,
        )
        harness.register_tool(
            ToolSpec(
                name="build_context",
                description="按照上下文预算压缩候选片段，同时保留时间戳、分数和证据文本。",
                input_keys=["query", "candidates"],
                output_keys=["items", "evidence_tags", "used_chars", "dropped_segment_ids"],
                input_schema={"query": "str", "candidates": "list"},
            ),
            lambda query, candidates: self._context_payload(context_manager.build(query, candidates), video_path=video_path),
        )
        harness.register_tool(
            ToolSpec(
                name="assess_evidence",
                description="检查候选片段与问题是否有足够的词义或多模态相关性，决定回答或拒答。",
                input_keys=["query", "context"],
                output_keys=["sufficient", "reason", "max_query_coverage", "max_vlm_score"],
                input_schema={"query": "str", "context": "dict"},
            ),
            lambda query, context: self._assess_evidence(query, context, query_intent=query_intent),
        )
        harness.register_tool(
            ToolSpec(
                name="search_memory",
                description="从视频片段记忆中检索和当前问题相关的历史事实。",
                input_keys=["query"],
                output_keys=["memory_id", "text", "source_segment_id", "score"],
                input_schema={"query": "str"},
            ),
            lambda query: self._merge_memory_hits(memory_store.search(query, top_k=3), persistent_memory_hits, top_k=4),
        )
        harness.register_tool(
            ToolSpec(
                name="synthesize_answer",
                description="基于上下文窗口和记忆命中生成带时间戳证据的中文回答。",
                input_keys=["query", "context", "memory_hits", "grounding_decision"],
                output_keys=["answer"],
                input_schema={
                    "query": "str",
                    "context": "dict",
                    "memory_hits": "list",
                    "grounding_decision": "dict",
                },
                max_attempts=self.config.tool_retry_max_attempts,
            ),
            self._synthesize_answer,
        )
        harness.register_tool(
            ToolSpec(
                name="verify_answer",
                description="先执行确定性时间戳/claim 检查，再用可校准安全模型保守否决可疑回答。",
                input_keys=["query", "answer", "evidence_tags", "evidence_items", "grounding_decision"],
                output_keys=[
                    "ok",
                    "reason",
                    "coverage",
                    "unmatched_timestamp_refs",
                    "claim_support_ok",
                    "unsupported_claims",
                    "calibrated_verifier",
                    "calibrated_verifier_ok",
                ],
                input_schema={
                    "query": "str",
                    "answer": "str",
                    "evidence_tags": "list",
                    "evidence_items": "list",
                    "grounding_decision": "dict",
                },
            ),
            lambda query, answer, evidence_tags, evidence_items, grounding_decision: self._verify_payload(
                answer,
                evidence_tags,
                grounding_decision,
                evidence_items=evidence_items,
                query=query,
                calibrated_verifier=self.answer_verifier,
                fail_open=self.config.answer_verifier_fail_open,
            ),
        )

    @staticmethod
    def _context_payload(window, video_path: str = "") -> dict:
        payload = window.dump()
        payload["evidence_tags"] = window.evidence_tags()
        payload["video_path"] = video_path
        return payload

    @staticmethod
    def _verify_payload(
        answer: str,
        evidence_tags: list[str],
        grounding_decision: dict | None = None,
        *,
        evidence_items: list[dict] | None = None,
        query: str = "",
        calibrated_verifier: CalibratedAnswerVerifier | None = None,
        fail_open: bool = True,
    ) -> dict:
        result = inspect_answer_grounding(answer, evidence_tags, evidence_items=evidence_items)
        decision = grounding_decision or {}
        refusal_markers = ("证据不足", "无法确认", "不能确认", "无法回答")
        if decision.get("sufficient") and any(marker in answer for marker in refusal_markers):
            result = dict(result)
            result["ok"] = False
            result["reason"] = "answer refused despite sufficient evidence"
        calibrated = {
            "enabled": False,
            "passed": True,
            "reason": "calibrated verifier disabled",
        }
        if calibrated_verifier is not None and bool(result.get("ok")):
            try:
                calibrated = calibrated_verifier.assess(
                    query,
                    answer,
                    evidence_items or [],
                    decision,
                    result,
                )
                if not calibrated.get("passed", False):
                    result = dict(result)
                    result["ok"] = False
                    result["reason"] = "calibrated verifier rejected a low-confidence grounded answer"
            except Exception as exc:
                calibrated = {
                    "enabled": True,
                    "passed": bool(fail_open),
                    "error": f"{type(exc).__name__}: {exc}",
                }
                if not fail_open:
                    result = dict(result)
                    result["ok"] = False
                    result["reason"] = "calibrated verifier failed closed"
        result = dict(result)
        result["calibrated_verifier"] = calibrated
        result["calibrated_verifier_ok"] = bool(calibrated.get("passed", True))
        return result

    def _answer_verifier_metadata(self) -> dict:
        if self.answer_verifier is not None:
            return self.answer_verifier.metadata()
        payload = {
            "backend": self.config.answer_verifier_backend,
            "loaded": False,
            "checkpoint_path": self.config.answer_verifier_model_path,
            "threshold": self.config.answer_verifier_threshold,
            "fail_open": self.config.answer_verifier_fail_open,
        }
        if self.answer_verifier_load_error:
            payload["load_error"] = self.answer_verifier_load_error
        return payload

    def _synthesize_answer(
        self,
        query: str,
        context: dict,
        memory_hits: list[dict],
        grounding_decision: dict,
    ) -> str:
        if self.config.abstain_enabled and not grounding_decision.get("sufficient", False):
            items = list(context.get("items", []))
            lines = [f"问题：{query}", "结论：证据不足，无法确认视频中存在所问内容。"]
            for item in items[:2]:
                timestamp = f"{float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
                lines.append(
                    f"- {timestamp}：候选片段未提供足以支持该结论的可见或文本证据。 "
                    f"(timestamp={timestamp})"
                )
            return "\n".join(lines)
        return self.llm_client.generate_answer(query, context, memory_hits)

    def _assess_evidence(self, query: str, context: dict, query_intent=None) -> dict:
        if not self.config.abstain_enabled:
            return {
                "sufficient": True,
                "reason": "abstention gate disabled",
                "query_terms": [],
                "max_query_coverage": 0.0,
                "max_vlm_score": 0.0,
                "best_segment_id": "",
            }
        return assess_evidence_sufficiency(
            query,
            context,
            min_query_coverage=self.config.abstain_min_query_coverage,
            min_vlm_score=self.config.abstain_min_vlm_score,
            use_semantic_score=self.config.vlm_backend in {"clip", "siglip"},
            intent_kind=str(getattr(query_intent, "kind", "locate")),
            coverage_mode=str(getattr(query_intent, "coverage_mode", "local")),
        )

    @staticmethod
    def _merge_memory_hits(local_hits: list[dict], persistent_hits: list[dict], top_k: int) -> list[dict]:
        merged: dict[str, dict] = {}
        for item in persistent_hits + local_hits:
            key = item.get("memory_id", "")
            if not key:
                continue
            if key not in merged or float(item.get("score", 0.0)) > float(merged[key].get("score", 0.0)):
                merged[key] = item
        return sorted(merged.values(), key=lambda item: float(item.get("score", 0.0)), reverse=True)[:top_k]


def _min_max(values: list[float]) -> list[float]:
    if not values:
        return []
    if len(values) == 1:
        return [1.0]
    low = min(values)
    high = max(values)
    if high - low <= 1e-12:
        return [0.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def _cuda_memory_profile(reset_peak: bool) -> dict:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"available": False, "devices": []}
        devices = []
        for index in range(torch.cuda.device_count()):
            if reset_peak:
                torch.cuda.reset_peak_memory_stats(index)
            devices.append(
                {
                    "index": index,
                    "name": torch.cuda.get_device_name(index),
                    "allocated_mib": round(torch.cuda.memory_allocated(index) / (1024**2), 2),
                    "reserved_mib": round(torch.cuda.memory_reserved(index) / (1024**2), 2),
                    "peak_allocated_mib": round(torch.cuda.max_memory_allocated(index) / (1024**2), 2),
                    "peak_reserved_mib": round(torch.cuda.max_memory_reserved(index) / (1024**2), 2),
                }
            )
        return {"available": True, "devices": devices}
    except Exception as exc:
        return {"available": False, "devices": [], "error": f"{type(exc).__name__}: {exc}"}


def _cache_profile(understanding_report: dict, vlm_report: dict) -> dict:
    understanding_records = list(understanding_report.get("records", []))
    vlm_records = list(vlm_report.get("segments", []))
    vlm_statuses = [str(record.get("metadata", {}).get("cache", "unknown")) for record in vlm_records]
    return {
        "segment_understanding": {
            "generated": sum(record.get("status") == "generated" for record in understanding_records),
            "hits": sum(record.get("status") == "cache_hit" for record in understanding_records),
            "fallbacks": sum(record.get("status") == "fallback" for record in understanding_records),
        },
        "vlm": {
            "hits": sum(status == "hit" for status in vlm_statuses),
            "misses": sum(status == "miss" for status in vlm_statuses),
            "unknown": sum(status not in {"hit", "miss"} for status in vlm_statuses),
        },
    }
