from __future__ import annotations

"""Evaluate the evidence-answer adapter on the frozen cola product context.

This is a task-local regression, not a benchmark.  The pack fixes the video,
query, retrieved windows, and verifier contract; only the answer model changes
between the baseline and adapter runs.  A separate compare step writes the
explicit Web admission decision consumed by ``server._prepare_runtime_config``.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import (
    canonical_pack_sha256,
    file_sha256,
    runtime_environment,
    source_fingerprint,
)
from videomemo.llm.qwen35_local import Qwen35LocalClient
from videomemo.verifier import inspect_answer_grounding


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-evaluate-qwen35-adapter")
    parser.add_argument("--variant", choices=["baseline", "adapter", "compare"], required=True)
    parser.add_argument("--pack", default=None)
    parser.add_argument("--model", default="/lavender/models/Qwen3.5-9B")
    parser.add_argument("--adapter", default="outputs/models/qwen35_sft_adapter")
    parser.add_argument("--candidate-id", default="qwen35_sft")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--input-baseline", default="outputs/reports/qwen35_adapter_eval_baseline.json")
    parser.add_argument("--input-adapter", default="outputs/reports/qwen35_adapter_eval_adapter.json")
    parser.add_argument("--output", default="outputs/reports/qwen35_adapter_eval.json")
    args = parser.parse_args()

    if args.variant == "compare":
        report = compare_reports(_read_json(ROOT / args.input_baseline), _read_json(ROOT / args.input_adapter))
    else:
        pack_path = resolve_pack(args.pack)
        report = evaluate_variant(
            variant=args.variant,
            candidate_id=(args.candidate_id if args.variant == "adapter" else "base_model"),
            pack_path=pack_path,
            model_path=args.model,
            adapter_path=(args.adapter if args.variant == "adapter" else ""),
            device=args.device,
            dtype=args.dtype,
        )
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def resolve_pack(requested: str | None) -> Path:
    candidates = []
    if requested:
        candidates.append(Path(requested))
    candidates.extend(
        [
            ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json",
            ROOT / "outputs" / "cola_review_qwen35" / "knowledge_pack.json",
            ROOT / "outputs" / "runs" / "latest" / "knowledge_pack.json",
        ]
    )
    for candidate in candidates:
        path = candidate if candidate.is_absolute() else ROOT / candidate
        if path.exists():
            return path.resolve()
    raise FileNotFoundError("no canonical cola knowledge pack found: " + ", ".join(str(p) for p in candidates))


def evaluate_variant(
    *,
    variant: str,
    candidate_id: str,
    pack_path: Path,
    model_path: str,
    adapter_path: str,
    device: str,
    dtype: str,
) -> dict:
    pack = json.loads(pack_path.read_text(encoding="utf-8"))
    metadata = dict(pack.get("metadata") or {})
    agent = dict(metadata.get("agent_run") or {})
    context = dict(agent.get("context") or {})
    items = list(context.get("items") or [])
    if not items:
        items = [
            {
                "segment_id": str(item.get("segment_id", f"timeline-{index}")),
                "start_sec": float(item.get("start_sec", 0.0)),
                "end_sec": float(item.get("end_sec", 0.0)),
                "score": float(item.get("score", 0.0)),
                "text": str(item.get("text", "")),
            }
            for index, item in enumerate(pack.get("timeline", []))
        ]
    video_path = Path(str(pack.get("video_path", ""))).expanduser()
    if not video_path.exists():
        raise FileNotFoundError(f"frozen pack video does not exist: {video_path}")
    query = str(metadata.get("query") or "这个视频的整体流程是什么？请概括开场、分国家试喝和最后盲测三个阶段并给出时间戳。")
    evidence_tags = list(context.get("evidence_tags") or [])
    if not evidence_tags:
        evidence_tags = [
            f"timestamp={float(item['start_sec']):.1f}-{float(item['end_sec']):.1f}"
            for item in items
        ]
    context["items"] = items
    context["video_path"] = str(video_path)
    started = time.perf_counter()
    client = Qwen35LocalClient(
        model_path=model_path,
        adapter_path=adapter_path,
        device=device,
        dtype=dtype,
        max_new_tokens=900,
        num_frames_per_segment=2,
    )
    answer = client.generate_answer(query, context, [])
    elapsed = time.perf_counter() - started
    verification = inspect_answer_grounding(answer, evidence_tags, evidence_items=items)
    return {
        "schema_version": "videotrace-qwen35-adapter-eval-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "variant": variant,
        "candidate_id": candidate_id,
        "pack_path": str(pack_path),
        # Admission binds to the stable pack identity.  The raw file SHA is
        # intentionally kept in artifact_manifest only: embedding it here
        # would make the evaluation hash depend on the admission metadata that
        # is written back into the pack.
        "pack_sha256": canonical_pack_sha256(pack_path),
        "pack_hash_scope": "canonical_json_without_metadata.llm_adapter",
        "video_path": str(video_path),
        "video_sha256": file_sha256(video_path),
        "query": query,
        "candidate_segment_ids": [str(item.get("segment_id", "")) for item in items],
        "evidence_tags": evidence_tags,
        "model_path": model_path,
        "adapter_path": adapter_path,
        "device": device,
        "dtype": dtype,
        "elapsed_seconds": round(elapsed, 3),
        "answer": answer,
        "answer_sha256": _text_sha256(answer),
        "verification": verification,
        "timestamp_binding_ok": not bool(verification.get("unmatched_timestamp_refs")),
        "claim_support_ok": bool(verification.get("claim_support_ok")),
        "runtime": runtime_environment(),
        "source_sha256": source_fingerprint(ROOT),
    }


def compare_reports(baseline: dict, adapter: dict) -> dict:
    baseline_verification = dict(baseline.get("verification") or {})
    adapter_verification = dict(adapter.get("verification") or {})
    baseline_coverage = float(baseline_verification.get("coverage", 0.0))
    adapter_coverage = float(adapter_verification.get("coverage", 0.0))
    adapter_ok = bool(adapter_verification.get("ok"))
    binding_ok = bool(adapter.get("timestamp_binding_ok"))
    claim_support_ok = bool(adapter.get("claim_support_ok"))
    # A small tolerance protects against harmless wording changes while
    # disallowing a checkpoint that loses evidence coverage or emits an
    # unbound timestamp.  The baseline remains the safe fallback otherwise.
    coverage_non_regression = adapter_coverage + 0.05 >= baseline_coverage
    validated_for_web = bool(adapter_ok and binding_ok and claim_support_ok and coverage_non_regression)
    return {
        "schema_version": "videotrace-qwen35-adapter-eval-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "candidate_id": adapter.get("candidate_id", "adapter"),
        "baseline": baseline,
        "adapter": adapter,
        "comparison": {
            "baseline_verified": bool(baseline_verification.get("ok")),
            "adapter_verified": adapter_ok,
            "baseline_coverage": baseline_coverage,
            "adapter_coverage": adapter_coverage,
            "coverage_delta": round(adapter_coverage - baseline_coverage, 6),
            "adapter_timestamp_binding_ok": binding_ok,
            "adapter_claim_support_ok": claim_support_ok,
            "coverage_non_regression": coverage_non_regression,
            "answer_changed": baseline.get("answer_sha256") != adapter.get("answer_sha256"),
            "validated_for_web": validated_for_web,
            "decision_reason": (
                "adapter passed frozen evidence, claim-support and non-regression gates"
                if validated_for_web
                else "keep baseline: adapter failed frozen evidence or non-regression gates"
            ),
        },
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _text_sha256(value: str) -> str:
    import hashlib

    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
