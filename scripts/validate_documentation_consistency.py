from __future__ import annotations

"""Check that interview-facing documents agree with canonical machine reports.

This is intentionally a small, deterministic consistency check. It does not
claim that prose is complete or that task-local metrics generalize; it only
prevents a stale job id, hash, or training number from surviving a product
refresh.
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.eval.reproducibility import source_fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-documentation-consistency")
    parser.add_argument("--output", default="outputs/reports/documentation_consistency.json")
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "also require the machine reports to have been produced by the current "
            "source tree. Use this for a release snapshot; it can only pass right "
            "after a full revalidation on the GPU host (see docs/REVALIDATION.md), "
            "so CI runs without it."
        ),
    )
    args = parser.parse_args()
    report = validate_documentation(ROOT)
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    if ROOT.resolve() not in (output, *output.parents):
        raise SystemExit("documentation report must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the digest of this tracked report identical on
    # Windows and Linux; the delivery validator hashes these bytes.
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["snapshot_current"] and not args.strict:
        # Never silent: the stale binding is stated even when it is not fatal.
        print(
            "WARNING: 机器产物由较早的源码树产生，证据链需要在 iboy 上重跑一次；"
            "见 docs/REVALIDATION.md。文档内容一致性检查本身已通过。",
            file=sys.stderr,
        )
    raise SystemExit(0 if (report["valid"] if args.strict else report["content_valid"]) else 1)


def validate_documentation(root: Path) -> dict:
    root = root.resolve()
    dpo = _load(root / "outputs/models/qwen35_dpo_metrics.json")
    resume = _load(root / "outputs/reports/qwen35_dpo_resume_validation.json")
    performance = _load(root / "outputs/reports/performance_report.json")
    browser = _load(root / "outputs/reports/browser_e2e.json")
    manifest = _load(root / "outputs/reports/artifact_manifest.json")
    length_bias = _load(root / "outputs/reports/dpo_length_bias.json")
    delivery = _load(root / "outputs/reports/delivery_readiness.json")

    source_sha = str(manifest.get("source_sha256") or "")
    current_source_sha = source_fingerprint(root)
    # The dated acceptance documents describe one specific validation run, so
    # they must quote *that run's* source fingerprint -- taken from the reports
    # the run produced -- not whatever the regenerated manifest says today.
    # Rewriting a dated record to match a newer tree would falsify it; whether
    # the run is still current is a separate question, answered by the
    # `current-product-source` check below.
    snapshot_source_sha = str(browser.get("source_sha256") or "")
    job_id = str(browser.get("job_id") or "")
    job_elapsed = _number_text(browser.get("job", {}).get("elapsed_sec"))
    browser_check_count = str(len(browser.get("checks") or {}))
    artifact_count = str(len(manifest.get("artifacts") or {}))
    admission_count = str(len(manifest.get("adapter_admission_history") or []))
    browser_sha = _sha256(root / "outputs/reports/browser_e2e.json")
    delivery_current = f"{delivery.get('checks_passed')}/{delivery.get('checks_total')}"
    dpo_eval = dpo.get("evaluations", {})
    dpo_facts = [
        _number_text(dpo_eval.get("train", {}).get("mean_reward_margin")),
        _number_text(dpo_eval.get("dev", {}).get("mean_reward_margin")),
        _number_text(dpo_eval.get("frozen_test", {}).get("mean_reward_margin")),
        _number_text(resume.get("evaluations", {}).get("train", {}).get("mean_reward_margin")),
        _number_text(resume.get("evaluations", {}).get("dev", {}).get("mean_reward_margin")),
        _number_text(resume.get("evaluations", {}).get("frozen_test", {}).get("mean_reward_margin")),
        _number_text(resume.get("train_loss_last")),
        _number_text(resume.get("tokens_per_second")),
        _number_text(resume.get("peak_cuda_memory_mib")),
    ]
    performance_facts = [
        _number_text(performance.get("model_residency", {}).get("cold_pipeline_construction_and_run_seconds")),
        _number_text(performance.get("model_residency", {}).get("warm_cache_hit_run_seconds")),
        _number_text(performance.get("model_residency", {}).get("speedup")),
    ]
    required_by_doc = {
        "docs/MODEL_CARD_DPO.md": dpo_facts,
        "docs/TRAINING.md": dpo_facts + ["GRPO", "PPO/RLHF"],
        # The length-bias diagnostic is the answer to the standard "is your
        # margin just length?" follow-up, so the guide must quote the numbers
        # the analyzer currently produces rather than a remembered version.
        "docs/POST_TRAINING_DECISION_GUIDE.md": [
            _number_text(length_bias.get("pearson_token_delta_vs_reward_margin")),
            _number_text(length_bias.get("token_delta", {}).get("mean")),
            _number_text(length_bias.get("token_delta", {}).get("max")),
            f"{length_bias.get('reward_margin_positive')}/{length_bias.get('pairs')}",
            f"{length_bias.get('reference_prefers_chosen_sum')}/{length_bias.get('pairs')}",
            f"{length_bias.get('reference_prefers_chosen_per_token')}/{length_bias.get('pairs')}",
            f"{length_bias.get('policy_flipped_vs_reference')}/{length_bias.get('pairs')}",
            _number_text(length_bias.get("beta")),
        ],
        "docs/FINAL_ACCEPTANCE_20260820.md": [
            snapshot_source_sha,
            job_id,
            browser_sha,
            job_elapsed,
            f"{browser_check_count} 项",
            f"{artifact_count} 个",
            f"{admission_count} 份",
            *performance_facts,
        ],
        "docs/INTERVIEW_READINESS_REPORT.md": [
            snapshot_source_sha,
            job_id,
            browser_sha,
            job_elapsed,
            f"{browser_check_count} 项",
        ],
        "docs/AGENT_TRACE_AUDIT_20260820.md": [job_id, job_elapsed],
        "docs/INTERVIEW_GUIDE.md": ["SFT", "DPO", "GRPO", "PPO/RLHF", "reference-relative"],
        # Present-tense status documents must quote the delivery validator's
        # *current* result. The dated acceptance records above legitimately keep
        # their original numbers; these three do not, because they claim to
        # describe the repository as it stands today.
        "docs/READINESS_CHECKLIST.md": [delivery_current],
        "docs/ROADMAP.md": [delivery_current],
        "docs/JOB_READINESS_MATRIX.md": [delivery_current],
    }
    checks = []
    for relative, facts in required_by_doc.items():
        path = root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        missing = [fact for fact in facts if fact and fact not in text]
        checks.append({"path": relative, "passed": not missing, "missing": missing})

    stale_tokens = [
        "0.22957693",
        "0.14952278",
        "0.19810181",
        "0.60302391",
        "60.581",
        "11099.59",
        "e13fb43e439447438207e9235dc757f3",
        "13b7f3c2efe14ff99c1ab0a72c51456c",
        "a4c17f72e1ad4f7391c815dc9a01042c",
        "0f1c812fa9dd2dd3b7846e860cc82499707400be0c2016d134ed799cea3ee03e",
        "ea3da56cae302bb0bf7d08dd14a82f835b38c646d100ec6fac88b1c1ef4beaea",
        "97.633",
        "27.048",
        "15.203",
        "1.779",
        "26.483",
        "15.072",
        "1.757",
        "29.477",
    ]
    stale_hits = []
    for relative in required_by_doc:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        stale_hits.extend({"path": relative, "token": token} for token in stale_tokens if token in text)
    checks.append({"path": "interview-facing-docs", "passed": not stale_hits, "stale_hits": stale_hits})
    machine_sources = {
        "artifact_manifest": source_sha,
        "browser_e2e": str(browser.get("source_sha256") or ""),
        "performance_report": str(performance.get("source_sha256") or ""),
    }
    checks.append(
        {
            "path": "current-product-source",
            # This is the only check that binds the evidence to the *current*
            # source tree rather than checking internal agreement. It goes red
            # on any product-source edit and only goes green again after a full
            # revalidation on the GPU host -- see docs/REVALIDATION.md. It is
            # therefore tagged separately so the unit test can assert document
            # correctness without requiring a GPU run on every commit;
            # validate_delivery_package.py still enforces the binding.
            "snapshot_binding": True,
            "passed": all(value == current_source_sha for value in machine_sources.values()),
            "current_source_sha256": current_source_sha,
            "machine_sources": machine_sources,
        }
    )
    valid = all(bool(check.get("passed")) for check in checks)
    content_checks = [check for check in checks if not check.get("snapshot_binding")]
    return {
        "schema_version": "videotrace-documentation-consistency-v1",
        "valid": valid,
        "content_valid": all(bool(check.get("passed")) for check in content_checks),
        "snapshot_current": all(
            bool(check.get("passed")) for check in checks if check.get("snapshot_binding")
        ),
        "source_sha256": source_sha,
        "current_source_sha256": current_source_sha,
        "browser_report_sha256": browser_sha,
        "job_id": job_id,
        "checks_passed": sum(bool(check.get("passed")) for check in checks),
        "checks_total": len(checks),
        "failures": [check for check in checks if not check.get("passed")],
    }


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _number_text(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
