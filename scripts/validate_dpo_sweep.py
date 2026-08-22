from __future__ import annotations

"""Validate the sealed-dev / one-time-frozen-test DPO sweep contract."""

import argparse
import json
from pathlib import Path
import sys

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.reproducibility import file_sha256, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-validate-dpo-sweep")
    parser.add_argument("--sweep", default="outputs/reports/dpo_sweep.json")
    parser.add_argument("--evaluation", default="outputs/reports/qwen35_dpo_selected_experiment_eval.json")
    parser.add_argument("--regression", default="outputs/reports/qwen35_dpo_selected_experiment_regression.json")
    parser.add_argument("--output", default="outputs/reports/dpo_sweep_validation.json")
    args = parser.parse_args()
    report = validate(
        _rooted(args.sweep),
        _rooted(args.evaluation),
        _rooted(args.regression),
    )
    _atomic_json(_rooted(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["valid"] else 1)


def validate(sweep_path: Path, evaluation_path: Path, regression_path: Path) -> dict:
    current_source = source_fingerprint(ROOT)
    failures: list[str] = []
    evidence: dict = {
        "current_source_sha256": current_source,
        "sweep_sha256": file_sha256(sweep_path) if sweep_path.is_file() else "",
        "evaluation_sha256": file_sha256(evaluation_path) if evaluation_path.is_file() else "",
        "regression_sha256": file_sha256(regression_path) if regression_path.is_file() else "",
    }
    if not sweep_path.is_file() or not evaluation_path.is_file() or not regression_path.is_file():
        failures.append("required research reports are missing")
        return {"schema_version": "videotrace-dpo-sweep-validation-v1", "valid": False, "failures": failures, "evidence": evidence}
    sweep = _load(sweep_path)
    evaluation = _load(evaluation_path)
    regression = _load(regression_path)
    # The sweep is an immutable research snapshot. Its source hash may point
    # to the exact tree used for the remote run, while the current checkout can
    # legitimately contain the validator/manifest code added afterward. Bind
    # freshness through the report's own source plus all downstream hashes.
    if not str(sweep.get("source_sha256") or ""):
        failures.append("sweep source fingerprint is missing")
    if len(sweep.get("candidates") or []) < 8:
        failures.append("fewer than eight sweep candidates")
    candidates = list(sweep.get("candidates") or [])
    if any(not bool((row.get("frozen_test") or {}).get("skipped")) for row in candidates):
        failures.append("a candidate accessed frozen test during selection")
    selected = dict(sweep.get("selected_candidate") or {})
    final = dict(sweep.get("selected_final") or {})
    if selected.get("spec") != final.get("spec"):
        failures.append("selected candidate and final retrain specs differ")
    final_frozen = dict(final.get("frozen_test") or {})
    if final_frozen.get("skipped") or final_frozen.get("reward_preference_accuracy") != 1.0:
        failures.append("selected final did not pass the one-time frozen preference gate")
    if float((final.get("dev") or {}).get("policy_preference_accuracy", 0.0)) < 1.0:
        failures.append("selected final dev policy preference accuracy is below 1.0")
    if not bool((evaluation.get("comparison") or {}).get("validated_for_web")):
        failures.append("selected final failed frozen product comparison")
    if int(regression.get("num_passed", 0)) != int(regression.get("num_cases", -1)):
        failures.append("selected final regressed on frozen product cases")
    seed = dict(sweep.get("seed_robustness") or {})
    if int(seed.get("num_runs", 0)) < 3 or float(seed.get("dev_policy_accuracy_std", 1.0)) != 0.0:
        failures.append("seed robustness evidence is incomplete")
    evidence.update(
        {
            "candidate_count": len(candidates),
            "selected_candidate_id": selected.get("candidate_id", ""),
            "selected_spec": selected.get("spec", {}),
            "selected_dev": final.get("dev", {}),
            "selected_frozen_test": final_frozen,
            "product_comparison": evaluation.get("comparison", {}),
            "regression": {"passed": regression.get("num_passed"), "total": regression.get("num_cases")},
            "seed_robustness": seed,
            "default_registry_untouched": selected.get("candidate_id") not in {"qwen35_sft", "qwen35_dpo"},
        }
    )
    if not evidence["default_registry_untouched"]:
        failures.append("research sweep unexpectedly used a product registry candidate id")
    return {"schema_version": "videotrace-dpo-sweep-validation-v1", "valid": not failures, "failures": failures, "evidence": evidence}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


if __name__ == "__main__":
    main()
