from __future__ import annotations

"""Run a dev-only DPO sweep, then unlock frozen test once for the selected setup."""

import argparse
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys

try:
    from _bootstrap import ensure_src_path
except ModuleNotFoundError:
    from scripts._bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.reproducibility import file_sha256, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Candidate:
    beta: float
    max_steps: int
    seed: int

    @property
    def candidate_id(self) -> str:
        return f"beta{str(self.beta).replace('.', 'p')}_steps{self.max_steps}_seed{self.seed}"


DEFAULT_CANDIDATES = (
    Candidate(0.1, 1, 43),
    Candidate(0.1, 2, 43),
    Candidate(0.1, 3, 43),
    Candidate(0.1, 4, 43),
    Candidate(0.05, 2, 43),
    Candidate(0.05, 2, 44),
    Candidate(0.05, 2, 45),
    Candidate(0.2, 2, 43),
    Candidate(0.1, 2, 44),
    Candidate(0.1, 2, 45),
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-run-dpo-sweep")
    parser.add_argument("--config", default="configs/qwen35_dpo.yaml")
    parser.add_argument("--work-dir", default="outputs/experiments/dpo_sweep")
    parser.add_argument("--output", default="outputs/reports/dpo_sweep.json")
    parser.add_argument("--selected-output", default="outputs/reports/qwen35_dpo_selected_experiment.json")
    parser.add_argument("--selected-model-card", default="outputs/models/qwen35_dpo_selected_experiment_model_card.json")
    parser.add_argument("--skip-product-regression", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    work_dir = _rooted(args.work_dir)
    rows = [_run_candidate(c, args.config, work_dir, force=args.force, frozen=False) for c in DEFAULT_CANDIDATES]
    selected = select_candidate(rows)
    spec = Candidate(**selected["spec"])
    final = _run_candidate(
        spec,
        args.config,
        work_dir / "selected_final",
        force=args.force,
        frozen=True,
        metrics_path=_rooted(args.selected_output),
        model_card_path=_rooted(args.selected_model_card),
    )
    frozen = final.get("frozen_test") or {}
    if frozen.get("reward_preference_accuracy") != 1.0:
        raise SystemExit("selected DPO experiment failed the frozen preference gate")
    report = {
        "schema_version": "videotrace-dpo-sweep-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "selection_policy": (
            "seal frozen test for every candidate; maximize dev policy preference accuracy, "
            "then minimize mean absolute implicit reward, then use fewer steps"
        ),
        "source_sha256": source_fingerprint(ROOT),
        "config_path": _display(_rooted(args.config)),
        "config_sha256": file_sha256(_rooted(args.config)),
        "candidates": rows,
        "seed_robustness": seed_robustness(rows, float(selected["spec"]["beta"]), int(selected["spec"]["max_steps"])),
        "selected_candidate": selected,
        "selected_final": final,
        "frozen_test_policy": "candidate runs skip frozen test; selected setup is evaluated exactly once",
    }
    _atomic_json(_rooted(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def select_candidate(rows: list[dict]) -> dict:
    eligible = [
        row for row in rows
        if float(row["dev"].get("reward_preference_accuracy", 0.0)) >= 1.0
        and float(row["dev"].get("mean_reward_margin", 0.0)) > 0
    ] or rows
    return max(
        eligible,
        key=lambda row: (
            float(row["dev"].get("policy_preference_accuracy", 0.0)),
            -float(row["dev"].get("mean_abs_implicit_reward", 1e9)),
            -int(row["spec"]["max_steps"]),
            -abs(float(row["spec"]["beta"]) - 0.1),
        ),
    )


def seed_robustness(rows: list[dict], beta: float, max_steps: int) -> dict:
    selected = [
        row for row in rows
        if math.isclose(float(row["spec"]["beta"]), beta)
        and int(row["spec"]["max_steps"]) == max_steps
    ]
    margins = [float(row["dev"].get("mean_reward_margin", 0.0)) for row in selected]
    accuracies = [float(row["dev"].get("policy_preference_accuracy", 0.0)) for row in selected]
    return {
        "beta": beta,
        "max_steps": max_steps,
        "seeds": [int(row["spec"]["seed"]) for row in selected],
        "num_runs": len(selected),
        "dev_reward_margin_mean": _mean(margins),
        "dev_reward_margin_std": _std(margins),
        "dev_policy_accuracy_mean": _mean(accuracies),
        "dev_policy_accuracy_std": _std(accuracies),
    }


def _run_candidate(candidate: Candidate, config: str, work_dir: Path, *, force: bool, frozen: bool, metrics_path: Path | None = None, model_card_path: Path | None = None) -> dict:
    root = work_dir if frozen else work_dir / "candidates" / candidate.candidate_id
    adapter = root / "adapter"
    metrics = metrics_path or root / "metrics.json"
    card = model_card_path or root / "model_card.json"
    log = root / "train.log"
    root.mkdir(parents=True, exist_ok=True)
    if metrics.is_file() and not force:
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    else:
        command = [
            sys.executable, "scripts/train_qwen35_dpo.py", "--config", str(_rooted(config)),
        "--output-dir", str(adapter), "--metrics-path", str(metrics), "--model-card-path", str(card),
            "--reference-logprobs-path", str(ROOT / "outputs/models/qwen35_dpo_reference_logprobs.json"),
            "--beta", str(candidate.beta), "--learning-rate", "0.00005", "--seed", str(candidate.seed),
            "--max-steps", str(candidate.max_steps), "--num-train-epochs", str(max(1, candidate.max_steps)), "--force",
        ]
        if not frozen:
            command.append("--skip-frozen-eval")
        completed = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"DPO candidate failed: {candidate.candidate_id}; see {log}")
        payload = json.loads(metrics.read_text(encoding="utf-8"))
    evaluations = payload.get("evaluations") or {}
    return {
        "candidate_id": candidate.candidate_id,
        "spec": asdict(candidate),
        "metrics_path": _display(metrics),
        "metrics_sha256": file_sha256(metrics),
        "adapter_path": _display(adapter),
        "adapter_sha256": file_sha256(adapter / "adapter_model.safetensors"),
        "steps": int(payload.get("steps", 0)),
        "tokens_per_second": float(payload.get("tokens_per_second", 0.0)),
        "peak_cuda_memory_mib": float(payload.get("peak_cuda_memory_mib", 0.0)),
        "dev": _compact(evaluations.get("dev") or {}),
        "frozen_test": _compact(evaluations.get("frozen_test") or {}),
    }


def _compact(payload: dict) -> dict:
    keys = (
        "num_pairs", "mean_loss", "mean_reward_margin", "mean_chosen_reward", "mean_rejected_reward",
        "mean_abs_implicit_reward", "reward_preference_accuracy", "policy_preference_accuracy",
        "reference_preference_accuracy", "policy_preference_accuracy_per_token",
        "reference_preference_accuracy_per_token", "policy_flip_count_vs_reference", "by_negative_type", "skipped", "reason",
    )
    return {key: payload[key] for key in keys if key in payload}


def _rooted(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def _display(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _mean(values: list[float]) -> float:
    return round(sum(values) / len(values), 8) if values else 0.0


def _std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return round(math.sqrt(sum((value - mean) ** 2 for value in values) / len(values)), 8)


if __name__ == "__main__":
    main()
