from __future__ import annotations

"""Compare local VLM/LLM model variants on the same frozen evidence pack.

This is a model-selection/inference experiment, not a benchmark claim. Each
variant runs in a fresh subprocess so weights are not co-resident and the
reported latency is attributable to that model path.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

from _bootstrap import ensure_src_path

ensure_src_path()

from videomemo.eval.reproducibility import file_sha256, source_fingerprint

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-run-model-variant-benchmark")
    parser.add_argument("--pack", default="outputs/iboy_qwen35/cola_review/knowledge_pack.json")
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="ID=PATH",
        help="repeatable model variant, e.g. qwen35=/lavender/models/Qwen3.5-9B",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--work-dir", default="outputs/experiments/model_variants")
    parser.add_argument("--output", default="outputs/reports/model_variant_benchmark.json")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    pack = _rooted(args.pack)
    work_dir = _rooted(args.work_dir)
    rows = []
    for item in args.model:
        variant_id, separator, model_path = item.partition("=")
        if not separator or not variant_id.strip() or not model_path.strip():
            raise SystemExit(f"invalid --model value: {item!r}; expected ID=PATH")
        rows.append(
            _run_variant(
                variant_id.strip(),
                model_path.strip(),
                pack,
                work_dir,
                args.device,
                force=args.force,
            )
        )
    report = {
        "schema_version": "videotrace-model-variant-benchmark-v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "same frozen cola evidence pack; model-selection and inference trade-off only; not a public benchmark",
        "source_sha256": source_fingerprint(ROOT),
        "pack_path": _display(pack),
        "pack_sha256": file_sha256(pack),
        "variants": rows,
        "interpretation": _interpret(rows),
    }
    _atomic_json(_rooted(args.output), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _run_variant(variant_id: str, model_path: str, pack: Path, work_dir: Path, device: str, *, force: bool) -> dict:
    target = work_dir / variant_id
    target.mkdir(parents=True, exist_ok=True)
    output = target / "evaluation.json"
    log = target / "run.log"
    if output.is_file() and not force:
        payload = json.loads(output.read_text(encoding="utf-8"))
    else:
        command = [
            sys.executable,
            "scripts/evaluate_qwen35_adapter.py",
            "--variant",
            "baseline",
            "--pack",
            str(pack),
            "--model",
            model_path,
            "--device",
            device,
            "--output",
            str(output),
        ]
        completed = subprocess.run(
            command,
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        log.write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(f"model variant {variant_id} failed; see {log}")
        payload = json.loads(output.read_text(encoding="utf-8"))
    verification = dict(payload.get("verification") or {})
    return {
        "variant_id": variant_id,
        "model_path": model_path,
        "model_exists_at_run": Path(model_path).is_dir(),
        "evaluation_path": _display(output),
        "evaluation_sha256": file_sha256(output),
        "elapsed_seconds": payload.get("elapsed_seconds"),
        "answer_sha256": payload.get("answer_sha256", ""),
        "verified": bool(verification.get("ok")),
        "coverage": verification.get("coverage"),
        "claim_support_ok": bool(payload.get("claim_support_ok")),
        "timestamp_binding_ok": bool(payload.get("timestamp_binding_ok")),
        "runtime": payload.get("runtime", {}),
    }


def _interpret(rows: list[dict]) -> dict:
    if not rows:
        return {"status": "no_variants"}
    fastest = min(rows, key=lambda row: float(row.get("elapsed_seconds") or 1e18))
    verified = [row for row in rows if row.get("verified") and row.get("claim_support_ok") and row.get("timestamp_binding_ok")]
    return {
        "fastest_verified_variant": fastest.get("variant_id") if fastest in verified else None,
        "fastest_overall_variant": fastest.get("variant_id"),
        "all_verified": len(verified) == len(rows),
        "rule": "choose the smallest/fastest model only after frozen grounding and timestamp gates; parameter count alone is not an optimization result",
    }


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


if __name__ == "__main__":
    main()
