from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time


@dataclass(frozen=True)
class GPUState:
    index: int
    uuid: str
    total_mib: int
    used_mib: int
    free_mib: int
    utilization_pct: int
    compute_pids: tuple[int, ...] = ()

    def safe(self, max_used_mib: int, max_utilization_pct: int) -> bool:
        return (
            not self.compute_pids
            and self.used_mib <= max_used_mib
            and self.utilization_pct <= max_utilization_pct
        )


def snapshot() -> list[tuple[int, int]]:
    """Backward-compatible memory-only snapshot."""
    return [(state.index, state.free_mib) for state in snapshot_states()]


def snapshot_states() -> list[GPUState]:
    gpu_rows = _query(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ]
    )
    uuid_to_pids: dict[str, list[int]] = {}
    process_rows = _query(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv,noheader,nounits",
        ],
        allow_failure=True,
    )
    for row in process_rows:
        if len(row) < 2:
            continue
        try:
            uuid_to_pids.setdefault(row[0], []).append(int(row[1]))
        except ValueError:
            continue

    states: list[GPUState] = []
    for row in gpu_rows:
        if len(row) < 6:
            continue
        try:
            states.append(
                GPUState(
                    index=int(row[0]),
                    uuid=row[1],
                    total_mib=int(row[2]),
                    used_mib=int(row[3]),
                    free_mib=int(row[4]),
                    utilization_pct=int(row[5]),
                    compute_pids=tuple(sorted(uuid_to_pids.get(row[1], []))),
                )
            )
        except ValueError:
            continue
    if not states:
        raise RuntimeError("nvidia-smi returned no parseable GPU state")
    return states


def select_pair(rows: list[tuple[int, int]], qwen_min: int, siglip_min: int) -> tuple[int, int] | None:
    """Compatibility selector for callers that only have (index, free_mib)."""
    ordered = sorted(rows, key=lambda item: item[1], reverse=True)
    for qwen_index, qwen_free in ordered:
        if qwen_free < qwen_min:
            continue
        for siglip_index, siglip_free in ordered:
            if siglip_index != qwen_index and siglip_free >= siglip_min:
                return qwen_index, siglip_index
    return None


def select_safe_pair(
    states: list[GPUState],
    qwen_min: int,
    siglip_min: int,
    max_used_mib: int,
    max_utilization_pct: int,
    preferred_qwen: int | None = None,
    preferred_siglip: int | None = None,
) -> tuple[int, int] | None:
    safe = {
        state.index: state
        for state in states
        if state.safe(max_used_mib, max_utilization_pct)
    }
    if preferred_qwen is not None or preferred_siglip is not None:
        if preferred_qwen is None or preferred_siglip is None or preferred_qwen == preferred_siglip:
            return None
        qwen = safe.get(preferred_qwen)
        siglip = safe.get(preferred_siglip)
        if qwen and siglip and qwen.free_mib >= qwen_min and siglip.free_mib >= siglip_min:
            return preferred_qwen, preferred_siglip
        return None

    ordered = sorted(safe.values(), key=lambda state: (state.free_mib, -state.index), reverse=True)
    for qwen in ordered:
        if qwen.free_mib < qwen_min:
            continue
        for siglip in ordered:
            if siglip.index != qwen.index and siglip.free_mib >= siglip_min:
                return qwen.index, siglip.index
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Wait for a non-invasive, stable GPU pair.")
    parser.add_argument("--wait-seconds", type=int, default=0)
    parser.add_argument("--poll-seconds", type=int, default=20)
    parser.add_argument("--stable-checks", type=int, default=3)
    parser.add_argument("--stable-interval-seconds", type=int, default=5)
    parser.add_argument("--qwen-min-free-mib", type=int, default=20500)
    parser.add_argument("--siglip-min-free-mib", type=int, default=6000)
    parser.add_argument("--max-used-mib", type=int, default=512)
    parser.add_argument("--max-utilization-pct", type=int, default=5)
    parser.add_argument("--preferred-qwen-index", type=int, default=None)
    parser.add_argument("--preferred-siglip-index", type=int, default=None)
    parser.add_argument("--audit-log", default="")
    args = parser.parse_args()

    required_checks = max(1, args.stable_checks)
    minimum_probe_window = max(0, required_checks - 1) * max(1, args.stable_interval_seconds)
    deadline = time.monotonic() + max(max(0, args.wait_seconds), minimum_probe_window)
    stable_pair: tuple[int, int] | None = None
    stable_count = 0
    probes: list[dict] = []
    while True:
        try:
            states = snapshot_states()
            selected = select_safe_pair(
                states,
                qwen_min=args.qwen_min_free_mib,
                siglip_min=args.siglip_min_free_mib,
                max_used_mib=args.max_used_mib,
                max_utilization_pct=args.max_utilization_pct,
                preferred_qwen=args.preferred_qwen_index,
                preferred_siglip=args.preferred_siglip_index,
            )
        except (OSError, RuntimeError, subprocess.SubprocessError) as exc:
            selected = None
            states = []
            _log(f"GPU probe failed: {type(exc).__name__}: {exc}")

        if selected == stable_pair and selected is not None:
            stable_count += 1
        else:
            stable_pair = selected
            stable_count = 1 if selected is not None else 0

        probes.append(
            {
                "observed_at_utc": datetime.now(timezone.utc).isoformat(),
                "selected_pair": list(selected) if selected is not None else None,
                "stable_count": stable_count,
                "states": [_state_payload(state) for state in states],
            }
        )

        if selected is not None and stable_count >= required_checks:
            _write_audit(args.audit_log, args, probes, "selected", selected)
            print(f"{selected[0]},{selected[1]}")
            return

        detail = _format_states(states)
        if time.monotonic() >= deadline:
            _write_audit(args.audit_log, args, probes, "no_safe_pair", None)
            raise SystemExit(
                "No safe stable GPU pair available without interrupting existing jobs: "
                + (detail or "probe unavailable")
            )
        _log(f"Waiting for safe GPU pair; {detail or 'probe unavailable'}")
        sleep_for = args.stable_interval_seconds if selected is not None else args.poll_seconds
        time.sleep(max(1, sleep_for))


def _query(command: list[str], allow_failure: bool = False) -> list[list[str]]:
    try:
        output = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        if allow_failure:
            return []
        raise RuntimeError(exc.output.strip() or "nvidia-smi failed") from exc
    rows = []
    for line in output.splitlines():
        line = line.strip()
        if not line or "No running processes" in line:
            continue
        rows.append([part.strip() for part in line.split(",")])
    return rows


def _format_states(states: list[GPUState]) -> str:
    return ", ".join(
        f"gpu={state.index} free={state.free_mib}MiB used={state.used_mib}MiB "
        f"util={state.utilization_pct}% pids={list(state.compute_pids)}"
        for state in states
    )


def _state_payload(state: GPUState) -> dict:
    return {
        "index": state.index,
        "uuid": state.uuid,
        "total_mib": state.total_mib,
        "used_mib": state.used_mib,
        "free_mib": state.free_mib,
        "utilization_pct": state.utilization_pct,
        "compute_pids": list(state.compute_pids),
    }


def _write_audit(
    audit_log: str,
    args: argparse.Namespace,
    probes: list[dict],
    status: str,
    selected: tuple[int, int] | None,
) -> None:
    if not audit_log:
        return
    path = Path(audit_log).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    selected_states = []
    if selected is not None and probes:
        selected_states = [
            state
            for state in probes[-1]["states"]
            if state["index"] in selected
        ]
    payload = {
        "schema_version": "videotrace-gpu-selection-audit-v1",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "selected_physical_gpu_ids": list(selected) if selected is not None else [],
        "selected_states": selected_states,
        "required_stable_checks": max(1, args.stable_checks),
        "thresholds": {
            "qwen_min_free_mib": args.qwen_min_free_mib,
            "siglip_min_free_mib": args.siglip_min_free_mib,
            "max_used_mib": args.max_used_mib,
            "max_utilization_pct": args.max_utilization_pct,
        },
        "non_interference_policy": (
            "selection only; selected GPUs must have zero compute PIDs for every stable probe; "
            "the selector never terminates or signals existing processes"
        ),
        "probes": probes,
    }
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temp.replace(path)


def _log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
