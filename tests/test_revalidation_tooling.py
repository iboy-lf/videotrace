from __future__ import annotations

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]


def _tracked(pattern: str) -> list[Path]:
    result = subprocess.run(
        ["git", "ls-files", pattern],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return []
    return [ROOT / line.strip() for line in result.stdout.splitlines() if line.strip()]


def test_remote_shell_scripts_use_lf_in_the_working_tree():
    """Shell scripts reach the GPU host by scp, which bypasses git's newline
    normalization. A CRLF script fails there with an unreadable ``$'\\r':
    command not found``, so the bytes on disk must already be LF.
    """

    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in _tracked("*.sh")
        if path.is_file() and b"\r\n" in path.read_bytes()
    ]
    assert not offenders, f"CRLF in shell scripts that are copied to Linux: {offenders}"


def test_revalidation_script_supports_a_dry_run_and_stops_the_web_service():
    script = ROOT / "scripts" / "remote" / "revalidate_all.sh"
    text = script.read_text(encoding="utf-8")

    # A busy shared host must be inspectable without spending GPU time.
    assert "--dry-run" in text
    assert "DRY_RUN" in text
    # Every mutating stage must go through the guarded runner, otherwise
    # --dry-run would silently execute part of the pipeline.
    for stage in (
        "bash scripts/remote/run_tests.sh",
        "bash scripts/remote/run_qwen35_demo.sh",
        '"$PY" scripts/evaluate_qwen35_adapter.py',
        '"$PY" scripts/select_best_qwen35_adapter.py',
        '"$PY" scripts/profile_runtime.py',
        "bash scripts/remote/run_browser_e2e.sh",
    ):
        assert f"run {stage}" in text, f"stage not guarded by run(): {stage}"
    # Courtesy on a shared machine: do not leave a 9B model holding a card.
    assert "stop_web_service.sh" in text
    assert "--keep-web" in text


def test_sync_manifest_only_lists_paths_that_exist():
    """The sync archive is built with tar; a stale entry aborts the whole sync."""

    script = (ROOT / "scripts" / "remote" / "sync_to_iboy.ps1").read_text(encoding="utf-8")
    block = script.split("$include = @(", 1)[1].split(")", 1)[0]
    entries = [line.strip().strip(',').strip('"') for line in block.splitlines()]
    listed = [entry for entry in entries if entry]

    missing = [entry for entry in listed if not (ROOT / entry).exists()]
    assert not missing, f"sync manifest lists paths that no longer exist: {missing}"
