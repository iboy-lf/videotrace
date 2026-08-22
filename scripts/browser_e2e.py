from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin
import argparse
import hashlib
import json
import os
import time

from _bootstrap import ensure_src_path


ensure_src_path()

from videomemo.eval.reproducibility import source_fingerprint


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-browser-e2e")
    parser.add_argument("--base-url", default="http://127.0.0.1:7860")
    parser.add_argument("--video", default="", help="optional video path for a real upload+analysis run")
    parser.add_argument(
        "--query",
        default="这个视频主要讲了什么？请给出带时间戳的证据。",
    )
    parser.add_argument("--vlm-mode", default="auto_best")
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--output", default="outputs/reports/browser_e2e.json")
    args = parser.parse_args()

    try:
        from playwright.sync_api import Error as PlaywrightError
        from playwright.sync_api import TimeoutError as PlaywrightTimeoutError
        from playwright.sync_api import sync_playwright
    except ImportError as exc:
        raise SystemExit("Playwright is unavailable in this existing environment.") from exc

    base_url = str(args.base_url).rstrip("/") + "/"
    video = Path(args.video).expanduser().resolve() if args.video else None
    if video is not None and not video.is_file():
        raise SystemExit(f"video does not exist: {video}")
    timeout_ms = max(30, int(args.timeout_seconds)) * 1000
    console_errors: list[str] = []
    console_warnings: list[str] = []
    page_errors: list[str] = []
    report: dict = {
        "schema_version": "videotrace-browser-e2e-v1",
        "created_at": time.time(),
        "base_url": base_url,
        "video": str(video) if video else "",
        "video_sha256": _sha256(video) if video else "",
        "source_sha256": source_fingerprint(ROOT),
        "vlm_mode": args.vlm_mode,
        "checks": {},
        "valid": False,
    }

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1440, "height": 900})
            page = context.new_page()
            page.on("console", lambda message: _capture_console(message, console_errors, console_warnings))
            page.on("pageerror", lambda error: page_errors.append(str(error)))
            page.goto(base_url, wait_until="domcontentloaded", timeout=timeout_ms)
            page.wait_for_function(
                "() => window.VideoTraceDebug && window.VideoTraceDebug.capabilities() !== null",
                timeout=timeout_ms,
            )
            page.wait_for_function(
                "() => document.querySelectorAll('#vlmMode option').length > 0",
                timeout=timeout_ms,
            )
            page.wait_for_function(
                "() => document.querySelectorAll('#presetRow .preset').length >= 4",
                timeout=timeout_ms,
            )
            report["checks"]["core_controls_visible"] = all(
                page.locator(selector).is_visible()
                for selector in ("#videoUpload", "#query", "#vlmMode", "#runBtn")
            )
            report["checks"]["failure_retry_control_present"] = page.locator("#retryBtn").count() == 1
            report["checks"]["refusal_demo_preset_visible"] = any(
                "无证据拒答" in text
                for text in page.locator("#presetRow .preset").all_inner_texts()
            )
            mode_ids = page.locator("#vlmMode option").evaluate_all(
                "nodes => nodes.map(node => node.value).filter(Boolean)"
            )
            report["available_vlm_modes"] = mode_ids
            report["checks"]["requested_mode_available"] = args.vlm_mode in mode_ids
            if args.vlm_mode not in mode_ids:
                raise AssertionError(f"requested VLM mode is unavailable: {args.vlm_mode}; got {mode_ids}")
            report["checks"]["desktop_no_horizontal_overflow"] = _no_horizontal_overflow(page)

            if video is not None:
                page.locator("#videoUpload").set_input_files(str(video))
                preview_src = page.locator("#mainVideo").get_attribute("src") or ""
                report["checks"]["immediate_blob_preview"] = preview_src.startswith("blob:")
                page.wait_for_function(
                    "() => document.querySelector('#videoId').value && document.querySelector('#uploadState').textContent.includes('已上传')",
                    timeout=timeout_ms,
                )
                remote_src = page.locator("#mainVideo").get_attribute("src") or ""
                report["checks"]["upload_switched_to_remote_media"] = remote_src.startswith("/media?")
                page.locator("#vlmMode").select_option(args.vlm_mode)
                page.locator("#query").fill(args.query)
                page.locator("#runBtn").click()
                page.wait_for_function(
                    "() => Boolean(window.VideoTraceDebug.jobId())",
                    timeout=30_000,
                )
                job_id = page.evaluate("() => window.VideoTraceDebug.jobId()")
                report["job_id"] = job_id
                page.wait_for_function(
                    "jobId => { const meta = document.querySelector('#jobMeta'); const id = document.querySelector('#jobIdLabel')?.textContent || ''; const elapsed = document.querySelector('#jobElapsedLabel')?.textContent || ''; return meta && !meta.hidden && id.includes(jobId.slice(0, 12)) && elapsed.includes('已用时'); }",
                    arg=job_id,
                    timeout=30_000,
                )
                report["checks"]["job_identity_and_elapsed_visible"] = True
                job_url = urljoin(base_url, f"api/jobs/{job_id}")
                deadline = time.time() + max(30, int(args.timeout_seconds))
                final_job = None
                while time.time() < deadline:
                    response = context.request.get(job_url, timeout=30_000)
                    if not response.ok:
                        raise AssertionError(f"job polling failed: HTTP {response.status}")
                    final_job = response.json()
                    if final_job.get("status") in {"completed", "failed"}:
                        break
                    page.wait_for_timeout(1000)
                if not final_job or final_job.get("status") != "completed":
                    raise AssertionError(f"analysis did not complete: {final_job}")
                terminal_elapsed = _finite_float(final_job.get("elapsed_sec"))
                started_at = _finite_float(final_job.get("started_at"))
                completed_at = _finite_float(final_job.get("completed_at"))
                execution_elapsed = (
                    max(0.0, completed_at - started_at)
                    if started_at is not None and completed_at is not None
                    else None
                )
                page.wait_for_timeout(1200)
                recheck_response = context.request.get(job_url, timeout=30_000)
                if not recheck_response.ok:
                    raise AssertionError(f"terminal job recheck failed: HTTP {recheck_response.status}")
                rechecked_job = recheck_response.json()
                rechecked_elapsed = _finite_float(rechecked_job.get("elapsed_sec"))
                elapsed_stable = (
                    terminal_elapsed is not None
                    and rechecked_elapsed is not None
                    and abs(rechecked_elapsed - terminal_elapsed) <= 0.2
                )
                elapsed_matches_execution = (
                    terminal_elapsed is not None
                    and execution_elapsed is not None
                    and abs(terminal_elapsed - execution_elapsed) <= 0.2
                )
                report["checks"]["completed_elapsed_is_frozen"] = elapsed_stable
                report["checks"]["completed_elapsed_matches_execution_window"] = elapsed_matches_execution
                if not elapsed_stable or not elapsed_matches_execution:
                    raise AssertionError(
                        "completed job elapsed_sec is not a stable execution duration: "
                        f"terminal={terminal_elapsed}, recheck={rechecked_elapsed}, "
                        f"execution={execution_elapsed}"
                    )
                page.wait_for_function(
                    "() => document.querySelectorAll('.evidenceButton').length > 0 && document.querySelectorAll('.timelineItem').length > 0",
                    timeout=30_000,
                )
                report["job"] = {
                    "status": final_job.get("status"),
                    "phase": final_job.get("phase"),
                    "error_code": final_job.get("error_code", ""),
                    "elapsed_sec": terminal_elapsed,
                    "rechecked_elapsed_sec": rechecked_elapsed,
                    "execution_elapsed_sec": execution_elapsed,
                    "elapsed_delta_sec": (
                        round(rechecked_elapsed - terminal_elapsed, 3)
                        if terminal_elapsed is not None and rechecked_elapsed is not None
                        else None
                    ),
                    "started_at": started_at,
                    "completed_at": completed_at,
                    "events": final_job.get("events", []),
                    "persistence": final_job.get("persistence", {}),
                }
            else:
                page.wait_for_function(
                    "() => document.querySelectorAll('.evidenceButton').length > 0 && document.querySelectorAll('.timelineItem').length > 0",
                    timeout=timeout_ms,
                )

            evidence_count = page.locator(".evidenceButton").count()
            inline_citation_count = page.locator(".inlineCitation").count()
            timeline_count = page.locator(".timelineItem").count()
            report["evidence_count"] = evidence_count
            report["inline_citation_count"] = inline_citation_count
            report["timeline_count"] = timeline_count
            report["checks"]["answer_evidence_timeline_rendered"] = (
                bool(page.locator("#answer").inner_text().strip()) and evidence_count > 0 and timeline_count > 0
            )
            report["checks"]["inline_answer_citation_rendered"] = inline_citation_count > 0
            if not page.locator("#technicalDetails").evaluate("node => node.open"):
                page.locator("#technicalDetails > summary").click()
            page.wait_for_function(
                "() => document.querySelectorAll('.agentTraceItem').length > 0 && document.querySelectorAll('.verificationItem').length >= 3",
                timeout=30_000,
            )
            trace_count = page.locator(".agentTraceItem").count()
            verification_count = page.locator(".verificationItem").count()
            report["agent_trace_count"] = trace_count
            report["verification_check_count"] = verification_count
            report["checks"]["agent_trace_and_verifier_visible"] = (
                trace_count > 0
                and verification_count >= 3
                and "学习式否决器" in page.locator("#technicalContent").inner_text()
            )

            first_evidence = page.locator(".inlineCitation").first
            evidence_start = float(first_evidence.get_attribute("data-start") or 0)
            evidence_end = float(first_evidence.get_attribute("data-end") or evidence_start)
            first_evidence.click()
            page.wait_for_function(
                "start => Math.abs(document.querySelector('#mainVideo').currentTime - start) < 2",
                arg=evidence_start,
                timeout=30_000,
            )
            page.evaluate(
                "end => { const video = document.querySelector('#mainVideo'); video.currentTime = Math.max(0, end - 0.12); return video.play(); }",
                evidence_end,
            )
            page.wait_for_function(
                "() => { const v = document.querySelector('#mainVideo'); const s = window.VideoTraceDebug.playback(); return v.paused && !s.activeWindow; }",
                timeout=30_000,
            )
            report["checks"]["evidence_auto_pause_releases_window"] = True
            page.locator("#clearWindowBtn").click()
            page.wait_for_function("() => !document.querySelector('#mainVideo').paused", timeout=10_000)
            report["checks"]["continue_from_current_plays"] = True

            timeline = page.locator(".timelineItem").nth(1 if timeline_count > 1 else 0)
            timeline_start = float(timeline.get_attribute("data-start") or 0)
            timeline.click()
            page.wait_for_function(
                "start => { const v = document.querySelector('#mainVideo'); const s = window.VideoTraceDebug.playback(); return Math.abs(v.currentTime - start) < 2 && !s.activeWindow; }",
                arg=timeline_start,
                timeout=30_000,
            )
            page.evaluate(
                "start => { const video = document.querySelector('#mainVideo'); video.currentTime = start + 21; return video.play(); }",
                timeline_start,
            )
            page.wait_for_timeout(1200)
            report["checks"]["timeline_continues_without_window"] = page.evaluate(
                "() => !document.querySelector('#mainVideo').paused && !window.VideoTraceDebug.playback().activeWindow"
            )

            media_url = page.locator("#mainVideo").get_attribute("src") or ""
            range_response = context.request.get(
                urljoin(base_url, media_url.lstrip("/")),
                headers={"Range": "bytes=0-1023"},
                timeout=30_000,
            )
            report["range"] = {
                "status": range_response.status,
                "content_range": range_response.headers.get("content-range", ""),
                "content_length": range_response.headers.get("content-length", ""),
            }
            report["checks"]["http_range_206"] = (
                range_response.status == 206
                and range_response.headers.get("content-length") == "1024"
                and range_response.headers.get("content-range", "").startswith("bytes 0-1023/")
            )

            page.set_viewport_size({"width": 390, "height": 844})
            page.wait_for_timeout(250)
            report["checks"]["mobile_no_horizontal_overflow"] = _no_horizontal_overflow(page)
            report["console_errors"] = console_errors
            report["console_warnings"] = console_warnings
            report["page_errors"] = page_errors
            report["checks"]["console_clean"] = not console_errors and not console_warnings and not page_errors
            browser.close()
    except (AssertionError, PlaywrightError, PlaywrightTimeoutError) as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"

    report["valid"] = bool(report["checks"]) and all(report["checks"].values()) and not report.get("error")
    output = (ROOT / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output).resolve()
    if ROOT.resolve() not in (output, *output.parents):
        raise SystemExit("browser E2E output must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


def _capture_console(message, errors: list[str], warnings: list[str]) -> None:
    if message.type == "error":
        errors.append(message.text)
    elif message.type == "warning":
        warnings.append(message.text)


def _no_horizontal_overflow(page) -> bool:
    return bool(
        page.evaluate(
            """() => {
                const root = document.documentElement;
                const widthOk = root.scrollWidth <= window.innerWidth + 1;
                const visible = [...document.querySelectorAll('button, input, select, textarea, video')]
                  .filter(node => { const style = getComputedStyle(node); return style.display !== 'none' && style.visibility !== 'hidden'; });
                const boundsOk = visible.every(node => {
                  const rect = node.getBoundingClientRect();
                  return rect.left >= -1 && rect.right <= window.innerWidth + 1;
                });
                return widthOk && boundsOk;
            }"""
        )
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


if __name__ == "__main__":
    main()
