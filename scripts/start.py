from __future__ import annotations

from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
import json

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from videomemo.media_resolver import pack_has_playable_video


def main() -> None:
    host = "127.0.0.1"
    port = _select_port(host)
    url = f"http://{host}:{port}"

    if not _is_alive(url):
        latest_pack, playable = _preferred_pack()
        if latest_pack is None:
            raise RuntimeError(
                "缺少最终 Qwen3.5 分析结果；请同步 "
                "outputs/iboy_qwen35/cola_review/knowledge_pack.json。"
            )
        if not playable:
            # The knowledge pack is valid but its source video is absent or has
            # a different SHA-256. Serve the verified evidence read-only rather
            # than refusing to start: the answer, timestamps, agent trace and
            # technical panel are all still auditable, and the page states
            # explicitly that evidence cannot be replayed.
            print(
                "视频文件不可用，进入证据模式：回答、时间戳、脉络和技术面板可查看，"
                "证据不能回看。补齐 data/raw/cola_review.mp4（SHA-256 见产物清单）"
                "后重启即可恢复回放。详见 data/raw/README.md。"
            )
        env = dict(**__import__("os").environ)
        env["PYTHONPATH"] = "src"
        interview_config = ROOT / "configs" / "iboy_qwen35.yaml"
        if interview_config.exists():
            env["VIDEOTRACE_CONFIG"] = str(interview_config)
        command = [sys.executable, "scripts/run_web.py", "--host", host, "--port", str(port)]
        command += ["--latest-pack", str(latest_pack)]
        subprocess.Popen(
            command,
            cwd=str(ROOT),
            env=env,
            creationflags=getattr(subprocess, "CREATE_NEW_CONSOLE", 0),
        )
        _wait_until_alive(url)

    webbrowser.open(url)
    print(f"VideoTrace 已启动：{url}")


def _preferred_pack() -> tuple[Path | None, bool]:
    """Return the pack to serve and whether its source video is playable."""

    canonical = ROOT / "outputs" / "iboy_qwen35" / "cola_review" / "knowledge_pack.json"
    legacy = ROOT / "outputs" / "cola_review_qwen35" / "knowledge_pack.json"
    # If the current canonical artifact is present, never silently downgrade to
    # a stale legacy result -- not even when the canonical video is missing,
    # because a reviewer must see the current evidence rather than an older run.
    if canonical.exists():
        return canonical, _pack_points_to_existing_video(canonical)
    if legacy.exists():
        return legacy, _pack_points_to_existing_video(legacy)
    return None, False


def _pack_points_to_existing_video(pack: Path) -> bool:
    return pack_has_playable_video(pack, ROOT)


def _is_alive(url: str) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/health", timeout=1.0) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
            return resp.status == 200 and int(payload.get("product_version", 0)) >= 2
    except Exception:
        return False


def _select_port(host: str) -> int:
    for port in range(7860, 7871):
        url = f"http://{host}:{port}"
        if _is_alive(url) or _port_is_free(host, port):
            return port
    raise RuntimeError("7860-7870 端口均被占用。")


def _port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


def _wait_until_alive(url: str) -> None:
    for _ in range(30):
        if _is_alive(url):
            return
        time.sleep(0.5)
    raise RuntimeError(f"服务启动超时：{url}")


if __name__ == "__main__":
    main()
