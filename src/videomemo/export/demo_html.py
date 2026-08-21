from __future__ import annotations

from html import escape
import json
import os
from pathlib import Path

from ..models import KnowledgePack


def _rel(path: str, base: Path) -> str:
    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    try:
        return Path(os.path.relpath(target.resolve(), base.resolve())).as_posix()
    except ValueError:  # Different Windows drives cannot have a relative path.
        return target.resolve().as_posix()


def _answer_fields(answer: str) -> tuple[str, str]:
    question = ""
    conclusion = ""
    for raw_line in str(answer or "").splitlines():
        line = raw_line.strip()
        if line.startswith(("问题：", "用户问题：")):
            question = line.split("：", 1)[1].strip()
        elif line.startswith(("结论：", "总体结论：", "回答：")):
            conclusion = line.split("：", 1)[1].strip()
    return question, conclusion


def _evidence_text(value: object) -> str:
    text = str(value or "").strip()
    return text.split(" 场景：", 1)[0].strip() or text


def export_demo_html(pack: KnowledgePack, output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    html_path = out / "demo.html"
    video_src = _rel(pack.video_path, out)
    question, conclusion = _answer_fields(pack.answer)
    question = question or str(pack.metadata.get("query", "这个视频主要讲了什么？"))
    conclusion = conclusion or pack.answer or pack.summary
    agent_run = pack.metadata.get("agent_run", {})
    verification = agent_run.get("verification", {})
    timeline_json = json.dumps(pack.timeline, ensure_ascii=False).replace("</", "<\\/")

    evidence_html = "\n".join(
        f"""
        <button class="evidence" type="button" data-start="{float(item['start_sec']):.3f}" data-end="{float(item['end_sec']):.3f}">
          <time>{float(item['start_sec']):.1f}s - {float(item['end_sec']):.1f}s</time>
          <span>{escape(_evidence_text(item['text']))}</span>
          <b aria-hidden="true">▶</b>
        </button>
        """
        for item in pack.timeline
    )

    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>VideoTrace 分析结果</title>
  <style>
    :root {{ --ink:#18201e; --muted:#68736f; --line:#dce2df; --teal:#176b67; --page:#f4f5f6; --yellow:#e8b949; }}
    * {{ box-sizing:border-box; }}
    body {{ margin:0; color:var(--ink); background:var(--page); font-family:Arial,"Microsoft YaHei",sans-serif; }}
    header {{ height:68px; display:flex; align-items:center; justify-content:space-between; padding:0 max(20px,calc((100vw - 1180px)/2)); color:#fff; background:#15201e; border-bottom:3px solid var(--yellow); }}
    header strong {{ font-size:20px; }}
    header span {{ color:#b8dfc8; font-size:13px; }}
    main {{ width:min(1180px,100%); margin:0 auto; padding:24px; }}
    h1,h2,p {{ margin:0; }}
    .question {{ margin-bottom:18px; color:var(--muted); line-height:1.6; }}
    .grid {{ display:grid; grid-template-columns:minmax(0,1.35fr) minmax(330px,.65fr); background:#fff; border:1px solid var(--line); border-radius:8px; overflow:hidden; }}
    .videoPane,.answerPane {{ min-width:0; padding:18px; }}
    .videoPane {{ border-right:1px solid var(--line); }}
    video {{ display:block; width:100%; aspect-ratio:16/9; object-fit:contain; background:#080b0a; border-radius:6px; }}
    .playing {{ min-height:42px; margin-top:10px; padding:10px; color:#10514e; background:#eef7f5; border:1px solid #c9e0db; border-radius:6px; font-size:13px; }}
    .answerPane h1 {{ margin-bottom:12px; font-size:20px; }}
    .conclusion {{ padding:13px; line-height:1.65; background:#eef7f5; border-left:4px solid var(--teal); }}
    .answerPane h2 {{ margin:20px 0 10px; font-size:15px; }}
    .evidenceList {{ display:grid; gap:8px; }}
    .evidence {{ width:100%; display:grid; grid-template-columns:84px minmax(0,1fr) 20px; gap:9px; align-items:center; padding:10px; color:var(--ink); background:#fff; border:1px solid var(--line); border-radius:6px; text-align:left; cursor:pointer; }}
    .evidence:hover,.evidence.active {{ background:#f1f8f6; border-color:#9fc9c2; }}
    .evidence time {{ color:#10514e; font-size:12px; font-weight:700; }}
    .evidence span {{ display:-webkit-box; overflow:hidden; -webkit-box-orient:vertical; -webkit-line-clamp:3; line-height:1.45; font-size:13px; }}
    .evidence b {{ color:var(--teal); }}
    details {{ margin-top:22px; padding-top:4px; border-top:1px solid var(--line); }}
    summary {{ min-height:48px; display:flex; align-items:center; font-weight:700; cursor:pointer; }}
    details p {{ padding-bottom:16px; color:var(--muted); line-height:1.6; font-size:13px; }}
    @media (max-width:820px) {{ main {{ padding:14px; }} .grid {{ grid-template-columns:1fr; }} .videoPane {{ border-right:0; border-bottom:1px solid var(--line); }} }}
  </style>
</head>
<body>
  <header><strong>VideoTrace</strong><span>{'证据已核验' if agent_run.get('verified') else '证据待核验'}</span></header>
  <main>
    <p class="question">{escape(question)}</p>
    <section class="grid">
      <div class="videoPane">
        <video id="video" controls preload="metadata" src="{escape(video_src)}"></video>
        <div id="playing" class="playing">点击右侧证据，直接回看对应位置</div>
      </div>
      <div class="answerPane">
        <h1>回答与证据</h1>
        <p class="conclusion">{escape(conclusion)}</p>
        <h2>引用证据</h2>
        <div class="evidenceList">{evidence_html}</div>
      </div>
    </section>
    <details>
      <summary>技术实现</summary>
      <p>Qwen3.5 片段理解、SigLIP2 视觉检索、神经重排与结构化 Agent；证据覆盖率 {float(verification.get('coverage', 0.0)):.0%}。每条回答均绑定到原视频时间窗。</p>
    </details>
  </main>
  <script>
    const timeline = {timeline_json};
    const video = document.getElementById("video");
    const playing = document.getElementById("playing");
    let activeEnd = null;
    document.querySelectorAll(".evidence").forEach((button, index) => {{
      button.addEventListener("click", () => {{
        document.querySelectorAll(".evidence.active").forEach((item) => item.classList.remove("active"));
        button.classList.add("active");
        const item = timeline[index];
        activeEnd = Number(item.end_sec);
        video.currentTime = Number(item.start_sec);
        playing.textContent = `正在回看 ${{Number(item.start_sec).toFixed(1)}}s - ${{Number(item.end_sec).toFixed(1)}}s`;
        video.play().catch(() => {{ playing.textContent += "，请点击播放器播放"; }});
      }});
    }});
    video.addEventListener("timeupdate", () => {{
      if (activeEnd !== null && video.currentTime >= activeEnd - 0.05) {{
        video.pause();
        activeEnd = null;
        playing.textContent = "证据播放完成，可继续选择其他时间点";
      }}
    }});
  </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")
    return html_path
