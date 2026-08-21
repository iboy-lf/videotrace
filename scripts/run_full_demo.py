from __future__ import annotations

from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
from _bootstrap import ensure_src_path

ensure_src_path()

from scripts.sample_video import make_demo_video
from videomemo.config import VideoMemoConfig
from videomemo.eval.agent_metrics import evaluate_agent_run
from videomemo.eval.metrics import evaluate_pack
from videomemo.pipeline import VideoMemoPipeline
from videomemo.planner.train_data import write_planner_dataset, write_ranker_dataset
from videomemo.scorer import train_scorer, write_scorer_dataset


def main() -> None:
    video_path = make_demo_video(str(ROOT / "data" / "raw" / "sample.mp4"))
    query = "这个视频主要讲了什么？请给出带时间戳的证据。"

    bootstrap_cfg = VideoMemoConfig(
        output_dir=str(ROOT / "outputs"),
        scorer_model_path=str(ROOT / "outputs" / "missing_scorer_model.pkl"),
    )
    bootstrap_pack = VideoMemoPipeline(bootstrap_cfg).run(video_path, query=query)

    train_dir = ROOT / "outputs_train"
    train_dir.mkdir(parents=True, exist_ok=True)
    planner_path = write_planner_dataset(query, bootstrap_pack.metadata["ranked_segments"], str(train_dir / "planner.json"))
    ranker_path = write_ranker_dataset(query, bootstrap_pack.metadata["ranked_segments"], str(train_dir / "ranker.json"))
    scorer_path = write_scorer_dataset(query, bootstrap_pack.segments, str(train_dir / "scorer.json"))

    scorer_result = train_scorer([scorer_path], str(train_dir / "scorer_model.pkl"))
    runtime_model = ROOT / "outputs" / "scorer_model.pkl"
    runtime_model.parent.mkdir(parents=True, exist_ok=True)
    runtime_model.write_bytes(Path(scorer_result.model_path).read_bytes())
    (train_dir / "scorer_metrics.json").write_text(
        json.dumps(scorer_result.dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    final_cfg = VideoMemoConfig(output_dir=str(ROOT / "outputs"), scorer_model_path=str(runtime_model))
    final_pipeline = VideoMemoPipeline(final_cfg)
    demo_path = final_pipeline.run_and_export(video_path, query=query)
    final_pack = final_pipeline.run(video_path, query=query)
    eval_result = evaluate_pack(final_pack, query.split())
    agent_eval_result = evaluate_agent_run(final_pack)
    eval_dir = ROOT / "outputs_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "eval.json").write_text(json.dumps(eval_result.dump(), ensure_ascii=False, indent=2), encoding="utf-8")
    (eval_dir / "agent_eval.json").write_text(
        json.dumps(agent_eval_result.dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(
        {
            "video": video_path,
            "demo": str(demo_path),
            "knowledge_pack": str(ROOT / "outputs" / "sample" / "knowledge_pack.json"),
            "report": str(ROOT / "outputs" / "sample" / "report.md"),
            "planner_data": planner_path,
            "ranker_data": ranker_path,
            "scorer_data": scorer_path,
            "scorer_model": str(runtime_model),
            "eval": str(eval_dir / "eval.json"),
            "agent_eval": str(eval_dir / "agent_eval.json"),
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
