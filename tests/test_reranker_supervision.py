from pathlib import Path

from videomemo.reranker.supervision import load_reranker_supervision


def test_reranker_supervision_is_dev_only_and_excludes_final_demo():
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "supervision" / "reranker_annotations.json"
    cases = load_reranker_supervision(str(path))

    assert len(cases) == 11
    assert {case.split for case in cases} == {"dev"}
    assert {case.video_id for case in cases} == {"safedroid-demo", "yoga-action"}
    assert all("cola_review" not in case.video_path for case in cases)
