from __future__ import annotations

from videomemo.eval.artifact_manifest import _evidence_summary


def test_post_training_summary_reports_selected_registry_candidate():
    summary = _evidence_summary(
        {
            "sft_metrics": {
                "status": "completed",
                "steps": 1,
                "validated_for_web": False,
                "gradient_payload_sha256": "sft-gradient",
                "adapter_admission": {
                    "adapter_sha256": "sft-adapter",
                    "evaluation_sha256": "sft-evaluation",
                },
            },
            "dpo_metrics": {"status": "completed", "steps": 1},
            "best_adapter_registry": {
                "selected_candidate_id": "qwen35_dpo",
                "candidates": {
                    "qwen35_dpo": {
                        "method": "dpo",
                        "validated_for_web": True,
                        "adapter_sha256": "dpo-adapter",
                        "evaluation_sha256": "dpo-evaluation",
                    }
                },
            },
            "adapter_evaluation": {"comparison": {"validated_for_web": True}},
        }
    )

    post_training = summary["post_training"]
    assert post_training["sft_validated_for_web"] is False
    assert post_training["selected_candidate_id"] == "qwen35_dpo"
    assert post_training["selected_method"] == "dpo"
    assert post_training["selected_validated_for_web"] is True
    assert post_training["selected_adapter_sha256"] == "dpo-adapter"


def test_optional_dpo_research_artifacts_are_separate_from_product_contract(tmp_path):
    from videomemo.eval.artifact_manifest import OPTIONAL_ARTIFACT_PATHS, build_artifact_manifest

    assert "dpo_sweep_report" in OPTIONAL_ARTIFACT_PATHS
    assert "dpo_selected_experiment_metrics" in OPTIONAL_ARTIFACT_PATHS
