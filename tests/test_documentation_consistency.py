from pathlib import Path

from scripts.validate_documentation_consistency import validate_documentation


ROOT = Path(__file__).resolve().parents[1]


def test_interview_documents_match_canonical_metrics_and_job():
    """Documents must agree with whatever the machine reports currently say.

    This asserts ``content_valid`` rather than ``valid``: the remaining check,
    ``current-product-source``, binds the evidence to the exact source tree that
    produced it and can only be green immediately after a full revalidation on
    the GPU host. Asserting it here would make the test suite red for all normal
    development while proving nothing about the code under test.
    ``scripts/validate_delivery_package.py`` enforces that binding for a
    release, and ``docs/REVALIDATION.md`` documents the procedure.
    """

    report = validate_documentation(ROOT)
    assert report["content_valid"], report["failures"]


def test_source_binding_check_is_reported_separately_and_not_silently_dropped():
    """The snapshot binding must stay visible even when it is not asserted."""

    report = validate_documentation(ROOT)
    assert "snapshot_current" in report
    assert report["valid"] == (report["content_valid"] and report["snapshot_current"])


def test_dpo_beta_explanation_does_not_invert_the_kl_regularization_direction():
    guide = (ROOT / "docs" / "POST_TRAINING_DECISION_GUIDE.md").read_text(encoding="utf-8")

    assert "调大等于放松 KL" not in guide
    assert "更强的 KL 惩罚" in guide
