from __future__ import annotations

import scripts.analyze_dpo_length_bias as length_bias


def _metrics(pairs: list[dict]) -> dict:
    return {"hyperparameters": {"beta": 0.1}, "evaluations": {"train": {"pairs": pairs}}}


def test_length_bias_separates_reference_length_preference_from_reward_margin():
    """A reference that only prefers shorter answers must be visible as such."""

    metrics = _metrics(
        [
            {"pair_id": "long-chosen", "reward_margin": 0.2, "policy_preference_correct": False},
            {"pair_id": "short-chosen", "reward_margin": 0.2, "policy_preference_correct": True},
        ]
    )
    reference = {
        "records": [
            # Chosen is longer, so its unnormalized log-probability is lower even
            # though it is better per token.
            ["long-chosen", {
                "negative_type": "wrong_timestamp",
                "chosen_logp": -120.0,
                "rejected_logp": -60.0,
                "chosen_tokens": 200,
                "rejected_tokens": 60,
            }],
            ["short-chosen", {
                "negative_type": "unsupported_overclaim",
                "chosen_logp": -40.0,
                "rejected_logp": -80.0,
                "chosen_tokens": 60,
                "rejected_tokens": 100,
            }],
        ]
    }

    report = length_bias.analyze_length_bias(metrics, reference)

    assert report["pairs"] == 2
    assert report["reward_margin_positive"] == 2
    assert report["reference_prefers_chosen_sum"] == 1
    assert report["reference_prefers_chosen_per_token"] == 2
    assert report["policy_flipped_vs_reference"] == 0
    assert report["token_delta"]["chosen_longer"] == 1


def test_length_bias_reports_positive_correlation_when_margin_tracks_length():
    metrics = _metrics(
        [
            {"pair_id": "a", "reward_margin": 0.1, "policy_preference_correct": True},
            {"pair_id": "b", "reward_margin": 0.9, "policy_preference_correct": True},
        ]
    )
    reference = {
        "records": [
            ["a", {"negative_type": "x", "chosen_logp": -10.0, "rejected_logp": -20.0,
                   "chosen_tokens": 10, "rejected_tokens": 10}],
            ["b", {"negative_type": "x", "chosen_logp": -10.0, "rejected_logp": -20.0,
                   "chosen_tokens": 200, "rejected_tokens": 10}],
        ]
    }

    report = length_bias.analyze_length_bias(metrics, reference)

    assert report["pearson_token_delta_vs_reward_margin"] > 0.99


def test_length_bias_refuses_pairs_without_reference_logprobs():
    metrics = _metrics([{"pair_id": "missing", "reward_margin": 0.1, "policy_preference_correct": True}])

    try:
        length_bias.analyze_length_bias(metrics, {"records": []})
    except SystemExit as error:
        assert "missing" in str(error)
    else:  # pragma: no cover - the analyzer must not silently drop a pair
        raise AssertionError("a pair without reference log-probabilities must fail loudly")
