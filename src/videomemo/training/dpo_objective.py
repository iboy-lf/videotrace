from __future__ import annotations

import math


def dpo_statistics(
    policy_chosen_logp: float,
    policy_rejected_logp: float,
    reference_chosen_logp: float,
    reference_rejected_logp: float,
    *,
    beta: float,
) -> dict:
    """Return the scalar DPO objective and its log-probability derivatives.

    Keeping this scalar implementation separate from the training framework
    makes the sign convention directly unit-testable.  The trainer recomputes
    one chosen and one rejected forward pass and applies these derivatives,
    avoiding simultaneous 9B-model activation graphs on a 24 GiB GPU.
    """

    if beta <= 0:
        raise ValueError("DPO beta must be positive")
    policy_logratio = float(policy_chosen_logp) - float(policy_rejected_logp)
    reference_logratio = float(reference_chosen_logp) - float(reference_rejected_logp)
    logit = float(beta) * (policy_logratio - reference_logratio)
    loss = _softplus(-logit)
    sigmoid_negative = _sigmoid_negative(logit)
    chosen_gradient = -float(beta) * sigmoid_negative
    rejected_gradient = -chosen_gradient
    chosen_reward = float(beta) * (float(policy_chosen_logp) - float(reference_chosen_logp))
    rejected_reward = float(beta) * (float(policy_rejected_logp) - float(reference_rejected_logp))
    reward_margin = chosen_reward - rejected_reward
    return {
        "loss": loss,
        "logit": logit,
        "policy_logratio": policy_logratio,
        "reference_logratio": reference_logratio,
        "chosen_reward": chosen_reward,
        "rejected_reward": rejected_reward,
        "reward_margin": reward_margin,
        "reward_preference_correct": reward_margin > 0,
        "policy_preference_correct": policy_logratio > 0,
        "reference_preference_correct": reference_logratio > 0,
        "chosen_logp_gradient": chosen_gradient,
        "rejected_logp_gradient": rejected_gradient,
    }


def _softplus(value: float) -> float:
    if value > 0:
        return value + math.log1p(math.exp(-value))
    return math.log1p(math.exp(value))


def _sigmoid_negative(value: float) -> float:
    if value >= 0:
        exp_negative = math.exp(-value)
        return exp_negative / (1.0 + exp_negative)
    exp_positive = math.exp(value)
    return 1.0 / (1.0 + exp_positive)
