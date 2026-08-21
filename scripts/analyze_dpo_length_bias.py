from __future__ import annotations

"""Quantify how much of the DPO signal is a length artifact.

"Your reward margin is just length" is the standard follow-up to any DPO
result, and it cannot be answered with prose. This script recomputes the
diagnostic from two committed artifacts -- the frozen-reference log-probability
cache and the training metrics -- so the numbers quoted in
``docs/POST_TRAINING_DECISION_GUIDE.md`` can be re-derived by a reader who never
had access to the GPU host.

It reports three separable things:

* ``reward_margin`` vs the chosen/rejected token-count delta. The DPO reward is
  reference-relative, so a length preference shared by policy and reference
  cancels; a strong positive correlation here would mean it does not.
* Whether the *reference* model's preference is itself length-driven, by
  comparing its unnormalized sequence log-probability (what
  ``scripts/train_qwen35_dpo.py:_sequence_logp`` actually optimizes) against the
  per-token mean.
* Whether the policy's absolute preference differs from the reference's at all.

None of these are benchmark results. They characterize a 12-pair task-local
dataset and are reported precisely so the limits stay visible.
"""

import argparse
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(prog="videotrace-analyze-dpo-length-bias")
    parser.add_argument("--metrics", default="outputs/models/qwen35_dpo_metrics.json")
    parser.add_argument("--reference", default="outputs/models/qwen35_dpo_reference_logprobs.json")
    parser.add_argument("--output", default="outputs/reports/dpo_length_bias.json")
    args = parser.parse_args()

    report = analyze_length_bias(
        _load(ROOT / args.metrics),
        _load(ROOT / args.reference),
    )
    output = (ROOT / args.output).resolve()
    if ROOT.resolve() not in (output, *output.parents):
        raise SystemExit("length-bias report must remain inside the project")
    output.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" keeps the digest of this tracked report identical on
    # Windows and Linux; the delivery validator hashes these bytes.
    with output.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def analyze_length_bias(metrics: dict, reference: dict) -> dict:
    records = dict(reference.get("records") or [])
    pairs: list[dict] = []
    for split, evaluation in (metrics.get("evaluations") or {}).items():
        for pair in evaluation.get("pairs") or []:
            record = records.get(pair["pair_id"])
            if record is None:
                raise SystemExit(f"no reference log-probabilities for pair {pair['pair_id']}")
            chosen_tokens = int(record["chosen_tokens"])
            rejected_tokens = int(record["rejected_tokens"])
            chosen_logp = float(record["chosen_logp"])
            rejected_logp = float(record["rejected_logp"])
            pairs.append(
                {
                    "pair_id": pair["pair_id"],
                    "split": split,
                    "negative_type": record["negative_type"],
                    "token_delta": chosen_tokens - rejected_tokens,
                    "reward_margin": float(pair["reward_margin"]),
                    "policy_prefers_chosen": bool(pair["policy_preference_correct"]),
                    "reference_prefers_chosen_sum": chosen_logp > rejected_logp,
                    "reference_prefers_chosen_per_token": (
                        chosen_logp / chosen_tokens > rejected_logp / rejected_tokens
                    ),
                }
            )

    total = len(pairs)
    token_deltas = [pair["token_delta"] for pair in pairs]
    margins = [pair["reward_margin"] for pair in pairs]
    return {
        "schema_version": "videotrace-dpo-length-bias-v1",
        "pairs": total,
        "beta": (metrics.get("hyperparameters") or {}).get("beta"),
        "token_delta": {
            "mean": round(statistics.mean(token_deltas), 4),
            "min": min(token_deltas),
            "max": max(token_deltas),
            "chosen_longer": sum(1 for delta in token_deltas if delta > 0),
        },
        # The headline number: near zero means the reference-relative reward is
        # not explained by answer length.
        "pearson_token_delta_vs_reward_margin": round(_pearson(token_deltas, margins), 4),
        "reward_margin_positive": sum(1 for margin in margins if margin > 0),
        "policy_prefers_chosen": sum(1 for pair in pairs if pair["policy_prefers_chosen"]),
        # Unnormalized sequence log-probability is what the trainer uses, so a
        # gap between these two rows is a direct measure of the reference
        # model's own length preference on this dataset.
        "reference_prefers_chosen_sum": sum(1 for pair in pairs if pair["reference_prefers_chosen_sum"]),
        "reference_prefers_chosen_per_token": sum(
            1 for pair in pairs if pair["reference_prefers_chosen_per_token"]
        ),
        "policy_flipped_vs_reference": sum(
            1 for pair in pairs if pair["policy_prefers_chosen"] != pair["reference_prefers_chosen_sum"]
        ),
        "per_pair": pairs,
    }


def _pearson(xs: list[float], ys: list[float]) -> float:
    mean_x = statistics.mean(xs)
    mean_y = statistics.mean(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    denominator = (
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
