# VideoTrace Grounded Preference Data Card

## Purpose

This task-local dataset supports direct preference optimization for evidence-bound Chinese video answers. It teaches four product-visible preferences: use the supplied time window, include a playable timestamp, avoid unsupported visual details, and abstain when no candidate evidence exists.

It is not a public benchmark and is not intended to measure general video understanding.

## Artifacts

- Source positives: `data/sft/grounded_qa.jsonl`
- Authored negative annotations: `data/preference/preference_annotations.json`
- Training JSONL: `data/preference/grounded_dpo.jsonl`
- Validation summary: `data/preference/grounded_dpo.summary.json`

The builder copies the query, evidence, chosen answer, split, video identity and frozen-test flag from the SFT source. Annotators only supply the rejected answer, error type, rationale and provenance. This prevents a preference annotation from silently moving a source video into another split.

## Composition and isolation

The stable dataset contains 12 pairs: 7 SafeDroid train pairs, 4 Yoga dev pairs and 1 frozen cola test pair. Video groups do not cross splits. The designated cola review video is test-only and is excluded from `preference_gradient_payload_sha256`.

Negative categories are:

- `wrong_timestamp`: a claim is bound to a window not present in the supplied evidence.
- `missing_timestamp`: an otherwise plausible answer cannot be played or checked because it omits the time range.
- `hallucinated_detail`: the time range is plausible but the answer adds an unsupported action or state.
- `unsupported_overclaim`: an evidence-insufficient query receives a fabricated positive answer instead of an abstention.

## Quality controls

`scripts/build_preference_dataset.py` and `scripts/validate_preference_dataset.py` enforce unique pair IDs and payloads, non-identical chosen/rejected text, valid timestamps, explicit rationale/provenance, video-group split isolation, frozen-test isolation and negative-type-specific contracts. The summary records the dataset, source SFT, annotation and optimizer-payload hashes.

## Limitations

- The dataset is deliberately small and product-specific; it is evidence for a reproducible training loop, not a broad quality claim.
- The rejected answers are manually authored contrastive failures rather than production user ratings.
- Visual recognition remains supplied by the Qwen3.5/SigLIP2 inference path. Preference optimization targets answer grounding, abstention and timestamp behavior.
- The frozen cola pair is used only for regression/admission and never contributes gradients.
