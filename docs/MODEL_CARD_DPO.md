# VideoTrace Qwen3.5 DPO Adapter Model Card

## Model and intended use

This is a LoRA adapter initialized from the admitted VideoTrace Qwen3.5 SFT adapter. It is intended to improve task-local preference for timestamp-correct, evidence-supported Chinese answers and explicit abstention over wrong timestamps, omitted timestamps, hallucinated detail and unsupported overclaiming.

It is not a general-purpose alignment model and is not evidence of broad video benchmark improvement.

## Training method

The base Qwen3.5-9B weights stay frozen. Before any optimizer update, the admitted SFT adapter computes and stores chosen/rejected sequence log-probabilities for all preference pairs. Training then keeps one policy model resident and applies the standard DPO loss against those frozen reference values. This avoids co-resident policy/reference 9B models on a 24 GiB GPU.

The verified run uses BF16 LoRA, `beta=0.1`, learning rate `5e-5`, four preference pairs per optimizer group and one optimizer step. The installed bitsandbytes build is CPU-only, so no 4-bit/QLoRA success is claimed.

## Data and leakage policy

The source is `data/preference/grounded_dpo.jsonl` with train/dev/frozen-test counts `7/4/1`. Chosen answers come from verified SFT records; rejected answers are manually authored and categorized. The cola review pair is frozen test-only and does not contribute backward gradients. Dataset, train-only gradient payload, SFT adapter and reference log-prob artifacts are SHA-bound.

## Task-local results

- DPO loss at the update boundary: `0.69314718`
- Train tokens: `411`; throughput: `45.501 tokens/s`
- Reported peak CUDA allocation: `9715.34 MiB` with two-GPU model parallelism
- Post-update mean reference-relative reward margin: train `0.22440502`, dev `0.14014463`, frozen cola `0.09733963`
- Reference-relative preference accuracy: `1.0` on train, dev and frozen test; absolute policy preference accuracy is `0.571429/0.75/1.0` on train/dev/frozen test, so this is not evidence that every pair is absolutely solved.

The formal step-1 checkpoint was also restored without changing its dataset/reference/SFT/source/config contract. One additional optimizer step produced loss `0.59365082`, throughput `61.506 tokens/s`, peak allocation `11141.65 MiB`, and train/dev/frozen margins `0.46911376/0.29970732/0.37768097`. This separate checkpoint is recovery evidence; the product registry continues to admit the frozen-regression-tested formal step-1 adapter.

These values demonstrate a real, recoverable update and the intended local preference direction. They do not establish broad generalization.

## Product admission

The adapter is not enabled merely because its files exist. The server-side selector requires frozen cola grounding, timestamp binding, claim-support and baseline coverage non-regression, plus DPO provenance and positive dev/frozen reward margins. The registry verifies adapter/config/metrics/model-card/evaluation/source hashes. If DPO fails or is tampered with, the product falls back to the admitted SFT adapter; users cannot select adapters from the browser.

## Limitations

- Very small, manually authored preference set.
- One optimizer step, deliberately limiting overfit risk and training cost.
- Preference optimization targets answer grounding behavior, not visual encoder learning.
- Product verification combines authoritative deterministic timestamp/claim rules with a separate task-local calibrated veto; neither is general visual entailment.
- PPO/RLHF and broad catastrophic-forgetting studies are not implemented.
