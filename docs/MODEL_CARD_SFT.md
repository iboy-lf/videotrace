# Qwen3.5 evidence-grounded LoRA 模型卡

## 模型

- base: `/lavender/models/Qwen3.5-9B`
- method: BF16 LoRA SFT；base weights frozen
- targets: `q_proj`、`v_proj`、`o_proj`
- real run: 1 optimizer step
- trainable params: 3,276,800；total params: 9,413,090,544
- train loss: 1.4157356024；dev loss: 1.213108
- throughput: 51.271 tokens/s；peak CUDA memory: 19129.67 MiB
- checkpoint supports adapter/tokenizer/optimizer/RNG/trainer-state resume；从正式 step 1 恢复到 step 2 的新 step loss 为 0.6611789465、dev loss 为 1.072001、吞吐 50.775 tokens/s、峰值 19156.73 MiB

## 产品准入

adapter 通过 frozen cola evidence/timestamp regression，coverage 1.00，verified=true。Web 只接受 adapter SHA、evaluation SHA、pack/video SHA 和 source snapshot 全部匹配的 admission；否则回退 baseline。不可变 evaluation 副本位于 `outputs/reports/adapter_admissions/`。

## 量化与限制

4-bit QLoRA preflight 因远端 bitsandbytes CPU-only 明确失败，当前真实部署路径为 BF16 LoRA。数据只有 12 条，结果是链路和准入证据，不是公开 benchmark 或质量提升承诺。该 SFT adapter 同时是后续 DPO 的冻结 reference 和产品安全 fallback；PPO/RLHF 未实现。
