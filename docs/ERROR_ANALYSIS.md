# 错误分析

范围是冻结可乐视频上的 5 个任务内回归，不是公开 benchmark。

## 分类

- `retrieval_error`：未选中覆盖 gold span 的窗口。
- `visual_understanding_error`：窗口正确，但结构化视觉描述缺少所需事实。
- `temporal_coverage_error`：全局流程漏掉开场、中段或结尾。
- `generation_error`：有证据但回答漏事实、错拒答或格式错误。
- `claim_support_error`：时间戳合法，但事实子句不被对应窗口文本支持。
- `verifier_miss`：仍违反 case 预期却被整体 verifier 放行。

## 当前结果

`outputs/reports/error_analysis.json` 绑定产品源码 `7374516a…` 与指定视频 SHA，结果为 `5/5 passed`、`error_category_counts={"none": 5}`。案例覆盖全局流程、开场产品、约 300 秒配料/OCR、末尾盲测和无证据拒答。

canonical 的硬时间戳/claim-support 校验全通过；calibrated verifier 也实际运行并通过，概率 `0.92716`、阈值 `0.2`。学习式层只能否决硬规则已通过的答案，因此错误定位仍优先落到检索、视觉、时序、生成和确定性校验，不能用模型分数掩盖硬失败。

## 已完成的修复闭环

早期问题包括全局问题过度拒答、阶段提示未约束检索、有证据却拒答仍被放行。对应修复是 temporal coverage gate、`stage_local`/`time_local` 约束、中文长问题 bigram 支持、claim-support 子句检查，以及 task-local calibrated safety veto。每项改动都进入可见回答或冻结回归，而不是只写 TODO。
