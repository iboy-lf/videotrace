from videomemo.verifier.evidence_gate import assess_evidence_sufficiency
from videomemo.llm.qwen_vl_api import QwenVLAPIClient
from videomemo.vlm.segment_analyzer import _parse_analysis


def test_evidence_gate_abstains_for_unsupported_specific_query():
    context = {
        "items": [
            {
                "segment_id": "seg-1",
                "text": "一个人在室内展示饮料罐，并依次摆放在桌面上。",
                "vlm_score": 0.05,
            }
        ]
    }
    decision = assess_evidence_sufficiency(
        "视频里有没有购买链接和优惠价格？",
        context,
        use_semantic_score=False,
    )
    assert not decision["sufficient"]


def test_evidence_gate_accepts_supported_query():
    context = {
        "items": [
            {
                "segment_id": "seg-1",
                "text": "一个人在室内展示饮料罐，并依次摆放在桌面上。",
                "vlm_score": 0.05,
            }
        ]
    }
    decision = assess_evidence_sufficiency(
        "哪里展示了饮料罐？",
        context,
        use_semantic_score=False,
    )
    assert decision["sufficient"]
    assert decision["best_segment_id"] == "seg-1"


def test_evidence_gate_aggregates_structured_facts_for_ending_stage():
    context = {
        "items": [
            {
                "segment_id": "seg-ending-1",
                "text": "一名男子坐在桌前进行品尝。",
                "caption": "一名戴着眼罩的男子坐在桌前。",
                "entities": ["男子", "眼罩", "饮料瓶", "桌子", "吸管"],
                "actions": ["品尝饮料", "猜测"],
                "scene": "室内",
                "vlm_score": 0.09,
            },
            {
                "segment_id": "seg-ending-2",
                "text": "桌上摆放着多种饮料。",
                "caption": "男子继续使用吸管品尝饮料。",
                "entities": ["饮料", "吸管", "桌子"],
                "actions": ["品尝", "说话"],
                "scene": "室内",
                "vlm_score": 0.08,
            },
        ]
    }
    decision = assess_evidence_sufficiency(
        "最后盲测时主持人佩戴了什么，桌上有什么？请给出时间戳。",
        context,
        use_semantic_score=True,
        coverage_mode="stage_local",
    )
    assert decision["sufficient"]
    assert decision["best_segment_id"] == "seg-ending-1"
    assert decision["structured_match_count"] >= 3
    assert "wearing_or_blindfold" in decision["structured_matched_concepts"]
    assert "table_or_surface" in decision["structured_matched_concepts"]


def test_evidence_gate_keeps_unrelated_stage_local_question_abstained():
    context = {
        "items": [
            {
                "segment_id": "seg-ending",
                "text": "男子戴着眼罩坐在桌前品尝多种饮料。",
                "caption": "一名戴着眼罩的男子坐在桌前。",
                "entities": ["眼罩", "饮料", "桌子"],
                "actions": ["品尝"],
                "scene": "室内",
                "vlm_score": 0.09,
            }
        ]
    }
    decision = assess_evidence_sufficiency(
        "最后主持人有没有驾驶汽车穿越沙漠？请给出证据。",
        context,
        use_semantic_score=True,
        coverage_mode="stage_local",
    )
    assert not decision["sufficient"]


def test_verifier_rejects_unbound_timestamp_claim():
    from videomemo.verifier.simple_verifier import inspect_answer_grounding

    result = inspect_answer_grounding(
        "问题：哪里？\n结论：见证据。\n- 20 秒 (timestamp=20.0-40.0)\n- 证据外的时间 (timestamp=90.0-100.0)",
        ["timestamp=20.0-40.0"],
    )
    assert not result["ok"]
    assert result["unmatched_timestamp_refs"] == ["90.0-100.0"]


def test_claim_level_verifier_accepts_supported_timestamp_clauses():
    from videomemo.verifier.simple_verifier import inspect_answer_grounding

    result = inspect_answer_grounding(
        "问题：发生了什么？\n结论：有证据。\n- 20.0-40.0：男子举起可乐瓶，并将饮料倒入玻璃杯。(timestamp=20.0-40.0)",
        ["timestamp=20.0-40.0"],
        evidence_items=[
            {
                "start_sec": 20.0,
                "end_sec": 40.0,
                "text": "一名男子举起可乐瓶进行展示，随后将饮料倒入玻璃杯。",
            }
        ],
    )
    assert result["ok"] is True
    assert result["claim_support_checked"] is True
    assert result["claim_support_ok"] is True
    assert result["unsupported_claims"] == []


def test_claim_level_verifier_rejects_hallucinated_clause_inside_valid_window():
    from videomemo.verifier.simple_verifier import inspect_answer_grounding

    result = inspect_answer_grounding(
        "问题：发生了什么？\n结论：有证据。\n- 20.0-40.0：男子举起可乐瓶，随后驾驶汽车穿越沙漠。(timestamp=20.0-40.0)",
        ["timestamp=20.0-40.0"],
        evidence_items=[
            {
                "start_sec": 20.0,
                "end_sec": 40.0,
                "text": "一名男子在室内举起可乐瓶进行展示。",
            }
        ],
    )
    assert result["ok"] is False
    assert result["claim_support_ok"] is False
    assert any("驾驶汽车" in item["claim"] for item in result["unsupported_claims"])


def test_claim_level_verifier_preserves_timestamped_safe_abstention():
    from videomemo.verifier.simple_verifier import inspect_answer_grounding

    result = inspect_answer_grounding(
        "问题：有没有汽车？\n结论：证据不足，无法确认。\n- 10.0-20.0：候选片段未提供足以支持该结论的可见或文本证据。(timestamp=10.0-20.0)",
        ["timestamp=10.0-20.0"],
        evidence_items=[
            {
                "start_sec": 10.0,
                "end_sec": 20.0,
                "text": "男子在室内展示饮料瓶和玻璃杯。",
            }
        ],
    )
    assert result["ok"] is True
    assert result["claim_support_ok"] is True
    assert result["unsupported_claims"] == []


def test_pipeline_verifier_rejects_refusal_when_grounding_gate_is_sufficient():
    from videomemo.pipeline import VideoMemoPipeline

    result = VideoMemoPipeline._verify_payload(
        "问题：哪里？\n结论：证据不足。\n- 20.0-40.0：候选证据 (timestamp=20.0-40.0)",
        ["timestamp=20.0-40.0"],
        {"sufficient": True},
    )
    assert result["ok"] is False
    assert "despite sufficient" in result["reason"]


def test_qwen_segment_parser_strips_thinking_and_markdown():
    raw = """<think>hidden</think>
```json
{"summary":"桌上有三罐饮料","ocr_text":"COLA","entities":["饮料罐"],"actions":["摆放"],"scene":"室内桌面","confidence":0.92,"uncertainties":[]}
```"""
    parsed = _parse_analysis(raw)
    assert parsed["summary"] == "桌上有三罐饮料"
    assert parsed["ocr_text"] == "COLA"
    assert parsed["confidence"] == 0.92


def test_qwen_segment_parser_repairs_jsonish_nested_summary_cache():
    from videomemo.vlm.segment_analyzer import _normalize_analysis

    legacy = {
        "summary": (
            '{"summary":"男子展示可乐并举起非常可乐。", '
            '"ocr_text":"Coca-Cola", "entities":["男子","可乐罐"], '
            '"actions":["展示","举起"], "scene":"室内", "confidence":0.9}'
        ),
        "ocr_text": "",
        "entities": [],
        "actions": [],
        "scene": "",
        "confidence": 0.35,
    }
    repaired = _normalize_analysis(legacy)
    assert repaired["summary"] == "男子展示可乐并举起非常可乐。"
    assert repaired["ocr_text"] == "Coca-Cola"
    assert repaired["entities"] == ["男子", "可乐罐"]
    assert repaired["actions"] == ["展示", "举起"]
    assert repaired["scene"] == "室内"
    assert repaired["confidence"] == 0.9


def test_qwen_answer_parser_accepts_compact_json_contract():
    raw = '{"conclusion":"视频展示了检索模块。","evidence":[{"timestamp":"20.0-40.0","text":"画面出现 Multimodal retrieval"}]}'
    answer = QwenVLAPIClient._format_answer(
        raw,
        "哪里展示了检索？",
        [{"start_sec": 20.0, "end_sec": 40.0, "text": "fallback"}],
    )
    assert "视频展示了检索模块。" in answer
    assert "Multimodal retrieval" in answer
    assert answer.count("(timestamp=") == 1


def test_canonical_pack_hash_ignores_dynamic_adapter_metadata(tmp_path):
    import json
    from videomemo.eval.reproducibility import canonical_pack_sha256

    payload = {
        "video_path": "demo.mp4",
        "metadata": {"query": "q", "llm_adapter": {"evaluation_sha256": "old"}},
        "timeline": [],
    }
    path = tmp_path / "pack.json"
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    first = canonical_pack_sha256(path)
    payload["metadata"]["llm_adapter"] = {"evaluation_sha256": "new", "pack_sha256": "other"}
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    assert canonical_pack_sha256(path) == first


def test_qwen_answer_parser_handles_bold_markdown_sections():
    raw = """**问题：** 哪里展示了检索？
**结论：** 检索出现在第二段。
- timestamp=20.0-40.0：画面文字写有 Multimodal retrieval。 (timestamp=20.0-40.0)
"""
    answer = QwenVLAPIClient._format_answer(
        raw,
        "哪里展示了检索？",
        [{"start_sec": 20.0, "end_sec": 40.0, "text": "fallback"}],
    )
    assert "结论：检索出现在第二段。" in answer
    assert "问题：哪里展示了检索？" in answer
    assert "timestamp=20.0-40.0：" not in answer


def test_qwen_answer_parser_groups_module_heading_and_followup_observation():
    raw = """**结论：** 多模态检索模块
- 20.0-40.0：时间戳证据 (timestamp=20.0-40.0)
- 结论：视频在 20.0-40.0 秒展示了 Multimodal retrieval。
Agent planning and evidence verification
- 40.0-60.0：时间戳证据 (timestamp=40.0-60.0)
- 结论：视频展示 retrieve -> assess -> synthesize -> verify。
Trainable modules and evaluation
- 60.0-80.0：时间戳证据 (timestamp=60.0-80.0)
"""
    answer = QwenVLAPIClient._format_answer(
        raw,
        "问题",
        [
            {"start_sec": 20.0, "end_sec": 40.0, "text": "检索"},
            {"start_sec": 40.0, "end_sec": 60.0, "text": "规划"},
            {"start_sec": 60.0, "end_sec": 80.0, "text": "训练"},
        ],
    )
    assert "Multimodal retrieval" in answer
    assert "retrieve -> assess -> synthesize -> verify" in answer
    assert "Trainable modules and evaluation" in answer
    assert answer.count("(timestamp=") == 3


def test_qwen_answer_parser_backfills_truncated_timestamp_from_context():
    raw = """**结论：** 多模态检索模块
- 20.0-40.0：时间戳证据 (timestamp=20.0-40.0)
- 结论：视频展示了 Multimodal retrieval。
- 40.0-60.0：时间戳证据 (timestamp=40.0-60.0)
- 结论：视频展示 Agent planning and evidence verification。
- 60.0-80.0：时间戳证据 (timestamp=60.0-80.0)"""
    answer = QwenVLAPIClient._format_answer(
        raw,
        "问题",
        [
            {"start_sec": 20.0, "end_sec": 40.0, "text": "Multimodal retrieval"},
            {"start_sec": 40.0, "end_sec": 60.0, "text": "Agent planning and evidence verification"},
            {"start_sec": 60.0, "end_sec": 80.0, "text": "Trainable modules and evaluation"},
        ],
    )
    assert "Trainable modules and evaluation" in answer
    assert answer.count("(timestamp=") == 3


def test_qwen_answer_parser_completes_missing_selected_evidence_without_changing_conclusion():
    answer = QwenVLAPIClient._format_answer(
        '{"conclusion":"视频分为开场和盲测。","evidence":[{"timestamp":"0.0-20.0","text":"开场展示多款可乐"}]}',
        "整体流程是什么？",
        [
            {"start_sec": 0.0, "end_sec": 20.0, "text": "开场展示多款可乐"},
            {"start_sec": 400.0, "end_sec": 416.2, "text": "蒙眼完成最后盲测"},
        ],
    )
    assert "结论：视频分为开场和盲测。" in answer
    assert "timestamp=0.0-20.0" in answer
    assert "timestamp=400.0-416.2" in answer
    assert answer.count("(timestamp=") == 2


def test_qwen_answer_parser_does_not_attach_context_to_abstention():
    answer = QwenVLAPIClient._format_answer(
        '{"conclusion":"证据不足，无法确认。","evidence":[]}',
        "视频里是否出现火箭？",
        [{"start_sec": 0.0, "end_sec": 20.0, "text": "人物展示可乐"}],
    )
    assert "证据不足" in answer
    assert "timestamp=" not in answer
