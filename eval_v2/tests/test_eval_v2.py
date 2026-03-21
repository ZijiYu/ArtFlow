from __future__ import annotations

import json
from pathlib import Path

from guohua_eval.analyzer import GuohuaEvalV2Analyzer, _filter_terms
from guohua_eval.models import SemanticTerm
from guohua_eval.sentence_indexer import build_indexed_text, estimate_token_count


class _FakeUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class _FakeEmbeddingResponse:
    def __init__(self, embeddings: list[list[float]], total_tokens: int) -> None:
        self.data = [_FakeEmbeddingItem(embedding) for embedding in embeddings]
        self.usage = _FakeUsage(total_tokens)


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeChatResponse:
    def __init__(self, payload: dict, total_tokens: int) -> None:
        self.choices = [_FakeChoice(json.dumps(payload, ensure_ascii=False))]
        self.usage = _FakeUsage(total_tokens)


class _FakeEmbeddings:
    def create(self, *, input: list[str], **_: object) -> _FakeEmbeddingResponse:
        sentence_map = {
            "绢本设色，收藏于故宫博物院。": [1.0, 0.0, 0.0],
            "画面题跋署石涛，并钤有朱文印。": [0.0, 1.0, 0.0],
            "绢本设色，收藏于故宫博物院，再次强调馆藏。": [0.98, 0.02, 0.0],
            "画面只说山水苍润，意境悠远。": [0.0, 0.0, 1.0],
            "浅绛设色，披麻皴描绘山体。": [1.0, 0.0, 0.0],
            "右上角题跋署石涛，印章文字可辨。": [0.0, 1.0, 0.0],
            "现藏于故宫博物院，并见绢本材质。": [0.98, 0.02, 0.0],
        }
        return _FakeEmbeddingResponse([sentence_map[item] for item in input], total_tokens=11)


class _FakeCompletions:
    def create(self, *, messages: list[dict], **_: object) -> _FakeChatResponse:
        system_prompt = messages[0]["content"]
        user_prompt = messages[1]["content"]

        if "任务配置解析器" in system_prompt:
            return _FakeChatResponse(
                {
                    "slots": [
                        {
                            "slot_name": "馆藏",
                            "description": "关注馆藏机构与收藏地点",
                            "covered_terms": ["故宫博物院", "馆藏", "收藏于"],
                        },
                        {
                            "slot_name": "技法",
                            "description": "关注皴法、设色、笔法等",
                            "covered_terms": ["浅绛设色", "披麻皴", "绢本设色"],
                        },
                        {
                            "slot_name": "题跋与印章",
                            "description": "关注题跋作者、印章内容等",
                            "covered_terms": ["石涛", "朱文印", "印章文字"],
                        },
                    ]
                },
                total_tokens=13,
            )

        if "中国画术语抽取器" in system_prompt and "context_baseline" in user_prompt:
            return _FakeChatResponse(
                {
                    "terms": [
                        {
                            "term": "绢本设色",
                            "category": "材质",
                            "detail": "文本明确描述材质与设色",
                            "sentence_ids": [0],
                            "evidence_sentences": ["绢本设色，收藏于故宫博物院。"],
                        },
                        {
                            "term": "故宫博物院",
                            "category": "馆藏地址",
                            "detail": "指出收藏机构",
                            "sentence_ids": [0, 2],
                            "evidence_sentences": [
                                "绢本设色，收藏于故宫博物院。",
                                "绢本设色，收藏于故宫博物院，再次强调馆藏。",
                            ],
                        },
                        {
                            "term": "石涛",
                            "category": "题跋作者",
                            "detail": "题跋署名",
                            "sentence_ids": [1],
                            "evidence_sentences": ["画面题跋署石涛，并钤有朱文印。"],
                        },
                        {
                            "term": "朱文印",
                            "category": "印章内容",
                            "detail": "文本写到印章类型",
                            "sentence_ids": [1],
                            "evidence_sentences": ["画面题跋署石涛，并钤有朱文印。"],
                        },
                    ]
                },
                total_tokens=17,
            )

        if "中国画术语抽取器" in system_prompt and "context_enhanced" in user_prompt:
            return _FakeChatResponse(
                {
                    "terms": [
                        {
                            "term": "浅绛设色",
                            "category": "设色",
                            "detail": "直接点明设色方式",
                            "sentence_ids": [0],
                            "evidence_sentences": ["浅绛设色，披麻皴描绘山体。"],
                        },
                        {
                            "term": "披麻皴",
                            "category": "技法名称",
                            "detail": "具体皴法",
                            "sentence_ids": [0],
                            "evidence_sentences": ["浅绛设色，披麻皴描绘山体。"],
                        },
                        {
                            "term": "石涛",
                            "category": "题跋作者",
                            "detail": "题跋署名",
                            "sentence_ids": [1],
                            "evidence_sentences": ["右上角题跋署石涛，印章文字可辨。"],
                        },
                        {
                            "term": "故宫博物院",
                            "category": "馆藏地址",
                            "detail": "收藏机构",
                            "sentence_ids": [2],
                            "evidence_sentences": ["现藏于故宫博物院，并见绢本材质。"],
                        },
                    ]
                },
                total_tokens=19,
            )

        if "slot 命中判定器" in system_prompt and "context_baseline" in user_prompt:
            return _FakeChatResponse(
                {
                    "matches": [
                        {
                            "slot_name": "馆藏",
                            "matched_terms": ["故宫博物院"],
                            "matched_categories": ["馆藏地址"],
                            "sentence_ids": [0, 2],
                            "reason": "出现了明确馆藏机构。",
                        },
                        {
                            "slot_name": "题跋与印章",
                            "matched_terms": ["石涛", "朱文印"],
                            "matched_categories": ["题跋作者", "印章内容"],
                            "sentence_ids": [1],
                            "reason": "出现了题跋作者和印章信息。",
                        },
                    ]
                },
                total_tokens=7,
            )

        if "slot 命中判定器" in system_prompt and "context_enhanced" in user_prompt:
            return _FakeChatResponse(
                {
                    "matches": [
                        {
                            "slot_name": "馆藏",
                            "matched_terms": ["故宫博物院"],
                            "matched_categories": ["馆藏地址"],
                            "sentence_ids": [2],
                            "reason": "出现馆藏机构。",
                        },
                        {
                            "slot_name": "技法",
                            "matched_terms": ["浅绛设色", "披麻皴"],
                            "matched_categories": ["设色", "技法名称"],
                            "sentence_ids": [0],
                            "reason": "技法信息更完整。",
                        },
                        {
                            "slot_name": "题跋与印章",
                            "matched_terms": ["石涛"],
                            "matched_categories": ["题跋作者"],
                            "sentence_ids": [1],
                            "reason": "有题跋作者信息。",
                        },
                    ]
                },
                total_tokens=9,
            )

        if "视觉线索一致性判定器" in system_prompt and "context_baseline" in user_prompt:
            return _FakeChatResponse(
                {
                    "fidelity": [
                        {
                            "term": "绢本设色",
                            "category": "材质",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索支持绢本设色。",
                        },
                        {
                            "term": "故宫博物院",
                            "category": "馆藏地址",
                            "supported_by_ground_truth": False,
                            "reason": "视觉线索未给出馆藏机构。",
                        },
                        {
                            "term": "石涛",
                            "category": "题跋作者",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索提到题跋署石涛。",
                        },
                        {
                            "term": "朱文印",
                            "category": "印章内容",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索提到朱文印。",
                        },
                    ]
                },
                total_tokens=11,
            )

        if "视觉线索一致性判定器" in system_prompt and "context_enhanced" in user_prompt:
            return _FakeChatResponse(
                {
                    "fidelity": [
                        {
                            "term": "浅绛设色",
                            "category": "设色",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索支持浅绛设色。",
                        },
                        {
                            "term": "披麻皴",
                            "category": "技法名称",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索支持披麻皴。",
                        },
                        {
                            "term": "石涛",
                            "category": "题跋作者",
                            "supported_by_ground_truth": True,
                            "reason": "视觉线索支持石涛题跋。",
                        },
                        {
                            "term": "故宫博物院",
                            "category": "馆藏地址",
                            "supported_by_ground_truth": False,
                            "reason": "视觉线索未给出馆藏机构。",
                        },
                    ]
                },
                total_tokens=11,
            )

        if "专业中国画赏析评测裁判" in system_prompt:
            return _FakeChatResponse(
                {
                    "winner": "context_enhanced",
                    "textual_loss_for": "context_baseline",
                    "reasoning": "文本B在技法与 slot 覆盖率上更完整，同时保持了相近的信息密度。",
                    "textual_loss": "文本A的冗余主要来自重复强调“绢本设色、收藏于故宫博物院”的语义簇；同时忽略了视觉线索中可明确支持的“浅绛设色”“披麻皴”等技法信息。",
                },
                total_tokens=23,
            )

        raise AssertionError(f"未覆盖的调用: {system_prompt[:20]} / {user_prompt[:40]}")


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()
        self.embeddings = _FakeEmbeddings()


def test_sentence_indexer_and_token_estimate() -> None:
    indexed = build_indexed_text("第一句。第二句！第三句？")
    assert [item["sentence_id"] for item in indexed] == [0, 1, 2]
    assert indexed[1]["text"] == "第二句！"
    assert estimate_token_count("绢本设色 故宫博物院") >= 6


def test_filter_terms_removes_generic_words() -> None:
    filtered = _filter_terms(
        [
            SemanticTerm(
                term="笔墨",
                category="技法",
                detail="画面中的笔墨运用流畅自然。",
                sentence_ids=[0],
                evidence_sentences=["画面中的笔墨运用流畅自然。"],
            ),
            SemanticTerm(
                term="高低错落的构图",
                category="构图",
                detail="高低错落的构图增强空间感。",
                sentence_ids=[1],
                evidence_sentences=["画面采用高低错落的构图。"],
            ),
            SemanticTerm(
                term="石涛",
                category="题跋作者",
                detail="题跋署名为石涛。",
                sentence_ids=[2],
                evidence_sentences=["题跋署石涛。"],
            ),
            SemanticTerm(
                term="国立故宫博物院",
                category="馆藏机构",
                detail="作品现藏于国立故宫博物院。",
                sentence_ids=[3],
                evidence_sentences=["现藏于国立故宫博物院。"],
            ),
            SemanticTerm(
                term="雨点皴",
                category="皴法",
                detail="山壁采用雨点皴。",
                sentence_ids=[4],
                evidence_sentences=["山壁上的雨点皴拉长了视觉高度。"],
            ),
        ]
    )

    assert [item.term for item in filtered] == ["石涛", "国立故宫博物院", "雨点皴"]


def test_eval_v2_pipeline(tmp_path: Path) -> None:
    analyzer = GuohuaEvalV2Analyzer(client=_FakeClient(), duplicate_threshold=0.95)
    result = analyzer.evaluate(
        context_baseline=(
            "绢本设色，收藏于故宫博物院。"
            "画面题跋署石涛，并钤有朱文印。"
            "绢本设色，收藏于故宫博物院，再次强调馆藏。"
            "画面只说山水苍润，意境悠远。"
        ),
        context_enhanced=(
            "浅绛设色，披麻皴描绘山体。"
            "右上角题跋署石涛，印章文字可辨。"
            "现藏于故宫博物院，并见绢本材质。"
        ),
        slots_input="重点关注馆藏、技法、题跋与印章相关术语。",
        image_context_v="画面可见浅绛设色、披麻皴、题跋署石涛，并钤朱文印，材质为绢本。",
        output_dir=tmp_path,
    )

    assert result.slots_number == 3
    assert result.context_baseline_metrics.similar_semantic_num == 1
    assert result.context_baseline_metrics.duplicate_sentence_num == 2
    assert result.context_baseline_metrics.unique_semantic_num == 3
    assert result.context_baseline_metrics.term_num == 4
    assert result.context_baseline_metrics.slots_match == 2
    assert result.context_baseline_metrics.accuracy == 0.75
    assert result.context_enhanced_metrics.slots_match == 3
    assert result.context_enhanced_metrics.accuracy == 0.75
    assert result.final_judgment.winner == "context_enhanced"
    assert result.final_judgment.textual_loss_for == "context_baseline"
    assert "披麻皴" in result.final_judgment.textual_loss
    assert Path(result.result_json_path).exists()
    assert Path(result.context_baseline_metrics.duplicate_clusters_jsonl).exists()
    assert Path(result.context_enhanced_metrics.terms_jsonl).exists()
    assert result.llm_tokens == 132
