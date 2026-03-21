from __future__ import annotations

import json
from pathlib import Path

from textscore.analyzer import TextScoreAnalyzer
from textscore.sentence_indexer import build_indexed_text


class _FakeUsage:
    def __init__(self, total_tokens: int) -> None:
        self.total_tokens = total_tokens


class _FakeMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str, total_tokens: int) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(total_tokens)


class _FakeCompletions:
    def __init__(self) -> None:
        self._responses = [
            (
                {
                    "slots_analysis": [
                        {"sentence_ids": [0], "slot_name": "癌症", "slot_term": "肺癌", "score": 3},
                        {"sentence_ids": [0], "slot_name": "", "slot_term": "", "score": 1},
                        {"sentence_ids": [1], "slot_name": "分子分型", "slot_term": "EGFR突变", "score": 2},
                    ]
                },
                51,
            ),
            (
                {
                    "sentence_scores": [
                        {
                            "sentence_id": 0,
                            "score": 3,
                            "loss": 1,
                            "logic_score": 3,
                            "slot_relevance_score": 3,
                            "redundancy_score": 0,
                            "worth_optimizing": True,
                            "reasoning": "疾病对象明确，但诊疗信息还不算完整。",
                            "improvement_suggestion": "可继续补充具体病理分型、适应症或治疗路径。",
                            "matched_slots": ["癌症"],
                            "matched_terms": ["肺癌"],
                        },
                        {
                            "sentence_id": 1,
                            "score": 2,
                            "loss": 1,
                            "logic_score": 3,
                            "slot_relevance_score": 3,
                            "redundancy_score": 0,
                            "worth_optimizing": False,
                            "reasoning": "给出了分子分型术语，和任务高度相关。",
                            "improvement_suggestion": "可补充该突变对应的治疗决策或临床意义。",
                            "matched_slots": ["分子分型"],
                            "matched_terms": ["EGFR突变"],
                        },
                    ]
                },
                52,
            ),
            (
                {
                    "slots_analysis": [
                        {"sentence_ids": [0], "slot_name": "癌症", "slot_term": "肺癌", "score": 1}
                    ]
                },
                53,
            ),
            (
                {
                    "sentence_scores": [
                        {
                            "sentence_id": 0,
                            "score": 1,
                            "loss": 3,
                            "logic_score": 1,
                            "slot_relevance_score": 2,
                            "redundancy_score": 0,
                            "worth_optimizing": True,
                            "reasoning": "只说了肺癌治疗，信息较泛。",
                            "improvement_suggestion": "补充具体治疗方案、适用人群或病理条件。",
                            "matched_slots": ["癌症"],
                            "matched_terms": ["肺癌"],
                        },
                        {
                            "sentence_id": 1,
                            "score": 0,
                            "loss": 4,
                            "logic_score": 1,
                            "slot_relevance_score": 0,
                            "redundancy_score": 3,
                            "worth_optimizing": False,
                            "reasoning": "先做一些检查。",
                            "improvement_suggestion": "先做一些检查。",
                            "matched_slots": [],
                            "matched_terms": [],
                        },
                    ]
                },
                54,
            ),
        ]

    def create(self, **_: object) -> _FakeResponse:
        payload, total_tokens = self._responses.pop(0)
        return _FakeResponse(json.dumps(payload, ensure_ascii=False), total_tokens)


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeClient:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_sentence_indexer() -> None:
    indexed = build_indexed_text("第一句。第二句！第三句？")
    assert [item["sentence_id"] for item in indexed] == [0, 1, 2]
    assert indexed[1]["text"] == "第二句！"


def test_score_aggregation_and_visualization(tmp_path: Path) -> None:
    analyzer = TextScoreAnalyzer(client=_FakeClient())
    result, html_path = analyzer.score(
        context_1="肺癌治疗要先分型。EGFR突变可考虑靶向药。",
        context_2="肺癌需要治疗。先做一些检查。",
        slots=["癌症", "分子分型"],
        slots_number=2,
        visualization_path=tmp_path / "view.html",
    )

    assert result.context_1_slots_score == 5
    assert result.context_2_slots_score == 1
    assert len(result.context_1_slots_analysis) == 2
    assert result.context_1_score == 5
    assert result.context_2_score == 1
    assert result.tokens == 210
    assert result.context_1_sentence_scores[0].sentence_text == "肺癌治疗要先分型。"
    assert result.context_1_sentence_scores[0].logic_score == 3
    assert result.context_1_sentence_scores[0].worth_optimizing is True
    assert result.context_2_sentence_scores[1].score == 0
    assert result.context_2_sentence_scores[1].reasoning == "该句当前更像原文复述，缺少针对任务目标的显式诊断。"
    assert result.context_2_sentence_scores[1].improvement_suggestion == "补充与目标 slots 直接相关的术语、定义、机制或结论，避免空泛描述。"
    assert result.context_1_optimization_summary.need_optimization_count == 1
    assert result.context_2_optimization_summary.need_optimization_count == 1
    assert result.context_more_to_optimize == "context_2"
    assert Path(html_path).exists()
    html = Path(html_path).read_text(encoding="utf-8")
    assert "展开诊断与优化建议" in html
    assert "context_2 需要优化的句子更多。" in html
