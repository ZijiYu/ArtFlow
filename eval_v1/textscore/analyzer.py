from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import List, Sequence

from openai import OpenAI

from .models import (
    ContextOptimizationSummary,
    SentenceScore,
    SentenceScoresLLMOutput,
    SlotAnalysis,
    SlotsLLMOutput,
    TextScoreResult,
)
from .render import render_comparison_html
from .sentence_indexer import build_indexed_text


SLOTS_SYSTEM_PROMPT = """你是一个严格的垂直领域 slot 命中分析器。
你的任务只负责判断输入文本中哪些句子与给定 slots 明确匹配，并返回结构化命中结果。

必须遵守：
1. 绝对只依据输入的 context 与 slots 进行分析，禁止引入外部知识。
2. 只输出 `slots_analysis`，不要输出任何句子级优化字段。
3. 只有句子与某个 slot 明确 match 时才允许输出条目；未命中就不要输出。
4. `slot_name` 必须严格取自输入的 slots，禁止输出空字符串、近义改写或未提供的新 slot。
5. `score` 只有在明确 match 时才允许为 1-5；未 match 时不能给分。
6. `slot_term` 不能为空；如果只匹配到 slot 本身而没有更细术语，就把 `slot_term` 填成 `slot_name`。
7. 输出必须是合法 JSON。
"""


OPTIMIZATION_SYSTEM_PROMPT = """你是一个严格的垂直领域文本 TextLoss 优化判定器。
你的任务只负责单文本句子级优化分析，不负责 slots 命中判断。

必须遵守：
1. 只依据输入的 context 与 slots 进行分析，禁止引入外部知识。
2. 只输出 `sentence_scores`，不要输出任何 `slots_analysis` 字段。
3. 必须覆盖 context 中的每一个句子，不能遗漏。
4. 所有数值评分必须是 [0, 5] 之间的整数。
5. `logic_score` 评价逻辑是否清楚、因果是否连贯、表达是否自洽。
6. `slot_relevance_score` 评价句子与 slots 的直接相关性。
7. `redundancy_score` 评价句子是否与上下文其他句子重复。分数越高越冗余。
8. `loss` 评价句子的缺陷程度；`score` 评价句子的正向贡献。
9. `worth_optimizing=true` 表示该句值得优先优化。判断时综合 loss、逻辑、slot 相关性、冗余和信息价值。
10. 对过渡句、结构句、低价值重复句，可以设为 `worth_optimizing=false`。
11. `reasoning` 必须是诊断，不允许简单复述原句。
12. `improvement_suggestion` 必须给出具体优化建议。
13. 输出必须是合法 JSON。
"""


def _build_slots_prompt(
    context_name: str,
    context: Sequence[dict],
    slots: Sequence[str],
    slots_number: int,
) -> str:
    slot_text = "\n".join(f"- {slot}" for slot in slots)
    context_text = "\n".join(f"[{item['sentence_id']}] {item['text']}" for item in context)
    return f"""请只做 slot 命中判断，不要做优化评分。

context_name: {context_name}
slots_number: {slots_number}
slots:
{slot_text}

context:
{context_text}

请输出 JSON，字段为：
{{
  "slots_analysis": [{{"sentence_ids":[0], "slot_name":"slot", "slot_term":"term", "score":1}}]
}}

要求：
- 每个 slot 至少尝试判断当前 context 是否覆盖。
- `slots_analysis` 只有在句子与某个 slot 明确匹配时才允许输出条目；未命中就不要输出该条。
- `slot_name` 必须严格取自输入的 slots，禁止输出空字符串、近义改写或未提供的新 slot。
- `score` 只有在明确 match 时才允许为 1-5；未 match 时不能给分，且不要创建条目。
- `slot_term` 不能为空；如果只匹配到 slot 本身而没有更细术语，就把 `slot_term` 填成 `slot_name`，不要留空。
"""


def _build_optimization_prompt(
    context_name: str,
    context: Sequence[dict],
    slots: Sequence[str],
    slots_number: int,
) -> str:
    slot_text = "\n".join(f"- {slot}" for slot in slots)
    context_text = "\n".join(f"[{item['sentence_id']}] {item['text']}" for item in context)
    return f"""请只做句子级优化判定，不要做 slot 命中输出。

context_name: {context_name}
slots_number: {slots_number}
slots:
{slot_text}

context:
{context_text}

请输出 JSON，字段为：
{{
  "sentence_scores": [
    {{
      "sentence_id":0,
      "score":1,
      "loss":1,
      "logic_score":1,
      "slot_relevance_score":1,
      "redundancy_score":0,
      "worth_optimizing":true,
      "reasoning":"",
      "improvement_suggestion":"",
      "matched_slots":[""],
      "matched_terms":[""]
    }}
  ]
}}

要求：
- `sentence_scores` 必须覆盖 context 中的每一个句子。
- `matched_slots` 填该句命中的 slots 名称；未命中可为空数组。
- `matched_terms` 填该句命中的更细粒度术语；未命中可为空数组。
- `reasoning` 必须是诊断，不允许简单复述原句。
- `improvement_suggestion` 必须给出具体优化建议，说明应该补什么信息、改到什么程度更好。
- 如果句子更像重复、过渡句、低价值泛化描述，可以把 `redundancy_score` 调高，并将 `worth_optimizing` 设为 false。
- 如果句子内容有价值但表达弱、逻辑弱、slot 相关内容不足，应该将 `worth_optimizing` 设为 true。
"""


def _sentence_lookup(indexed_sentences: Sequence[dict]) -> dict[int, str]:
    return {item["sentence_id"]: item["text"] for item in indexed_sentences}


def _normalize_text(text: str) -> str:
    return re.sub(r"[\s，。！？；、,.!?;:：'\"“”‘’（）()\\-]", "", text).strip().lower()


def _is_trivial_copy(sentence_text: str, generated_text: str) -> bool:
    normalized_sentence = _normalize_text(sentence_text)
    normalized_generated = _normalize_text(generated_text)
    if not normalized_generated:
        return True
    return (
        normalized_generated == normalized_sentence
        or normalized_generated in normalized_sentence
        or normalized_sentence in normalized_generated
    )


def _hydrate_slot_analyses(
    items: List[SlotAnalysis],
    lookup: dict[int, str],
    allowed_slots: Sequence[str],
) -> List[SlotAnalysis]:
    hydrated: List[SlotAnalysis] = []
    allowed = {slot.strip() for slot in allowed_slots if slot.strip()}
    for item in items:
        slot_name = item.slot_name.strip()
        if not slot_name or slot_name not in allowed:
            continue
        slot_term = item.slot_term.strip() or slot_name
        sentences = [lookup[sentence_id] for sentence_id in item.sentence_ids if sentence_id in lookup]
        if not sentences:
            continue
        hydrated.append(item.model_copy(update={"slot_name": slot_name, "sentences": sentences, "slot_term": slot_term}))
    return hydrated


def _hydrate_sentence_scores(items: List[SentenceScore], lookup: dict[int, str]) -> List[SentenceScore]:
    hydrated: List[SentenceScore] = []
    for item in items:
        sentence_text = lookup.get(item.sentence_id, "")
        reasoning = item.reasoning.strip()
        improvement_suggestion = item.improvement_suggestion.strip()
        if _is_trivial_copy(sentence_text, reasoning):
            reasoning = "该句当前更像原文复述，缺少针对任务目标的显式诊断。"
        if _is_trivial_copy(sentence_text, improvement_suggestion):
            if item.matched_slots:
                slots_text = "、".join(item.matched_slots)
                improvement_suggestion = f"围绕 {slots_text} 补充更具体的术语、关系或结论，减少泛化表述。"
            else:
                improvement_suggestion = "补充与目标 slots 直接相关的术语、定义、机制或结论，避免空泛描述。"
        hydrated.append(
            item.model_copy(
                update={
                    "sentence_text": sentence_text,
                    "reasoning": reasoning,
                    "improvement_suggestion": improvement_suggestion,
                }
            )
        )
    return hydrated


def _build_optimization_summary(sentence_scores: List[SentenceScore]) -> ContextOptimizationSummary:
    worth_items = [item for item in sentence_scores if item.worth_optimizing]
    return ContextOptimizationSummary(
        need_optimization_count=len(worth_items),
        worth_optimizing_sentence_ids=[item.sentence_id for item in worth_items],
        total_loss=sum(item.loss for item in sentence_scores),
    )


def _pick_context_more_to_optimize(
    context_1_summary: ContextOptimizationSummary,
    context_2_summary: ContextOptimizationSummary,
) -> str:
    if context_1_summary.need_optimization_count > context_2_summary.need_optimization_count:
        return "context_1"
    if context_2_summary.need_optimization_count > context_1_summary.need_optimization_count:
        return "context_2"
    if context_1_summary.total_loss > context_2_summary.total_loss:
        return "context_1"
    if context_2_summary.total_loss > context_1_summary.total_loss:
        return "context_2"
    return "tie"


class TextScoreAnalyzer:
    def __init__(
        self,
        model: str = "openai/gpt-4.1",
        base_url: str = "https://api.zjuqx.cn/v1",
        client: OpenAI | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.client = client

    def score(
        self,
        context_1: str,
        context_2: str,
        slots: List[str],
        slots_number: int,
        visualization_path: str | os.PathLike[str] = "artifacts/textscore_comparison.html",
    ) -> tuple[TextScoreResult, str]:
        if slots_number != len(slots):
            raise ValueError(f"slots_number({slots_number}) 必须等于 slots 数量({len(slots)})")

        context_1_indexed = build_indexed_text(context_1)
        context_2_indexed = build_indexed_text(context_2)
        if not context_1_indexed or not context_2_indexed:
            raise ValueError("两个 context 都必须包含至少一句有效文本")

        client = self.client or OpenAI(base_url=self.base_url)
        response_1_slots = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SLOTS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_slots_prompt("context_1", context_1_indexed, slots, slots_number),
                },
            ],
        )
        response_1_optimization = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": OPTIMIZATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_optimization_prompt("context_1", context_1_indexed, slots, slots_number),
                },
            ],
        )
        response_2_slots = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SLOTS_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_slots_prompt("context_2", context_2_indexed, slots, slots_number),
                },
            ],
        )
        response_2_optimization = client.chat.completions.create(
            model=self.model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": OPTIMIZATION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_optimization_prompt("context_2", context_2_indexed, slots, slots_number),
                },
            ],
        )

        parsed_context_1_slots = SlotsLLMOutput.model_validate(
            json.loads(response_1_slots.choices[0].message.content or "{}")
        )
        parsed_context_1_optimization = SentenceScoresLLMOutput.model_validate(
            json.loads(response_1_optimization.choices[0].message.content or "{}")
        )
        parsed_context_2_slots = SlotsLLMOutput.model_validate(
            json.loads(response_2_slots.choices[0].message.content or "{}")
        )
        parsed_context_2_optimization = SentenceScoresLLMOutput.model_validate(
            json.loads(response_2_optimization.choices[0].message.content or "{}")
        )

        context_1_lookup = _sentence_lookup(context_1_indexed)
        context_2_lookup = _sentence_lookup(context_2_indexed)
        context_1_slots_analysis = _hydrate_slot_analyses(parsed_context_1_slots.slots_analysis, context_1_lookup, slots)
        context_2_slots_analysis = _hydrate_slot_analyses(parsed_context_2_slots.slots_analysis, context_2_lookup, slots)
        context_1_sentence_scores = _hydrate_sentence_scores(parsed_context_1_optimization.sentence_scores, context_1_lookup)
        context_2_sentence_scores = _hydrate_sentence_scores(parsed_context_2_optimization.sentence_scores, context_2_lookup)

        context_1_slots_score = sum(item.score for item in context_1_slots_analysis)
        context_2_slots_score = sum(item.score for item in context_2_slots_analysis)
        context_1_optimization_summary = _build_optimization_summary(context_1_sentence_scores)
        context_2_optimization_summary = _build_optimization_summary(context_2_sentence_scores)

        result = TextScoreResult(
            context_1_score=context_1_slots_score,
            context_2_score=context_2_slots_score,
            context_1_slots_score=context_1_slots_score,
            context_2_slots_score=context_2_slots_score,
            context_1_slots_analysis=context_1_slots_analysis,
            context_2_slots_analysis=context_2_slots_analysis,
            context_1_sentence_scores=context_1_sentence_scores,
            context_2_sentence_scores=context_2_sentence_scores,
            context_1_optimization_summary=context_1_optimization_summary,
            context_2_optimization_summary=context_2_optimization_summary,
            context_more_to_optimize=_pick_context_more_to_optimize(
                context_1_optimization_summary,
                context_2_optimization_summary,
            ),
            tokens=sum(
                item.usage.total_tokens
                for item in (response_1_slots, response_1_optimization, response_2_slots, response_2_optimization)
                if item.usage
            ),
        )

        path = Path(visualization_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        html_path = render_comparison_html(
            context_1_sentence_scores=context_1_sentence_scores,
            context_2_sentence_scores=context_2_sentence_scores,
            output_path=str(path),
            context_more_to_optimize=result.context_more_to_optimize,
        )
        return result, html_path
