from __future__ import annotations

import json

from ..common.prompt_utils import (
    background_knowledge,
    dedupe_texts,
    final_slot_coverage,
    guardrail_notes,
    meta_payload,
    qa_analysis_digest,
    slot_coverage_digest,
    supplementary_appreciation_digest,
    take,
)
from ..cot_layer.models import CrossValidationResult, DialogueState, DomainCoTRecord, SlotSchema


def build_rag_keyword_prompt(
    *,
    slot_name: str,
    focus_text: str,
    task_reason: str = "",
    max_keyword_blocks: int = 2,
    enable_web_search: bool = False,
) -> str:
    payload = {
        "slot_name": slot_name,
        "focus_text": focus_text,
        "task_reason": task_reason,
    }
    if enable_web_search:
        return (
            "你是国画检索路由规划器。请为当前问题同时判断更适合本地 RAG、联网搜索还是混合检索，并只输出 JSON。\n"
            'JSON 结构固定为 {"mode":"rag|web|hybrid","rag_queries":["短词"],"web_queries":["多词查询"],"reason":"原因"}。\n'
            f"`rag_queries` 每条只能保留 1 到 {max(1, int(max_keyword_blocks))} 个关键词块，用空格分隔；优先输出稳定术语、单实体、短检索词。\n"
            "`web_queries` 允许 2 到 6 个关键词块，可使用作品名、作者名、馆藏、年代、比较对象、题跋/印章释读等多词组合，但不要输出完整自然语言问句。\n"
            "判断原则：技法、构图、设色、图像学描述优先 rag；作者、作品名、馆藏、题跋释文、印章、年代、版本、流传、比较研究优先 web；两者都重要时用 hybrid。\n"
            "若 `mode=rag`，`web_queries` 允许为空；若 `mode=web`，`rag_queries` 允许为空；若 `mode=hybrid`，两者都尽量给出。\n"
            "每个数组最多 5 条，尽量互补。\n\n"
            f"输入信息: {json.dumps(payload, ensure_ascii=False)}\n"
        )
    return (
        "你是国画检索词规划器。请基于当前问题与上下文，尽可能生成能帮助 RAG 挖出更多信息的检索词与搜索短语。\n"
        "这些检索词的目标是为下游更强模型提供更多稀缺领域材料。请只输出 JSON。\n"
        'JSON 结构固定为 {"queries": ["检索词1", "检索词2"]}。\n'
        f"每条 query 只能保留 1 到 {max(1, int(max_keyword_blocks))} 个关键词块，用空格分隔；若无必要，优先输出单术语或单实体。\n"
        "不要输出完整问题、解释句、审美判断句、自然语言短句，也不要输出 3 个及以上关键词块的长串。\n"
        "优先补充：稳定术语、对象、技法、风格、时代、材料、构图、符号、图像细节、可关联的知识点。\n"
        "输出 3 到 5 条，尽量互补。\n\n"
        f"输入信息: {json.dumps(payload, ensure_ascii=False)}\n"
    )


def build_batch_rag_keyword_prompt(
    *,
    requests: list[dict[str, object]],
    max_keyword_blocks: int = 2,
    enable_web_search: bool = False,
) -> str:
    payload = []
    for item in requests:
        payload.append(
            {
                "request_id": str(item.get("request_id", "")).strip(),
                "slot_name": str(item.get("slot_name", "")).strip(),
                "focus_text": str(item.get("focus_text", "")).strip(),
                "task_reason": str(item.get("task_reason", "")).strip(),
            }
        )
    if enable_web_search:
        return (
            "你是国画检索路由批量规划器。请为每个请求分别判断更适合本地 RAG、联网搜索还是混合检索，并只输出 JSON。\n"
            'JSON 结构固定为 {"items":[{"request_id":"id","mode":"rag|web|hybrid","rag_queries":["短词"],"web_queries":["多词查询"],"reason":"原因"}]}。\n'
            f"`rag_queries` 每条只能保留 1 到 {max(1, int(max_keyword_blocks))} 个关键词块，用空格分隔；优先稳定术语、单实体、短检索词。\n"
            "`web_queries` 允许 2 到 6 个关键词块，可使用作品名、作者名、馆藏、年代、比较对象、题跋/印章释读等多词组合，但不要输出完整自然语言问句。\n"
            "判断原则：技法、构图、设色、图像学描述优先 rag；作者、作品名、馆藏、题跋释文、印章、年代、版本、流传、比较研究优先 web；两者都重要时用 hybrid。\n"
            "每个 request 的 `rag_queries` 和 `web_queries` 最多各 5 条，尽量互补。\n\n"
            f"输入信息: {json.dumps(payload, ensure_ascii=False)}\n"
        )
    return (
        "你是国画检索词批量规划器。请为每个请求分别生成适合搜索引擎的短 query，只输出 JSON。\n"
        'JSON 结构固定为 {"items":[{"request_id":"id","queries":["检索词1","检索词2"]}] }。\n'
        f"每条 query 只能保留 1 到 {max(1, int(max_keyword_blocks))} 个关键词块，用空格分隔；若无必要，优先输出单术语或单实体。\n"
        "不要输出完整问题、解释句、审美判断句、自然语言短句，也不要输出 2 个以上关键词块的长串。\n"
        "优先补充：稳定术语、对象、技法、风格、时代、材料、构图、符号、图像细节、可关联知识点。\n"
        "每个 request 输出 3 到 5 条，尽量互补。\n\n"
        f"输入信息: {json.dumps(payload, ensure_ascii=False)}\n"
    )


def build_slot_lifecycle_prompt(
    *,
    slot_schemas: list[SlotSchema],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
) -> str:
    slot_payload = [
        {
            "slot_name": slot.slot_name,
            "slot_term": slot.slot_term,
            "lifecycle": str(slot.metadata.get("lifecycle", "ACTIVE")).strip().upper() or "ACTIVE",
            "specific_questions": slot.specific_questions,
        }
        for slot in slot_schemas
    ]
    output_payload = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "question_coverage": [
                {
                    "question": item.question,
                    "answered": item.answered,
                    "support": item.support,
                }
                for item in output.question_coverage
            ],
            "visual_anchoring": [item.observation for item in output.visual_anchoring[:4]],
            "domain_decoding": [item.term for item in output.domain_decoding[:4]],
            "cultural_mapping": [item.insight for item in output.cultural_mapping[:3]],
            "unresolved_points": output.unresolved_points[:4],
        }
        for output in outputs
    ]
    review_payload = {
        "issues": [
            {
                "severity": issue.severity,
                "slots": issue.slot_names,
                "detail": issue.detail,
            }
            for issue in validation.issues
        ],
        "follow_up_questions": validation.round_table_follow_up_questions,
        "rag_needs": validation.round_table_rag_needs,
        "blind_spots": validation.round_table_blind_spots,
    }
    return (
        "你是 slot lifecycle 决策节点。你的任务不是写赏析，而是判断每个槽位后续是否还值得继续投入 CoT 或 RAG。\n"
        "请严格基于已有 answers、visual anchoring、domain decoding、round-table follow-up 和已有知识，输出 JSON。\n"
        "判断原则：\n"
        "1. ACTIVE：该 slot 仍有高价值专业问题未解决，继续分析可能带来新的图像学、术语或文献知识。\n"
        "2. STABLE：该 slot 的核心问题基本已覆盖，目前没有明显高价值新增方向，可暂时保持稳定。\n"
        "3. CLOSED：继续追问大多只会落入赏析、意义延展、修辞性解释，或现有信息已经足以回答，进一步调用专业 RAG 的价值很低。\n"
        "4. follow-up 若指向不存在的 slot、或本质上是在尝试发现新的分析维度，应标记为 downstream_discovery，而不是 cot。\n"
        "5. follow-up 若现有信息已经足以回答，或继续追问不会产生新的专业术语/图像学知识/可检索增益，应标记为 close。\n"
        "6. 只有当 follow-up 明确指向已有 slot 且仍可能产生新的专业知识时，才标记为 cot。\n"
        'JSON 结构固定为：{"slot_reviews":[{"slot_name":"槽位名","status":"ACTIVE|STABLE|CLOSED","reason":"原因"}],"follow_up_reviews":[{"slot_name":"原始槽位名","question":"问题","action":"cot|downstream_discovery|close","reason":"原因"}]}\n'
        f"context_meta: {json.dumps(meta_payload(meta), ensure_ascii=False)}\n"
        f"slot_schemas: {json.dumps(slot_payload, ensure_ascii=False)}\n"
        f"slot_outputs: {json.dumps(output_payload, ensure_ascii=False)}\n"
        f"round_table_review: {json.dumps(review_payload, ensure_ascii=False)}\n"
    )


def build_validation_review_prompt(
    *,
    slot_schemas: list[SlotSchema],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    enable_web_search: bool = False,
) -> str:
    output_payload = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "visual_anchoring": [item.observation for item in output.visual_anchoring[:4]],
            "domain_decoding": [item.term for item in output.domain_decoding[:4]],
            "cultural_mapping": [item.insight for item in output.cultural_mapping[:3]],
            "question_coverage": [
                {
                    "question": item.question,
                    "answered": item.answered,
                    "support": item.support,
                }
                for item in output.question_coverage[:6]
            ],
            "unresolved_points": output.unresolved_points[:4],
            "statuses": output.statuses[:3],
        }
        for output in outputs
    ]
    issues_payload = [
        {
            "type": issue.issue_type,
            "severity": issue.severity,
            "slots": issue.slot_names,
            "detail": issue.detail,
        }
        for issue in validation.issues[:12]
    ]
    slot_payload = [
        {
            "slot_name": slot.slot_name,
            "slot_term": slot.slot_term,
            "lifecycle": str(slot.metadata.get("lifecycle", "ACTIVE")).strip().upper() or "ACTIVE",
            "specific_questions": slot.specific_questions,
        }
        for slot in slot_schemas
    ]
    web_instruction = ""
    follow_up_schema = (
        '    {"slot_name": "槽位名", "question": "需要继续追问的问题", "why": "为什么值得追问", '
        '"priority": "high|medium", "rag_queries": ["检索词"]}\n'
    )
    if enable_web_search:
        web_instruction = (
            "若判断需要联网搜索，请在 follow_up_questions 中同时输出 retrieval_mode 和 web_queries。`web_queries` 可以是多词组合，例如作品名+作者名+馆藏、"
            "题跋释文+关键词、作者比较等；技法、构图、设色、图像学描述优先 rag，作者、作品名、馆藏、题跋、印章、年代、版本、流传、比较研究优先 web，两者都重要时用 hybrid。\n"
        )
        follow_up_schema = (
            '    {"slot_name": "槽位名", "question": "需要继续追问的问题", "why": "为什么值得追问", '
            '"priority": "high|medium", "retrieval_mode": "rag|web|hybrid", "rag_queries": ["短词"], '
            '"web_queries": ["多词查询"], "retrieval_reason": "为什么这么分"}\n'
        )
    return (
        "你是国画分析的综合复核与路由裁决节点。请在一次审查中完成两件事：\n"
        "1. 找出各槽位回答里仍然缺失的重要细节、值得继续追问的问题，以及还需要补检索的主题。\n"
        "2. 判断每个槽位的后续状态，以及每个追问更适合继续 CoT、转 downstream discovery 还是直接关闭。\n\n"
        "细节复核要求：\n"
        "1. 不要过度关注赏析、修辞或美学感受，重点关注对象身份、图像细节、图像与术语的对应、时代/宗教/人物属性、构图元素、材质技法、图像证据缺口。\n"
        "2. 如果现有回答只完成了对象命名、尊者编号、题材归类或术语贴标签，但没有展开可见物象、识别依据、属性组合和相关图像信息，应视为未充分回答，而不是已完成回答。\n"
        "3. 如果 context_meta 里已经提供了 post_rag_text_extraction 或 rag_cache，请先判断这些已有文献证据是否足够，只有在 cache 明显不足时，才提出新的 rag_queries 或 rag_needs。\n"
        "4. rag_queries 应优先输出稳定术语或单一实体名，不要输出自然语言短句。\n"
        f"{web_instruction}"
        "5. follow_up_questions 中的 slot_name 必须严格使用已有输出里出现过的 slot_name，不要自造新的槽位名。\n\n"
        "生命周期裁决要求：\n"
        "6. ACTIVE：该 slot 仍有高价值专业问题未解决，继续分析可能带来新的图像学、术语或文献知识。\n"
        "7. STABLE：该 slot 的核心问题基本已覆盖，目前没有明显高价值新增方向，可暂时保持稳定。\n"
        "8. CLOSED：继续追问大多只会落入赏析、意义延展、修辞性解释，或现有信息已经足以回答，进一步调用专业 RAG 的价值很低。\n"
        "9. follow_up 若指向不存在的 slot、或本质上是在尝试发现新的分析维度，应标记为 downstream_discovery，而不是 cot。\n"
        "10. follow_up 若现有信息已经足以回答，或继续追问不会产生新的专业术语、图像学知识或可检索增益，应标记为 close。\n"
        "11. 只有当 follow-up 明确指向已有 slot 且仍可能产生新的专业知识时，才标记为 cot。\n\n"
        "请只输出 JSON，不要输出解释性散文。\n"
        "JSON 结构固定为：\n"
        "{\n"
        '  "review_summary": "一句话总结当前回答最大的细节缺口",\n'
        '  "blind_spots": ["被忽略的细节或识别盲点"],\n'
        '  "follow_up_questions": [\n'
        f"{follow_up_schema}"
        "  ],\n"
        '  "rag_needs": [\n'
        '    {"topic": "需要补检索的信息主题", "reason": "为什么要查", "queries": ["检索词"]}\n'
        "  ],\n"
        '  "slot_reviews": [\n'
        '    {"slot_name":"槽位名","status":"ACTIVE|STABLE|CLOSED","reason":"原因"}\n'
        "  ],\n"
        '  "follow_up_reviews": [\n'
        '    {"slot_name":"原始槽位名","question":"问题","action":"cot|downstream_discovery|close","reason":"原因"}\n'
        "  ]\n"
        "}\n\n"
        f"context_meta: {json.dumps(meta_payload(meta), ensure_ascii=False)}\n"
        f"slot_schemas: {json.dumps(slot_payload, ensure_ascii=False)}\n"
        f"已有输出: {json.dumps(output_payload, ensure_ascii=False)}\n"
        f"本地校验: {json.dumps(issues_payload, ensure_ascii=False)}\n"
    )


def build_final_appreciation_prompt(
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    dialogue_state: DialogueState,
) -> str:
    qa_analysis = qa_analysis_digest(outputs)
    supplementary = supplementary_appreciation_digest(outputs)
    slot_coverage = slot_coverage_digest(outputs, meta)
    background = background_knowledge(meta)
    guardrails = guardrail_notes(meta, validation)
    paragraphs = [item for item in qa_analysis.split("\n\n") if item.strip()]
    paragraphs.extend(item for item in supplementary.split("\n\n") if item.strip())
    paragraphs.extend(item for item in slot_coverage.split("\n\n") if item.strip())
    if background:
        paragraphs.append("结合已有文献与背景线索，" + "；".join(background[:10]) + "。")
    if dialogue_state.unresolved_questions or guardrails:
        caution_items = dedupe_texts(
            list(dialogue_state.unresolved_questions[:6])
            + list(guardrails[:4])
        )
        if caution_items:
            paragraphs.append("需要保守说明的是，" + "；".join(caution_items[:6]) + "。")
    appreciation = "\n\n".join(dedupe_texts([item.strip() for item in paragraphs if item.strip()])).strip()
    return "## 赏析\n" + (appreciation or "当前尚未形成可整合的赏析内容。")


def build_final_answer_request_prompt(
    *,
    question: str,
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    dialogue_state: DialogueState,
) -> str:
    qa_analysis = qa_analysis_digest(outputs).strip()
    supplementary = supplementary_appreciation_digest(outputs).strip()
    slot_coverage = final_slot_coverage(outputs, meta)
    background = background_knowledge(meta)
    guardrails = guardrail_notes(meta, validation)
    resolved = take(dedupe_texts(list(dialogue_state.resolved_questions)), 10)
    unresolved = take(dedupe_texts(list(dialogue_state.unresolved_questions)), 10)
    compact_outputs = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "visual_anchoring": [item.observation for item in output.visual_anchoring[:6]],
            "domain_decoding": [
                (
                    f"{item.term}：{item.explanation}"
                    if item.term and item.explanation
                    else item.term or item.explanation
                )
                for item in output.domain_decoding[:6]
            ],
            "cultural_mapping": [item.insight for item in output.cultural_mapping[:5]],
            "answered_support": [
                str(item.support).strip()
                for item in output.question_coverage
                if item.answered and str(item.support).strip()
            ][:6],
            "unresolved_points": output.unresolved_points[:6],
        }
        for output in outputs
    ]
    payload = {
        "question": question,
        "qa_analysis": qa_analysis,
        "supplementary_analysis": supplementary,
        "required_slot_coverage": slot_coverage,
        "slot_summaries": meta.get("final_slot_summaries", []),
        "background_knowledge": background,
        "resolved_questions": resolved,
        "unresolved_questions": unresolved,
        "guardrail_notes": guardrails,
        "slots": compact_outputs,
        "context_meta": meta_payload(meta),
    }
    return (
        "请基于以下国画分析材料，直接回答用户问题。\n"
        "要求：\n"
        "1. 只围绕用户问题作答，不要复述流程。\n"
        "2. 优先使用已经确认的图像证据、术语解码和背景信息。\n"
        "3. 若某些判断证据不足，要在行文中明确保守表达，不得编造。\n"
        "4. 输出 Markdown，只保留一个标题 `## 赏析`，下面写 5 到 7 段完整赏析文字。\n"
        "5. 不要只围绕单一 slot 或最后一轮新增 slot，要尽量综合多个槽位的已确认信息。\n"
        "6. `required_slot_coverage` 中 `must_cover=true` 的槽位不得省略；每个此类槽位至少吸收一条 `must_include_facts`，若只能保守表述，也要明确写出，而不是直接跳过。\n"
        "7. 若作者时代流派槽位存在作者、时代、地域或流派事实，必须明确写出作者名、时代，并至少吸收一条地域/流派背景。\n"
        "8. 若尺寸规格/材质形制/收藏地槽位存在馆藏、材质、装裱或尺寸线索，必须明确写出已确认部分，并把未确认项用保守句式说明。\n"
        "9. 若题跋诗文或题跋/印章/用笔槽位的稳定结论是“未见题跋/印章”或“仅能确认用笔线质”，也应在文中交代，不要装作不存在。\n"
        "10. 若材料足够，请尽量覆盖：作品身份与时代、视觉主体与构图、笔墨/设色/材质、题跋印章与流传、题材寓意与审美价值。\n"
        "11. 引用背景知识时必须回扣画面证据，不要把文献知识堆成百科介绍。\n"
        "12. 结尾请用 1 段总结整件作品的艺术价值；如仍有关键未解点，可用保守句式顺带说明。\n\n"
        f"输入材料: {json.dumps(payload, ensure_ascii=False)}\n"
    )
