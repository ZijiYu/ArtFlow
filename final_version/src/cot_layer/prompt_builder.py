from __future__ import annotations

import json

from ..common.prompt_utils import meta_payload
from .models import SlotSchema
from .schema_loader import extract_slot_terms


def build_domain_cot_prompt(
    slot: SlotSchema,
    meta: dict,
    focus_question: str | None = None,
    analysis_round: int = 1,
    thread_context: dict | None = None,
    specific_questions: list[str] | None = None,
    retrieval_gain_enabled: bool = True,
    web_search_enabled: bool = False,
) -> str:
    focus_text = focus_question.strip() if focus_question else "无"
    raw_thread_payload = thread_context or {}
    scoped_questions = (specific_questions if specific_questions is not None else slot.specific_questions)[:3]
    slot_description = str(slot.description or "").strip()
    if len(slot_description) > 360:
        slot_description = slot_description[:357].rstrip() + "..."
    thread_payload = {
        "thread_id": str(raw_thread_payload.get("thread_id", "")).strip(),
        "reason": str(raw_thread_payload.get("reason", "")).strip(),
        "priority": raw_thread_payload.get("priority"),
        "attempts": raw_thread_payload.get("attempts"),
        "rag_terms": list(raw_thread_payload.get("rag_terms", []))[:6] if isinstance(raw_thread_payload.get("rag_terms"), list) else [],
        "latest_summary": str(raw_thread_payload.get("latest_summary", "")).strip()[:180],
        "prompt_questions": list(raw_thread_payload.get("prompt_questions", []))[:3] if isinstance(raw_thread_payload.get("prompt_questions"), list) else [],
        "image_mode": str(raw_thread_payload.get("image_mode", "")).strip(),
    }
    thread_payload = {key: value for key, value in thread_payload.items() if value not in ("", [], None)}
    slot_terms = extract_slot_terms(slot.slot_term, slot.metadata)
    controlled_vocabulary = slot.controlled_vocabulary[:6]
    retrieval_gain_instruction = ""
    retrieval_gain_schema = ""
    if retrieval_gain_enabled:
        retrieval_gain_web_schema = ""
        retrieval_gain_instruction = (
            "7. 在回答已有问题之后，请继续学习当前 slot description、memory card 和已有 RAG 描述中的专业信息，"
            "提炼出与当前图像仍可对齐的新术语、新对象名或新的检索方向。\n"
            "8. 只有当这些新信息与当前图像的视觉锚点、物象细节或已回答内容存在直接对应关系时，才认为它们有继续检索的价值；"
            "如果只是赏析延展、意义解释或没有新的专业术语，就不要继续扩展。\n"
            "9. 如果当前证据不足，就如实保留空缺，不要扩展出新的追问。\n\n"
        )
        if web_search_enabled:
            retrieval_gain_instruction += (
                "10. 若下一步更适合联网搜索，请输出 retrieval_mode=web 或 hybrid，并把多词 web_queries 与短词 rag_queries 区分开；"
                "技法、构图、设色、图像学描述优先 rag，作者、作品名、馆藏、题跋、印章、年代、版本、流传、比较研究优先 web。\n\n"
            )
            retrieval_gain_web_schema = (
                '    "retrieval_mode": "rag|web|hybrid",\n'
                '    "web_queries": ["适合联网搜索的多词 query"],\n'
            )
        retrieval_gain_schema = (
            '  "retrieval_gain": {\n'
            '    "has_new_value": false,\n'
            '    "focus": "回答当前问题后，下一步仍值得继续核验的图像相关焦点；如果没有则为空",\n'
            '    "related_terms": ["从当前 RAG 描述中学到、且和图像相关的新术语"],\n'
            '    "search_queries": ["适合继续检索的短 query"],\n'
            f"{retrieval_gain_web_schema}"
            '    "reason": "为什么这些新信息与图像相关、并仍值得继续检索"\n'
            "  },\n"
        )
    return (
        "你是 dynamic agent pipeline 的领域 CoT 信息扩展引擎。"
        "请围绕当前槽位，逐步回答已有问题，补充图像信息、术语解码和文化背景，并只输出 JSON。\n\n"
        "本轮目标：\n"
        "1. 充分吸收 slot schema、上下文、上一轮 memory card 和已有 RAG 痕迹，直接回答已有问题。\n"
        "2. 尽量把视觉线索转成支撑答案的结构化材料。\n"
        "3. 只围绕当前问题和 slot 原有的 specific_questions 作答，不要主动提出新问题，也不要改写、优化问题。\n"
        "4. 优先继承上一轮已确认的信息，并在此基础上补充本轮新增证据，实现滚雪球式累积。\n"
        "5. 如果识别出人物、尊像、尊者、器物、建筑、植物、动物或其他关键对象，不要停留在命名或编号，"
        "还要继续说明其可见属性、持物、姿态、服饰、座具、随侍、环境线索或组合关系，以及这些线索如何支撑识别。\n"
        "6. 对象分析要区分：哪些细节具有身份识别作用，哪些只是一般性的题材或风格特征；"
        "若只能做出初步识别，也要明确指出仍缺少哪类图像信息。\n"
        f"{retrieval_gain_instruction}"
        "如果当前证据不足，就如实保留空缺，不要扩展出新的追问。\n\n"
        f"当前槽位: {slot.slot_name}\n"
        f"当前槽位术语: {slot.slot_term}\n"
        f"当前槽位术语组: {json.dumps(slot_terms, ensure_ascii=False)}\n"
        f"补充聚焦问题: {focus_text}\n"
        f"分析轮次: {analysis_round}\n"
        f"受控词表: {json.dumps(controlled_vocabulary, ensure_ascii=False)}\n"
        f"槽位描述: {slot_description}\n"
        f"specific_questions: {json.dumps(scoped_questions, ensure_ascii=False)}\n"
        f"thread_context: {json.dumps(thread_payload, ensure_ascii=False)}\n"
        f"context_meta: {json.dumps(meta_payload(meta), ensure_ascii=False)}\n\n"
        "JSON 输出结构：\n"
        "{\n"
        '  "slot_name": "string",\n'
        '  "controlled_vocabulary_used": ["term"],\n'
        '  "visual_anchoring": [\n'
        '    {"observation": "客观视觉现象或物象细节", "evidence": "画面证据", "position": "位置"}\n'
        "  ],\n"
        '  "domain_decoding": [\n'
        '    {"term": "术语", "explanation": "术语如何对应图像，以及哪些物象支撑该判断", "status": "IDENTIFIED|UNIDENTIFIABLE_FEATURE", "reason": "原因"}\n'
        "  ],\n"
        '  "cultural_mapping": [\n'
        '    {"insight": "通俗但严谨的文化映射", "basis": "视觉依据", "risk_note": "若不确定则说明"}\n'
        "  ],\n"
        '  "specific_question_coverage": [\n'
        '    {"question": "问题", "answered": true, "support": "回答时应包含图像依据，以及必要的物象与识别信息"}\n'
        "  ],\n"
        f"{retrieval_gain_schema}"
        '  "generated_questions": [],\n'
        '  "unresolved_points": ["仍不确定的点"],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )
