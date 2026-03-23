from __future__ import annotations

import json

from .models import CrossValidationResult, DialogueState, DomainCoTRecord, SlotSchema


def _dedupe_texts(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        text = " ".join(str(item or "").strip().split())
        if not text:
            continue
        key = text.casefold()
        if key in seen:
            continue
        seen.add(key)
        deduped.append(text)
    return deduped


def _take(items: list[str], limit: int) -> list[str]:
    return items[: max(0, limit)]


def _format_bullets(items: list[str], *, empty_text: str = "- 无") -> str:
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def _profile_snapshot(meta: dict) -> list[str]:
    profile = meta.get("domain_profile") or meta.get("system_metadata") or {}
    lines: list[str] = []
    name = str(profile.get("name", "")).strip()
    category = str(profile.get("category", "")).strip()
    subject = str(profile.get("subject", "")).strip()
    scene_summary = str(profile.get("scene_summary", "")).strip()
    if name:
        lines.append(f"作品暂定信息：{name}")
    if category:
        lines.append(f"画种或类别：{category}")
    if subject:
        lines.append(f"主题对象：{subject}")
    if scene_summary:
        lines.append(f"画面概况：{scene_summary}")
    return lines


def _slot_section(
    output: DomainCoTRecord,
    *,
    include_answered: bool = False,
    include_unresolved: bool = False,
) -> str:
    visual = _take(
        _dedupe_texts(
            [
                f"{item.observation}{f'（{item.position}）' if item.position else ''}"
                for item in output.visual_anchoring
            ]
        ),
        4,
    )
    decoding = _take(
        _dedupe_texts(
            [
                (
                    f"{item.term}：{item.explanation}"
                    if item.term and item.explanation
                    else item.term or item.explanation
                )
                for item in output.domain_decoding
            ]
        ),
        4,
    )
    mapping = _take(_dedupe_texts([item.insight for item in output.cultural_mapping]), 3)
    answered = _take(_dedupe_texts([item.question for item in output.question_coverage if item.answered]), 3)
    unresolved = _take(_dedupe_texts(output.unresolved_points), 2)
    lines = [
        f"### {output.slot_name}",
        f"- 术语锚点：{output.slot_term or output.slot_name}",
        "- 视觉证据：",
        _format_bullets(visual, empty_text="- 暂无稳定视觉证据"),
        "- 专业解码：",
        _format_bullets(decoding, empty_text="- 暂无稳定术语解码"),
        "- 文化映射：",
        _format_bullets(mapping, empty_text="- 暂无稳定文化映射"),
    ]
    if include_answered and answered:
        lines.extend(
            [
                "- 已覆盖问题：",
                _format_bullets(answered),
            ]
        )
    if include_unresolved and unresolved:
        lines.extend(
            [
                "- 仍需保守处理：",
                _format_bullets(unresolved),
            ]
        )
    return "\n".join(lines)


def _guardrail_notes(meta: dict, validation: CrossValidationResult) -> list[str]:
    notes = meta.get("closed_loop_notes", [])
    candidate_notes = [str(note).strip() for note in notes if isinstance(note, str)]
    candidate_notes.extend(issue.detail for issue in validation.issues if issue.severity != "low")
    filtered = [
        note
        for note in candidate_notes
        if any(keyword in note for keyword in ("冲突", "跨度", "误导", "不可辨", "不符", "待考", "无法"))
    ]
    return _take(_dedupe_texts(filtered), 6)


def _background_knowledge(meta: dict) -> list[str]:
    profile = meta.get("domain_profile") or meta.get("system_metadata") or {}
    lines: list[str] = []
    knowledge = profile.get("knowledge_background", [])
    if isinstance(knowledge, list):
        lines.extend(str(item).strip() for item in knowledge if str(item).strip())
    for item in meta.get("post_rag_text_extraction", [])[:6]:
        if not isinstance(item, dict):
            continue
        term = str(item.get("term", "")).strip()
        description = " ".join(str(item.get("description", "")).strip().split())
        if not term and not description:
            continue
        if term and description:
            lines.append(f"{term}：{description[:180]}")
        else:
            lines.append(term or description[:180])
    return _take(_dedupe_texts(lines), 10)


def _image_detail_lines(meta: dict) -> list[str]:
    profile = meta.get("domain_profile") or meta.get("system_metadata") or {}
    details = profile.get("image_details", [])
    lines: list[str] = []
    if isinstance(details, list):
        lines.extend(str(item).strip() for item in details if str(item).strip())
    return _take(_dedupe_texts(lines), 10)


def _long_slot_digest(output: DomainCoTRecord) -> str:
    visual = _take(
        _dedupe_texts(
            [
                f"{item.observation}{f'（位置：{item.position}）' if item.position else ''}"
                for item in output.visual_anchoring
            ]
        ),
        8,
    )
    decoding = _take(
        _dedupe_texts(
            [
                (
                    f"{item.term}：{item.explanation}"
                    if item.term and item.explanation
                    else item.term or item.explanation
                )
                for item in output.domain_decoding
            ]
        ),
        8,
    )
    mapping = _take(_dedupe_texts([item.insight for item in output.cultural_mapping]), 6)
    answered = _take(_dedupe_texts([item.question for item in output.question_coverage if item.answered]), 6)
    unresolved = _take(_dedupe_texts(output.unresolved_points), 4)
    lines = [
        f"### {output.slot_name}",
        f"- 槽位术语：{output.slot_term or output.slot_name}",
        "- 图像信息：",
        _format_bullets(visual, empty_text="- 暂无稳定图像信息"),
        "- 分析与解码：",
        _format_bullets(decoding, empty_text="- 暂无稳定分析结果"),
        "- 背景与意义：",
        _format_bullets(mapping, empty_text="- 暂无稳定背景解释"),
    ]
    if answered:
        lines.extend(["- 本轮及历史已覆盖的重要问题：", _format_bullets(answered)])
    if unresolved:
        lines.extend(["- 仍需保守处理的点：", _format_bullets(unresolved)])
    return "\n".join(lines)


def _meta_payload(meta: dict) -> dict:
    return {
        "system_metadata": meta.get("system_metadata", {}),
        "domain_profile": meta.get("domain_profile", {}),
        "post_rag_text_extraction": meta.get("post_rag_text_extraction", []),
        "ontology_updates": meta.get("ontology_updates", []),
        "downstream_updates": meta.get("downstream_updates", []),
        "closed_loop_notes": meta.get("closed_loop_notes", []),
        "dialogue_turns": meta.get("dialogue_turns", []),
        "round_memories": meta.get("round_memories", [])[-2:],
    }


def build_domain_cot_prompt(
    slot: SlotSchema,
    meta: dict,
    focus_question: str | None = None,
    analysis_round: int = 1,
    thread_context: dict | None = None,
) -> str:
    focus_text = focus_question.strip() if focus_question else "无"
    thread_payload = thread_context or {}
    return (
        "你是 dynamic agent pipeline 的领域 CoT 推理引擎。"
        "请严格围绕当前槽位，对高分辨率国画图像做三段式推理，并只输出 JSON。\n\n"
        "执行规则：\n"
        "1. 先解析 slot schema，把 slot_term 与 description 中的核心术语当成受控词表。\n"
        "2. 在专业解码层优先使用这些术语，避免随意改写。\n"
        "3. 若题跋、印章、草书或模糊细节无法确认，必须输出状态 [STATUS: UNIDENTIFIABLE_FEATURE]，不能猜。\n"
        "4. 绝不允许跨朝代、跨流派强行拼接常识。\n"
        "5. 若当前槽位不适合常规术语框架，就退回物理现象描述。\n\n"
        f"当前槽位: {slot.slot_name}\n"
        f"当前槽位术语: {slot.slot_term}\n"
        f"补充聚焦问题: {focus_text}\n"
        f"分析轮次: {analysis_round}\n"
        f"受控词表: {json.dumps(slot.controlled_vocabulary, ensure_ascii=False)}\n"
        f"槽位描述: {slot.description}\n"
        f"specific_questions: {json.dumps(slot.specific_questions, ensure_ascii=False)}\n"
        f"thread_context: {json.dumps(thread_payload, ensure_ascii=False)}\n"
        f"context_meta: {json.dumps(_meta_payload(meta), ensure_ascii=False)}\n\n"
        "JSON 输出结构：\n"
        "{\n"
        '  "slot_name": "string",\n'
        '  "controlled_vocabulary_used": ["term"],\n'
        '  "visual_anchoring": [\n'
        '    {"observation": "客观视觉现象", "evidence": "画面证据", "position": "位置"}\n'
        "  ],\n"
        '  "domain_decoding": [\n'
        '    {"term": "术语", "explanation": "术语如何对应图像", "status": "IDENTIFIED|UNIDENTIFIABLE_FEATURE", "reason": "原因"}\n'
        "  ],\n"
        '  "cultural_mapping": [\n'
        '    {"insight": "通俗但严谨的文化映射", "basis": "视觉依据", "risk_note": "若不确定则说明"}\n'
        "  ],\n"
        '  "specific_question_coverage": [\n'
        '    {"question": "问题", "answered": true, "support": "图像依据"}\n'
        "  ],\n"
        '  "generated_questions": ["新问题"],\n'
        '  "unresolved_points": ["仍不确定的点"],\n'
        '  "confidence": 0.0\n'
        "}\n"
    )


def build_round_table_prompt(
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
) -> str:
    payload = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "visual_anchoring": [item.observation for item in output.visual_anchoring],
            "domain_decoding": [item.term for item in output.domain_decoding],
            "cultural_mapping": [item.insight for item in output.cultural_mapping],
            "question_coverage": [item.question for item in output.question_coverage if item.answered],
            "unresolved_points": output.unresolved_points,
            "statuses": output.statuses,
        }
        for output in outputs
    ]
    issues = [
        {
            "type": issue.issue_type,
            "severity": issue.severity,
            "slots": issue.slot_names,
            "detail": issue.detail,
        }
        for issue in validation.issues
    ]
    return (
        "你是圆桌交叉验证节点。请基于各槽位草稿与已有本地校验结果，再做一次严格的逻辑复核。"
        "重点寻找：年代与技法冲突、视觉证据不足、问题覆盖缺口、重复描述，以及应交给 RAG 的术语。\n"
        "输出自然语言短报告，不要 JSON。\n\n"
        f"context_meta: {json.dumps(_meta_payload(meta), ensure_ascii=False)}\n"
        f"已有输出: {json.dumps(payload, ensure_ascii=False)}\n"
        f"本地校验: {json.dumps(issues, ensure_ascii=False)}\n"
    )


def build_summary_prompt(outputs: list[DomainCoTRecord], validation: CrossValidationResult, meta: dict) -> str:
    payload = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "visual_anchoring": [
                {"observation": item.observation, "evidence": item.evidence, "position": item.position}
                for item in output.visual_anchoring
            ],
            "domain_decoding": [
                {"term": item.term, "explanation": item.explanation, "status": item.status}
                for item in output.domain_decoding
            ],
            "cultural_mapping": [
                {"insight": item.insight, "basis": item.basis, "risk_note": item.risk_note}
                for item in output.cultural_mapping
            ],
        }
        for output in outputs
    ]
    return (
        "你是最终总结节点。请把已经排重并通过常识校验的内容，整理成颗粒度细、通俗易懂、学术严谨的国画赏析 Markdown。\n"
        "必须满足：\n"
        "1. 使用 Markdown 标题、加粗、列表。\n"
        "2. 所有判断都要落到图像证据。\n"
        "3. 使用专业术语时要顺带解释，避免堆砌。\n"
        "4. 对仍不确定之处要明说“特征模糊”或“暂不可辨”。\n"
        "5. 不要为了简洁删掉稳定细节；每个槽位尽量保留多条视觉证据、专业解码和已解决问题。\n"
        "6. 不要输出 JSON。\n\n"
        f"context_meta: {json.dumps(_meta_payload(meta), ensure_ascii=False)}\n"
        f"validated_issues: {json.dumps([issue.detail for issue in validation.issues], ensure_ascii=False)}\n"
        f"deduped_content: {json.dumps(payload, ensure_ascii=False)}\n"
    )


def build_final_appreciation_prompt(
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    dialogue_state: DialogueState,
) -> str:
    prompt_mode = "final" if dialogue_state.converged else "intermediate"
    profile_lines = _profile_snapshot(meta)
    image_details = _image_detail_lines(meta)
    background_knowledge = _background_knowledge(meta)
    slot_sections = "\n\n".join(_long_slot_digest(output) for output in outputs)
    guardrails = _guardrail_notes(meta, validation)
    resolved = _take(_dedupe_texts(dialogue_state.resolved_questions), 18)
    open_questions = _take(
        _dedupe_texts(
            dialogue_state.unresolved_questions
            + validation.missing_points
            + [issue.detail for issue in validation.issues if issue.severity != "low"]
        ),
        15,
    )
    prompt_label = "最终综合赏析提示单" if dialogue_state.converged else "阶段性综合赏析提示单"
    direct_prompt = (
        "请基于上方“图像背景”“图像信息”“背景知识”“综合分析”“已覆盖问题”“综合性问题”六部分，"
        "撰写一份长篇中文综合赏析提示文本。"
        "文本必须同时包含：对画面内容的描述、对关键技法与图像结构的分析、与中国画传统相关的背景知识、"
        "以及值得继续深入研究的综合性问题。"
        "若某条信息主要来自 RAG 或一般艺术史知识，请用“常见于”“通常用于”“往往会”“可理解为”这类表述；"
        "若某条判断有明确视觉支撑，则要把图像证据一起写出。"
        "对不能确认之处必须明确写成“暂不可辨”或“证据不足”；"
        "禁止引入新的作品名、作者名、朝代结论或与当前主题无关的类比例子。"
    )
    return (
        f"# {prompt_label}\n\n"
        "## 当前状态\n"
        f"- 模式：{prompt_mode}\n"
        f"- 是否收敛：{'是' if dialogue_state.converged else '否'}\n"
        f"- 停止或阶段说明：{dialogue_state.convergence_reason or '仍在补证据阶段'}\n\n"
        "## 作品画像\n"
        f"{_format_bullets(profile_lines, empty_text='- 当前未提供稳定的作品画像信息')}\n\n"
        "## 图像信息\n"
        f"{_format_bullets(image_details, empty_text='- 当前未额外记录稳定的图像细节清单')}\n\n"
        "## 背景知识\n"
        f"{_format_bullets(background_knowledge, empty_text='- 当前未提取出额外背景知识')}\n\n"
        "## 写作目标\n"
        "- 生成一份对人类读者友好的长文本，覆盖图像背景、图像信息、分析结论和后续研究问题。\n"
        "- 重点说明技法、构图、题材身份、题跋或物理状态在中国画语境中的一般含义与常见作用。\n"
        "- 所有术语都要带简短解释，避免只堆术语不落地。\n"
        "- 若某条信息来自 RAG 的一般知识而非当前图像独有证据，必须写成概括性表达。\n\n"
        "## 写作边界\n"
        "- 只能使用下方已经验证过的视觉证据与术语。\n"
        "- 不得把其他作品、其他题材的 RAG 示例直接写成当前画作事实。\n"
        "- 遇到模糊处要保守表达，不得补写不存在的题跋、印章、作者或年代。\n"
        "- 除非图像证据足够强，否则不要写“增强了”“证明了”“体现了该画家必然如何”，优先改写为“通常用于”“常被用来”“可帮助形成”。\n\n"
        "## 主题与证据风险提醒\n"
        f"{_format_bullets(guardrails, empty_text='- 当前未发现额外的高风险主题漂移提示')}\n\n"
        "## 综合分析\n"
        f"{slot_sections or '### 暂无稳定槽位输出'}\n\n"
        "## 已覆盖的重要问题\n"
        f"{_format_bullets(resolved, empty_text='- 当前尚未记录已覆盖的问题')}\n\n"
        "## 综合性问题\n"
        f"{_format_bullets(open_questions, empty_text='- 当前没有额外的综合问题')}\n\n"
        "## 可直接交给生成模型的指令\n"
        "```text\n"
        f"{direct_prompt}\n"
        "```\n"
    )
