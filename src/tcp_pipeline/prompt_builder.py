from __future__ import annotations

import json

from .slots import slot_label, slot_labels


def build_baseline_prompt(image_path: str, meta: dict, slots: list[str]) -> str:
    _ = image_path
    _ = meta
    slots_text = "、".join(slot_labels(slots)) if slots else "整体"
    return f"请从{slots_text}等方面赏析这幅中国画。"


def build_enhanced_prompt(image_path: str, meta: dict, slot_context: dict[str, str]) -> str:
    _ = image_path
    _ = meta
    blocks: list[str] = []
    for slot, context_text in slot_context.items():
        if not context_text.strip():
            continue
        blocks.append(f"### {slot_label(slot)}\n{context_text}")

    return (
        "请结合图像与以下分领域的补充信息内容，对这幅中国画进行赏析。请注意不能直接抄补充信息的内容，而是在补充信息的基础上充分赏析图片\n\n"
        "## 补充信息\n"
        + "\n\n".join(blocks)
    )


def build_solitary_first_prompt() -> str:
    return (
        "请基于图像进行中国画鉴赏。\n"
        "要求：输出自然语言，不要JSON"
    )


def build_solitary_reflection_prompt(previous_analysis: str, round_index: int) -> str:
    return (
        f"请继续赏析这幅中国画。\n\n"
        f"以下是第{round_index - 1}轮赏析结果，请结合图像进行反思并重新鉴赏：\n"
        "1. 是否存在错误或不准确描述；是否有证据不足或推断过度；是否有没有鉴赏到的点。成对列出不足和修改方式（反思要点对）\n"
        "2. 在反思要点对的基础上，继续对图片进行赏析，并生成本轮的完整赏析内容\n"
        "3. 输出必须严格包含以下两个分段标题：\n"
        "【反思要点】\n"
        "【本轮赏析】\n"
        "后续轮次只会读取【本轮赏析】部分。\n\n"
        f"### 第{round_index - 1}轮赏析结果\n"
        f"{previous_analysis}\n"
    )


def build_communal_guest1_prompt() -> str:
    return (
        "这是一幅中国画，请你作为群赏中的第一位客人，先行题评。\n"
        "请进行个人化鉴赏，不限定维度。\n"
        "要求：\n"
        "1. 保持你个人兴趣点，不必面面俱到\n"
        "2. 输出为人类可读的自然语言，不要JSON"
    )


def build_communal_guest_next_prompt(previous_guest_texts: list[str], guest_index: int) -> str:
    history = "\n\n".join(f"【前序客人{i + 1}】\n{text}" for i, text in enumerate(previous_guest_texts))
    return (
        f"这是一幅中国画，你是第{guest_index}位客人。\n"
        "图片和上面几位客人的鉴赏都在，请在不重复前人表述的前提下，"
        "根据你自己的兴趣点给出个人化鉴赏。\n\n"
        "要求：\n"
        "1. 必须结合图像本身，不要只复述前文\n"
        "2. 明确指出你新增了什么观察或判断\n"
        "3. 可保留分歧观点，但要有图像依据。如果有分歧点也需要写进文本内容\n"
        "4. 输出为人类可读的自然语言，不要JSON\n\n"
        "### 前序客人鉴赏\n"
        f"{history}\n"
    )


def build_slot_pipe_agent_prompt(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    slot: str,
    meta: dict,
    previous_layer_slot_points: list[dict] | None,
    judge_feedback: str,
    agent_index: int,
    attempt_index: int,
) -> str:
    if layer_index == 1:
        return build_slot_pipe_agent_prompt_layer1(
            slot=slot,
            meta=meta,
            previous_layer_slot_points=previous_layer_slot_points,
            judge_feedback=judge_feedback,
            agent_index=agent_index,
            attempt_index=attempt_index,
        )
    if layer_index == 2:
        return build_slot_pipe_agent_prompt_layer2(
            slot=slot,
            meta=meta,
            previous_layer_slot_points=previous_layer_slot_points,
            judge_feedback=judge_feedback,
            agent_index=agent_index,
            attempt_index=attempt_index,
        )
    return build_slot_pipe_agent_prompt_layer3(
        slot=slot,
        meta=meta,
        previous_layer_slot_points=previous_layer_slot_points,
        judge_feedback=judge_feedback,
        agent_index=agent_index,
        attempt_index=attempt_index,
    )


def _build_slot_pipe_agent_prompt_base(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    slot: str,
    meta: dict,
    previous_layer_slot_points: list[dict] | None,
    judge_feedback: str,
    agent_index: int,
    attempt_index: int,
    extra_constraints: list[str],
) -> str:
    prev_points_text = "无（第一层）"
    if previous_layer_slot_points:
        prev_points_text = json.dumps(previous_layer_slot_points, ensure_ascii=False, indent=2)

    feedback_text = judge_feedback.strip() if judge_feedback.strip() else "无（首次执行）"
    extra_text = "\n".join(f"5. {line}" for line in extra_constraints)
    return (
        f"你是slot_pipe模式中的第{layer_index}层agent。\n"
        f"层级主题：{layer_name}\n"
        f"层级目标：{layer_goal}\n"
        f"当前slot：{slot_label(slot)}\n"
        f"agent编号：{agent_index}\n"
        f"当前尝试：第{attempt_index}次\n\n"
        "请你基于国画图像完成该slot任务，并仅输出自然语言要点。\n"
        "输出要求：\n"
        "1. 输出5到8条要点，每条以“- ”开头\n"
        "2. 必须体现图像证据，不要空泛修辞\n"
        "3. 严禁输出JSON\n"
        "4. 只围绕本slot，不要扩散到其他slot\n"
        f"{extra_text}\n\n"
        f"meta信息：{json.dumps(meta, ensure_ascii=False)}\n"
        "上一层该slot要点（如有）：\n"
        f"{prev_points_text}\n\n"
        "judge反馈（如有）：\n"
        f"{feedback_text}\n"
    )


def build_slot_pipe_agent_prompt_layer1(
    slot: str,
    meta: dict,
    previous_layer_slot_points: list[dict] | None,
    judge_feedback: str,
    agent_index: int,
    attempt_index: int,
) -> str:
    return _build_slot_pipe_agent_prompt_base(
        layer_index=1,
        layer_name="要素整理",
        layer_goal="提炼该slot在图像中的视觉要素与证据。",
        slot=slot,
        meta=meta,
        previous_layer_slot_points=previous_layer_slot_points,
        judge_feedback=judge_feedback,
        agent_index=agent_index,
        attempt_index=attempt_index,
        extra_constraints=[
            "优先输出可被图像直接验证的客观观察，不做跨图推断。",
            "尽量覆盖形、势、色、笔触或材料痕迹中的至少两类。",
        ],
    )


def build_slot_pipe_agent_prompt_layer2(
    slot: str,
    meta: dict,
    previous_layer_slot_points: list[dict] | None,
    judge_feedback: str,
    agent_index: int,
    attempt_index: int,
) -> str:
    return _build_slot_pipe_agent_prompt_base(
        layer_index=2,
        layer_name="内容要点",
        layer_goal="整理可用于最终鉴赏写作的内容要点与论述方向。",
        slot=slot,
        meta=meta,
        previous_layer_slot_points=previous_layer_slot_points,
        judge_feedback=judge_feedback,
        agent_index=agent_index,
        attempt_index=attempt_index,
        extra_constraints=[
            "每条要点需包含“观察依据 -> 鉴赏判断”的关系。",
            "避免只罗列名词，尽量给出可直接用于写作的论述线索。",
        ],
    )


def build_slot_pipe_agent_prompt_layer3(
    slot: str,
    meta: dict,
    previous_layer_slot_points: list[dict] | None,
    judge_feedback: str,
    agent_index: int,
    attempt_index: int,
) -> str:
    return _build_slot_pipe_agent_prompt_base(
        layer_index=3,
        layer_name="专业表达",
        layer_goal="将要点提升为更专业、准确、可落地的鉴赏表达建议。",
        slot=slot,
        meta=meta,
        previous_layer_slot_points=previous_layer_slot_points,
        judge_feedback=judge_feedback,
        agent_index=agent_index,
        attempt_index=attempt_index,
        extra_constraints=[
            "强调术语准确、语气克制、论证完整，避免空泛修辞。",
            "输出应可直接转写为最终鉴赏段落的表达要求。",
        ],
    )


def build_slot_pipe_slot_judge_prompt(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    slot: str,
    max_retry: int,
    retry_round: int,
    pooled_points: list[dict],
    latest_agent_outputs: list[dict],
) -> str:
    if layer_index == 1:
        return build_slot_pipe_slot_judge_prompt_layer1(
            slot=slot,
            max_retry=max_retry,
            retry_round=retry_round,
            pooled_points=pooled_points,
            latest_agent_outputs=latest_agent_outputs,
        )
    if layer_index == 2:
        return build_slot_pipe_slot_judge_prompt_layer2(
            slot=slot,
            max_retry=max_retry,
            retry_round=retry_round,
            pooled_points=pooled_points,
            latest_agent_outputs=latest_agent_outputs,
        )
    return build_slot_pipe_slot_judge_prompt_layer3(
        slot=slot,
        max_retry=max_retry,
        retry_round=retry_round,
        pooled_points=pooled_points,
        latest_agent_outputs=latest_agent_outputs,
    )


def _build_slot_pipe_slot_judge_prompt_base(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    slot: str,
    max_retry: int,
    retry_round: int,
    pooled_points: list[dict],
    latest_agent_outputs: list[dict],
    quality_focus: str,
) -> str:
    return (
        "你是slot_pipe的slot级评审judge，请严格输出JSON。\n"
        f"当前层：{layer_index}-{layer_name}\n"
        f"层级目标：{layer_goal}\n"
        f"本层评审重点：{quality_focus}\n"
        f"slot：{slot_label(slot)}\n"
        f"当前返工轮次：{retry_round}，最大返工轮次：{max_retry}\n\n"
        "这是语义池化后的要点：\n"
        f"{json.dumps(pooled_points, ensure_ascii=False, indent=2)}\n\n"
        "这是各agent最新输出：\n"
        f"{json.dumps(latest_agent_outputs, ensure_ascii=False, indent=2)}\n\n"
        "请只输出一个JSON对象，结构如下：\n"
        "{\n"
        '  "slot_status": "good|needs_improve",\n'
        '  "reason": "简短说明",\n'
        '  "agents_to_retry": [1,2],\n'
        '  "feedback_by_agent": {"1": "具体改进方向"}\n'
        "}\n"
        "要求：\n"
        "1. 只有在确实有明显问题时才给needs_improve\n"
        "2. agents_to_retry必须是已有agent编号\n"
        "3. 如果slot_status是good，则agents_to_retry返回空数组\n"
        "4. 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_slot_judge_prompt_layer1(
    slot: str,
    max_retry: int,
    retry_round: int,
    pooled_points: list[dict],
    latest_agent_outputs: list[dict],
) -> str:
    return _build_slot_pipe_slot_judge_prompt_base(
        layer_index=1,
        layer_name="要素整理",
        layer_goal="提炼该slot在图像中的视觉要素与证据。",
        slot=slot,
        max_retry=max_retry,
        retry_round=retry_round,
        pooled_points=pooled_points,
        latest_agent_outputs=latest_agent_outputs,
        quality_focus="要点是否客观、可被图像验证，是否避免空泛主观判断。",
    )


def build_slot_pipe_slot_judge_prompt_layer2(
    slot: str,
    max_retry: int,
    retry_round: int,
    pooled_points: list[dict],
    latest_agent_outputs: list[dict],
) -> str:
    return _build_slot_pipe_slot_judge_prompt_base(
        layer_index=2,
        layer_name="内容要点",
        layer_goal="整理可用于最终鉴赏写作的内容要点与论述方向。",
        slot=slot,
        max_retry=max_retry,
        retry_round=retry_round,
        pooled_points=pooled_points,
        latest_agent_outputs=latest_agent_outputs,
        quality_focus="要点是否形成“观察依据->判断”链路，是否具备写作可用性。",
    )


def build_slot_pipe_slot_judge_prompt_layer3(
    slot: str,
    max_retry: int,
    retry_round: int,
    pooled_points: list[dict],
    latest_agent_outputs: list[dict],
) -> str:
    return _build_slot_pipe_slot_judge_prompt_base(
        layer_index=3,
        layer_name="专业表达",
        layer_goal="将要点提升为更专业、准确、可落地的鉴赏表达建议。",
        slot=slot,
        max_retry=max_retry,
        retry_round=retry_round,
        pooled_points=pooled_points,
        latest_agent_outputs=latest_agent_outputs,
        quality_focus="术语是否准确、语气是否克制、表达是否可直接落地到最终文稿。",
    )


def build_slot_pipe_layer_judge_prompt(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    current_slots: list[str],
    slot_global_payload: dict[str, dict],
    max_retry: int,
    retry_round: int,
) -> str:
    if layer_index == 1:
        return build_slot_pipe_layer_judge_prompt_layer1(
            current_slots=current_slots,
            slot_global_payload=slot_global_payload,
            max_retry=max_retry,
            retry_round=retry_round,
        )
    if layer_index == 2:
        return build_slot_pipe_layer_judge_prompt_layer2(
            current_slots=current_slots,
            slot_global_payload=slot_global_payload,
            max_retry=max_retry,
            retry_round=retry_round,
        )
    return build_slot_pipe_layer_judge_prompt_layer3(
        current_slots=current_slots,
        slot_global_payload=slot_global_payload,
        max_retry=max_retry,
        retry_round=retry_round,
    )


def _build_slot_pipe_layer_judge_prompt_base(
    layer_index: int,
    layer_name: str,
    layer_goal: str,
    current_slots: list[str],
    slot_global_payload: dict[str, dict],
    max_retry: int,
    retry_round: int,
    quality_focus: str,
) -> str:
    return (
        "你是slot_pipe的全局judge。你需要一次性完成两件事：\n"
        "1) 判断当前层是否需要继续返工（并指定要返工的slot+agent）；\n"
        "2) 当本层达到可接受质量时，给出slot更新与next_slots。\n"
        "严格输出JSON。\n"
        f"当前层：{layer_index}-{layer_name}\n"
        f"层级目标：{layer_goal}\n"
        f"本层评审重点：{quality_focus}\n"
        f"当前评审轮次：{retry_round}，最大返工轮次：{max_retry}\n"
        f"当前slot集合：{json.dumps(current_slots, ensure_ascii=False)}\n\n"
        "每个slot的全局结果如下（含池化结果、各agent最新输出、历史尝试记录）：\n"
        f"{json.dumps(slot_global_payload, ensure_ascii=False, indent=2)}\n\n"
        "请只输出一个JSON对象，结构如下：\n"
        "{\n"
        '  "layer_ok": true,\n'
        '  "summary": "本层总结",\n'
        '  "slot_decisions": [\n'
        '    {"slot": "笔墨", "status": "good|needs_improve", "reason": "原因"}\n'
        "  ],\n"
        '  "retry_tasks": [\n'
        '    {"slot": "笔墨", "agent_index": 1, "feedback": "具体返工要求"}\n'
        "  ],\n"
        '  "slot_updates": [\n'
        '    {"slot": "笔墨", "action": "rename|reduce|split", "new_slot": "", "new_slots": []}\n'
        "  ],\n"
        '  "next_slots": ["笔墨", "章法经营"]\n'
        "}\n"
        "few-shot 示例（只作格式与意图参考，禁止照抄slot名）：\n"
        "示例1：rename（重命名）\n"
        "{\n"
        '  "layer_ok": true,\n'
        '  "summary": "概念边界更准确，进行重命名",\n'
        '  "slot_decisions": [{"slot": "设色", "status": "good", "reason": "维度有效"}],\n'
        '  "retry_tasks": [],\n'
        '  "slot_updates": [{"slot": "设色", "action": "rename", "new_slot": "设色层次"}],\n'
        '  "next_slots": ["设色层次", "笔墨", "章法"]\n'
        "}\n"
        "示例2：reduce（2变1，审慎）\n"
        "{\n"
        '  "layer_ok": true,\n'
        '  "summary": "两个slot高度重合，且证据重复，谨慎合并",\n'
        '  "slot_decisions": [\n'
        '    {"slot": "气韵", "status": "good", "reason": "与神采重合较高"},\n'
        '    {"slot": "神采", "status": "good", "reason": "与气韵重合较高"}\n'
        "  ],\n"
        '  "retry_tasks": [],\n'
        '  "slot_updates": [\n'
        '    {"slot": "气韵", "action": "reduce", "new_slot": "气韵神采"},\n'
        '    {"slot": "神采", "action": "reduce", "new_slot": "气韵神采"}\n'
        "  ],\n"
        '  "next_slots": ["气韵神采", "笔墨", "章法"]\n'
        "}\n"
        "示例3：split（1变2）\n"
        "{\n"
        '  "layer_ok": true,\n'
        '  "summary": "单一slot内部包含两类稳定子问题，拆分更利于后续写作",\n'
        '  "slot_decisions": [{"slot": "构图", "status": "good", "reason": "可拆为章法与空间"}],\n'
        '  "retry_tasks": [],\n'
        '  "slot_updates": [{"slot": "构图", "action": "split", "new_slots": ["章法经营", "空间层次"]}],\n'
        '  "next_slots": ["章法经营", "空间层次", "笔墨", "设色"]\n'
        "}\n"
        "要求：\n"
        "1. 若 layer_ok=false，则必须给出 retry_tasks（可为空但需解释）\n"
        "2. retry_tasks 中 agent_index 必须是正整数\n"
        "3. 若 layer_ok=true，next_slots 不能为空；若无需变更，返回当前slot集合\n"
        "4. slot_updates 记录每个slot处理决定：\n"
        "   - rename：1变1，必须给 new_slot\n"
        "   - reduce：2变1（或多变1），每个被合并slot都要给 action=reduce 且 new_slot 相同\n"
        "   - split：1变2（或1变多），必须给 new_slots（长度>=2）\n"
        "5. next_slots 与 slot_updates.new_slot/new_slots（若提供）必须是中文命名，且必须使用国画鉴赏理论概念（如笔墨、章法、气韵、设色、虚实、留白、意境等）\n"
        "6. 严禁使用 SlotA/SlotB、A/B、topic1 等占位命名\n"
        "7. reduce 必须审慎：仅在证据高度重合、区分收益明显不足时才允许；默认优先 rename 或 split\n"
        "8. next_slots 必须与当前slot形成清晰边界，避免语义重叠或合并已有维度（如已有“笔墨”“气韵”时，不应再给“笔墨气韵”）\n"
        "9. slot_updates.new_slot/new_slots 也必须满足不重叠原则，避免把两个既有维度拼接成一个新slot\n"
        "10. 每层最多减少1个slot；若无强证据，不要大幅缩减slot集合\n"
        "11. 仅输出JSON"
    )


def build_slot_pipe_layer_judge_prompt_layer1(
    current_slots: list[str],
    slot_global_payload: dict[str, dict],
    max_retry: int,
    retry_round: int,
) -> str:
    return _build_slot_pipe_layer_judge_prompt_base(
        layer_index=1,
        layer_name="要素整理",
        layer_goal="提炼该slot在图像中的视觉要素与证据。",
        current_slots=current_slots,
        slot_global_payload=slot_global_payload,
        max_retry=max_retry,
        retry_round=retry_round,
        quality_focus="定位客观要素不足或证据薄弱的slot，并派发定向返工任务。",
    )


def build_slot_pipe_layer_judge_prompt_layer2(
    current_slots: list[str],
    slot_global_payload: dict[str, dict],
    max_retry: int,
    retry_round: int,
) -> str:
    return _build_slot_pipe_layer_judge_prompt_base(
        layer_index=2,
        layer_name="内容要点",
        layer_goal="整理可用于最终鉴赏写作的内容要点与论述方向。",
        current_slots=current_slots,
        slot_global_payload=slot_global_payload,
        max_retry=max_retry,
        retry_round=retry_round,
        quality_focus="定位论述链路不足或写作可用性不足的slot，并派发定向返工任务。",
    )


def build_slot_pipe_layer_judge_prompt_layer3(
    current_slots: list[str],
    slot_global_payload: dict[str, dict],
    max_retry: int,
    retry_round: int,
) -> str:
    return _build_slot_pipe_layer_judge_prompt_base(
        layer_index=3,
        layer_name="专业表达",
        layer_goal="将要点提升为更专业、准确、可落地的鉴赏表达建议。",
        current_slots=current_slots,
        slot_global_payload=slot_global_payload,
        max_retry=max_retry,
        retry_round=retry_round,
        quality_focus="定位术语、语气和可落地表达不足的slot，并派发定向返工任务。",
    )


def build_slot_pipe_final_prompt(slot_pipe_layers: list[dict]) -> str:
    layer_slot_points: dict[str, dict[str, list[str]]] = {
        "要素整理": {},
        "内容要点": {},
        "专业表达": {},
    }
    slot_order: list[str] = []
    seen_slots: set[str] = set()

    for layer in slot_pipe_layers or []:
        layer_name = str(layer.get("layer_name", "")).strip()
        if layer_name not in layer_slot_points:
            continue
        for slot_item in layer.get("slots") or []:
            if not isinstance(slot_item, dict):
                continue
            slot_name = str(slot_item.get("slot", "")).strip()
            if not slot_name:
                continue
            if slot_name not in seen_slots:
                seen_slots.add(slot_name)
                slot_order.append(slot_name)
            points_text: list[str] = []
            for point_item in slot_item.get("final_points") or []:
                if not isinstance(point_item, dict):
                    continue
                text = str(point_item.get("point", "")).strip()
                if text:
                    points_text.append(text)
            if points_text:
                layer_slot_points[layer_name][slot_name] = points_text

    if not slot_order:
        return (
            "以下内容是slot_pipe三层迭代后的结构化补充信息。\n"
            "最重要原则：最终鉴赏必须首先结合图片本身进行观察与判断，再利用以下辅助文本进行补强，不得以文本替代看图。\n"
            "请在最终国画鉴赏中吸收这些要点，但不要机械复述。\n\n"
            "## 辅助材料说明\n"
            "- 要素点：偏客观观察要点，用于视觉理解与图像证据定位。\n"
            "- 内容点：偏写作指令（需带例子），用于组织论述内容与分析线索。\n"
            "- 表达点：偏表述指令（需带例子），用于约束语气、术语、论证与可读性。\n\n"
            "## 要素材料（客观要点）\n- 无\n\n"
            "## 内容材料（带例子的写作指令）\n- 无\n\n"
            "## 表达材料（带例子的表述指令）\n- 无\n"
        )

    def _first_example(points: list[str]) -> str:
        if not points:
            return "图像证据"
        sample = points[0].strip().replace("\n", " ")
        return sample[:60] + ("..." if len(sample) > 60 else "")

    element_blocks: list[str] = []
    content_blocks: list[str] = []
    expression_blocks: list[str] = []

    for slot_name in slot_order:
        element_points = layer_slot_points.get("要素整理", {}).get(slot_name, [])
        content_points = layer_slot_points.get("内容要点", {}).get(slot_name, [])
        expression_points = layer_slot_points.get("专业表达", {}).get(slot_name, [])

        if element_points:
            element_lines = [f"### {slot_name}"] + [f"- {p}" for p in element_points]
            element_blocks.append("\n".join(element_lines))

        if content_points:
            ex = _first_example(content_points)
            content_lines = [f"### {slot_name}"]
            for p in content_points:
                content_lines.append(
                    f"- 指令：围绕“{slot_name}”组织论述，必须给出图像依据与判断关系；可参考例子“{ex}”，并吸收要点“{p}”。"
                )
            content_blocks.append("\n".join(content_lines))

        if expression_points:
            ex = _first_example(expression_points)
            expression_lines = [f"### {slot_name}"]
            for p in expression_points:
                expression_lines.append(
                    f"- 指令：在“{slot_name}”段落中使用专业且克制的鉴赏表述，避免空泛修辞；可参考例子“{ex}”，并落实要求“{p}”。"
                )
            expression_blocks.append("\n".join(expression_lines))

    element_body = "\n\n".join(element_blocks).strip() or "- 无"
    content_body = "\n\n".join(content_blocks).strip() or "- 无"
    expression_body = "\n\n".join(expression_blocks).strip() or "- 无"

    return (
        "以下内容是slot_pipe三层迭代后的结构化补充信息。\n"
        "最重要原则：最终鉴赏必须首先结合图片本身进行观察与判断，再利用以下辅助文本进行补强，不得以文本替代看图。\n"
        "请在最终国画鉴赏中吸收这些要点，但不要机械复述。\n\n"
        "## 辅助材料说明\n"
        "- 要素点：偏客观观察要点，用于视觉理解与图像证据定位（看图可验证）。\n"
        "- 内容点：偏写作指令（需带例子），用于组织论述内容、因果关系与分析线索。\n"
        "- 表达点：偏表述指令（需带例子），用于规范术语、语气、论证强度与可读性。\n\n"
        "## 要素材料（客观要点）\n"
        f"{element_body}\n\n"
        "## 内容材料（带例子的写作指令）\n"
        f"{content_body}\n\n"
        "## 表达材料（带例子的表述指令）\n"
        f"{expression_body}\n"
    )


def build_slot_pipe_v4_checker_prompt(slot: dict, meta: dict) -> str:
    slot_payload = {
        "slot_name": str(slot.get("slot_name", "")).strip(),
        "slot_term": str(slot.get("slot_term", "")).strip(),
        "description_ref": str(slot.get("description", "")).strip(),
    }
    return (
        "你是国画要素核验checker。\n"
        "请结合图片，对当前slot给出0-5分（5分=图像中证据非常充分，0分=几乎无图像证据）。\n"
        "注意：description仅作术语背景参考，不能当作图像事实。\n"
        "请严格返回JSON：{\"score\": 0-5数字, \"reason\": \"简明图像证据\"}\n"
        f"slot数据：{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n"
        f"meta：{json.dumps(meta or {}, ensure_ascii=False)}\n"
        "要求：\n"
        "1) reason必须指向图像中可见证据\n"
        "2) score支持小数，范围必须在0到5之间\n"
        "2) 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_v4_reviewer_prompt(slot: dict, checker_result: dict, meta: dict) -> str:
    slot_payload = {
        "slot_name": str(slot.get("slot_name", "")).strip(),
        "slot_term": str(slot.get("slot_term", "")).strip(),
        "description_ref": str(slot.get("description", "")).strip(),
    }
    return (
        "你是国画领域专家reviewer。你将复核checker结论是否成立。\n"
        "请结合图片、slot信息和checker分数，给出你的置信度（0-5，5=你非常确认该slot是有效鉴赏点）。\n"
        "注意：description仅作术语背景参考，不能当作图像事实。\n"
        "请严格返回JSON：{\"confidence\": 0-5数字, \"reason\": \"复核理由\", \"review_comment\": \"可选补充\"}\n"
        f"slot数据：{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n"
        f"checker结果：{json.dumps(checker_result or {}, ensure_ascii=False, indent=2)}\n"
        f"meta：{json.dumps(meta or {}, ensure_ascii=False)}\n"
        "要求：\n"
        "1) confidence支持小数，范围必须在0到5之间\n"
        "2) 如与你对checker结论不一致，明确说明冲突点\n"
        "2) 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_v4_content_prompt(slot: dict, verify_result: dict, meta: dict, agent_index: int) -> str:
    slot_payload = {
        "slot_name": str(slot.get("slot_name", "")).strip(),
        "slot_term": str(slot.get("slot_term", "")).strip(),
        "specific_questions": slot.get("specific_questions") if isinstance(slot.get("specific_questions"), list) else [],
    }
    return (
        "你是内容要点层agent，请围绕当前slot产出可用于鉴赏写作的内容要点，并转成问题。\n"
        "可使用通用艺术史与国画知识（可模拟web_search风格），但必须回扣当前图像可观察证据。\n"
        "请严格返回JSON：\n"
        "{\"content_points\": [\"...\"], \"questions\": [\"...？\"]}\n"
        f"agent_index={agent_index}\n"
        f"slot数据：{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n"
        f"verify结果：{json.dumps(verify_result or {}, ensure_ascii=False, indent=2)}\n"
        f"meta：{json.dumps(meta or {}, ensure_ascii=False)}\n"
        "要求：\n"
        "1) content_points给3-6条，强调“观察依据->鉴赏判断”\n"
        "2) questions给3-6条，采用问题句式，能够激发推理\n"
        "3) 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_v4_guest_prompt(
    slot: dict,
    question: str,
    previous_guest_rounds: list[dict],
    guest_index: int,
) -> str:
    history = ""
    if previous_guest_rounds:
        blocks: list[str] = []
        for item in previous_guest_rounds:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("guest_index", 0) or 0)
            ans = str(item.get("answer", "")).strip()
            insight = str(item.get("insight", "")).strip()
            blocks.append(f"客人{idx}回答: {ans}\\n客人{idx}心得: {insight}")
        history = "\n\n".join(blocks)

    slot_payload = {
        "slot_name": str(slot.get("slot_name", "")).strip(),
        "slot_term": str(slot.get("slot_term", "")).strip(),
    }

    return (
        "你在参加国画群赏。请结合图片，围绕指定slot问题给出回答与心得。\n"
        "心得是你如何回答该问题的经验与提示，可供后续客人参考。\n"
        "请严格返回JSON：{\"answer\": \"...\", \"insight\": \"...\"}\n"
        f"当前客人序号：{guest_index}\n"
        f"slot数据：{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n"
        f"问题：{question}\n"
        "前序客人内容（如有）：\n"
        f"{history if history else '无'}\n"
        "要求：\n"
        "1) answer必须直接回答问题并体现图像依据\n"
        "2) insight需可复用、可操作\n"
        "3) 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_v4_expression_summary_prompt(slot: dict, question: str, guest_rounds: list[dict]) -> str:
    slot_payload = {
        "slot_name": str(slot.get("slot_name", "")).strip(),
        "slot_term": str(slot.get("slot_term", "")).strip(),
    }
    return (
        "你是群赏汇总者。请对同一问题下多位客人的回答和心得做池化、降重、摘要。\n"
        "请严格返回JSON：\n"
        "{\"answer\": \"综合回答\", \"insights\": [\"心得1\", \"心得2\"], \"tips\": [\"表达提示1\", \"表达提示2\"]}\n"
        f"slot数据：{json.dumps(slot_payload, ensure_ascii=False, indent=2)}\n"
        f"问题：{question}\n"
        f"客人轮次：{json.dumps(guest_rounds, ensure_ascii=False, indent=2)}\n"
        "要求：\n"
        "1) answer应综合但不重复\n"
        "2) insights与tips分别保留2-5条\n"
        "3) 不允许输出JSON之外任何文字"
    )


def build_slot_pipe_v4_final_prompt(final_slots: list[dict]) -> str:
    if not final_slots:
        return (
            "请结合图片进行国画鉴赏。\n"
            "当前没有可用的slot结构化结果，请直接基于画面进行客观、专业、克制的赏析。"
        )

    blocks: list[str] = []
    for slot in final_slots:
        if not isinstance(slot, dict):
            continue
        name = str(slot.get("slot_name", "")).strip() or "未命名slot"
        term = str(slot.get("slot_term", "")).strip()
        score = slot.get("slot_score")
        confidence = slot.get("slot_confidence")
        questions = slot.get("questions") if isinstance(slot.get("questions"), list) else []
        expression_points = slot.get("expression_points") if isinstance(slot.get("expression_points"), list) else []

        lines = [f"## {name}（{term or '无术语'}）"]
        lines.append(f"- 鉴赏点占比信号（分/置信度）：score={score if score is not None else '0'} / confidence={confidence if confidence is not None else '0'}")
        lines.append("- 内容问题（鉴赏文本中应尽量覆盖并回答）：")
        for q in questions:
            q_text = str(q).strip()
            if q_text:
                lines.append(f"  - {q_text}")
        if not any(str(q).strip() for q in questions):
            lines.append("  - 无")

        lines.append("- 鉴赏表达要点（写作规范与表述要求）：")
        for p in expression_points:
            p_text = str(p).strip()
            if p_text:
                lines.append(f"  - {p_text}")
        if not any(str(p).strip() for p in expression_points):
            lines.append("  - 无")
        blocks.append("\n".join(lines))

    body = "\n\n".join(blocks) if blocks else "无"
    return (
        "你是国画鉴赏专家。最重要原则：必须先看图，再使用以下结构化信息作为辅助，禁止照抄。\n"
        "下面每个slot都包含三类信息：\n"
        "1) 分/置信度：表示该slot作为真鉴赏点在全文中的建议占比，分越高可分配更多篇幅。\n"
        "2) 内容问题：是鉴赏中需要回答的问题。\n"
        "3) 鉴赏表达要点：是写作时需要满足的规范与要求。\n"
        "请输出完整、可读、专业且有图像依据的国画鉴赏文本。\n\n"
        "### 结构化辅助信息\n"
        f"{body}\n"
    )
