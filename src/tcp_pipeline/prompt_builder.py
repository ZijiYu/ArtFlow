from __future__ import annotations

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
