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
