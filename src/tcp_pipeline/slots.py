from __future__ import annotations

SLOT = {
    "brush-and-ink": "笔墨",
    "composition": "构图",
    "color": "色彩",
    "spirit": "气韵",
    "school": "流派",
    "history": "历史",
    "literary": "题跋印章",
    "medium": "材质",
}


def slot_label(slot: str) -> str:
    return SLOT.get(slot, slot)


def slot_labels(slots: list[str]) -> list[str]:
    return [slot_label(x) for x in slots]


def normalize_slots(raw_slots: list[str] | None) -> list[str]:
    all_slots = list(SLOT.keys())
    valid_slots = set(SLOT.keys())

    if not raw_slots:
        return all_slots

    normalized_raw = [x.strip().lower() for x in raw_slots if x and x.strip()]
    if not normalized_raw or "all" in normalized_raw:
        return all_slots

    out: list[str] = []
    seen: set[str] = set()
    for slot in raw_slots:
        key = slot.strip().lower()
        if not key or key == "all" or key not in valid_slots or key in seen:
            continue
        seen.add(key)
        out.append(key)

    return out or all_slots
