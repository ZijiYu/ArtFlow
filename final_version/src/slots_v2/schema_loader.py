from __future__ import annotations

import json
import re
from pathlib import Path

from .models import SlotSchema


_TERM_TOKEN_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9/\-· ]{1,24}|[\u4e00-\u9fff]{2,18}")
_QUOTE_PATTERN = re.compile(r"[“\"'《](.+?)[”\"'》]")
_STOP_TERMS = {
    "中国画",
    "山水画",
    "国画",
    "画作",
    "作品",
    "技法特点",
    "实际创作",
    "具体运用",
    "艺术境界",
    "表现",
    "特点",
    "研究",
    "搭配",
}


def load_slot_schemas(path: str) -> list[SlotSchema]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Slot schema file not found: {path}")

    items: list[SlotSchema] = []
    for raw_line in file_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        slot_name = str(payload.get("slot_name", "")).strip()
        if not slot_name:
            continue
        slot_term = str(payload.get("slot_term", "")).strip()
        description = str(payload.get("description", "")).strip()
        questions = payload.get("specific_questions") if isinstance(payload.get("specific_questions"), list) else []
        metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
        controlled_vocabulary = extract_controlled_vocabulary(slot_term=slot_term, description=description)
        items.append(
            SlotSchema(
                slot_name=slot_name,
                slot_term=slot_term,
                description=description,
                specific_questions=[str(item).strip() for item in questions if str(item).strip()],
                metadata=metadata,
                controlled_vocabulary=controlled_vocabulary,
            )
        )
    return items


def extract_controlled_vocabulary(slot_term: str, description: str) -> list[str]:
    candidates: list[str] = []
    for text in (slot_term, description):
        text = str(text or "").strip()
        if not text:
            continue
        candidates.extend(_QUOTE_PATTERN.findall(text))
        candidates.extend(_TERM_TOKEN_PATTERN.findall(text))
    if slot_term.strip():
        candidates.insert(0, slot_term.strip())
    expanded: list[str] = []
    for candidate in candidates:
        expanded.extend(_expand_candidate(candidate))
    return _dedupe_terms(expanded)


def _dedupe_terms(candidates: list[str]) -> list[str]:
    results: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        term = _clean_term(candidate)
        if not term or term in _STOP_TERMS:
            continue
        key = re.sub(r"\s+", "", term)
        if key in seen:
            continue
        seen.add(key)
        results.append(term)
    return results


def _expand_candidate(candidate: str) -> list[str]:
    text = _clean_term(candidate)
    if not text:
        return []
    text = re.sub(r"^(包含|包括|例如|比如|合并近义术语[:：]?)+", "", text)
    parts = re.split(r"[、，,；;]|和|与|及|包含|包括", text)
    expanded: list[str] = []
    for part in parts:
        piece = _clean_term(part)
        if not piece:
            continue
        if "是" in piece:
            piece = piece.split("是", 1)[0].strip()
        if piece:
            expanded.append(piece)
    return expanded or [text]


def _clean_term(value: str) -> str:
    term = re.sub(r"^[\s:：,，;；、]+|[\s:：,，;；、]+$", "", str(value or ""))
    if not term:
        return ""
    if len(term) == 1:
        return ""
    if term.startswith("如"):
        term = term[1:].strip()
    if "合并近义术语" in term:
        term = term.split("合并近义术语", 1)[-1].strip(":： ")
    return term
