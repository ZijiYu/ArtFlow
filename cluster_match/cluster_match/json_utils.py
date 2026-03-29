from __future__ import annotations

import csv
import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from .categories import (
    ACADEMIC_V2_SCHEMA,
    LEGACY_CATEGORY_SPECS,
    RELEVANCE_VALUES,
    SIMPLE_V1_FIELD_SPECS,
    default_academic_leaf,
    default_entry,
    default_schema,
)


SUBJECTIVE_KEYWORD_EXACT = {
    "富丽",
    "富贵气象",
    "清雅",
    "浓丽清雅",
    "工笔细腻",
    "质感突出",
    "哲理意蕴",
    "美学追求",
    "绚烂之极归于平淡",
    "衰中见盛",
    "艺术价值",
    "文化意义",
    "特别吸引人",
    "特别的艺术特色",
}

SUBJECTIVE_KEYWORD_CONTAINS = (
    "浓丽清雅",
    "美学追求",
    "哲理意蕴",
    "艺术价值",
    "文化意义",
)

PROFESSIONAL_TERMS = tuple(
    sorted(
        {
            "斧劈皴",
            "披麻皴",
            "荷叶皴",
            "解索皴",
            "云头皴",
            "折带皴",
            "米点皴",
            "雨点皴",
            "牛毛皴",
            "白描",
            "双钩",
            "勾勒",
            "点染",
            "罩染",
            "渲染",
            "泼墨",
            "破墨",
            "积墨",
            "焦墨",
            "宿墨",
            "淡墨",
            "浓墨",
            "干笔",
            "湿笔",
            "侧锋",
            "中锋",
            "逆锋",
            "顺锋",
            "正锋",
            "皴擦",
            "没骨法",
            "没骨",
            "界画",
            "工笔",
            "写意",
            "工写结合",
            "院体画",
            "文人画",
            "青绿",
            "浅绛",
            "水墨",
            "重彩",
            "淡彩",
            "设色",
            "石青",
            "石绿",
            "花青",
            "藤黄",
            "朱砂",
            "赭石",
            "三远法",
            "高远法",
            "平远法",
            "深远法",
            "高远",
            "平远",
            "深远",
            "留白",
            "计白当黑",
            "气韵生动",
            "以形写神",
            "托物言志",
            "太湖石",
        },
        key=len,
        reverse=True,
    )
)

GENERAL_SEPARATOR_RE = re.compile(r"[，,；;。:：/|·•、\n\r\t]+")
CONNECTOR_SEPARATOR_RE = re.compile(r"(?:以及|并且|并与|与其|与|及|和|兼有|兼具|并|或)")
VERB_SEPARATOR_RE = re.compile(
    r"(?:描绘|描写|刻画|表现出|表现|勾勒|点染|渲染|采用|使用|运用|通过|形成|营造|体现|突出|展现|呈现|配合|衬托|暗示|说明)"
)
BRACKET_RE = re.compile(r"[“”\"'《》〈〉【】\[\]（）()]")
WHITESPACE_RE = re.compile(r"\s+")
LEADING_FILLER_RE = re.compile(r"^(?:使用了|使用|采用了|采用|运用|通过|以|将|把|对|其|并以|并将|多以|常以|再以|来|作|为)+")
TRAILING_FILLER_RE = re.compile(r"(?:为主|之一|特点|特征|方式|手法|形式|关系|内容)+$")
SIMPLE_V1_GLOSSARY_FIELDS = {"技法", "构图", "题材", "形制", "材质", "设色方式"}

SIMPLE_V1_MAX_ITEMS = {
    "画名": 1,
    "作者": 1,
    "朝代": 1,
    "技法": None,
    "构图": None,
    "题材": None,
    "形制": None,
    "材质": None,
    "设色方式": None,
    "题跋": None,
    "印章": None,
}

SIMPLE_V1_TECHNIQUE_HINTS = ("皴", "描", "勾", "染", "点", "墨", "笔", "白描", "没骨", "工笔", "写意", "界画")
SIMPLE_V1_COMPOSITION_HINTS = ("构图", "布局", "留白", "全景", "纵深", "三段", "高远", "平远", "深远", "层次", "空间")
SIMPLE_V1_COLOR_HINTS = ("青绿", "浅绛", "水墨", "重彩", "淡彩", "设色", "朱砂", "石青", "石绿", "花青", "赭石", "藤黄")
SIMPLE_V1_TECHNIQUE_STRONG_HINTS = ("皴", "描", "勾", "白描", "没骨", "工笔", "写意", "界画")
SIMPLE_V1_TECHNIQUE_WEAK_HINTS = ("染", "点", "墨", "笔")
SIMPLE_V1_FORMAT_HINTS = ("立轴", "手卷", "长卷", "册页", "扇面", "横披", "中堂", "屏风", "卷", "轴")
SIMPLE_V1_MATERIAL_HINTS = ("绢本", "纸本", "绫本", "绢", "纸", "绫", "绢素")
SIMPLE_V1_SUBJECT_HINTS = (
    "山水",
    "花鸟",
    "人物",
    "仕女",
    "走兽",
    "草虫",
    "博古",
    "佛教",
    "宗教",
    "历史故事",
    "人物画",
    "花鸟画",
    "山水画",
)


def strip_code_fences(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines:
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines.pop()
        cleaned = "\n".join(lines).strip()
    return cleaned


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fences(text)
    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    for index, char in enumerate(cleaned):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(cleaned[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("Model response does not contain a valid JSON object.")


def _normalize_item(item: dict[str, Any]) -> dict[str, str]:
    keyword = str(item.get("关键词", "")).strip()
    relevance = str(item.get("相关性", "")).strip()
    source_sentence = str(item.get("原句", "")).strip()

    if relevance not in RELEVANCE_VALUES:
        relevance = "不相关"

    if relevance == "不相关":
        return {"关键词": "", "相关性": "不相关", "原句": "不相关"}
    if not source_sentence:
        source_sentence = "未涉及"
    return {"关键词": keyword, "相关性": relevance, "原句": source_sentence}


def _is_subjective_keyword(keyword: str) -> bool:
    compact = keyword.replace(" ", "").strip()
    if compact in SUBJECTIVE_KEYWORD_EXACT:
        return True
    return any(token in compact for token in SUBJECTIVE_KEYWORD_CONTAINS)


def _clean_atomic_part(text: str) -> str:
    cleaned = BRACKET_RE.sub("", str(text))
    cleaned = WHITESPACE_RE.sub("", cleaned)
    cleaned = LEADING_FILLER_RE.sub("", cleaned)
    cleaned = TRAILING_FILLER_RE.sub("", cleaned)
    cleaned = cleaned.strip("，,；;。:：/|、·•- ")
    cleaned = cleaned.strip()
    if cleaned in {"未涉及", "不相关"}:
        return ""
    return cleaned


def _normalize_term_key(text: str) -> str:
    return WHITESPACE_RE.sub("", str(text or "")).strip()


@lru_cache(maxsize=1)
def _load_glossary_terms() -> tuple[set[str], tuple[str, ...]]:
    glossary_path = Path(__file__).resolve().parent.parent / "术语表.csv"
    terms: list[str] = []
    seen: set[str] = set()

    if not glossary_path.exists():
        return set(), tuple()

    with glossary_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            for cell in row:
                term = _clean_atomic_part(cell)
                if not term:
                    continue
                key = _normalize_term_key(term)
                if not key or key in seen:
                    continue
                seen.add(key)
                terms.append(term)

    return seen, tuple(terms)


def _is_glossary_valid_term(term: str) -> bool:
    normalized = _normalize_term_key(term)
    if not normalized:
        return False

    glossary_keys, glossary_terms = _load_glossary_terms()
    if normalized in glossary_keys:
        return True

    if len(normalized) < 2:
        return False

    return any(normalized in _normalize_term_key(glossary_term) for glossary_term in glossary_terms)


def _professional_term_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for term in PROFESSIONAL_TERMS:
        start = 0
        while True:
            index = text.find(term, start)
            if index < 0:
                break
            end = index + len(term)
            if any(not (end <= existing_start or index >= existing_end) for existing_start, existing_end, _ in spans):
                start = index + 1
                continue
            spans.append((index, end, term))
            start = end
    spans.sort(key=lambda item: item[0])
    return spans


def _split_general_fragment(text: str) -> list[str]:
    fragment = _clean_atomic_part(text)
    if not fragment:
        return []

    fragment = GENERAL_SEPARATOR_RE.sub("|", fragment)
    fragment = VERB_SEPARATOR_RE.sub("|", fragment)
    fragment = CONNECTOR_SEPARATOR_RE.sub("|", fragment)

    parts: list[str] = []
    for chunk in fragment.split("|"):
        token = _clean_atomic_part(chunk)
        if not token:
            continue
        if len(token) == 1:
            continue
        parts.append(token)
    return parts


def split_atomic_facts(text: str) -> list[str]:
    raw_text = _clean_atomic_part(text)
    if not raw_text:
        return []

    spans = _professional_term_spans(raw_text)
    if not spans:
        parts = _split_general_fragment(raw_text)
        return parts or [raw_text]

    items: list[str] = []
    cursor = 0
    for start, end, term in spans:
        if start > cursor:
            items.extend(_split_general_fragment(raw_text[cursor:start]))
        items.append(term)
        cursor = end
    if cursor < len(raw_text):
        items.extend(_split_general_fragment(raw_text[cursor:]))

    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = _clean_atomic_part(item)
        if not cleaned or _is_subjective_keyword(cleaned):
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped or [raw_text]


def _normalize_legacy_result(raw: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    normalized = default_schema("legacy")

    for category, _ in LEGACY_CATEGORY_SPECS:
        value = raw.get(category)
        items: list[dict[str, str]] = []

        if isinstance(value, dict):
            items = [_normalize_item(value)]
        elif isinstance(value, list):
            for entry in value:
                if isinstance(entry, dict):
                    items.append(_normalize_item(entry))

        if not items:
            normalized[category] = [default_entry()]
            continue

        relevant_items = [item for item in items if item["相关性"] in {"强相关", "弱相关"}]
        relevant_items = [item for item in relevant_items if not _is_subjective_keyword(item["关键词"])]
        if relevant_items:
            seen: set[tuple[str, str, str]] = set()
            deduped: list[dict[str, str]] = []
            for item in relevant_items:
                key = (item["关键词"], item["相关性"], item["原句"])
                if key in seen:
                    continue
                seen.add(key)
                deduped.append(item)
            normalized[category] = deduped
        else:
            normalized[category] = [default_entry()]

    return dict(normalized)


def _normalize_academic_leaf(node: Any) -> dict[str, Any]:
    if not isinstance(node, dict):
        return default_academic_leaf()

    relevance = str(node.get("相关性", "")).strip()
    if relevance not in RELEVANCE_VALUES:
        relevance = "不相关"

    raw_items = node.get("要素列表", [])
    if isinstance(raw_items, str):
        raw_items = [raw_items]
    elif not isinstance(raw_items, list):
        raw_items = []

    facts: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        for atomic_item in split_atomic_facts(str(item)):
            if relevance == "不相关" and atomic_item in {"不相关", "未涉及"}:
                continue
            if _is_subjective_keyword(atomic_item):
                continue
            if atomic_item in seen:
                continue
            seen.add(atomic_item)
            facts.append(atomic_item)

    if relevance == "不相关":
        return {"相关性": "不相关", "要素列表": []}

    if not facts:
        return {"相关性": "不相关", "要素列表": []}

    return {"相关性": relevance, "要素列表": facts}


def _normalize_academic_result(raw: dict[str, Any]) -> dict[str, Any]:
    normalized = default_schema("academic_v2")

    for group_name, leaf_specs in ACADEMIC_V2_SCHEMA.items():
        raw_group = raw.get(group_name, {})
        if not isinstance(raw_group, dict):
            raw_group = {}
        for leaf_name in leaf_specs:
            normalized[group_name][leaf_name] = _normalize_academic_leaf(raw_group.get(leaf_name))

    return dict(normalized)


def _coerce_simple_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        candidates: list[str] = []
        for key in ("要素列表", "关键词", "原句", "value", "values"):
            if key not in value:
                continue
            extracted = _coerce_simple_values(value[key])
            if extracted:
                candidates.extend(extracted)
        return candidates
    if isinstance(value, list):
        values: list[str] = []
        for item in value:
            values.extend(_coerce_simple_values(item))
        return values
    return []


def _filter_simple_candidates(field_name: str, candidates: list[str]) -> list[str]:
    if field_name == "技法":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_TECHNIQUE_HINTS)]
        return filtered or candidates

    if field_name == "构图":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_COMPOSITION_HINTS)]
        return filtered or candidates

    if field_name == "设色方式":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_COLOR_HINTS)]
        filtered = [item for item in filtered if item != "设色"] or filtered
        return filtered or candidates

    if field_name == "形制":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_FORMAT_HINTS)]
        return filtered or candidates

    if field_name == "材质":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_MATERIAL_HINTS)]
        return filtered or candidates

    if field_name == "题材":
        filtered = [item for item in candidates if any(token in item for token in SIMPLE_V1_SUBJECT_HINTS)]
        return filtered or candidates

    return candidates


def _normalize_simple_field(field_name: str, raw_value: Any) -> list[str]:
    max_items = SIMPLE_V1_MAX_ITEMS[field_name]
    values: list[str] = []
    seen: set[str] = set()
    candidates: list[str] = []

    for raw_item in _coerce_simple_values(raw_value):
        atomic_items = split_atomic_facts(raw_item)
        if not atomic_items:
            atomic_items = [_clean_atomic_part(raw_item)]
        for item in atomic_items:
            cleaned = _clean_atomic_part(item)
            if not cleaned:
                continue
            if cleaned in {"未涉及", "不相关"}:
                continue
            if _is_subjective_keyword(cleaned):
                continue
            if cleaned in seen:
                continue
            seen.add(cleaned)
            candidates.append(cleaned)

    candidates = _filter_simple_candidates(field_name, candidates)
    if field_name in SIMPLE_V1_GLOSSARY_FIELDS:
        candidates = [item for item in candidates if _is_glossary_valid_term(item)]

    if field_name == "技法":
        candidates.sort(
            key=lambda item: (
                not any(token in item for token in SIMPLE_V1_TECHNIQUE_STRONG_HINTS),
                not any(token in item for token in SIMPLE_V1_TECHNIQUE_WEAK_HINTS),
                any(token in item for token in SIMPLE_V1_COLOR_HINTS),
                -len(item),
            )
        )
    elif field_name == "构图":
        candidates.sort(
            key=lambda item: (
                not any(token in item for token in SIMPLE_V1_COMPOSITION_HINTS),
                -len(item),
            )
        )
    elif field_name == "设色方式":
        candidates.sort(
            key=lambda item: (
                not any(token in item for token in SIMPLE_V1_COLOR_HINTS),
                item == "设色",
                len(item),
            )
        )

    for candidate in candidates:
        values.append(candidate)
        if max_items is not None and len(values) >= max_items:
            break

    return values


def _normalize_simple_result(raw: dict[str, Any]) -> dict[str, list[str]]:
    normalized = default_schema("simple_v1")
    for field_name, _ in SIMPLE_V1_FIELD_SPECS:
        normalized[field_name] = _normalize_simple_field(field_name, raw.get(field_name))
    return dict(normalized)


def normalize_result(raw: dict[str, Any], schema_profile: str = "legacy") -> dict[str, Any]:
    if schema_profile == "legacy":
        return _normalize_legacy_result(raw)
    if schema_profile == "simple_v1":
        return _normalize_simple_result(raw)
    if schema_profile == "academic_v2":
        return _normalize_academic_result(raw)
    raise ValueError(f"Unsupported schema profile: {schema_profile}")
