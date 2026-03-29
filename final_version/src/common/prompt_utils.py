from __future__ import annotations

import re
from typing import Any

from ..cot_layer.models import CrossValidationResult, DomainCoTRecord, SlotSchema


def dedupe_texts(items: list[str]) -> list[str]:
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


def take(items: list[str], limit: int) -> list[str]:
    return items[: max(0, limit)]


def format_bullets(items: list[str], *, empty_text: str = "- 无") -> str:
    if not items:
        return empty_text
    return "\n".join(f"- {item}" for item in items)


def _compact_text(value: object, limit: int = 160) -> str:
    text = " ".join(str(value or "").strip().split())
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 3)].rstrip() + "..."


def _tail(items: object, limit: int) -> list[Any]:
    if not isinstance(items, list):
        return []
    if limit <= 0:
        return []
    return list(items[-limit:])


def _compact_text_list(items: object, *, limit: int, text_limit: int = 120) -> list[str]:
    values = [
        _compact_text(item, text_limit)
        for item in _tail(items, limit)
        if _compact_text(item, text_limit)
    ]
    return dedupe_texts(values)


def _compact_text_updates(items: object, *, limit: int = 4) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in _tail(items, limit):
        if not isinstance(item, dict):
            continue
        compact: dict[str, Any] = {}
        term = _compact_text(item.get("term", ""), 48)
        description = _compact_text(item.get("description", ""), 180)
        text_evidence = _compact_text_list(item.get("text_evidence", []), limit=2, text_limit=100)
        if term:
            compact["term"] = term
        if description:
            compact["description"] = description
        if text_evidence:
            compact["text_evidence"] = text_evidence
        if compact:
            results.append(compact)
    return results


def _compact_downstream_updates(items: object, *, limit: int = 3) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for item in _tail(items, limit):
        if not isinstance(item, dict):
            continue
        compact = {
            "slot_name": _compact_text(item.get("slot_name", ""), 40),
            "reason": _compact_text(item.get("reason", ""), 40),
            "focus": _compact_text(item.get("focus", ""), 140),
            "status": _compact_text(item.get("status", ""), 24),
            "resolved_questions": _compact_text_list(item.get("resolved_questions", []), limit=2, text_limit=100),
            "open_questions": _compact_text_list(item.get("open_questions", []), limit=2, text_limit=100),
            "notes": _compact_text_list(item.get("notes", []), limit=2, text_limit=100),
        }
        queries = []
        for query in _tail(item.get("search_queries", []), 2):
            if not isinstance(query, dict):
                continue
            query_text = _compact_text(query.get("query_text", ""), 48)
            if query_text:
                queries.append(query_text)
        if queries:
            compact["search_queries"] = dedupe_texts(queries)
        compact = {key: value for key, value in compact.items() if value}
        if compact:
            results.append(compact)
    return results


def _compact_dialogue_turns(items: object, *, limit: int = 4) -> list[dict[str, str]]:
    results: list[dict[str, str]] = []
    for item in _tail(items, limit):
        if not isinstance(item, dict):
            continue
        role = _compact_text(item.get("role", ""), 16)
        content = _compact_text(item.get("content", ""), 140)
        if role and content:
            results.append({"role": role, "content": content})
    return results


def background_knowledge(meta: dict) -> list[str]:
    profile = meta.get("domain_profile") or meta.get("system_metadata") or {}
    lines: list[str] = []
    knowledge = profile.get("knowledge_background", [])
    if isinstance(knowledge, list):
        lines.extend(str(item).strip() for item in knowledge if str(item).strip())
    for item in meta.get("retained_facts", [])[:8]:
        if not isinstance(item, dict):
            continue
        fact = " ".join(str(item.get("fact", "")).strip().split())
        slot_name = str(item.get("slot_name", "")).strip()
        if not fact:
            continue
        lines.append(f"{slot_name}：{fact}" if slot_name else fact)
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
    return take(dedupe_texts(lines), 10)


def _clean_slot_description(description: object) -> str:
    text = " ".join(str(description or "").strip().split())
    if not text:
        return ""
    text = re.sub(r"当前推进 term：[^。！？；]*[。！？；]?", " ", text)
    text = re.sub(r"当前轮仍需结合 `[^`]+` 补充：[^。！？；]*[。！？；]?", " ", text)
    text = text.replace("补充证据：", "")
    return " ".join(text.strip("；。 ").split())


def _slot_description_highlights(description: object, *, limit: int = 4) -> list[str]:
    cleaned = _clean_slot_description(description)
    if not cleaned:
        return []
    fragments = [
        _compact_text(fragment.strip(" ，,；;"), 220)
        for fragment in re.split(r"[。！？；]+", cleaned)
        if fragment.strip(" ，,；;")
    ]
    return take(dedupe_texts(fragments), limit)


def build_slot_summary_payload(slot_schemas: list[SlotSchema]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for slot in slot_schemas:
        description_highlights = _slot_description_highlights(slot.description)
        summary = {
            "slot_name": slot.slot_name,
            "slot_term": _compact_text(slot.slot_term, 48),
            "lifecycle": _compact_text(slot.metadata.get("lifecycle", "ACTIVE"), 16) or "ACTIVE",
            "slot_mode": _compact_text(slot.metadata.get("slot_mode", ""), 16),
            "description_highlights": description_highlights,
            "specific_questions": take(dedupe_texts([str(item).strip() for item in slot.specific_questions if str(item).strip()]), 5),
        }
        summaries.append({key: value for key, value in summary.items() if value})
    return summaries


def _slot_summary_map(meta: dict) -> dict[str, dict[str, Any]]:
    summaries = meta.get("final_slot_summaries", [])
    if not isinstance(summaries, list):
        return {}
    summary_map: dict[str, dict[str, Any]] = {}
    for item in summaries:
        if not isinstance(item, dict):
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        if not slot_name:
            continue
        summary_map[slot_name] = item
    return summary_map


def final_slot_coverage(outputs: list[DomainCoTRecord], meta: dict) -> list[dict[str, Any]]:
    summary_map = _slot_summary_map(meta)
    output_map = {output.slot_name: output for output in outputs}
    retained_by_slot: dict[str, list[str]] = {}
    for item in meta.get("retained_facts", []):
        if not isinstance(item, dict):
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        fact = _compact_text(item.get("fact", ""), 180)
        if not slot_name or not fact:
            continue
        retained_by_slot.setdefault(slot_name, []).append(fact)

    ordered_slot_names = list(summary_map)
    for output in outputs:
        if output.slot_name not in ordered_slot_names:
            ordered_slot_names.append(output.slot_name)

    coverage: list[dict[str, Any]] = []
    for slot_name in ordered_slot_names:
        summary = summary_map.get(slot_name, {})
        output = output_map.get(slot_name)
        retained_facts = take(dedupe_texts(retained_by_slot.get(slot_name, [])), 4)
        description_highlights = take(
            dedupe_texts([_compact_text(item, 180) for item in summary.get("description_highlights", []) if _compact_text(item, 180)]),
            4,
        )
        confirmed_points: list[str] = []
        if output is not None:
            confirmed_points.extend(
                [
                    str(item.support).strip()
                    for item in output.question_coverage
                    if item.answered and str(item.support).strip()
                ][:3]
            )
            confirmed_points.extend([item.observation for item in output.visual_anchoring[:2]])
            confirmed_points.extend(
                [
                    (
                        f"{item.term}：{item.explanation}"
                        if item.term and item.explanation
                        else item.term or item.explanation
                    )
                    for item in output.domain_decoding[:2]
                ]
            )
            confirmed_points.extend([item.insight for item in output.cultural_mapping[:2]])
        confirmed_points = take(dedupe_texts([_compact_text(item, 180) for item in confirmed_points if _compact_text(item, 180)]), 5)
        unresolved_points = take(
            dedupe_texts([_compact_text(item, 160) for item in (output.unresolved_points if output is not None else []) if _compact_text(item, 160)]),
            2,
        )
        must_include = take(dedupe_texts(retained_facts + description_highlights), 3)
        key_points = take(dedupe_texts(must_include + confirmed_points), 6)
        if not key_points and not unresolved_points:
            continue
        lifecycle = str(summary.get("lifecycle", "")).strip().upper() or "ACTIVE"
        coverage.append(
            {
                "slot_name": slot_name,
                "slot_term": _compact_text(
                    summary.get("slot_term", getattr(output, "slot_term", "") if output is not None else ""),
                    48,
                ),
                "lifecycle": lifecycle,
                "must_cover": True,
                "must_include_facts": must_include,
                "key_points": key_points,
                "unresolved_points": unresolved_points,
            }
        )
    return coverage


def slot_coverage_digest(outputs: list[DomainCoTRecord], meta: dict) -> str:
    sections: list[str] = []
    for item in final_slot_coverage(outputs, meta):
        fragments = list(item.get("must_include_facts", [])) + [
            point for point in item.get("key_points", []) if point not in item.get("must_include_facts", [])
        ]
        fragments = take(dedupe_texts(fragments), 4)
        if item.get("unresolved_points"):
            fragments.append("仍需保守说明：" + str(item["unresolved_points"][0]).strip())
        if not fragments:
            continue
        slot_term = str(item.get("slot_term", "")).strip() or str(item.get("slot_name", "")).strip()
        sections.append(
            f"从{item['slot_name']}看，可围绕{slot_term}这一线索理解，" + "；".join(fragments) + "。"
        )
    return "\n\n".join(sections)


def long_slot_digest(output: DomainCoTRecord) -> str:
    visual = take(
        dedupe_texts(
            [
                f"{item.observation}{f'（位置：{item.position}）' if item.position else ''}"
                for item in output.visual_anchoring
            ]
        ),
        8,
    )
    decoding = take(
        dedupe_texts(
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
    mapping = take(dedupe_texts([item.insight for item in output.cultural_mapping]), 6)
    lines = [
        f"### {output.slot_name}",
        f"- 槽位术语：{output.slot_term or output.slot_name}",
        "- 图像信息：",
        format_bullets(visual, empty_text="- 暂无稳定图像信息"),
        "- 分析与解码：",
        format_bullets(decoding, empty_text="- 暂无稳定分析结果"),
        "- 背景与意义：",
        format_bullets(mapping, empty_text="- 暂无稳定背景解释"),
    ]
    return "\n".join(lines)


def qa_analysis_digest(outputs: list[DomainCoTRecord]) -> str:
    sections: list[str] = []
    for output in outputs:
        answered_supports = [
            item
            for item in output.question_coverage
            if item.answered and str(item.support).strip()
        ]
        if not answered_supports:
            continue
        merged_supports = dedupe_texts([str(item.support).strip() for item in answered_supports])
        narrative = " ".join(merged_supports)
        sections.append(f"围绕{output.slot_name}，{narrative}")
    return "\n\n".join(sections)


def supplementary_appreciation_digest(outputs: list[DomainCoTRecord]) -> str:
    sections: list[str] = []
    for output in outputs:
        fragments = take(
            dedupe_texts(
                [
                    *[item.observation for item in output.visual_anchoring],
                    *[
                        (
                            f"{item.term}：{item.explanation}"
                            if item.term and item.explanation
                            else item.term or item.explanation
                        )
                        for item in output.domain_decoding
                    ],
                    *[item.insight for item in output.cultural_mapping],
                ]
            ),
            4,
        )
        if not fragments:
            continue
        sections.append(f"补充来看，{output.slot_name}可从{output.slot_term or output.slot_name}这一线索展开，" + "；".join(fragments))
    return "\n\n".join(sections)


def meta_payload(meta: dict) -> dict:
    round_memories = meta.get("round_memories", [])
    return {
        "system_metadata": meta.get("system_metadata", {}),
        "domain_profile": meta.get("domain_profile", {}),
        "retained_facts": _compact_retained_facts(meta.get("retained_facts", []), limit=6),
        "post_rag_text_extraction": _compact_text_updates(meta.get("post_rag_text_extraction", []), limit=4),
        "rag_cache": _compact_text_updates(meta.get("rag_cache", []), limit=4),
        "ontology_updates": _compact_text_list(meta.get("ontology_updates", []), limit=6, text_limit=120),
        "downstream_updates": _compact_downstream_updates(meta.get("downstream_updates", []), limit=3),
        "closed_loop_notes": _compact_text_list(meta.get("closed_loop_notes", []), limit=6, text_limit=120),
        "dialogue_turns": _compact_dialogue_turns(meta.get("dialogue_turns", []), limit=4),
        "memory_card": memory_card(round_memories),
        "round_memories": round_memories[-1:] if isinstance(round_memories, list) else [],
    }


def memory_card(round_memories: object) -> dict:
    if not isinstance(round_memories, list):
        return {}
    latest = next((item for item in reversed(round_memories) if isinstance(item, dict)), None)
    if latest is None:
        return {}

    info_gain = latest.get("info_gain", {}) if isinstance(latest.get("info_gain", {}), dict) else {}
    carry_over = latest.get("carry_over", {}) if isinstance(latest.get("carry_over", {}), dict) else {}
    cumulative = latest.get("cumulative_snapshot", {}) if isinstance(latest.get("cumulative_snapshot", {}), dict) else {}
    slot_gains = info_gain.get("slot_gains", []) if isinstance(info_gain.get("slot_gains", []), list) else []
    compact_slot_gains = []
    for item in slot_gains[:4]:
        if not isinstance(item, dict):
            continue
        compact_slot_gains.append(
            {
                "slot_name": item.get("slot_name", ""),
                "new_visual_anchoring": take(dedupe_texts(item.get("new_visual_anchoring", [])), 3),
                "new_domain_decoding": take(dedupe_texts(item.get("new_domain_decoding", [])), 3),
                "new_cultural_mapping": take(dedupe_texts(item.get("new_cultural_mapping", [])), 2),
                "new_answered_questions": take(dedupe_texts(item.get("new_answered_questions", [])), 3),
            }
        )

    return {
        "round_index": latest.get("round_index", 0),
        "previous_round_index": latest.get("previous_round_index", 0),
        "total_new_items": info_gain.get("total_new_items", 0),
        "new_resolved_questions": take(dedupe_texts(info_gain.get("new_resolved_questions", [])), 8),
        "new_unresolved_questions": take(dedupe_texts(info_gain.get("new_unresolved_questions", [])), 8),
        "new_issues": take(dedupe_texts(info_gain.get("new_issues", [])), 6),
        "carry_over_questions": take(dedupe_texts(carry_over.get("focus_questions", [])), 8),
        "carry_over_issues": take(dedupe_texts(carry_over.get("issues", [])), 6),
        "focus_slots": take(dedupe_texts(carry_over.get("focus_slots", [])), 6),
        "resolved_questions": take(dedupe_texts(cumulative.get("resolved_questions", [])), 8),
        "unresolved_questions": take(dedupe_texts(cumulative.get("unresolved_questions", [])), 8),
        "slot_gains": compact_slot_gains,
    }


def _compact_retained_facts(items: object, *, limit: int = 6) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []
    compacted: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        slot_name = _compact_text(item.get("slot_name", ""), 24)
        slot_term = _compact_text(item.get("slot_term", ""), 24)
        fact = _compact_text(item.get("fact", ""), 180)
        source = _compact_text(item.get("source", ""), 24)
        if not fact:
            continue
        marker = f"{slot_name}::{slot_term}::{fact}"
        if marker in seen:
            continue
        seen.add(marker)
        payload = {
            "slot_name": slot_name,
            "fact": fact,
            "source": source or "slot_description",
        }
        if slot_term:
            payload["slot_term"] = slot_term
        compacted.append(payload)
        if len(compacted) >= limit:
            break
    return compacted


def guardrail_notes(meta: dict, validation: CrossValidationResult) -> list[str]:
    notes = meta.get("closed_loop_notes", [])
    candidate_notes = [str(note).strip() for note in notes if isinstance(note, str)]
    candidate_notes.extend(issue.detail for issue in validation.issues if issue.severity != "low")
    filtered = [
        note
        for note in candidate_notes
        if any(keyword in note for keyword in ("冲突", "跨度", "误导", "不可辨", "不符", "待考", "无法"))
    ]
    return take(dedupe_texts(filtered), 6)
