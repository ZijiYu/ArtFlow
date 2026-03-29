from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_HEADING_PATTERN = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def load_context_meta(path: str) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        raise FileNotFoundError(f"Context file not found: {path}")

    text = file_path.read_text(encoding="utf-8")
    sections = _parse_markdown_sections(text)

    domain_profile = _first_json_item(sections.get("Domain Profile", []))
    post_rag_text_extraction = _json_items(sections.get("Post-RAG Text Extraction", []))
    rag_cache = _json_items(sections.get("RAG Cache", []))
    ontology_updates = _bullet_items(sections.get("Ontology Updates", []))
    downstream_updates = _json_items(sections.get("Downstream Updates", []))
    closed_loop_notes = _bullet_items(sections.get("Closed-loop Notes", []))
    round_memories = _json_items(sections.get("Round Memories", []))

    system_metadata: dict[str, Any] = {}
    if isinstance(domain_profile, dict):
        for key in ("domain", "name", "category", "subject", "scene_summary", "knowledge_background", "image_details"):
            if key in domain_profile:
                system_metadata[key] = domain_profile[key]

    return {
        "system_metadata": system_metadata,
        "domain_profile": domain_profile if isinstance(domain_profile, dict) else {},
        "post_rag_text_extraction": post_rag_text_extraction,
        "rag_cache": rag_cache,
        "ontology_updates": ontology_updates,
        "downstream_updates": downstream_updates,
        "closed_loop_notes": closed_loop_notes,
        "round_memories": round_memories,
        "context_source": str(file_path),
    }


def merge_meta(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_meta(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_markdown_sections(text: str) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_title = ""
    current_level = 10

    for line in text.splitlines():
        match = _HEADING_PATTERN.match(line)
        if match:
            hashes, raw_title = match.groups()
            level = len(hashes)
            title = re.sub(r"\s*\[[^\]]+\]\s*$", "", raw_title).strip()
            current_title = title
            current_level = level
            sections.setdefault(current_title, [])
            continue

        if not current_title:
            continue
        sections[current_title].append(line)

    return sections


def _first_json_item(lines: list[str]) -> Any:
    for item in _json_items(lines):
        return item
    return {}


def _json_items(lines: list[str]) -> list[Any]:
    items: list[Any] = []
    for line in lines:
        payload = line.strip()
        if payload.startswith("- "):
            payload = payload[2:].strip()
        if not payload.startswith("{"):
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        items.append(parsed)
    return items


def _bullet_items(lines: list[str]) -> list[str]:
    items: list[str] = []
    for line in lines:
        payload = line.strip()
        if not payload:
            continue
        if payload.startswith("- "):
            payload = payload[2:].strip()
        items.append(payload)
    return items
