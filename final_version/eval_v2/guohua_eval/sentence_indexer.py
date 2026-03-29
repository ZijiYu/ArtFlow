from __future__ import annotations

import re
from typing import List


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])\s*")
TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip())]
    return [part for part in parts if part]


def build_indexed_text(text: str) -> List[dict]:
    return [{"sentence_id": idx, "text": sentence} for idx, sentence in enumerate(split_sentences(text))]


def estimate_token_count(text: str) -> int:
    return len(TOKEN_RE.findall(text))
