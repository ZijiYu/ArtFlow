from __future__ import annotations

import re
from typing import List


SENTENCE_SPLIT_RE = re.compile(r"(?<=[。！？!?；;\n])\s+|(?<=[。！？!?；;])")


def split_sentences(text: str) -> List[str]:
    parts = [part.strip() for part in SENTENCE_SPLIT_RE.split(text.strip())]
    return [part for part in parts if part]


def build_indexed_text(text: str) -> List[dict]:
    sentences = split_sentences(text)
    return [{"sentence_id": idx, "text": sentence} for idx, sentence in enumerate(sentences)]
