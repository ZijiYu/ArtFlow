from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.slots_v2.meta_loader import load_context_meta, merge_meta


class MetaLoaderTests(unittest.TestCase):
    def test_load_context_meta_extracts_requested_sections(self) -> None:
        markdown = """# Dynamic Ontology Context

## Domain Profile [2026-03-21 16:08:56]
- {"domain":"guohua","name":"溪山行旅图","category":"山水画","knowledge_background":["北宋山水画"]}

## Post-RAG Text Extraction [2026-03-21 16:09:13]
- {"term":"雨点皴","description":"以点为主的皴法"}

## Ontology Updates [2026-03-21 16:10:41]
- `雨点皴` is-a `皴法`
"""
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "context.md"
            path.write_text(markdown, encoding="utf-8")
            meta = load_context_meta(str(path))

        self.assertEqual("guohua", meta["system_metadata"]["domain"])
        self.assertEqual("溪山行旅图", meta["domain_profile"]["name"])
        self.assertEqual("雨点皴", meta["post_rag_text_extraction"][0]["term"])
        self.assertIn("皴法", meta["ontology_updates"][0])

    def test_merge_meta_keeps_nested_fields(self) -> None:
        base = {"system_metadata": {"dynasty": "北宋", "artist": "范宽"}}
        override = {"system_metadata": {"artist": "传范宽"}, "note": "局部图"}
        merged = merge_meta(base, override)
        self.assertEqual("北宋", merged["system_metadata"]["dynasty"])
        self.assertEqual("传范宽", merged["system_metadata"]["artist"])
        self.assertEqual("局部图", merged["note"])


if __name__ == "__main__":
    unittest.main()
