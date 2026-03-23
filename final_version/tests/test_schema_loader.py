from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.slots_v2.schema_loader import extract_controlled_vocabulary, load_slot_schemas


class SchemaLoaderTests(unittest.TestCase):
    def test_extract_controlled_vocabulary_keeps_core_terms(self) -> None:
        terms = extract_controlled_vocabulary(
            "雨点皴",
            "雨点皴是中国画中的皴法，包含钉头皴、芝麻皴和豆瓣皴。合并近义术语：范宽。",
        )
        self.assertIn("雨点皴", terms)
        self.assertIn("钉头皴", terms)
        self.assertIn("芝麻皴", terms)
        self.assertIn("豆瓣皴", terms)
        self.assertIn("范宽", terms)

    def test_load_slot_schemas_reads_jsonl(self) -> None:
        payload = {
            "slot_name": "皴法",
            "slot_term": "雨点皴",
            "description": "包含钉头皴与芝麻皴。",
            "specific_questions": ["它如何表现北方山石？"],
            "metadata": {"confidence": 0.9},
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "slots.jsonl"
            path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")
            slots = load_slot_schemas(str(path))
        self.assertEqual(1, len(slots))
        self.assertEqual("皴法", slots[0].slot_name)
        self.assertIn("雨点皴", slots[0].controlled_vocabulary)


if __name__ == "__main__":
    unittest.main()

