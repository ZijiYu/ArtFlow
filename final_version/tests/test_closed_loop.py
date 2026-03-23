from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.slots_v2.closed_loop import ClosedLoopCoordinator
from src.slots_v2.meta_loader import load_context_meta
from src.slots_v2.new_api_client import NewAPIClient
from src.slots_v2.models import CrossValidationIssue, CrossValidationResult, DialogueState, PipelineConfig, SlotSchema, SpawnTask


class ClosedLoopTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = ClosedLoopCoordinator(
            slots_config=PipelineConfig(output_dir="artifacts_test"),
        )
        self.slot = SlotSchema(
            slot_name="皴法",
            slot_term="雨点皴",
            description="用于表现北方山石质感。",
            specific_questions=["它如何表现北方山石？"],
            metadata={"confidence": 0.9, "source_id": "0"},
            controlled_vocabulary=["雨点皴", "北方山石"],
        )

    def test_prepare_input_image_resizes_before_bootstrap(self) -> None:
        try:
            from PIL import Image
        except ModuleNotFoundError:
            self.skipTest("Pillow not installed")

        with tempfile.TemporaryDirectory() as tmpdir:
            image_path = Path(tmpdir) / "large.png"
            Image.new("RGB", (100, 100), color="white").save(image_path)
            coordinator = ClosedLoopCoordinator(
                slots_config=PipelineConfig(max_pixel=2_500, output_dir="artifacts_test"),
            )
            prepared = coordinator._prepare_input_image(str(image_path))
            self.assertTrue(prepared.was_resized)
            self.assertLessEqual(prepared.prepared_pixels or 0, 2_500)
            self.assertTrue(Path(prepared.path).exists())

    def test_build_perception_config_uses_closed_loop_api_settings(self) -> None:
        coordinator = ClosedLoopCoordinator(
            slots_config=PipelineConfig(output_dir="artifacts_test"),
            api_client=NewAPIClient(
                api_key="test-key",
                base_url="https://api.openai.com/v1",
                model="gpt-4.1-mini",
                timeout=30,
            ),
        )
        config = coordinator._build_perception_config(
            context_path=Path("/tmp/context.md"),
            rag_search_record_path=Path("/tmp/rag.md"),
            llm_chat_record_path=Path("/tmp/chat.jsonl"),
            output_path=Path("/tmp/slots.jsonl"),
        )
        self.assertEqual("https://api.openai.com/v1", config.base_url)
        self.assertEqual("gpt-4.1-mini", config.judge_model)
        self.assertEqual("text-embedding-3-small", config.embedding_model)

    def test_build_downstream_payload_contains_reflection_context(self) -> None:
        task = SpawnTask(
            slot_name="皴法",
            reason="specific_question_unanswered",
            prompt_focus="它如何表现北方山石？",
            rag_terms=["雨点皴"],
            priority=4,
        )
        fake_result = SimpleNamespace(
            routing=SimpleNamespace(
                action="SPAWN_COT",
                convergence_reason="仍需补证据",
                removed_questions=["重复问题"],
                merged_duplicates=["重复描述已合并"],
            ),
            dialogue_state=DialogueState(
                resolved_questions=["已解决问题"],
                unresolved_questions=["它如何表现北方山石？"],
            ),
            cross_validation=CrossValidationResult(
                issues=[
                    CrossValidationIssue(
                        issue_type="question_gap",
                        severity="medium",
                        slot_names=["皴法"],
                        detail="皴法 尚未充分回答问题：它如何表现北方山石？",
                    )
                ],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
        )
        payload = self.coordinator._build_downstream_payload(
            task=task,
            slot_schemas=[self.slot],
            meta={
                "domain_profile": {"name": "溪山行旅图"},
                "ontology_updates": ["- `雨点皴` is-a `皴法`"],
                "post_rag_text_extraction": [{"term": "雨点皴", "description": "点状皴法"}],
            },
            slots_result=fake_result,
            external_rag={
                "queries": ["北方山石", "雨点皴"],
                "documents": [{"term": "北方山石", "description": "山石相关外部证据", "text_evidence": ["证据A"]}],
            },
        )
        self.assertEqual("溪山行旅图", payload["painting_profile"]["name"])
        self.assertEqual("SPAWN_COT", payload["reflection_context"]["routing_action"])
        self.assertTrue(any(item.startswith("issue=") for item in payload["extra_constraints"]))
        self.assertEqual(["北方山石", "雨点皴"], payload["external_rag_queries"])
        self.assertTrue(any(item.startswith("downstream_rag_query=") for item in payload["extra_constraints"]))
        self.assertEqual(2, len(payload["rag_documents"]))

    def test_apply_downstream_response_updates_slots_and_meta(self) -> None:
        response = {
            "status": "applied",
            "merge_candidates": [
                {
                    "target_slot_term": "雨点皴",
                    "description_append": "文字证据指出其通过密集短点强化山石坚硬感。",
                    "additional_questions": ["雨点皴与芝麻皴如何区分？", "在花卉（如紫蝶牡丹）的写生中，铁线描与兰叶描结合使用时如何分配主次？"],
                }
            ],
            "new_slots": [
                {
                    "slot_name": "题跋",
                    "slot_term": "范中立谿山行旅圖",
                    "description": "右上角题跋可作为作品识别线索。",
                    "specific_questions": ["题跋如何参与作品识别？"],
                    "metadata": {"confidence": 0.6, "source_id": "downstream"},
                }
            ],
            "ontology_updates": ["`范中立谿山行旅圖` is-a `题跋识读`"],
            "text_evidence_updates": [
                {
                    "term": "雨点皴",
                    "description": "文字证据补充其与北方山石质感相关。",
                    "text_evidence": ["相关文献提到北方山石的坚硬感。"],
                }
            ],
            "notes": ["仍需谨慎区分雨点皴与近似点皴。"],
        }
        slots, meta, applied = self.coordinator._apply_downstream_response(
            response=response,
            slot_schemas=[self.slot],
            meta={
                "domain_profile": {"name": "罗汉图", "category": "道释人物", "subject": "罗汉、侍者与如意", "scene_summary": "主尊持如意，侍者奉经函。"},
                "post_rag_text_extraction": [],
                "ontology_updates": [],
                "downstream_updates": [],
                "closed_loop_notes": [],
                "dialogue_turns": [],
            },
            task=SpawnTask(slot_name="皴法", reason="specific_question_unanswered", prompt_focus="它如何表现北方山石？"),
        )
        self.assertTrue(applied["changed"])
        self.assertEqual(2, len(slots))
        self.assertIn("补充证据", slots[0].description)
        self.assertIn("雨点皴与芝麻皴如何区分？", slots[0].specific_questions)
        self.assertTrue(all("紫蝶牡丹" not in question for question in slots[0].specific_questions))
        self.assertEqual(1, len(meta["post_rag_text_extraction"]))
        self.assertEqual(1, len(meta["ontology_updates"]))
        self.assertTrue(meta["downstream_updates"])
        self.assertTrue(meta["dialogue_turns"])

    def test_write_context_markdown_round_trips_new_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "context.md"
            self.coordinator._write_context_markdown(
                {
                    "domain_profile": {"name": "溪山行旅图"},
                    "post_rag_text_extraction": [{"term": "雨点皴", "description": "点皴"}],
                    "ontology_updates": ["`雨点皴` is-a `皴法`"],
                    "downstream_updates": [{"slot_name": "皴法", "status": "applied"}],
                    "closed_loop_notes": ["round 1 changed"],
                    "round_memories": [{"round_index": 1, "resolved_questions": ["问题A"]}],
                },
                path,
            )
            meta = load_context_meta(str(path))
            self.assertEqual("溪山行旅图", meta["domain_profile"]["name"])
            self.assertEqual("雨点皴", meta["post_rag_text_extraction"][0]["term"])
            self.assertEqual("`雨点皴` is-a `皴法`", meta["ontology_updates"][0])
            self.assertEqual("皴法", meta["downstream_updates"][0]["slot_name"])
            self.assertEqual("round 1 changed", meta["closed_loop_notes"][0])
            self.assertEqual(1, meta["round_memories"][0]["round_index"])


if __name__ == "__main__":
    unittest.main()
