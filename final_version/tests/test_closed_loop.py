from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.cot_layer.closed_loop import ClosedLoopCoordinator
from src.cot_layer.new_api_client import ChatResult
from src.cot_layer.new_api_client import NewAPIClient
from src.cot_layer.models import (
    CrossValidationIssue,
    CrossValidationResult,
    DecodingItem,
    DialogueState,
    DomainCoTRecord,
    EvidenceItem,
    MappingItem,
    PipelineConfig,
    PipelineResult,
    PreparedImage,
    QuestionCoverage,
    RoutingDecision,
    SlotSchema,
    SpawnTask,
)
from src.common.web_search_client import WebSearchHit


class FakeAPIClient:
    def __init__(self, *, enabled: bool = False, responses: list[str] | None = None) -> None:
        self.enabled = enabled
        self._responses = list(responses or [])
        self.calls: list[dict[str, object]] = []

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        image_path: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "temperature": temperature,
                "image_path": image_path,
                "model": model,
            }
        )
        content = self._responses.pop(0) if self._responses else None
        return ChatResult(
            content=content,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=model or "fake-model",
            endpoint="fake://chat",
            status_code=200,
            image_attached=bool(image_path),
            duration_ms=0.0,
        )


class FakeWebSearchClient:
    def __init__(self) -> None:
        self.enabled = True
        self.queries: list[str] = []
        self.urls: list[str] = []

    def search(self, query: str, *, top_k: int = 5) -> list[WebSearchHit]:
        self.queries.append(query)
        return [
            WebSearchHit(
                title="溪山行旅图馆藏说明",
                url="https://museum.example/xi-shan",
                snippet="介绍《溪山行旅图》与范宽、馆藏信息。",
                source="organic",
                position=1,
            ),
            WebSearchHit(
                title="博客聚合页",
                url="https://blog.example/post",
                snippet="转载多篇相关内容。",
                source="organic",
                position=2,
            ),
        ][:top_k]

    def fetch_page(self, url: str, *, max_chars: int = 6_000) -> dict[str, object]:
        self.urls.append(url)
        if "museum.example" in url:
            return {
                "url": url,
                "domain": "museum.example",
                "title": "溪山行旅图馆藏说明",
                "description": "作品馆藏与作者归属说明。",
                "content": "《溪山行旅图》通常归于范宽名下，现藏台北故宫博物院。",
            }
        return {}


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
        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "rag:",
                        '  endpoint: "http://example.com/rag"',
                        "  top_k: 9",
                        '  collection_name: "general_collection"',
                        '  info_collection_name: "image_blip"',
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            coordinator = ClosedLoopCoordinator(
                slots_config=PipelineConfig(output_dir="artifacts_test"),
                api_client=NewAPIClient(
                    api_key="test-key",
                    base_url="https://api.openai.com/v1",
                    model="gpt-4.1-mini",
                    timeout=30,
                    config_path=str(config_path),
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
        self.assertEqual("http://example.com/rag", config.rag_endpoint)
        self.assertEqual(9, config.rag_top_k)
        self.assertEqual("general_collection", config.rag_collection_name)
        self.assertEqual("image_blip", config.rag_info_collection_name)

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
                "retained_facts": [
                    {
                        "slot_name": "皴法",
                        "slot_term": "雨点皴",
                        "fact": "北方山石的坚硬感常通过密集短点来强化。",
                        "source": "slot_description",
                    },
                    {
                        "slot_name": "作者时代流派",
                        "slot_term": "范宽",
                        "fact": "范宽活动于北宋。",
                        "source": "slot_description",
                    },
                ],
                "rag_cache": [
                    {
                        "term": "北方山石",
                        "description": "与当前作品山石结构相关的缓存证据",
                        "text_evidence": ["证据A"],
                        "source_slot": "皴法",
                        "origin_stage": "downstream_rag",
                    }
                ],
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
        self.assertEqual(["北方山石"], payload["cached_terms"])
        self.assertEqual("北方山石", payload["cached_documents"][0]["term"])
        self.assertEqual(1, len(payload["retained_facts"]))
        self.assertIn("北方山石的坚硬感", payload["retained_facts"][0]["fact"])
        self.assertTrue(any(item.startswith("retained_fact=") for item in payload["extra_constraints"]))
        self.assertTrue(any(item.startswith("downstream_rag_query=") for item in payload["extra_constraints"]))
        self.assertEqual(3, len(payload["rag_documents"]))

    def test_reuse_existing_bootstrap_copies_files_for_ablation(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_slots = tmp_path / "source_slots.jsonl"
            source_context = tmp_path / "source_context.md"
            source_slots.write_text('{"slot_name":"皴法"}\n', encoding="utf-8")
            source_context.write_text("# context\n", encoding="utf-8")
            coordinator = ClosedLoopCoordinator(
                slots_config=PipelineConfig(output_dir="artifacts_test"),
                closed_loop_config=self.coordinator.closed_loop_config.__class__(
                    output_dir=str(tmp_path / "outputs"),
                    disable_preception_layer=True,
                    bootstrap_slots_file=str(source_slots),
                    bootstrap_context_file=str(source_context),
                ),
            )
            bootstrap = coordinator._reuse_existing_bootstrap(run_dir=tmp_path / "run")

            self.assertEqual(source_slots.read_text(encoding="utf-8"), Path(bootstrap["slots_file"]).read_text(encoding="utf-8"))
            self.assertEqual(source_context.read_text(encoding="utf-8"), Path(bootstrap["context_file"]).read_text(encoding="utf-8"))

    def test_run_task_retrieval_uses_web_search_for_web_mode(self) -> None:
        coordinator = ClosedLoopCoordinator(
            slots_config=PipelineConfig(
                output_dir="artifacts_test",
                enable_web_search=True,
                web_search_url="https://google.serper.dev/search",
                web_search_api_key="test-serper-key",
                web_search_top_k=5,
                web_search_fetch_top_n=1,
                web_search_use_llm_rerank=False,
            ),
            api_client=FakeAPIClient(enabled=False),
        )
        web_client = FakeWebSearchClient()
        task = SpawnTask(
            slot_name="作者时代流派",
            reason="round_table_follow_up",
            prompt_focus="这幅作品与范宽、李成、郭熙的关系应如何比较？",
            rag_terms=["范宽"],
            retrieval_mode="web",
            web_queries=["溪山行旅图 范宽 馆藏"],
            dispatch_target="downstream",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(coordinator, "_build_web_search_client", return_value=web_client):
                payload = coordinator._run_task_retrieval(
                    task=task,
                    image_path="/tmp/nonexistent.png",
                    run_dir=Path(tmpdir),
                    round_index=1,
                    task_index=1,
                )
        self.assertEqual(["溪山行旅图 范宽 馆藏"], payload["queries"])
        self.assertEqual(1, len(payload["documents"]))
        self.assertEqual("downstream_web", payload["documents"][0]["origin_stage"])
        self.assertEqual(["溪山行旅图 范宽 馆藏"], web_client.queries)
        self.assertEqual(["https://museum.example/xi-shan"], web_client.urls)

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
        self.assertTrue(any("北方山石质感" in item["fact"] for item in meta["retained_facts"]))
        self.assertGreaterEqual(applied["retained_facts_added"], 1)

    def test_apply_downstream_response_preserves_unconsumed_description_facts(self) -> None:
        slot = SlotSchema(
            slot_name="作者时代流派",
            slot_term="金大受",
            description=(
                "本作为南宋金大受所绘，其在宁波地区活跃，擅长罗汉与佛教人物画。 "
                "当前推进 term：南宋。 当前轮仍需结合 `南宋` 补充：需查阅馆藏。"
            ),
            specific_questions=["作者归属有哪些依据？"],
            metadata={"confidence": 0.8, "source_id": "0"},
            controlled_vocabulary=["金大受", "南宋"],
        )
        response = {
            "status": "applied",
            "merge_candidates": [
                {
                    "target_slot_name": "作者时代流派",
                    "description_append": "现藏东京国立博物馆，可与作者背景互证。",
                    "additional_questions": [],
                }
            ],
        }
        slots, meta, applied = self.coordinator._apply_downstream_response(
            response=response,
            slot_schemas=[slot],
            meta={
                "domain_profile": {"name": "十六罗汉图"},
                "post_rag_text_extraction": [],
                "ontology_updates": [],
                "downstream_updates": [],
                "closed_loop_notes": [],
                "dialogue_turns": [],
            },
            task=SpawnTask(slot_name="作者时代流派", reason="round_table_follow_up", prompt_focus="作者归属有哪些依据？"),
        )
        self.assertTrue(applied["changed"])
        self.assertEqual(1, len(slots))
        retained_facts = meta["retained_facts"]
        self.assertTrue(any("宁波地区活跃" in item["fact"] for item in retained_facts))
        self.assertTrue(any("东京国立博物馆" in item["fact"] for item in retained_facts))
        self.assertFalse(any("当前推进 term" in item["fact"] for item in retained_facts))
        self.assertFalse(any("当前轮仍需结合" in item["fact"] for item in retained_facts))

    def test_dedupe_spawn_tasks_merges_rephrased_followups(self) -> None:
        tasks = self.coordinator._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="是否有落款或印章可供辨认？",
                    rag_terms=["落款", "印章"],
                    priority=4,
                ),
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="是否有落款或相关印章可供辨认？",
                    rag_terms=["题跋", "印章"],
                    priority=5,
                ),
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="题跋书写者与画家有何关系？",
                    rag_terms=["题跋书写者"],
                    priority=4,
                ),
            ]
        )

        self.assertEqual(2, len(tasks))
        self.assertEqual("是否有落款或相关印章可供辨认？", tasks[0].prompt_focus)
        self.assertEqual(["题跋", "印章", "落款"], tasks[0].rag_terms)

    def test_has_unseen_followups_ignores_rephrased_history(self) -> None:
        seen_tasks = self.coordinator._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="是否有落款或印章可供辨认？",
                    rag_terms=["落款", "印章"],
                    priority=4,
                )
            ]
        )
        current_tasks = self.coordinator._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="是否有落款或相关印章可供辨认？",
                    rag_terms=["题跋", "印章"],
                    priority=5,
                )
            ]
        )
        distinct_tasks = self.coordinator._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="题跋诗文审美语言",
                    reason="round_table_follow_up",
                    prompt_focus="题跋书写者与画家有何关系？",
                    rag_terms=["题跋书写者"],
                    priority=4,
                )
            ]
        )

        self.assertFalse(self.coordinator._has_unseen_followups(current_tasks, seen_tasks))
        self.assertTrue(self.coordinator._has_unseen_followups(distinct_tasks, seen_tasks))

    def test_build_slot_questions_uses_new_fixed_slot_templates(self) -> None:
        composition_questions = self.coordinator._build_slot_questions(
            slot_name="构图/空间/布局",
            term="高远构图",
            slot_terms=["高远构图"],
            output=None,
            spawned_tasks=[],
        )
        material_questions = self.coordinator._build_slot_questions(
            slot_name="尺寸规格/材质形制/收藏地",
            term="绢本",
            slot_terms=["绢本"],
            output=None,
            spawned_tasks=[],
        )
        mood_questions = self.coordinator._build_slot_questions(
            slot_name="意境/题材/象征",
            term="山水画",
            slot_terms=["山水画"],
            output=None,
            spawned_tasks=[],
        )

        self.assertTrue(any("空间层次" in question or "布局经营" in question for question in composition_questions))
        self.assertTrue(any("尺寸" in question or "材质" in question for question in material_questions))
        self.assertTrue(any("意境" in question or "象征" in question for question in mood_questions))

    def test_advance_fixed_slot_progresses_term_and_creates_downstream_task(self) -> None:
        slot = SlotSchema(
            slot_name="画作背景",
            slot_term="十六罗汉图",
            description="围绕作品主干做背景分析。",
            specific_questions=["十六罗汉图如何界定当前作品主干？"],
            metadata={
                "confidence": 0.9,
                "source_id": "bootstrap",
                "slot_mode": "progressive",
                "used_terms": ["十六罗汉图"],
                "pending_terms": ["第十一尊者", "加诺迦伐蹉尊者"],
                "candidate_terms": ["十六罗汉图", "第十一尊者", "加诺迦伐蹉尊者"],
                "lifecycle": "ACTIVE",
            },
            controlled_vocabulary=["十六罗汉图"],
        )
        output = SimpleNamespace(
            question_coverage=[SimpleNamespace(question="十六罗汉图如何界定当前作品主干？", answered=True)],
            unresolved_points=["仍需进一步确认第十一尊者身份。"],
            retrieval_gain_terms=["第十一尊者", "加诺迦伐蹉尊者"],
            domain_decoding=[],
        )

        updated_slot, downstream_task, event = self.coordinator._advance_single_fixed_slot(
            slot=slot,
            output=output,
            spawned_tasks=[
                SpawnTask(
                    slot_name="画作背景",
                    reason="round_table_follow_up",
                    prompt_focus="第十一尊者与当前画面主尊的对应关系是什么？",
                    rag_terms=["第十一尊者"],
                    priority=5,
                )
            ],
            meta={"domain_profile": {"name": "十六罗汉图第十一尊者"}},
        )

        self.assertEqual("progressed", event)
        self.assertEqual("第十一尊者", updated_slot.slot_term)
        self.assertIn("第十一尊者", updated_slot.metadata["used_terms"])
        self.assertNotIn("第十一尊者", updated_slot.metadata["pending_terms"])
        self.assertEqual("slot_term_progression", downstream_task.reason)
        self.assertIn("第十一尊者", downstream_task.rag_terms[0])

    def test_advance_fixed_slot_skips_downstream_when_rag_verification_disabled(self) -> None:
        coordinator = ClosedLoopCoordinator(
            slots_config=PipelineConfig(output_dir="artifacts_test", enable_rag_verification=False),
        )
        slot = SlotSchema(
            slot_name="画作背景",
            slot_term="十六罗汉图",
            description="围绕作品主干做背景分析。",
            specific_questions=["十六罗汉图如何界定当前作品主干？"],
            metadata={
                "confidence": 0.9,
                "source_id": "bootstrap",
                "slot_mode": "progressive",
                "used_terms": ["十六罗汉图"],
                "pending_terms": ["第十一尊者"],
                "candidate_terms": ["十六罗汉图", "第十一尊者"],
                "lifecycle": "ACTIVE",
            },
            controlled_vocabulary=["十六罗汉图"],
        )
        output = SimpleNamespace(
            question_coverage=[SimpleNamespace(question="十六罗汉图如何界定当前作品主干？", answered=True)],
            unresolved_points=[],
            retrieval_gain_terms=["第十一尊者"],
            domain_decoding=[],
        )

        updated_slot, downstream_task, event = coordinator._advance_single_fixed_slot(
            slot=slot,
            output=output,
            spawned_tasks=[],
            meta={"domain_profile": {"name": "十六罗汉图第十一尊者"}},
        )

        self.assertEqual("progressed", event)
        self.assertEqual("第十一尊者", updated_slot.slot_term)
        self.assertIsNone(downstream_task)

    def test_advance_fixed_slot_closes_when_no_more_terms(self) -> None:
        slot = SlotSchema(
            slot_name="题跋诗文审美语言",
            slot_term="题跋",
            description="围绕题跋与审美语言做分析。",
            specific_questions=["题跋在当前作品中是否可见？"],
            metadata={
                "confidence": 0.7,
                "source_id": "bootstrap",
                "slot_mode": "enumerative",
                "used_terms": ["题跋"],
                "pending_terms": [],
                "candidate_terms": ["题跋"],
                "lifecycle": "ACTIVE",
            },
            controlled_vocabulary=["题跋"],
        )
        output = SimpleNamespace(
            question_coverage=[SimpleNamespace(question="题跋在当前作品中是否可见？", answered=True)],
            unresolved_points=[],
            retrieval_gain_terms=[],
            domain_decoding=[],
        )

        updated_slot, downstream_task, event = self.coordinator._advance_single_fixed_slot(
            slot=slot,
            output=output,
            spawned_tasks=[],
            meta={"domain_profile": {"name": "罗汉图"}},
        )

        self.assertEqual("closed", event)
        self.assertIsNone(downstream_task)
        self.assertEqual("CLOSED", updated_slot.metadata["lifecycle"])
        self.assertIn("没有更多可推进 term", updated_slot.metadata["lifecycle_reason"])

    def test_advance_fixed_slot_closes_when_same_term_repeats(self) -> None:
        slot = SlotSchema(
            slot_name="墨法设色技法",
            slot_term="披麻皴",
            description="围绕披麻皴做技法分析。",
            specific_questions=["披麻皴如何表现山体结构？"],
            metadata={
                "confidence": 0.82,
                "source_id": "bootstrap",
                "slot_mode": "progressive",
                "used_terms": ["披麻皴"],
                "pending_terms": ["墨色渲染"],
                "candidate_terms": ["披麻皴", "墨色渲染"],
                "repeat_guard_signatures": ["墨法设色技法::披麻皴"],
                "lifecycle": "ACTIVE",
            },
            controlled_vocabulary=["披麻皴"],
        )
        output = SimpleNamespace(
            question_coverage=[SimpleNamespace(question="披麻皴如何表现山体结构？", answered=True)],
            unresolved_points=["仍在重复讨论披麻皴的同一视觉证据。"],
            retrieval_gain_terms=["披麻皴"],
            domain_decoding=[],
        )

        updated_slot, downstream_task, event = self.coordinator._advance_single_fixed_slot(
            slot=slot,
            output=output,
            spawned_tasks=[],
            meta={"domain_profile": {"name": "溪山行旅图"}},
        )

        self.assertEqual("closed", event)
        self.assertIsNone(downstream_task)
        self.assertEqual("CLOSED", updated_slot.metadata["lifecycle"])
        self.assertIn("重复出现", updated_slot.metadata["lifecycle_reason"])

    def test_advance_fixed_slot_blocks_cross_slot_term_drift(self) -> None:
        slot = SlotSchema(
            slot_name="作者时代流派",
            slot_term="金大受",
            description="围绕作者与时代做分析。",
            specific_questions=["作者身份如何确认？"],
            metadata={
                "confidence": 0.9,
                "source_id": "bootstrap",
                "slot_mode": "enumerative",
                "used_terms": ["金大受"],
                "pending_terms": ["铁线描", "南宋", "人物画，道释画"],
                "candidate_terms": ["金大受", "铁线描", "南宋", "人物画，道释画"],
                "lifecycle": "ACTIVE",
            },
            controlled_vocabulary=["金大受", "南宋"],
        )
        output = SimpleNamespace(
            question_coverage=[SimpleNamespace(question="作者身份如何确认？", answered=True)],
            unresolved_points=[],
            retrieval_gain_terms=["铁线描"],
            domain_decoding=[SimpleNamespace(term="铁线描")],
        )

        updated_slot, downstream_task, event = self.coordinator._advance_single_fixed_slot(
            slot=slot,
            output=output,
            spawned_tasks=[],
            meta={"domain_profile": {"name": "十六罗汉图第十一尊者"}},
        )

        self.assertEqual("progressed", event)
        self.assertEqual("南宋", updated_slot.slot_term)
        self.assertNotIn("铁线描", updated_slot.metadata["pending_terms"])
        self.assertIsNotNone(downstream_task)

    def test_build_slot_questions_uses_slot_terms_phrase(self) -> None:
        questions = self.coordinator._build_slot_questions(
            slot_name="墨法设色技法",
            term="披麻皴",
            slot_terms=["披麻皴", "雨点皴"],
            output=None,
            spawned_tasks=[],
        )

        self.assertTrue(any("披麻皴、雨点皴" in question for question in questions))

    def test_match_slot_index_matches_secondary_slot_term(self) -> None:
        slots = [
            SlotSchema(
                slot_name="墨法设色技法",
                slot_term="披麻皴",
                description="围绕皴法分析。",
                specific_questions=[],
                metadata={"slot_terms": ["披麻皴", "雨点皴"]},
                controlled_vocabulary=["披麻皴", "雨点皴"],
            )
        ]

        self.assertEqual(0, self.coordinator._match_slot_index(slots, "雨点皴"))

    def test_run_omits_runtime_context_markdown_outputs(self) -> None:
        class FakeSlotsPipeline:
            def __init__(self, *, config: PipelineConfig, api_client: object) -> None:
                self.config = config
                self.api_client = api_client

            def run(self, image_path: str, meta: dict | None = None) -> PipelineResult:
                slot = SlotSchema(
                    slot_name="画作背景",
                    slot_term="溪山行旅图",
                    description="围绕作品背景展开。",
                    specific_questions=["这幅画的基本信息是什么？"],
                    metadata={"confidence": 0.8, "source_id": "bootstrap"},
                    controlled_vocabulary=["溪山行旅图"],
                )
                return PipelineResult(
                    image_path=image_path,
                    prepared_image=PreparedImage(path=image_path),
                    slot_schemas=[slot],
                    domain_outputs=[],
                    cross_validation=CrossValidationResult(
                        issues=[],
                        semantic_duplicates=[],
                        missing_points=[],
                        rag_terms=[],
                        removed_questions=[],
                    ),
                    routing=RoutingDecision(
                        action="PAUSE_COT",
                        rationale=["已无后续任务。"],
                        paused_slots=[],
                        spawned_tasks=[],
                        removed_questions=[],
                        merged_duplicates=[],
                        converged=True,
                        convergence_reason="complete",
                    ),
                    dialogue_state=DialogueState(
                        resolved_questions=[],
                        unresolved_questions=[],
                        converged=True,
                        convergence_reason="complete",
                    ),
                    cot_threads=[],
                    round_memory={"round_index": 1, "resolved_questions": []},
                    final_appreciation_prompt="最终赏析提示词",
                    api_logs=[],
                    execution_log=[],
                )

            def finalize_result(self, result: PipelineResult, *, meta: dict | None = None) -> PipelineResult:
                return result

            def save_result(self, result: PipelineResult, *, output_dir: str) -> dict[str, str]:
                run_dir = Path(output_dir) / "fake_run"
                run_dir.mkdir(parents=True, exist_ok=True)
                report_path = run_dir / "report.json"
                report_path.write_text(
                    json.dumps({"routing": asdict(result.routing)}, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                return {"run_dir": str(run_dir), "report": str(report_path)}

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            input_path.write_bytes(b"fake-image")
            bootstrap_slots = Path(tmpdir) / "bootstrap" / "slots.jsonl"
            bootstrap_slots.parent.mkdir(parents=True, exist_ok=True)
            bootstrap_slots.write_text(
                json.dumps(
                    {
                        "slot_name": "画作背景",
                        "slot_term": "溪山行旅图",
                        "description": "围绕作品背景展开。",
                        "specific_questions": ["这幅画的基本信息是什么？"],
                        "metadata": {"confidence": 0.8, "source_id": "bootstrap"},
                    },
                    ensure_ascii=False,
                )
                + "\n",
                encoding="utf-8",
            )
            bootstrap_context = Path(tmpdir) / "bootstrap" / "context.md"
            bootstrap_context.write_text("# Context\n", encoding="utf-8")
            coordinator = ClosedLoopCoordinator(
                slots_config=PipelineConfig(output_dir=tmpdir),
                closed_loop_config=self.coordinator.closed_loop_config.__class__(output_dir=tmpdir, max_closed_loop_rounds=1),
                slots_pipeline_factory=FakeSlotsPipeline,
            )

            with (
                patch.object(coordinator, "_prepare_input_image", return_value=PreparedImage(path=str(input_path))),
                patch.object(
                    coordinator,
                    "_run_perception_bootstrap",
                    return_value={"slots_file": bootstrap_slots, "context_file": bootstrap_context},
                ),
            ):
                result = coordinator.run(image_path=str(input_path), input_text="测试输入", meta={})

            run_dir = Path(result.run_dir)
            runtime_state_dir = run_dir / "runtime_state"
            self.assertTrue((runtime_state_dir / "slots_final.jsonl").exists())
            self.assertFalse((runtime_state_dir / "context_final.md").exists())
            self.assertFalse(any(runtime_state_dir.glob("context_round_*.md")))

            report = json.loads((run_dir / "closed_loop_report.json").read_text(encoding="utf-8"))
            self.assertNotIn("final_context_file", report)

    def test_build_closed_loop_final_appreciation_uses_cumulative_outputs(self) -> None:
        api_client = FakeAPIClient(
            enabled=True,
            responses=["## 赏析\n这是一段累计后的最终赏析。"],
        )
        coordinator = ClosedLoopCoordinator(
            slots_config=PipelineConfig(final_answer_model="gpt-4.1", output_dir="artifacts_test"),
            api_client=api_client,
        )
        author_slot = SlotSchema(
            slot_name="作者时代流派",
            slot_term="范宽",
            description="围绕作者与时代展开。",
            specific_questions=["作者身份如何确认？"],
            metadata={"confidence": 0.9},
            controlled_vocabulary=["范宽", "北宋"],
        )
        technique_slot = SlotSchema(
            slot_name="墨法设色技法",
            slot_term="雨点皴",
            description="围绕技法展开。",
            specific_questions=["山体技法如何体现？"],
            metadata={"confidence": 0.9},
            controlled_vocabulary=["雨点皴"],
        )
        latest_output = DomainCoTRecord(
            slot_name="墨法设色技法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面布满密集短点")],
            domain_decoding=[DecodingItem(term="雨点皴", explanation="通过密集短点表现山体质感。")],
            cultural_mapping=[MappingItem(insight="雨点皴强化山体厚重感。")],
            question_coverage=[
                QuestionCoverage(
                    question="山体技法如何体现？",
                    answered=True,
                    support="山石表面布满密集短点，显示范宽典型雨点皴。",
                )
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.94,
        )
        latest_result = PipelineResult(
            image_path="/tmp/demo.png",
            prepared_image=PreparedImage(path="/tmp/demo.png"),
            slot_schemas=[author_slot, technique_slot],
            domain_outputs=[latest_output],
            cross_validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            routing=RoutingDecision(
                action="SPAWN_COT",
                rationale=["仍有后续任务。"],
                paused_slots=[],
                spawned_tasks=[],
                removed_questions=[],
                merged_duplicates=[],
                converged=False,
                convergence_reason="not_converged",
            ),
            dialogue_state=DialogueState(),
            cot_threads=[],
            round_memory={},
            final_appreciation_prompt="",
            api_logs=[],
            execution_log=[],
        )
        meta = {
            "final_user_question": "请给出完整赏析。",
            "round_memories": [
                {
                    "round_index": 1,
                    "slots": [
                        {
                            "slot_name": "作者时代流派",
                            "slot_term": "范宽",
                            "visual_anchoring": [
                                {"observation": "画面巨山高耸，气势雄伟", "evidence": "主峰占据画面主体", "position": "中上部"}
                            ],
                            "domain_decoding": [
                                {"term": "北宋山水画", "explanation": "巨山主体与高远构图体现北宋山水典型风格。"}
                            ],
                            "cultural_mapping": [
                                {"insight": "范宽是北宋山水画代表画家，强调山川雄伟。"}
                            ],
                            "question_coverage": [
                                {
                                    "question": "作者身份如何确认？",
                                    "answered": True,
                                    "support": "高远构图、巨山主体与雨点皴共同指向范宽的典型风格。",
                                }
                            ],
                            "unresolved_points": [],
                            "generated_questions": [],
                            "confidence": 0.92,
                        }
                    ],
                    "issues": [],
                }
            ],
        }

        appreciation = coordinator._build_closed_loop_final_appreciation(
            slot_schemas=[author_slot, technique_slot],
            meta=meta,
            latest_result=latest_result,
        )

        self.assertEqual("## 赏析\n这是一段累计后的最终赏析。", appreciation)
        self.assertEqual(1, len(api_client.calls))
        user_prompt = str(api_client.calls[0]["user_prompt"])
        self.assertIn("作者时代流派", user_prompt)
        self.assertIn("墨法设色技法", user_prompt)
        self.assertIn("高远构图、巨山主体与雨点皴共同指向范宽的典型风格", user_prompt)
        self.assertIn("山石表面布满密集短点，显示范宽典型雨点皴", user_prompt)

    def test_merge_external_rag_into_meta_promotes_stable_facts(self) -> None:
        meta = {"rag_cache": [], "retained_facts": []}
        external_rag = {
            "documents": [
                {
                    "term": "金大受",
                    "description": "金大受为南宋前期在宁波地区活跃的道释画家。",
                    "text_evidence": [],
                    "source_id": "doc-author",
                    "origin_stage": "downstream_rag",
                    "source_slot": "作者时代流派",
                    "query": "金大受",
                },
                {
                    "term": "绢本",
                    "description": "绢本设色，纵120.5×横55.2厘米，现藏于东京国立博物馆。",
                    "text_evidence": [],
                    "source_id": "doc-material",
                    "origin_stage": "downstream_rag",
                    "source_slot": "尺寸规格/材质形制/收藏地",
                    "query": "绢本",
                },
            ]
        }

        self.coordinator._merge_external_rag_into_meta(
            meta,
            external_rag=external_rag,
            task=SpawnTask(
                slot_name="尺寸规格/材质形制/收藏地",
                reason="round_table_follow_up",
                prompt_focus="请补充材质与收藏信息。",
            ),
        )

        self.assertEqual(2, len(meta["rag_cache"]))
        retained_facts = [item["fact"] for item in meta["retained_facts"]]
        self.assertTrue(any("宁波地区活跃" in fact for fact in retained_facts))
        self.assertTrue(any("东京国立博物馆" in fact for fact in retained_facts))
        self.assertTrue(any("纵120.5×横55.2厘米" in fact for fact in retained_facts))

    def test_run_task_rag_prefers_cached_documents_before_external_search(self) -> None:
        searched_queries: list[str] = []

        class FakeRagClient:
            def __init__(self, endpoint: str) -> None:
                self.endpoint = endpoint

            def search(self, *, query_text: str, **_: object) -> list[object]:
                searched_queries.append(query_text)
                return []

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            cache_path = run_dir / "runtime_state" / "downstream_rag_cache.json"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(
                json.dumps(
                    [
                        {
                            "query": "竹林",
                            "query_signature": self.coordinator._query_signature("竹林"),
                            "documents": [
                                {
                                    "term": "竹林",
                                    "description": "缓存中的文档",
                                    "text_evidence": ["缓存证据"],
                                    "source_id": "cache-1",
                                    "query": "竹林",
                                    "origin_stage": "downstream_rag",
                                }
                            ],
                        }
                    ],
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            image_path = run_dir / "input.png"
            image_path.write_bytes(b"fake-image")
            task = SpawnTask(
                slot_name="佛教人物画",
                reason="specific_question_unanswered",
                prompt_focus="竹林在画面中承担什么环境作用？",
                rag_terms=["竹林"],
            )
            with patch("src.cot_layer.closed_loop._load_perception_rag_client", return_value=FakeRagClient):
                result = self.coordinator._run_task_rag(
                    task=task,
                    image_path=str(image_path),
                    run_dir=run_dir,
                    round_index=1,
                    task_index=1,
                )

            self.assertEqual([], searched_queries)
            self.assertEqual(["竹林"], result["queries"])
            self.assertEqual("缓存中的文档", result["documents"][0]["description"])

    def test_cleanup_downstream_task_logs_removes_context_and_chat_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir)
            context_path = task_dir / "context.md"
            chat_path = task_dir / "llm_chat_record.jsonl"
            task_path = task_dir / "task_01.json"
            context_path.write_text("context", encoding="utf-8")
            chat_path.write_text("[]\n", encoding="utf-8")
            task_path.write_text("{}", encoding="utf-8")

            self.coordinator._cleanup_downstream_task_logs(task_dir)

            self.assertFalse(context_path.exists())
            self.assertFalse(chat_path.exists())
            self.assertTrue(task_path.exists())

    def test_run_task_rag_skips_queries_already_searched_twice(self) -> None:
        class FakeRagItem:
            def __init__(self, content: str, source_id: str, score: float) -> None:
                self.content = content
                self.source_id = source_id
                self.score = score
                self.metadata = {}

        searched_queries: list[str] = []

        class FakeRagClient:
            def __init__(self, endpoint: str) -> None:
                self.endpoint = endpoint

            def search(self, *, query_text: str, **_: object) -> list[FakeRagItem]:
                searched_queries.append(query_text)
                return [FakeRagItem(content=f"{query_text} 证据", source_id="doc-1", score=0.9)]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            record_path = run_dir / "downstream_rounds" / "round_01" / "rag_search_record.md"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                "# RAG Search Record\n\n"
                "## Search Batch [2026-03-25 10:00:00]\n"
                "- image: `/tmp/demo.png`\n"
                "- query_text: `光环`\n"
                "  - image_attached: `true`\n"
                "  - sources: `a`\n"
                "  - alignment_scores: `0.9`\n"
                "## Search Batch [2026-03-25 10:01:00]\n"
                "- image: `/tmp/demo.png`\n"
                "- query_text: `光环`\n"
                "  - image_attached: `true`\n"
                "  - sources: `b`\n"
                "  - alignment_scores: `0.8`\n",
                encoding="utf-8",
            )
            image_path = run_dir / "input.png"
            image_path.write_bytes(b"fake-image")
            task = SpawnTask(
                slot_name="佛教人物画",
                reason="specific_question_unanswered",
                prompt_focus="光环如何增强神圣感？",
                rag_terms=["光环", "竹林"],
            )
            with patch("src.cot_layer.closed_loop._load_perception_rag_client", return_value=FakeRagClient):
                result = self.coordinator._run_task_rag(
                    task=task,
                    image_path=str(image_path),
                    run_dir=run_dir,
                    round_index=2,
                    task_index=1,
                )

            self.assertEqual(["竹林"], searched_queries)
            self.assertEqual(["竹林"], result["queries"])

    def test_normalize_downstream_queries_filters_descriptive_phrases(self) -> None:
        queries = self.coordinator._normalize_downstream_queries(
            ["整体色调为墨色加淡赭色", "淡赭设色", "如何表现山石质感？"],
            slot_name="设色",
        )

        self.assertEqual(["淡赭设色"], queries)

    def test_run_task_rag_skips_reordered_equivalent_queries(self) -> None:
        class FakeRagItem:
            def __init__(self, content: str, source_id: str, score: float) -> None:
                self.content = content
                self.source_id = source_id
                self.score = score
                self.metadata = {}

        searched_queries: list[str] = []

        class FakeRagClient:
            def __init__(self, endpoint: str) -> None:
                self.endpoint = endpoint

            def search(self, *, query_text: str, **_: object) -> list[FakeRagItem]:
                searched_queries.append(query_text)
                return [FakeRagItem(content=f"{query_text} 证据", source_id="doc-2", score=0.88)]

        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            record_path = run_dir / "downstream_rounds" / "round_01" / "rag_search_record.md"
            record_path.parent.mkdir(parents=True, exist_ok=True)
            record_path.write_text(
                "# RAG Search Record\n\n"
                "## Search Batch [2026-03-25 10:00:00]\n"
                "- image: `/tmp/demo.png`\n"
                "- query_text: `十六罗汉图 光环`\n"
                "  - image_attached: `true`\n"
                "  - sources: `a`\n"
                "  - alignment_scores: `0.9`\n"
                "## Search Batch [2026-03-25 10:01:00]\n"
                "- image: `/tmp/demo.png`\n"
                "- query_text: `十六罗汉图 光环`\n"
                "  - image_attached: `true`\n"
                "  - sources: `b`\n"
                "  - alignment_scores: `0.8`\n",
                encoding="utf-8",
            )
            image_path = run_dir / "input.png"
            image_path.write_bytes(b"fake-image")
            task = SpawnTask(
                slot_name="佛教人物画",
                reason="specific_question_unanswered",
                prompt_focus="光环如何增强神圣感？",
                rag_terms=["光环 十六罗汉图", "竹林"],
            )
            with patch("src.cot_layer.closed_loop._load_perception_rag_client", return_value=FakeRagClient):
                result = self.coordinator._run_task_rag(
                    task=task,
                    image_path=str(image_path),
                    run_dir=run_dir,
                    round_index=2,
                    task_index=1,
                )

            self.assertEqual(["竹林"], searched_queries)
            self.assertEqual(["竹林"], result["queries"])


if __name__ == "__main__":
    unittest.main()
