from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.cot_layer.models import (
    CoTThread,
    CrossValidationIssue,
    CrossValidationResult,
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
from src.cot_layer.new_api_client import ChatResult
from src.cot_layer.pipeline import DynamicAgentPipeline
from src.cot_layer.prompt_builder import build_domain_cot_prompt
from src.common.prompt_utils import build_slot_summary_payload, meta_payload
from src.reflection_layer.prompt_builder import build_final_answer_request_prompt, build_final_appreciation_prompt
from src.reflection_layer.service import batch_task_rag_terms, batch_task_retrieval_plans, clean_search_query, clean_web_search_query


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


class PipelineLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = DynamicAgentPipeline(
            config=PipelineConfig(max_threads_per_round=3, thread_attempt_limit=2),
            api_client=FakeAPIClient(enabled=False),
        )
        self.slot_schema = SlotSchema(
            slot_name="皴法",
            slot_term="雨点皴",
            description="用于表现北方山石质感。",
            specific_questions=["它如何表现北方山石？", "变体如何搭配？"],
            metadata={"confidence": 0.9},
            controlled_vocabulary=["雨点皴", "钉头皴", "芝麻皴"],
        )

    def test_initialize_threads_creates_one_overview_per_slot(self) -> None:
        threads = self.pipeline._initialize_threads([self.slot_schema])
        self.assertEqual(1, len(threads))
        self.assertEqual("slot_overview", threads[0].reason)
        self.assertEqual("皴法", threads[0].slot_name)

    def test_initialize_threads_skips_stable_and_closed_slots(self) -> None:
        stable_slot = SlotSchema(
            slot_name="人物画技法",
            slot_term="传神写照",
            description="稳定槽位",
            specific_questions=["问题"],
            metadata={"lifecycle": "STABLE"},
            controlled_vocabulary=["传神写照"],
        )
        closed_slot = SlotSchema(
            slot_name="画作名称",
            slot_term="洛神赋图",
            description="关闭槽位",
            specific_questions=["问题"],
            metadata={"lifecycle": "CLOSED"},
            controlled_vocabulary=["洛神赋图"],
        )
        threads = self.pipeline._initialize_threads([self.slot_schema, stable_slot, closed_slot])
        self.assertEqual(["皴法"], [thread.slot_name for thread in threads])

    def test_initialize_threads_prefers_external_cot_tasks_from_meta(self) -> None:
        threads = self.pipeline._initialize_threads(
            [self.slot_schema],
            meta={
                "pending_cot_tasks": [
                    {
                        "slot_name": "皴法",
                        "reason": "round_table_follow_up",
                        "prompt_focus": "钉头皴与芝麻皴如何区分？",
                        "rag_terms": ["钉头皴", "芝麻皴"],
                        "priority": 5,
                    }
                ]
            },
        )
        self.assertEqual(1, len(threads))
        self.assertEqual("round_table_follow_up", threads[0].reason)
        self.assertEqual("钉头皴与芝麻皴如何区分？", threads[0].focus)
        self.assertEqual(["钉头皴", "芝麻皴"], threads[0].rag_terms)

    def test_plan_spawn_tasks_prioritizes_unanswered_questions(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="更接近北方山石的厚重感")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=False),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="已回答"),
            ],
            unresolved_points=["钉头皴与芝麻皴具体分布待确认"],
            generated_questions=[],
            statuses=[],
            confidence=0.7,
        )
        validation = CrossValidationResult(
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
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertTrue(any(task.reason == "specific_question_unanswered" for task in tasks))
        self.assertEqual("它如何表现北方山石？", tasks[0].prompt_focus)
        self.assertEqual([], tasks[0].rag_terms)

    def test_plan_spawn_tasks_ignores_unresolved_points_without_round_table_followups(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="更接近北方山石的厚重感")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="已回答"),
            ],
            unresolved_points=["钉头皴与芝麻皴具体分布待确认"],
            generated_questions=["是否还存在地域风格差异？"],
            statuses=[],
            confidence=0.7,
        )
        validation = CrossValidationResult(
            issues=[
                CrossValidationIssue(
                    issue_type="missing_visual_anchor",
                    severity="high",
                    slot_names=["皴法"],
                    detail="皴法 缺少稳定的视觉锚点。",
                )
            ],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual([], tasks)

    def test_plan_spawn_tasks_routes_web_followup_to_downstream(self) -> None:
        output = DomainCoTRecord(
            slot_name="作者时代流派",
            slot_term="范宽",
            analysis_round=1,
            controlled_vocabulary=["范宽", "北宋"],
            visual_anchoring=[EvidenceItem(observation="主峰高耸，山石厚重")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[QuestionCoverage(question="作者归属如何判断？", answered=True, support="已回答")],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.7,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "作者时代流派",
                    "question": "这幅作品与范宽、李成、郭熙的关系应如何比较？",
                    "priority": "high",
                    "retrieval_mode": "web",
                    "rag_queries": [],
                    "web_queries": ["范宽 李成 郭熙 风格比较", "溪山行旅图 范宽"],
                    "retrieval_reason": "涉及作者比较与作品归属，更适合联网搜索。",
                }
            ],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("downstream", tasks[0].dispatch_target)
        self.assertEqual("web", tasks[0].retrieval_mode)
        self.assertEqual(["范宽 李成 郭熙 风格比较", "溪山行旅图 范宽"], tasks[0].web_queries)

    def test_batch_task_retrieval_plans_keeps_multiword_web_queries(self) -> None:
        api_client = FakeAPIClient(
            enabled=True,
            responses=[
                """{
                  "mode": "hybrid",
                  "rag_queries": ["披麻皴"],
                  "web_queries": ["溪山行旅图 范宽 馆藏", "范宽 李成 郭熙 风格比较"],
                  "reason": "技法适合 RAG，作者比较适合联网。"
                }"""
            ],
        )
        pipeline = DynamicAgentPipeline(
            config=PipelineConfig(enable_web_search=True, web_search_top_k=5),
            api_client=api_client,
        )
        plans = batch_task_retrieval_plans(
            pipeline,
            [
                {
                    "request_id": "req-1",
                    "slot_name": "作者时代流派",
                    "focus_text": "这幅作品与范宽、李成、郭熙的关系应如何比较？",
                    "fallback_terms": ["范宽", "李成", "郭熙"],
                    "task_reason": "round_table_follow_up",
                }
            ],
        )
        self.assertEqual("hybrid", plans["req-1"]["mode"])
        self.assertEqual(["披麻皴"], plans["req-1"]["rag_queries"])
        self.assertEqual(["溪山行旅图 范宽 馆藏", "范宽 李成 郭熙 风格比较"], plans["req-1"]["web_queries"])

    def test_clean_web_search_query_allows_multiword_phrase(self) -> None:
        self.assertEqual("溪山行旅图 范宽 馆藏", clean_web_search_query("溪山行旅图 范宽 馆藏"))

    def test_domain_prompt_for_follow_up_only_uses_focus_and_unanswered_questions(self) -> None:
        thread = CoTThread(
            thread_id="皴法-followup-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="钉头皴与芝麻皴如何区分？",
            reason="round_table_follow_up",
        )
        thread.latest_record = {
            "question_coverage": [
                {"question": "它如何表现北方山石？", "answered": False},
                {"question": "变体如何搭配？", "answered": False},
                {"question": "哪些局部最能体现雨点皴？", "answered": False},
                {"question": "变体如何搭配？", "answered": True},
            ]
        }
        prompt_questions = self.pipeline._domain_prompt_questions(self.slot_schema, thread)
        self.assertEqual(
            ["钉头皴与芝麻皴如何区分？", "它如何表现北方山石？", "变体如何搭配？"],
            prompt_questions,
        )

        prompt = build_domain_cot_prompt(
            slot=self.slot_schema,
            meta={},
            focus_question=thread.focus,
            analysis_round=2,
            thread_context={},
            specific_questions=prompt_questions,
            retrieval_gain_enabled=False,
        )
        self.assertIn(
            "specific_questions: [\"钉头皴与芝麻皴如何区分？\", \"它如何表现北方山石？\", \"变体如何搭配？\"]",
            prompt,
        )
        self.assertNotIn('"retrieval_gain"', prompt)
        self.assertNotIn("哪些局部最能体现雨点皴？", prompt)

    def test_domain_prompt_includes_web_retrieval_gain_schema_when_enabled(self) -> None:
        prompt = build_domain_cot_prompt(
            slot=self.slot_schema,
            meta={},
            focus_question="它如何表现北方山石？",
            analysis_round=1,
            thread_context={},
            specific_questions=self.slot_schema.specific_questions[:1],
            retrieval_gain_enabled=True,
            web_search_enabled=True,
        )
        self.assertIn('"retrieval_mode": "rag|web|hybrid"', prompt)
        self.assertIn('"web_queries": ["适合联网搜索的多词 query"]', prompt)

    def test_run_single_thread_uses_text_only_after_stable_visual_anchor(self) -> None:
        api_client = FakeAPIClient(
            enabled=True,
            responses=[
                """{
                  "slot_name": "皴法",
                  "controlled_vocabulary_used": ["雨点皴"],
                  "visual_anchoring": [{"observation":"山石表面密集短点排列","evidence":"画面可见","position":"中部"}],
                  "domain_decoding": [],
                  "cultural_mapping": [],
                  "specific_question_coverage": [{"question":"它如何表现北方山石？","answered":true,"support":"通过密集短点表现山石质感。"}],
                  "generated_questions": [],
                  "unresolved_points": [],
                  "confidence": 0.8,
                  "retrieval_gain": {"has_new_value": true, "focus": "不应被读取", "related_terms": ["钉头皴"], "search_queries": ["钉头皴"], "reason": "测试"}
                }"""
            ],
        )
        pipeline = DynamicAgentPipeline(
            config=PipelineConfig(max_threads_per_round=3, thread_attempt_limit=2, retrieval_gain=False),
            api_client=api_client,
        )
        thread = CoTThread(
            thread_id="皴法-followup-2",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="round_table_follow_up",
            latest_confidence=0.7,
        )
        thread.latest_record = {
            "visual_anchoring": [{"observation": "山石表面密集短点排列"}],
            "question_coverage": [{"question": "它如何表现北方山石？", "answered": False}],
        }

        record, logs = pipeline._run_single_thread(
            self.slot_schema,
            thread,
            "/tmp/demo.png",
            {},
            2,
        )
        self.assertFalse(logs[0]["image_attached"])
        self.assertTrue(logs[0]["text_only"])
        self.assertEqual(1, logs[0]["prompt_question_count"])
        self.assertFalse(record.retrieval_gain_has_value)
        self.assertEqual([], record.retrieval_gain_terms)

    def test_plan_spawn_tasks_includes_high_priority_round_table_followups(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀", "光环"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="主尊身份如何识别？", answered=True, support="存在佛陀形象"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "佛教人物画",
                    "question": "画中主尊具体是哪一尊佛？",
                    "why": "需要更具体身份",
                    "priority": "high",
                    "rag_queries": ["十六罗汉图 第十一尊者", "佛教画 光环"],
                },
                {
                    "slot_name": "佛教人物画",
                    "question": "光环的具体形态如何？",
                    "why": "补充细节",
                    "priority": "medium",
                    "rag_queries": ["佛教画 光环"],
                },
            ],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("round_table_follow_up", tasks[0].reason)
        self.assertEqual("画中主尊具体是哪一尊佛？", tasks[0].prompt_focus)
        self.assertEqual(["十六罗汉图 第十一尊者", "佛教画 光环"], tasks[0].rag_terms)

    def test_plan_spawn_tasks_includes_retrieval_gain_after_questions_answered(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="罗汉",
            analysis_round=2,
            controlled_vocabulary=["罗汉", "光环"],
            visual_anchoring=[EvidenceItem(observation="人物头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="主尊身份如何识别？", answered=True, support="已回答"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.85,
            retrieval_gain_focus="继续核验侍从所执长柄器物的专业名称。",
            retrieval_gain_terms=["长柄法器", "云头状执具"],
            retrieval_gain_queries=["长柄法器", "云头状执具"],
            retrieval_gain_reason="RAG 描述出现了与图中器物形制对应的新术语。",
            retrieval_gain_has_value=True,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("retrieval_gain", tasks[0].reason)
        self.assertEqual("佛教人物画", tasks[0].slot_name)
        self.assertEqual(["长柄法器", "云头状执具"], tasks[0].rag_terms)

    def test_plan_spawn_tasks_ignores_retrieval_gain_without_novel_terms(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="罗汉",
            analysis_round=2,
            controlled_vocabulary=["罗汉", "光环"],
            visual_anchoring=[EvidenceItem(observation="人物头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="主尊身份如何识别？", answered=True, support="已回答"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.85,
            retrieval_gain_focus="继续核验罗汉身份。",
            retrieval_gain_terms=["罗汉"],
            retrieval_gain_queries=[],
            retrieval_gain_reason="没有新的术语增益。",
            retrieval_gain_has_value=True,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual([], tasks)

    def test_plan_spawn_tasks_remaps_invalid_round_table_slot_name_from_question_context(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀", "光环"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="主尊身份如何识别？", answered=True, support="存在佛陀形象"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "持物",
                    "question": "主尊身份如何识别？",
                    "why": "圆桌误填了槽位名",
                    "priority": "high",
                    "rag_queries": ["佛教人物画 光环"],
                }
            ],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("佛教人物画", tasks[0].slot_name)

    def test_plan_spawn_tasks_routes_unresolvable_round_table_slot_name_to_downstream(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀", "光环"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="主尊身份如何识别？", answered=True, support="存在佛陀形象"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "持物",
                    "question": "法器具体为何物？",
                    "why": "圆桌误填了槽位名",
                    "priority": "high",
                    "rag_queries": ["法器 识别"],
                }
            ],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("downstream", tasks[0].dispatch_target)
        self.assertEqual("持物", tasks[0].slot_name)

    def test_plan_spawn_tasks_routes_new_slot_followup_to_downstream_discovery(self) -> None:
        output = DomainCoTRecord(
            slot_name="人物画技法",
            slot_term="传神写照",
            analysis_round=1,
            controlled_vocabulary=["传神写照"],
            visual_anchoring=[EvidenceItem(observation="人物衣纹线条流畅")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="顾恺之如何通过线条和神韵表现洛神的神态与气质？", answered=True, support="已回答"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "持物",
                    "question": "洛神是否持有可识别的器物？",
                    "why": "可能需要新增物象维度",
                    "priority": "high",
                    "rag_queries": ["洛神赋图 持物"],
                }
            ],
            follow_up_task_reviews=[
                {
                    "slot_name": "持物",
                    "question": "洛神是否持有可识别的器物？",
                    "action": "downstream_discovery",
                    "reason": "先交给 downstream 判断是否值得新建槽位",
                }
            ],
        )
        tasks = self.pipeline._plan_spawn_tasks([output], validation, [])
        self.assertEqual(1, len(tasks))
        self.assertEqual("downstream", tasks[0].dispatch_target)
        self.assertEqual("持物", tasks[0].slot_name)

    def test_check_convergence_requires_all_questions_answered(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="北方山石的硬度被强化")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="短促点皴强化硬度"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="前后景山体点皴节奏不同"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        dialogue_state = DialogueState(no_new_info_rounds=0)
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="specific_question_unanswered",
            status="ANSWERED",
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        snapshot = self.pipeline._check_convergence([self.slot_schema], [output], validation, [thread], dialogue_state)
        self.assertTrue(snapshot["converged"])

    def test_check_convergence_ignores_unresolved_points_when_questions_are_answered(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="北方山石的硬度被强化")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="短促点皴强化硬度"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="前后景山体点皴节奏不同"),
            ],
            unresolved_points=["局部仍可继续细化"],
            generated_questions=["是否还需要补充更多问题？"],
            statuses=[],
            confidence=0.9,
        )
        dialogue_state = DialogueState(no_new_info_rounds=0)
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="specific_question_unanswered",
            status="ANSWERED",
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        snapshot = self.pipeline._check_convergence([self.slot_schema], [output], validation, [thread], dialogue_state)
        self.assertTrue(snapshot["converged"])

    def test_check_convergence_waits_for_pending_downstream_tasks(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="北方山石的硬度被强化")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="短促点皴强化硬度"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="前后景山体点皴节奏不同"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        dialogue_state = DialogueState(no_new_info_rounds=0)
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="specific_question_unanswered",
            status="ANSWERED",
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        snapshot = self.pipeline._check_convergence(
            [self.slot_schema],
            [output],
            validation,
            [thread],
            dialogue_state,
            [SpawnTask(slot_name="持物", reason="round_table_follow_up", prompt_focus="是否持物", dispatch_target="downstream")],
        )
        self.assertFalse(snapshot["converged"])

    def test_sync_threads_reopens_matching_blocked_thread(self) -> None:
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="specific_question_unanswered",
            status="BLOCKED",
            attempts=1,
            max_attempts=3,
        )
        tasks = self.pipeline._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="皴法",
                    reason="specific_question_unanswered",
                    prompt_focus="它如何表现北方山石？",
                    rag_terms=["北方山石"],
                    priority=4,
                )
            ]
        )
        new_thread_ids = self.pipeline._sync_threads_with_tasks([thread], tasks)
        self.assertEqual([], new_thread_ids)
        self.assertEqual("OPEN", thread.status)

    def test_sync_threads_reuses_semantically_similar_thread(self) -> None:
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="人物题材",
            slot_term="罗汉",
            focus="作品中光环的运用如何增强罗汉的神圣感？",
            reason="round_table_follow_up",
            status="BLOCKED",
            attempts=1,
            max_attempts=3,
            rag_terms=["光环", "罗汉"],
            priority=4,
        )
        tasks = self.pipeline._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="人物题材",
                    reason="round_table_follow_up",
                    prompt_focus="画中光环的形态、色彩及位置如何设计？其如何增强罗汉的神圣感？",
                    rag_terms=["十六罗汉图", "南宋"],
                    priority=5,
                )
            ]
        )
        new_thread_ids = self.pipeline._sync_threads_with_tasks([thread], tasks)
        self.assertEqual([], new_thread_ids)
        self.assertEqual("OPEN", thread.status)
        self.assertEqual(["光环", "罗汉", "十六罗汉图", "南宋"], thread.rag_terms)
        self.assertEqual(5, thread.priority)

    def test_sync_threads_does_not_merge_light_overlap_only(self) -> None:
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="人物题材",
            slot_term="罗汉",
            focus="光环与背景自然元素的关系如何增强画面宗教氛围？",
            reason="specific_question_unanswered",
            status="BLOCKED",
            attempts=1,
            max_attempts=3,
            rag_terms=["光环", "自然元素"],
            priority=4,
        )
        tasks = self.pipeline._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="人物题材",
                    reason="specific_question_unanswered",
                    prompt_focus="光环色彩层次如何体现南宋中间色赋彩技法？",
                    rag_terms=["光环", "中间色赋彩"],
                    priority=4,
                )
            ]
        )
        new_thread_ids = self.pipeline._sync_threads_with_tasks([thread], tasks)
        self.assertEqual(1, len(new_thread_ids))
        self.assertEqual("BLOCKED", thread.status)

    def test_dedupe_spawn_tasks_merges_semantically_similar_focuses(self) -> None:
        tasks = self.pipeline._dedupe_spawn_tasks(
            [
                SpawnTask(
                    slot_name="人物题材",
                    reason="round_table_follow_up",
                    prompt_focus="作品中光环的运用如何增强罗汉的神圣感？",
                    rag_terms=["光环", "罗汉", "宗教符号"],
                    priority=4,
                ),
                SpawnTask(
                    slot_name="人物题材",
                    reason="round_table_follow_up",
                    prompt_focus="画中光环的形态、色彩及位置如何设计？其如何增强罗汉的神圣感？",
                    rag_terms=["十六罗汉图", "南宋"],
                    priority=5,
                ),
                SpawnTask(
                    slot_name="人物题材",
                    reason="specific_question_unanswered",
                    prompt_focus="法器的具体材质和装饰细节有哪些？",
                    rag_terms=["法器", "材质", "装饰"],
                    priority=4,
                ),
            ]
        )
        self.assertEqual(2, len(tasks))
        self.assertEqual(
            "画中光环的形态、色彩及位置如何设计？其如何增强罗汉的神圣感？",
            tasks[0].prompt_focus,
        )
        self.assertEqual(
            ["十六罗汉图", "南宋", "光环", "罗汉", "宗教符号"],
            tasks[0].rag_terms,
        )
        self.assertEqual("法器的具体材质和装饰细节有哪些？", tasks[1].prompt_focus)

    def test_suppress_redundant_tasks_pauses_stalled_duplicate_thread(self) -> None:
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="它如何表现北方山石？",
            reason="specific_question_unanswered",
            status="RETRY",
            attempts=2,
            max_attempts=4,
            latest_new_info_gain=0,
            stale_rounds=2,
        )
        kept_tasks, paused_thread_ids = self.pipeline._suppress_redundant_tasks(
            [
                SpawnTask(
                    slot_name="皴法",
                    reason="specific_question_unanswered",
                    prompt_focus="它如何表现北方山石？",
                    rag_terms=["北方山石"],
                    priority=4,
                )
            ],
            [thread],
        )
        self.assertEqual([], kept_tasks)
        self.assertEqual(["thread-1"], paused_thread_ids)
        self.assertEqual("PAUSED", thread.status)
        self.assertEqual("duplicate_stalled", thread.pause_reason)

    def test_task_rag_terms_uses_llm_queries_instead_of_question_fragments(self) -> None:
        self.pipeline.api_client = FakeAPIClient(
            enabled=True,
            responses=['{"queries":["佛教人物画 光环 象征","袈裟 设色 平涂","人物衣纹 细线勾勒"]}'],
        )
        terms = self.pipeline._task_rag_terms(
            focus_text="如何通过细线勾勒表现佛教人物的神圣气质？",
            fallback_terms=["细线勾勒", "神圣气质", "光环"],
            slot_name="佛教人物画",
            task_reason="generated_question",
        )
        self.assertEqual(
            ["佛教人物画 光环", "袈裟 设色", "人物衣纹 细线勾勒"],
            terms,
        )
        self.assertTrue(all("？" not in item and "?" not in item for item in terms))
        self.assertTrue(all(not item.startswith("如何") for item in terms))
        self.assertTrue(all(len(item.split()) <= 2 for item in terms))

    def test_task_rag_terms_returns_empty_when_llm_returns_no_valid_queries(self) -> None:
        self.pipeline.api_client = FakeAPIClient(
            enabled=True,
            responses=['{"queries":["如何表现北方山石","雨点皴有什么作用？","北方山石 质感"]}'],
        )
        terms = self.pipeline._task_rag_terms(
            focus_text="它如何表现北方山石？",
            fallback_terms=["雨点皴", "北方山石", "皴法"],
            slot_name="皴法",
            task_reason="specific_question_unanswered",
        )
        self.assertEqual(["北方山石 质感"], terms)

    def test_batch_task_rag_terms_respects_single_block_limit(self) -> None:
        self.pipeline.config.rag_query_max_blocks = 1
        self.pipeline.api_client = FakeAPIClient(
            enabled=True,
            responses=[
                '{"items":['
                '{"request_id":"q1","queries":["十六罗汉图 光环","佛教人物画"]},'
                '{"request_id":"q2","queries":["袈裟 设色","矿物颜料"]}'
                ']}'
            ],
        )
        results = batch_task_rag_terms(
            self.pipeline,
            [
                {
                    "request_id": "q1",
                    "slot_name": "佛教人物画",
                    "focus_text": "光环如何辅助身份识别？",
                    "fallback_terms": ["佛教人物画", "光环"],
                    "task_reason": "specific_question_unanswered",
                },
                {
                    "request_id": "q2",
                    "slot_name": "设色",
                    "focus_text": "设色层次如何体现？",
                    "fallback_terms": ["设色", "矿物颜料"],
                    "task_reason": "specific_question_unanswered",
                },
            ],
        )
        self.assertEqual(["十六罗汉图", "佛教人物画"], results["q1"])
        self.assertEqual(["袈裟", "矿物颜料"], results["q2"])

    def test_clean_search_query_rejects_descriptive_single_block_phrase(self) -> None:
        self.assertEqual("", clean_search_query("整体色调为墨色加淡赭色", slot_name="设色", max_keyword_blocks=1))
        self.assertEqual("淡赭设色", clean_search_query("淡赭设色", slot_name="设色", max_keyword_blocks=1))

    def test_round_table_review_extracts_followup_questions_and_rag_needs(self) -> None:
        fake_client = FakeAPIClient(
            enabled=True,
            responses=[
                (
                    '{"review_summary":"佛教人物身份细节不足",'
                    '"blind_spots":["佛陀身份未细分","光环形制未说明"],'
                    '"follow_up_questions":[{"slot_name":"佛教人物画","question":"画中主尊具体是哪一尊佛？","why":"当前只说佛陀，身份不够具体","priority":"high","rag_queries":["释迦牟尼佛 图像特征","阿弥陀佛 图像特征"]}],'
                    '"rag_needs":[{"topic":"佛教人物身份识别","reason":"需要区分主尊身份","queries":["佛教人物画 佛陀 手印 光环","佛像身份识别 图像线索"]}]}'
                )
            ],
        )
        self.pipeline.api_client = fake_client
        self.pipeline.vlm_runner.api_client = fake_client
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀", "光环"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[QuestionCoverage(question="主尊身份如何识别？", answered=True, support="存在佛陀形象")],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )

        updated = self.pipeline._augment_round_table_review([output], validation, {}, [])

        self.assertEqual(["佛陀身份未细分", "光环形制未说明"], updated.round_table_blind_spots)
        self.assertEqual("画中主尊具体是哪一尊佛？", updated.round_table_follow_up_questions[0]["question"])
        self.assertEqual("high", updated.round_table_follow_up_questions[0]["priority"])
        self.assertIn("释迦牟尼佛 图像特征", updated.rag_terms)
        self.assertEqual("佛教人物身份识别", updated.round_table_rag_needs[0]["topic"])

    def test_validation_review_extracts_round_table_and_slot_lifecycle(self) -> None:
        fake_client = FakeAPIClient(
            enabled=True,
            responses=[
                (
                    '{"review_summary":"佛教人物身份细节不足",'
                    '"blind_spots":["佛陀身份未细分","光环形制未说明"],'
                    '"follow_up_questions":[{"slot_name":"佛教人物画","question":"画中主尊具体是哪一尊佛？","why":"当前只说佛陀，身份不够具体","priority":"high","rag_queries":["释迦牟尼佛 图像特征","阿弥陀佛 图像特征"]}],'
                    '"rag_needs":[{"topic":"佛教人物身份识别","reason":"需要区分主尊身份","queries":["佛教人物画 佛陀 手印 光环"]}],'
                    '"slot_reviews":[{"slot_name":"佛教人物画","status":"ACTIVE","reason":"仍有高价值身份识别问题未解决。"}],'
                    '"follow_up_reviews":[{"slot_name":"佛教人物画","question":"画中主尊具体是哪一尊佛？","action":"cot","reason":"仍需继续 CoT 核验图像特征。"}]}'
                )
            ],
        )
        self.pipeline.api_client = fake_client
        self.pipeline.vlm_runner.api_client = fake_client
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀", "光环"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[QuestionCoverage(question="主尊身份如何识别？", answered=True, support="存在佛陀形象")],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        slot_schema = SlotSchema(
            slot_name="佛教人物画",
            slot_term="佛陀",
            description="围绕主尊身份做图像学识别。",
            specific_questions=["主尊身份如何识别？"],
            metadata={"confidence": 0.8},
            controlled_vocabulary=["佛陀", "光环"],
        )

        updated = self.pipeline._review_validation_bundle([slot_schema], [output], validation, {}, [])

        self.assertEqual(["佛陀身份未细分", "光环形制未说明"], updated.round_table_blind_spots)
        self.assertEqual("画中主尊具体是哪一尊佛？", updated.round_table_follow_up_questions[0]["question"])
        self.assertEqual("佛教人物身份识别", updated.round_table_rag_needs[0]["topic"])
        self.assertEqual("ACTIVE", updated.slot_lifecycle_reviews[0]["status"])
        self.assertEqual("cot", updated.follow_up_task_reviews[0]["action"])
        self.assertIn("释迦牟尼佛 图像特征", updated.rag_terms)

    def test_round_table_trigger_only_runs_on_event_rounds(self) -> None:
        answered_output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="已回答"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="已回答"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        unanswered_output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="已回答"),
                QuestionCoverage(question="变体如何搭配？", answered=False, support=""),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        self.assertTrue(
            self.pipeline._should_run_round_table_review(
                round_index=1,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=2,
            )
        )
        self.assertFalse(
            self.pipeline._should_run_round_table_review(
                round_index=2,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=2,
            )
        )
        self.assertTrue(
            self.pipeline._should_run_round_table_review(
                round_index=2,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[answered_output],
                new_info_count=2,
            )
        )
        self.assertTrue(
            self.pipeline._should_run_round_table_review(
                round_index=3,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=0,
            )
        )

    def test_lifecycle_trigger_only_runs_on_event_rounds(self) -> None:
        answered_output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="已回答"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="已回答"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        unanswered_output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="已回答"),
                QuestionCoverage(question="变体如何搭配？", answered=False, support=""),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        self.assertTrue(
            self.pipeline._should_run_slot_lifecycle_review(
                round_index=1,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=2,
            )
        )
        self.assertFalse(
            self.pipeline._should_run_slot_lifecycle_review(
                round_index=2,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=2,
            )
        )
        self.assertTrue(
            self.pipeline._should_run_slot_lifecycle_review(
                round_index=2,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[answered_output],
                new_info_count=2,
            )
        )
        self.assertTrue(
            self.pipeline._should_run_slot_lifecycle_review(
                round_index=4,
                max_rounds=4,
                slot_schemas=[self.slot_schema],
                outputs=[unanswered_output],
                new_info_count=2,
            )
        )

    def test_build_round_memory_rolls_forward_previous_round_gain(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=2,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[
                EvidenceItem(observation="山石表面密集短点排列"),
                EvidenceItem(observation="前景山体边缘皴擦更厚重"),
            ],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="北方山石的硬度被强化")],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="短促点皴强化硬度"),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="前后景山体点皴节奏不同"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        dialogue_state = DialogueState(
            resolved_questions=["它如何表现北方山石？", "变体如何搭配？"],
            unresolved_questions=[],
            converged=True,
            convergence_reason="原始问题均已覆盖。",
            final_round_index=2,
        )
        thread = CoTThread(
            thread_id="thread-1",
            slot_name="皴法",
            slot_term="雨点皴",
            focus="变体如何搭配？",
            reason="specific_question_unanswered",
            status="ANSWERED",
            latest_new_info_gain=2,
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
        )
        previous_memory = {
            "round_index": 1,
            "resolved_questions": ["它如何表现北方山石？"],
            "unresolved_questions": ["变体如何搭配？"],
            "issues": [],
            "slots": [
                {
                    "slot_name": "皴法",
                    "visual_anchoring": [{"observation": "山石表面密集短点排列"}],
                    "domain_decoding": [],
                    "cultural_mapping": [],
                    "question_coverage": [
                        {"question": "它如何表现北方山石？", "answered": True},
                        {"question": "变体如何搭配？", "answered": False},
                    ],
                }
            ],
        }

        memory = self.pipeline._build_round_memory(
            [output],
            validation,
            dialogue_state,
            [thread],
            prior_round_memories=[previous_memory],
        )

        self.assertEqual(1, memory["previous_round_index"])
        self.assertEqual(["变体如何搭配？"], memory["info_gain"]["new_resolved_questions"])
        self.assertEqual(["变体如何搭配？"], memory["info_gain"]["cleared_questions"])
        self.assertEqual(1, len(memory["info_gain"]["slot_gains"]))
        self.assertIn("前景山体边缘皴擦更厚重", memory["info_gain"]["slot_gains"][0]["new_visual_anchoring"])
        self.assertEqual([], memory["carry_over"]["focus_questions"])
        self.assertGreaterEqual(memory["info_gain"]["total_new_items"], 2)

    def test_final_appreciation_prompt_lists_question_answers(self) -> None:
        output = DomainCoTRecord(
            slot_name="佛教人物画",
            slot_term="佛陀",
            analysis_round=1,
            controlled_vocabulary=["佛陀"],
            visual_anchoring=[EvidenceItem(observation="主尊头后有圆形光环")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="宗教身份通过主尊位置被强化")],
            question_coverage=[
                QuestionCoverage(
                    question="画中主尊身份如何识别？",
                    answered=True,
                    support="主尊居中端坐，头后有圆形光环，侍者陪立，显示其为佛教尊像核心人物。",
                )
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )
        prompt = build_final_appreciation_prompt(
            outputs=[output],
            validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            meta={
                "retained_facts": [
                    {
                        "slot_name": "作者时代流派",
                        "slot_term": "金大受",
                        "fact": "金大受为南宋前期在宁波地区活跃的道释画家。",
                        "source": "slot_description",
                    }
                ]
            },
            dialogue_state=DialogueState(resolved_questions=["画中主尊身份如何识别？"]),
        )
        self.assertIn("## 赏析", prompt)
        self.assertIn("围绕佛教人物画，主尊居中端坐，头后有圆形光环", prompt)
        self.assertIn("补充来看，佛教人物画可从佛陀这一线索展开", prompt)
        self.assertIn("金大受为南宋前期在宁波地区活跃的道释画家", prompt)
        self.assertNotIn("问题：画中主尊身份如何识别？", prompt)
        self.assertNotIn("## 回答整合分析", prompt)
        self.assertNotIn("## 补充分析材料", prompt)

    def test_generate_final_appreciation_uses_separate_model_call(self) -> None:
        api_client = FakeAPIClient(
            enabled=True,
            responses=["## 赏析\n这是一段基于问题的最终回答。"],
        )
        pipeline = DynamicAgentPipeline(
            config=PipelineConfig(final_answer_model="gpt-5.4"),
            api_client=api_client,
        )
        output = DomainCoTRecord(
            slot_name="山水画",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="山体质感厚重")],
            question_coverage=[
                QuestionCoverage(question="这幅画的山石技法如何？", answered=True, support="山石以密集短点皴擦表现厚重质感。"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.8,
        )

        prompt = pipeline._generate_final_appreciation_prompt(
            [output],
            CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            {"final_user_question": "这幅画的山石技法如何？"},
            DialogueState(resolved_questions=["这幅画的山石技法如何？"]),
            api_logs=[],
        )

        self.assertEqual("## 赏析\n这是一段基于问题的最终回答。", prompt)
        self.assertEqual("gpt-5.4", api_client.calls[-1]["model"])
        self.assertIn("这幅画的山石技法如何？", str(api_client.calls[-1]["user_prompt"]))

    def test_build_slot_summary_payload_keeps_description_facts_and_strips_progress_notes(self) -> None:
        payload = build_slot_summary_payload(
            [
                SlotSchema(
                    slot_name="作者时代流派",
                    slot_term="金大受",
                    description=(
                        "本作《十六罗汉图》为南宋时期画家金大受所绘。"
                        "相关背景资料明确指出金大受为南宋前期宁波地区活跃的道释画家。"
                        " 当前推进 term：南宋。 当前轮仍需结合 `南宋` 补充：人物面部细节暂不可辨。"
                        " 补充证据：其作品以精致工笔、自然协调著称。"
                    ),
                    specific_questions=["画面中有哪些细节最能体现金大受的个人风格？"],
                    metadata={"lifecycle": "ACTIVE", "slot_mode": "enumerative"},
                )
            ]
        )
        self.assertEqual(1, len(payload))
        highlights = payload[0]["description_highlights"]
        self.assertTrue(any("宁波地区活跃的道释画家" in item for item in highlights))
        self.assertTrue(any("精致工笔、自然协调著称" in item for item in highlights))
        self.assertFalse(any("当前推进 term" in item for item in highlights))
        self.assertFalse(any("当前轮仍需结合" in item for item in highlights))

    def test_build_final_appreciation_prompt_uses_slot_coverage_for_author_background(self) -> None:
        output = DomainCoTRecord(
            slot_name="作者时代流派",
            slot_term="金大受",
            analysis_round=1,
            controlled_vocabulary=["金大受", "南宋"],
            visual_anchoring=[EvidenceItem(observation="人物衣纹细劲流畅，设色沉稳。")],
            domain_decoding=[],
            cultural_mapping=[MappingItem(insight="南宋道释人物画注重庄重肃穆的宗教气氛。")],
            question_coverage=[
                QuestionCoverage(
                    question="画面中有哪些细节最能体现金大受的个人风格？",
                    answered=True,
                    support="人物形态准确、衣纹线条流畅、设色以中间色为主，显示金大受个人风格。",
                )
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.9,
        )
        meta = {
            "final_slot_summaries": [
                {
                    "slot_name": "作者时代流派",
                    "slot_term": "金大受",
                    "description_highlights": [
                        "金大受为南宋前期宁波地区活跃的道释画家。",
                        "其作品以精致工笔、自然协调著称。",
                    ],
                }
            ],
            "retained_facts": [
                {
                    "slot_name": "作者时代流派",
                    "slot_term": "金大受",
                    "fact": "金大受为南宋前期宁波地区活跃的道释画家。",
                    "source": "slot_description",
                }
            ],
        }
        prompt = build_final_appreciation_prompt(
            outputs=[output],
            validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            meta=meta,
            dialogue_state=DialogueState(resolved_questions=["画面中有哪些细节最能体现金大受的个人风格？"]),
        )
        self.assertIn("从作者时代流派看，可围绕金大受这一线索理解", prompt)
        self.assertIn("金大受为南宋前期宁波地区活跃的道释画家", prompt)

    def test_build_final_answer_request_prompt_includes_required_slot_coverage(self) -> None:
        outputs = [
            DomainCoTRecord(
                slot_name="作者时代流派",
                slot_term="金大受",
                analysis_round=1,
                controlled_vocabulary=["金大受", "南宋"],
                visual_anchoring=[EvidenceItem(observation="人物衣纹细劲流畅，设色沉稳。")],
                domain_decoding=[],
                cultural_mapping=[],
                question_coverage=[
                    QuestionCoverage(
                        question="画面中有哪些细节最能体现金大受的个人风格？",
                        answered=True,
                        support="人物形态准确、衣纹线条流畅、设色以中间色为主，显示金大受个人风格。",
                    )
                ],
                unresolved_points=[],
                generated_questions=[],
                statuses=[],
                confidence=0.9,
            ),
            DomainCoTRecord(
                slot_name="尺寸规格/材质形制/收藏地",
                slot_term="东京国立博物馆",
                analysis_round=1,
                controlled_vocabulary=["东京国立博物馆", "绢本设色"],
                visual_anchoring=[EvidenceItem(observation="画面为纵向立轴，四周有织锦边框。")],
                domain_decoding=[],
                cultural_mapping=[],
                question_coverage=[
                    QuestionCoverage(
                        question="该作品现藏何处？",
                        answered=True,
                        support="现藏东京国立博物馆，画面装裱与保存状态符合馆藏古画特征。",
                    )
                ],
                unresolved_points=["未见官方尺寸与收藏编号。"],
                generated_questions=[],
                statuses=[],
                confidence=0.88,
            ),
        ]
        meta = {
            "final_slot_summaries": [
                {
                    "slot_name": "作者时代流派",
                    "slot_term": "金大受",
                    "description_highlights": ["金大受为南宋前期宁波地区活跃的道释画家。"],
                },
                {
                    "slot_name": "尺寸规格/材质形制/收藏地",
                    "slot_term": "东京国立博物馆",
                    "description_highlights": ["本作现藏东京国立博物馆，画面为绢本设色，装裱形式推测为立轴。"],
                },
            ],
            "retained_facts": [
                {
                    "slot_name": "作者时代流派",
                    "slot_term": "金大受",
                    "fact": "金大受为南宋前期宁波地区活跃的道释画家。",
                    "source": "slot_description",
                },
                {
                    "slot_name": "尺寸规格/材质形制/收藏地",
                    "slot_term": "东京国立博物馆",
                    "fact": "本作现藏东京国立博物馆，画面为绢本设色，装裱形式推测为立轴。",
                    "source": "slot_description",
                },
            ],
        }
        prompt = build_final_answer_request_prompt(
            question="请给出完整赏析。",
            outputs=outputs,
            validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            meta=meta,
            dialogue_state=DialogueState(
                resolved_questions=["画面中有哪些细节最能体现金大受的个人风格？", "该作品现藏何处？"],
                unresolved_questions=["官方尺寸与收藏编号仍待确认。"],
            ),
        )
        self.assertIn('"required_slot_coverage"', prompt)
        self.assertIn("宁波地区活跃的道释画家", prompt)
        self.assertIn("东京国立博物馆", prompt)
        self.assertIn("must_cover", prompt)

    def test_meta_payload_trims_large_closed_loop_context(self) -> None:
        payload = meta_payload(
            {
                "retained_facts": [
                    {
                        "slot_name": "作者时代流派",
                        "slot_term": "金大受",
                        "fact": f"宁波活动信息{i}" * 10,
                        "source": "slot_description",
                    }
                    for i in range(8)
                ],
                "post_rag_text_extraction": [
                    {"term": f"术语{i}", "description": f"描述{i}" * 20, "text_evidence": [f"证据{i}" * 10]}
                    for i in range(6)
                ],
                "ontology_updates": [f"本体{i}" * 20 for i in range(8)],
                "downstream_updates": [
                    {
                        "slot_name": "皴法",
                        "reason": "round_table_follow_up",
                        "focus": f"问题{i}" * 20,
                        "status": "applied",
                        "search_queries": [{"query_text": f"检索{i}" * 10}],
                        "resolved_questions": [f"已解决{i}" * 10],
                        "open_questions": [f"未解决{i}" * 10],
                        "notes": [f"说明{i}" * 10],
                    }
                    for i in range(5)
                ],
                "closed_loop_notes": [f"note-{i}" * 20 for i in range(8)],
                "dialogue_turns": [{"role": "system", "content": f"turn-{i}" * 30} for i in range(6)],
                "round_memories": [{"round_index": 1}],
            }
        )

        self.assertEqual(6, len(payload["retained_facts"]))
        self.assertEqual("金大受", payload["retained_facts"][0]["slot_term"])
        self.assertEqual(4, len(payload["post_rag_text_extraction"]))
        self.assertEqual(6, len(payload["ontology_updates"]))
        self.assertEqual(3, len(payload["downstream_updates"]))
        self.assertEqual(6, len(payload["closed_loop_notes"]))
        self.assertEqual(4, len(payload["dialogue_turns"]))

    def test_finalize_result_suppresses_repeated_follow_up_threads_without_new_info(self) -> None:
        api_client = FakeAPIClient(
            enabled=True,
            responses=["## 赏析\n不应在未收敛时触发。"],
        )
        pipeline = DynamicAgentPipeline(
            config=PipelineConfig(final_answer_model="gpt-4.1"),
            api_client=api_client,
        )
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="通过密集短点表现山石质感。"),
            ],
            unresolved_points=["北方山石的空间层次仍需核验"],
            generated_questions=[],
            statuses=[],
            confidence=0.7,
        )
        result = PipelineResult(
            image_path="/tmp/demo.png",
            prepared_image=PreparedImage(path="/tmp/demo.png"),
            slot_schemas=[self.slot_schema],
            domain_outputs=[output],
            cross_validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            routing=RoutingDecision(
                action="PAUSE_COT",
                rationale=["placeholder"],
                paused_slots=[],
                spawned_tasks=[],
                removed_questions=[],
                merged_duplicates=[],
            ),
            dialogue_state=DialogueState(),
            cot_threads=[
                CoTThread(
                    thread_id="皴法-followup-1",
                    slot_name="皴法",
                    slot_term="雨点皴",
                    focus="它如何表现北方山石？",
                    reason="round_table_follow_up",
                    status="RETRY",
                    attempts=2,
                    max_attempts=4,
                    latest_new_info_gain=0,
                    stale_rounds=2,
                )
            ],
            round_memory={},
            final_appreciation_prompt="",
            api_logs=[],
            execution_log=[],
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "皴法",
                    "question": "它如何表现北方山石？",
                    "why": "需要更多证据",
                    "priority": "high",
                    "rag_queries": ["北方山石"],
                }
            ],
            slot_lifecycle_reviews=[
                {"slot_name": "皴法", "status": "ACTIVE", "reason": "仍有未核验细节。"}
            ],
        )

        pipeline._cross_validate = lambda *_args, **_kwargs: validation
        pipeline._review_validation_bundle = lambda *_args, **_kwargs: validation

        finalized = pipeline.finalize_result(result, meta={"final_user_question": "它如何表现北方山石？"})

        self.assertEqual("PAUSE_COT", finalized.routing.action)
        self.assertEqual([], finalized.routing.spawned_tasks)
        self.assertEqual("PAUSED", finalized.cot_threads[0].status)
        self.assertEqual("duplicate_stalled", finalized.cot_threads[0].pause_reason)
        self.assertEqual([], api_client.calls)

    def test_finalize_result_recovers_previous_round_threads_for_duplicate_suppression(self) -> None:
        pipeline = DynamicAgentPipeline(
            config=PipelineConfig(final_answer_model="gpt-4.1"),
            api_client=FakeAPIClient(enabled=False),
        )
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=True, support="通过密集短点表现山石质感。"),
            ],
            unresolved_points=[],
            generated_questions=[],
            statuses=[],
            confidence=0.7,
        )
        result = PipelineResult(
            image_path="/tmp/demo.png",
            prepared_image=PreparedImage(path="/tmp/demo.png"),
            slot_schemas=[self.slot_schema],
            domain_outputs=[output],
            cross_validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            routing=RoutingDecision(
                action="PAUSE_COT",
                rationale=["placeholder"],
                paused_slots=[],
                spawned_tasks=[],
                removed_questions=[],
                merged_duplicates=[],
            ),
            dialogue_state=DialogueState(),
            cot_threads=[
                CoTThread(
                    thread_id="皴法-overview-1",
                    slot_name="皴法",
                    slot_term="雨点皴",
                    focus="它如何表现北方山石？",
                    reason="slot_overview",
                    status="ANSWERED",
                )
            ],
            round_memory={},
            final_appreciation_prompt="",
            api_logs=[],
            execution_log=[],
        )
        validation = CrossValidationResult(
            issues=[],
            semantic_duplicates=[],
            missing_points=[],
            rag_terms=[],
            removed_questions=[],
            round_table_follow_up_questions=[
                {
                    "slot_name": "皴法",
                    "question": "它如何表现北方山石？",
                    "why": "需要更多证据",
                    "priority": "high",
                    "rag_queries": ["北方山石"],
                }
            ],
            slot_lifecycle_reviews=[
                {"slot_name": "皴法", "status": "ACTIVE", "reason": "仍有未核验细节。"}
            ],
        )

        pipeline._cross_validate = lambda *_args, **_kwargs: validation
        pipeline._review_validation_bundle = lambda *_args, **_kwargs: validation

        finalized = pipeline.finalize_result(
            result,
            meta={
                "round_memories": [
                    {
                        "round_index": 1,
                        "threads": [
                            {
                                "thread_id": "皴法-followup-1",
                                "slot_name": "皴法",
                                "focus": "它如何表现北方山石？",
                                "reason": "round_table_follow_up",
                                "status": "RETRY",
                                "attempts": 2,
                                "latest_new_info_gain": 0,
                                "stale_rounds": 2,
                                "pause_reason": "",
                                "answered_questions": [],
                                "unresolved_points": [],
                                "latest_summary": "未获得新信息",
                            }
                        ],
                    }
                ],
                "final_user_question": "它如何表现北方山石？",
            },
        )

        self.assertEqual("PAUSE_COT", finalized.routing.action)
        self.assertEqual([], finalized.routing.spawned_tasks)

    def test_finalize_result_populates_routing_after_cot_only_run(self) -> None:
        output = DomainCoTRecord(
            slot_name="皴法",
            slot_term="雨点皴",
            analysis_round=1,
            controlled_vocabulary=["雨点皴"],
            visual_anchoring=[EvidenceItem(observation="山石表面密集短点排列")],
            domain_decoding=[],
            cultural_mapping=[],
            question_coverage=[
                QuestionCoverage(question="它如何表现北方山石？", answered=False),
                QuestionCoverage(question="变体如何搭配？", answered=True, support="已回答"),
            ],
            unresolved_points=["北方山石的空间层次仍需补充"],
            generated_questions=[],
            statuses=[],
            confidence=0.7,
        )
        result = PipelineResult(
            image_path="/tmp/demo.png",
            prepared_image=PreparedImage(path="/tmp/demo.png"),
            slot_schemas=[self.slot_schema],
            domain_outputs=[output],
            cross_validation=CrossValidationResult(
                issues=[],
                semantic_duplicates=[],
                missing_points=[],
                rag_terms=[],
                removed_questions=[],
            ),
            routing=RoutingDecision(
                action="PAUSE_COT",
                rationale=["placeholder"],
                paused_slots=[],
                spawned_tasks=[],
                removed_questions=[],
                merged_duplicates=[],
            ),
            dialogue_state=DialogueState(),
            cot_threads=[
                CoTThread(
                    thread_id="皴法-overview-1",
                    slot_name="皴法",
                    slot_term="雨点皴",
                    focus="它如何表现北方山石？",
                    reason="slot_overview",
                    status="ANSWERED",
                )
            ],
            round_memory={},
            final_appreciation_prompt="",
            api_logs=[],
            execution_log=[],
        )

        finalized = self.pipeline.finalize_result(result, meta={})

        self.assertEqual("SPAWN_COT", finalized.routing.action)
        self.assertTrue(finalized.routing.spawned_tasks)
        self.assertEqual("它如何表现北方山石？", finalized.routing.spawned_tasks[0].prompt_focus)
        self.assertTrue(finalized.execution_log)

    def test_save_result_omits_memory_markdown_file(self) -> None:
        result = PipelineResult(
            image_path="/tmp/demo.png",
            prepared_image=PreparedImage(path="/tmp/demo.png"),
            slot_schemas=[self.slot_schema],
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
                rationale=["placeholder"],
                paused_slots=[],
                spawned_tasks=[],
                removed_questions=[],
                merged_duplicates=[],
            ),
            dialogue_state=DialogueState(),
            cot_threads=[],
            round_memory={"round_index": 1, "resolved_questions": ["问题A"]},
            final_appreciation_prompt="最终提示词",
            api_logs=[],
            execution_log=[],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outputs = self.pipeline.save_result(result, output_dir=tmpdir)
            run_dir = Path(outputs["run_dir"])

            self.assertNotIn("memory", outputs)
            self.assertNotIn("routing", outputs)
            self.assertNotIn("report", outputs)
            self.assertFalse((run_dir / "memory.md").exists())
            self.assertFalse((run_dir / "routing.json").exists())
            self.assertFalse((run_dir / "report.json").exists())
            self.assertTrue((run_dir / "memory.json").exists())
            payload = json.loads((run_dir / "memory.json").read_text(encoding="utf-8"))
            self.assertEqual(1, payload["round_index"])
            dialogue_state = json.loads((run_dir / "dialogue_state.json").read_text(encoding="utf-8"))
            self.assertEqual("PAUSE_COT", dialogue_state["routing"]["action"])
            self.assertEqual([], dialogue_state["execution_log"])
            self.assertEqual("/tmp/demo.png", dialogue_state["image_path"])


if __name__ == "__main__":
    unittest.main()
