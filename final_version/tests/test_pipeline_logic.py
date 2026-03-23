from __future__ import annotations

import unittest

from src.slots_v2.models import (
    CoTThread,
    CrossValidationIssue,
    CrossValidationResult,
    DialogueState,
    DomainCoTRecord,
    EvidenceItem,
    MappingItem,
    PipelineConfig,
    QuestionCoverage,
    SlotSchema,
    SpawnTask,
)
from src.slots_v2.pipeline import DynamicAgentPipeline


class PipelineLogicTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = DynamicAgentPipeline(config=PipelineConfig(max_threads_per_round=3, thread_attempt_limit=2))
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
        self.assertTrue(tasks[0].rag_terms)

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


if __name__ == "__main__":
    unittest.main()
