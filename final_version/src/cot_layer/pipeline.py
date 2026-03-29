from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
from time import perf_counter
from typing import Any

from ..cross_validation_layer.service import (
    augment_round_table_review as run_round_table_review,
    cross_validate as run_cross_validate,
    parse_round_table_review as parse_round_table_review_payload,
    round_table_priority as normalize_round_table_priority,
)
from ..reflection_layer.service import (
    build_routing as build_reflection_routing,
    check_convergence as check_reflection_convergence,
    clean_search_query as clean_reflection_search_query,
    fallback_rag_terms as build_fallback_rag_terms,
    find_matching_thread as find_reflection_matching_thread,
    generate_final_appreciation_prompt as build_final_reflection_prompt,
    normalize_search_queries as normalize_reflection_queries,
    normalize_web_search_queries as normalize_reflection_web_queries,
    parse_rag_terms_response as parse_reflection_rag_terms,
    plan_spawn_tasks as plan_reflection_spawn_tasks,
    review_validation_bundle as review_reflection_validation_bundle,
    review_slot_lifecycle as review_reflection_slot_lifecycle,
    suppress_redundant_tasks as suppress_reflection_tasks,
    sync_threads_with_tasks as sync_reflection_threads,
    task_already_resolved as is_reflection_task_resolved,
    task_rag_terms as plan_task_rag_terms,
    should_pause_duplicate_task as should_pause_reflection_task,
)
from ..reflection_layer.prompt_builder import build_final_appreciation_prompt as build_final_prompt_fallback
from ..common.prompt_utils import build_slot_summary_payload
from .image_utils import prepare_image
from .models import (
    CoTThread,
    CrossValidationIssue,
    CrossValidationResult,
    DecodingItem,
    DialogueState,
    DialogueTurn,
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
from .new_api_client import NewAPIClient
from .prompt_builder import build_domain_cot_prompt
from .schema_loader import extract_controlled_vocabulary, extract_slot_terms, load_slot_schemas
from .vlm_runner import VLMRunner


_DYNASTIES = ("先秦", "汉", "魏晋", "隋", "唐", "五代", "北宋", "南宋", "元", "明", "清", "近现代", "当代")
_THREAD_STATUS_ORDER = {"OPEN": 0, "RETRY": 1, "BLOCKED": 2, "PAUSED": 3, "ANSWERED": 4, "MERGED": 5}
_DUPLICATE_STALE_ROUNDS = 2


class DynamicAgentPipeline:
    def __init__(self, config: PipelineConfig | None = None, api_client: NewAPIClient | None = None) -> None:
        self.config = config or PipelineConfig()
        self.api_client = api_client or NewAPIClient()
        self.vlm_runner = VLMRunner(self.api_client)
        self._rag_term_cache: dict[tuple[str, str, str, tuple[str, ...]], list[str]] = {}
        self._retrieval_plan_cache: dict[tuple[str, str, str], dict[str, Any]] = {}

    def run(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        meta = meta or {}
        slot_schemas = load_slot_schemas(self.config.slots_file)
        prepared_image = prepare_image(
            image_path=image_path,
            max_pixel=max(1, int(self.config.max_pixel or 1)),
            resize=bool(self.config.resize_image),
        )
        api_logs: list[dict[str, Any]] = []
        execution_log: list[dict[str, Any]] = [
            {
                "stage": "prepare_image",
                "prepared_image": asdict(prepared_image),
                "slot_count": len(slot_schemas),
            }
        ]

        dialogue_state = self._initialize_dialogue_state(meta)
        threads = self._initialize_threads(slot_schemas, meta=meta)
        dialogue_state.threads = [self._clone_thread(thread) for thread in threads]
        self._print_progress(
            stage="init",
            round_index=0,
            max_rounds=1,
            threads=threads,
            slot_schemas=slot_schemas,
            outputs=[],
            note=f"prepared_image={prepared_image.path}",
        )

        domain_outputs: list[DomainCoTRecord] = []
        validation = CrossValidationResult(issues=[], semantic_duplicates=[], missing_points=[], rag_terms=[], removed_questions=[])
        routing = RoutingDecision(
            action="PAUSE_COT",
            rationale=["CoT-only run completed; external orchestration should finalize validation and routing."],
            paused_slots=[],
            spawned_tasks=[],
            removed_questions=[],
            merged_duplicates=[],
        )

        executable_threads = [thread for thread in threads if thread.slot_name in {slot.slot_name for slot in slot_schemas}]
        self._print_progress(
            stage="round_start",
            round_index=1,
            max_rounds=1,
            threads=threads,
            slot_schemas=slot_schemas,
            outputs=domain_outputs,
            executable_threads=executable_threads,
        )
        round_stage_started = perf_counter()
        round_outputs, new_info_count = self._run_thread_round(
            threads=executable_threads,
            prepared_image=prepared_image,
            meta=meta,
            api_logs=api_logs,
            analysis_round=1,
        )
        stage_timings = {"domain_cot_parallel_s": round(perf_counter() - round_stage_started, 4)}
        domain_outputs = self._merge_domain_outputs(domain_outputs, round_outputs)
        if new_info_count > 0:
            dialogue_state.no_new_info_rounds = 0
        else:
            dialogue_state.no_new_info_rounds = 1
        dialogue_state.resolved_questions = self._collect_answered_questions(domain_outputs)
        dialogue_state.unresolved_questions = self._collect_unresolved_questions(domain_outputs, validation)
        dialogue_state.final_round_index = 1
        dialogue_state.threads = [self._clone_thread(thread) for thread in threads]
        dialogue_state.turns.append(
            DialogueTurn(
                round_index=1,
                active_thread_ids=[],
                executed_thread_ids=[thread.thread_id for thread in executable_threads],
                spawned_thread_ids=[],
                answered_thread_ids=[thread.thread_id for thread in executable_threads if thread.status == "ANSWERED"],
                blocked_thread_ids=[thread.thread_id for thread in executable_threads if thread.status == "BLOCKED"],
                paused_thread_ids=[thread.thread_id for thread in executable_threads if thread.status == "PAUSED"],
                merged_thread_ids=[],
                routing_action=routing.action,
                notes=routing.rationale,
                new_information_count=new_info_count,
                convergence_snapshot={"converged": False, "reason": "cot_only_pending_external_postprocess"},
            )
        )
        execution_log.append(
            {
                "stage": "round_1",
                "routing": asdict(routing),
                "new_information_count": new_info_count,
                "stage_timings": stage_timings,
                "note": "cot_only_round",
            }
        )
        self._print_progress(
            stage="round_end",
            round_index=1,
            max_rounds=1,
            threads=threads,
            slot_schemas=slot_schemas,
            outputs=domain_outputs,
            executable_threads=executable_threads,
            routing=routing,
            note=f"new_info={new_info_count} slowest_substage={self._slowest_stage_label(stage_timings)}",
        )

        round_memory = self._build_round_memory(
            domain_outputs,
            validation,
            dialogue_state,
            threads,
            prior_round_memories=meta.get("round_memories", []),
        )
        final_appreciation_prompt = build_final_prompt_fallback(
            domain_outputs,
            validation,
            meta,
            dialogue_state,
        )

        return PipelineResult(
            image_path=image_path,
            prepared_image=prepared_image,
            slot_schemas=slot_schemas,
            domain_outputs=domain_outputs,
            cross_validation=validation,
            routing=routing,
            dialogue_state=dialogue_state,
            cot_threads=threads,
            round_memory=round_memory,
            final_appreciation_prompt=final_appreciation_prompt,
            api_logs=api_logs,
            execution_log=execution_log,
        )

    def finalize_result(self, result: PipelineResult, *, meta: dict | None = None) -> PipelineResult:
        meta = meta or {}
        planning_threads = self._planning_reference_threads(result, meta)
        stage_timings: dict[str, float] = {}
        round_stage_started = perf_counter()
        validation = self._cross_validate(result.domain_outputs, result.slot_schemas, meta, result.api_logs)
        stage_timings["cross_validate_s"] = round(perf_counter() - round_stage_started, 4)
        round_stage_started = perf_counter()
        validation = self._review_validation_bundle(
            result.slot_schemas,
            result.domain_outputs,
            validation,
            meta,
            result.api_logs,
            use_llm=True,
        )
        stage_timings["validation_review_s"] = round(perf_counter() - round_stage_started, 4)
        round_stage_started = perf_counter()
        # CoT-only mode lets the outer coordinator own task queues, so follow-up planning
        # should not be suppressed just because the current round already executed a thread.
        candidate_tasks = self._plan_spawn_tasks(result.domain_outputs, validation, planning_threads, result.api_logs)
        stage_timings["plan_spawn_tasks_s"] = round(perf_counter() - round_stage_started, 4)
        round_stage_started = perf_counter()
        candidate_tasks, duplicate_paused_thread_ids = self._suppress_redundant_tasks(candidate_tasks, planning_threads)
        stage_timings["suppress_duplicate_tasks_s"] = round(perf_counter() - round_stage_started, 4)
        round_stage_started = perf_counter()
        convergence = self._check_convergence(
            result.slot_schemas,
            result.domain_outputs,
            validation,
            result.cot_threads,
            result.dialogue_state,
            candidate_tasks,
        )
        stage_timings["check_convergence_s"] = round(perf_counter() - round_stage_started, 4)
        routing = self._build_routing(result.domain_outputs, validation, candidate_tasks, convergence)

        result.cross_validation = validation
        result.routing = routing
        result.dialogue_state.resolved_questions = self._collect_answered_questions(result.domain_outputs)
        result.dialogue_state.unresolved_questions = self._collect_unresolved_questions(result.domain_outputs, validation)
        result.dialogue_state.removed_questions = validation.removed_questions
        result.dialogue_state.merged_duplicates = validation.semantic_duplicates
        result.dialogue_state.converged = convergence["converged"]
        result.dialogue_state.convergence_reason = convergence["reason"]
        result.dialogue_state.threads = [self._clone_thread(thread) for thread in result.cot_threads]

        if result.dialogue_state.turns:
            latest_turn = result.dialogue_state.turns[-1]
            latest_turn.active_thread_ids = []
            latest_turn.spawned_thread_ids = []
            latest_turn.paused_thread_ids = self._dedupe_text_list(
                [thread.thread_id for thread in result.cot_threads if thread.status == "PAUSED"] + duplicate_paused_thread_ids
            )
            latest_turn.answered_thread_ids = [thread.thread_id for thread in result.cot_threads if thread.status == "ANSWERED"]
            latest_turn.blocked_thread_ids = [thread.thread_id for thread in result.cot_threads if thread.status == "BLOCKED"]
            latest_turn.routing_action = routing.action
            latest_turn.notes = routing.rationale
            latest_turn.convergence_snapshot = convergence

        result.round_memory = self._build_round_memory(
            result.domain_outputs,
            validation,
            result.dialogue_state,
            result.cot_threads,
            prior_round_memories=meta.get("round_memories", []),
        )
        final_meta = dict(meta)
        final_meta["final_slot_summaries"] = build_slot_summary_payload(result.slot_schemas)
        fallback_prompt = build_final_prompt_fallback(
            result.domain_outputs,
            validation,
            final_meta,
            result.dialogue_state,
        )
        if convergence["converged"]:
            result.final_appreciation_prompt = self._generate_final_appreciation_prompt(
                result.domain_outputs,
                validation,
                final_meta,
                result.dialogue_state,
                api_logs=result.api_logs,
            )
        else:
            result.final_appreciation_prompt = fallback_prompt
        result.execution_log.append(
            {
                "stage": "postprocess",
                "routing": asdict(routing),
                "convergence": convergence,
                "duplicate_paused_thread_ids": duplicate_paused_thread_ids,
                "stage_timings": stage_timings,
            }
        )
        return result

    def _planning_reference_threads(self, result: PipelineResult, meta: dict[str, Any]) -> list[CoTThread]:
        planning_threads = [thread for thread in result.cot_threads if thread.reason != "slot_overview"]
        if planning_threads:
            return planning_threads
        return self._threads_from_round_memory(result.slot_schemas, self._latest_round_memory(meta.get("round_memories", [])))

    def _threads_from_round_memory(self, slot_schemas: list[SlotSchema], memory: dict[str, Any]) -> list[CoTThread]:
        if not isinstance(memory, dict):
            return []
        raw_threads = memory.get("threads", [])
        if not isinstance(raw_threads, list):
            return []
        slot_terms = {
            slot.slot_name: slot.slot_term
            for slot in slot_schemas
        }
        recovered: list[CoTThread] = []
        for item in raw_threads:
            if not isinstance(item, dict):
                continue
            slot_name = str(item.get("slot_name", "")).strip()
            focus = str(item.get("focus", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if not slot_name or not focus or reason == "slot_overview":
                continue
            recovered.append(
                CoTThread(
                    thread_id=str(item.get("thread_id", "")).strip() or f"{self._slug(slot_name)}-memory-{len(recovered) + 1}",
                    slot_name=slot_name,
                    slot_term=slot_terms.get(slot_name, slot_name),
                    focus=focus,
                    reason=reason,
                    status=str(item.get("status", "OPEN")).strip() or "OPEN",
                    attempts=max(0, int(item.get("attempts", 0) or 0)),
                    max_attempts=max(1, int(self.config.thread_attempt_limit or 1)),
                    latest_new_info_gain=int(item.get("latest_new_info_gain", 0) or 0),
                    stale_rounds=max(0, int(item.get("stale_rounds", 0) or 0)),
                    pause_reason=str(item.get("pause_reason", "")).strip(),
                    answered_questions=self._dedupe_text_list(item.get("answered_questions", [])),
                    unresolved_points=self._dedupe_text_list(item.get("unresolved_points", [])),
                    latest_summary=str(item.get("latest_summary", "")).strip(),
                )
            )
        return recovered

    def _print_progress(
        self,
        *,
        stage: str,
        round_index: int,
        max_rounds: int,
        threads: list[CoTThread],
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        executable_threads: list[CoTThread] | None = None,
        routing: RoutingDecision | None = None,
        note: str = "",
    ) -> None:
        solved_count, remaining_count, total_count = self._question_progress(slot_schemas, outputs)
        executable_count = len(executable_threads or [])
        active_count = len([thread for thread in threads if thread.status in {"OPEN", "RETRY", "BLOCKED"}])
        summary = (
            f"[slots_v2] stage={stage} round={round_index}/{max_rounds} "
            f"cot_total={len(threads)} cot_active={active_count} cot_running={executable_count} "
            f"questions_solved={solved_count} questions_remaining={remaining_count}/{total_count}"
        )
        if routing is not None:
            summary += f" routing={routing.action} converged={str(routing.converged).lower()}"
        if note:
            summary += f" note={note}"
        print(summary, flush=True)

    def _question_progress(
        self,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
    ) -> tuple[int, int, int]:
        total_questions = [question for slot in slot_schemas for question in slot.specific_questions]
        answered_questions = self._collect_answered_questions(outputs)
        solved_count = sum(
            1 for question in total_questions if any(self._text_similarity(question, answered) >= 0.8 for answered in answered_questions)
        )
        total_count = len(total_questions)
        remaining_count = max(0, total_count - solved_count)
        return solved_count, remaining_count, total_count

    @staticmethod
    def _slowest_stage_label(stage_timings: dict[str, float]) -> str:
        if not stage_timings:
            return "none"
        stage_name, duration = max(stage_timings.items(), key=lambda item: item[1])
        return f"{stage_name}:{duration:.2f}s"

    def _initialize_dialogue_state(self, meta: dict) -> DialogueState:
        history: list[str] = []
        dialogue_turns = meta.get("dialogue_turns", [])
        if isinstance(dialogue_turns, list):
            for item in dialogue_turns:
                if isinstance(item, dict):
                    role = str(item.get("role", "user")).strip() or "user"
                    content = str(item.get("content", "")).strip()
                    if content:
                        history.append(f"{role}: {content}")
                else:
                    text = str(item).strip()
                    if text:
                        history.append(text)
        latest_user_text = str(meta.get("latest_user_message", "")).strip()
        if latest_user_text:
            history.append(f"user: {latest_user_text}")
        return DialogueState(conversation_history=history)

    def _initialize_threads(self, slot_schemas: list[SlotSchema], meta: dict | None = None) -> list[CoTThread]:
        task_threads = self._initialize_threads_from_meta(slot_schemas, meta or {})
        if task_threads:
            return task_threads
        return self._initialize_overview_threads(slot_schemas)

    def _initialize_overview_threads(self, slot_schemas: list[SlotSchema]) -> list[CoTThread]:
        threads: list[CoTThread] = []
        for index, slot in enumerate(slot_schemas, start=1):
            lifecycle = str(slot.metadata.get("lifecycle", "ACTIVE")).strip().upper() or "ACTIVE"
            if lifecycle in {"STABLE", "CLOSED"}:
                continue
            slot_terms = extract_slot_terms(slot.slot_term, slot.metadata)
            slot_terms_phrase = "、".join(slot_terms) if slot_terms else (slot.slot_term or slot.slot_name)
            primary_focus = slot.specific_questions[0] if slot.specific_questions else f"围绕 {slot_terms_phrase} 完成整体分析。"
            priority = 2 + min(3, len(slot.specific_questions))
            if float(slot.metadata.get("confidence", 0.0) or 0.0) >= 0.9:
                priority += 1
            threads.append(
                CoTThread(
                    thread_id=f"{self._slug(slot.slot_name)}-overview-{index}",
                    slot_name=slot.slot_name,
                    slot_term=slot.slot_term,
                    focus=primary_focus,
                    reason="slot_overview",
                    rag_terms=slot.controlled_vocabulary[:5],
                    priority=priority,
                    max_attempts=max(1, int(self.config.thread_attempt_limit or 1)),
                )
            )
        return threads

    def _initialize_threads_from_meta(self, slot_schemas: list[SlotSchema], meta: dict) -> list[CoTThread]:
        slot_map = {slot.slot_name: slot for slot in slot_schemas}
        raw_tasks = meta.get("pending_cot_tasks", [])
        if not isinstance(raw_tasks, list):
            return []
        threads: list[CoTThread] = []
        for index, item in enumerate(raw_tasks, start=1):
            if not isinstance(item, dict):
                continue
            slot_name = str(item.get("slot_name", "")).strip()
            focus = str(item.get("prompt_focus", "")).strip()
            if not slot_name or not focus or slot_name not in slot_map:
                continue
            slot = slot_map[slot_name]
            threads.append(
                CoTThread(
                    thread_id=f"{self._slug(slot_name)}-{self._slug(str(item.get('reason', 'external_cot_task')))}-{index}",
                    slot_name=slot_name,
                    slot_term=slot.slot_term,
                    focus=focus,
                    reason=str(item.get("reason", "external_cot_task")).strip() or "external_cot_task",
                    rag_terms=self._dedupe_text_list(item.get("rag_terms", [])),
                    priority=max(1, int(item.get("priority", 1) or 1)),
                    max_attempts=max(1, int(self.config.thread_attempt_limit or 1)),
                    parent_thread_id=str(item.get("source_thread_id", "")).strip() or None,
                )
            )
        return threads

    def _select_threads_for_round(self, threads: list[CoTThread]) -> list[CoTThread]:
        candidates = [
            thread
            for thread in threads
            if thread.status in {"OPEN", "RETRY", "BLOCKED"} and thread.attempts < thread.max_attempts
        ]
        candidates.sort(
            key=lambda thread: (
                _THREAD_STATUS_ORDER.get(thread.status, 99),
                -int(thread.priority),
                int(thread.attempts),
                int(thread.last_round),
            )
        )
        limit = max(1, int(self.config.max_threads_per_round or 1))
        return candidates[:limit]

    def _run_thread_round(
        self,
        *,
        threads: list[CoTThread],
        prepared_image: PreparedImage,
        meta: dict,
        api_logs: list[dict[str, Any]],
        analysis_round: int,
    ) -> tuple[list[DomainCoTRecord], int]:
        slot_map = {slot.slot_name: slot for slot in load_slot_schemas(self.config.slots_file)}
        outputs: list[DomainCoTRecord] = []
        new_information_count = 0
        executable_threads: list[CoTThread] = []
        for thread in threads:
            if thread.slot_name in slot_map:
                executable_threads.append(thread)
                continue
            thread.status = "PAUSED"
            thread.pause_reason = "slot_schema_missing"
            thread.latest_new_info_gain = 0
        if not executable_threads:
            return outputs, new_information_count
        workers = min(max(1, len(executable_threads)), max(1, int(self.config.concurrent_workers or 1)))
        started_at = perf_counter()
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    self._run_single_thread,
                    slot_map[thread.slot_name],
                    thread,
                    prepared_image.path,
                    meta,
                    analysis_round,
                ): thread
                for thread in executable_threads
            }
            for future in as_completed(futures):
                thread = futures[future]
                record, logs = future.result()
                api_logs.extend(logs)
                outputs.append(record)
                new_information_count += self._update_thread(thread, record, analysis_round)
        outputs.sort(key=lambda item: item.slot_name)
        print(
            f"[slots_v2_stage] stage=domain_cot_parallel round={analysis_round} "
            f"cot_running={len(executable_threads)} elapsed_s={round(perf_counter() - started_at, 2):.2f}",
            flush=True,
        )
        return outputs, new_information_count

    def _run_single_thread(
        self,
        slot: SlotSchema,
        thread: CoTThread,
        image_path: str,
        meta: dict,
        analysis_round: int,
    ) -> tuple[DomainCoTRecord, list[dict[str, Any]]]:
        prompt_questions = self._domain_prompt_questions(slot, thread)
        attach_image = self._should_attach_image_for_thread(thread, analysis_round)
        prompt = build_domain_cot_prompt(
            slot=slot,
            meta=meta,
            focus_question=thread.focus,
            analysis_round=analysis_round,
            specific_questions=prompt_questions,
            retrieval_gain_enabled=bool(getattr(self.config, "retrieval_gain", False)),
            web_search_enabled=bool(getattr(self.config, "enable_web_search", False)),
            thread_context={
                "thread_id": thread.thread_id,
                "reason": thread.reason,
                "priority": thread.priority,
                "attempts": thread.attempts,
                "rag_terms": thread.rag_terms,
                "latest_summary": thread.latest_summary,
                "prompt_questions": prompt_questions,
                "image_mode": "vision" if attach_image else "text_only",
            },
        )
        raw, api_log = self.vlm_runner.analyze(
            image_path=image_path if attach_image else None,
            prompt=prompt,
            system_prompt="你是严谨的中国画多线程分析引擎，只能基于图像证据输出内容。",
            temperature=self.config.domain_temperature,
            model=self.config.domain_model,
            stage=f"domain_cot_round_{analysis_round}",
        )
        api_log["slot_name"] = slot.slot_name
        api_log["analysis_round"] = analysis_round
        api_log["thread_id"] = thread.thread_id
        api_log["prompt_question_count"] = len(prompt_questions)
        api_log["text_only"] = not attach_image
        if raw.strip():
            return self._parse_domain_record(slot, raw, analysis_round), [api_log]
        return self._fallback_domain_record(slot, analysis_round, reason=api_log.get("error") or "empty_response"), [api_log]

    def _domain_prompt_questions(self, slot: SlotSchema, thread: CoTThread) -> list[str]:
        focus = str(thread.focus or "").strip()
        if thread.reason == "slot_overview":
            return list(slot.specific_questions[:3])
        latest_record = thread.latest_record if isinstance(thread.latest_record, dict) else {}
        unresolved_questions = [
            str(item.get("question", "")).strip()
            for item in latest_record.get("question_coverage", [])
            if isinstance(item, dict) and not bool(item.get("answered", False)) and str(item.get("question", "")).strip()
        ]
        if unresolved_questions:
            scoped = []
            if focus:
                scoped.append(focus)
            scoped.extend(unresolved_questions[:2])
            return self._dedupe_text_list(scoped)[:3]
        return [focus] if focus else list(slot.specific_questions[:1])

    @staticmethod
    def _should_attach_image_for_thread(thread: CoTThread, analysis_round: int) -> bool:
        if analysis_round <= 1:
            return True
        latest_record = thread.latest_record if isinstance(thread.latest_record, dict) else {}
        visual_anchoring = latest_record.get("visual_anchoring", []) if isinstance(latest_record.get("visual_anchoring", []), list) else []
        latest_confidence = float(getattr(thread, "latest_confidence", 0.0) or 0.0)
        if visual_anchoring and latest_confidence >= 0.35:
            return False
        return True

    def _parse_domain_record(self, slot: SlotSchema, raw: str, analysis_round: int) -> DomainCoTRecord:
        data = self._extract_json_object(raw)
        if not data:
            return self._fallback_domain_record(slot, analysis_round, reason="json_parse_failed", raw=raw)

        visual = []
        for item in data.get("visual_anchoring", []):
            if not isinstance(item, dict):
                continue
            observation = str(item.get("observation", "")).strip()
            if not observation:
                continue
            visual.append(
                EvidenceItem(
                    observation=observation,
                    evidence=str(item.get("evidence", "")).strip(),
                    position=str(item.get("position", "")).strip(),
                )
            )

        decoding = []
        statuses: list[str] = []
        for item in data.get("domain_decoding", []):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            explanation = str(item.get("explanation", "")).strip()
            if not term and not explanation:
                continue
            status = str(item.get("status", "IDENTIFIED")).strip() or "IDENTIFIED"
            reason = str(item.get("reason", "")).strip()
            decoding.append(DecodingItem(term=term, explanation=explanation, status=status, reason=reason))
            if status != "IDENTIFIED":
                statuses.append(status)

        mapping = []
        for item in data.get("cultural_mapping", []):
            if not isinstance(item, dict):
                continue
            insight = str(item.get("insight", "")).strip()
            if not insight:
                continue
            mapping.append(
                MappingItem(
                    insight=insight,
                    basis=str(item.get("basis", "")).strip(),
                    risk_note=str(item.get("risk_note", "")).strip(),
                )
            )

        coverage: list[QuestionCoverage] = [QuestionCoverage(question=question, answered=False, support="") for question in slot.specific_questions]
        incoming = data.get("specific_question_coverage", [])
        if isinstance(incoming, list):
            merged: dict[str, QuestionCoverage] = {item.question: item for item in coverage}
            for item in incoming:
                if not isinstance(item, dict):
                    continue
                question = str(item.get("question", "")).strip()
                if not question:
                    continue
                merged[question] = QuestionCoverage(
                    question=question,
                    answered=bool(item.get("answered", False)),
                    support=str(item.get("support", "")).strip(),
                )
            coverage = list(merged.values())

        unresolved_points = self._dedupe_text_list(data.get("unresolved_points", []))
        generated_questions: list[str] = []
        confidence = self._safe_confidence(data.get("confidence", 0.0))
        controlled_used = self._dedupe_text_list(data.get("controlled_vocabulary_used", [])) or slot.controlled_vocabulary
        if bool(getattr(self.config, "retrieval_gain", False)):
            retrieval_gain = data.get("retrieval_gain", {})
            if not isinstance(retrieval_gain, dict):
                retrieval_gain = {}
            retrieval_gain_terms = self._dedupe_text_list(retrieval_gain.get("related_terms", []))
            retrieval_gain_queries = self._normalize_search_queries(retrieval_gain.get("search_queries", []))
            retrieval_gain_mode = self._normalize_retrieval_mode(retrieval_gain.get("retrieval_mode"))
            retrieval_gain_web_queries = self._normalize_web_search_queries(retrieval_gain.get("web_queries", []))
            retrieval_gain_focus = str(retrieval_gain.get("focus", "")).strip()
            retrieval_gain_reason = str(retrieval_gain.get("reason", "")).strip()
            retrieval_gain_has_value = self._parse_truthy_flag(retrieval_gain.get("has_new_value", False))
        else:
            retrieval_gain_terms = []
            retrieval_gain_queries = []
            retrieval_gain_mode = "rag"
            retrieval_gain_web_queries = []
            retrieval_gain_focus = ""
            retrieval_gain_reason = ""
            retrieval_gain_has_value = False

        return DomainCoTRecord(
            slot_name=slot.slot_name,
            slot_term=slot.slot_term,
            analysis_round=analysis_round,
            controlled_vocabulary=controlled_used,
            visual_anchoring=visual,
            domain_decoding=decoding,
            cultural_mapping=mapping,
            question_coverage=coverage,
            unresolved_points=unresolved_points,
            generated_questions=generated_questions,
            statuses=self._dedupe_text_list(statuses),
            confidence=confidence,
            retrieval_gain_focus=retrieval_gain_focus,
            retrieval_gain_terms=retrieval_gain_terms,
            retrieval_gain_queries=retrieval_gain_queries,
            retrieval_gain_mode=retrieval_gain_mode,
            retrieval_gain_web_queries=retrieval_gain_web_queries,
            retrieval_gain_reason=retrieval_gain_reason,
            retrieval_gain_has_value=retrieval_gain_has_value,
            raw_response=raw,
        )

    def _fallback_domain_record(
        self,
        slot: SlotSchema,
        analysis_round: int,
        reason: str,
        raw: str = "",
    ) -> DomainCoTRecord:
        return DomainCoTRecord(
            slot_name=slot.slot_name,
            slot_term=slot.slot_term,
            analysis_round=analysis_round,
            controlled_vocabulary=slot.controlled_vocabulary,
            visual_anchoring=[],
            domain_decoding=[
                DecodingItem(
                    term=slot.slot_term or slot.slot_name,
                    explanation="当前轮次未能产出可靠视觉解码结果。",
                    status="UNIDENTIFIABLE_FEATURE",
                    reason=reason,
                )
            ],
            cultural_mapping=[],
            question_coverage=[QuestionCoverage(question=question, answered=False, support="") for question in slot.specific_questions],
            unresolved_points=[f"{slot.slot_name} 尚待补充视觉证据"],
            generated_questions=[],
            statuses=["UNIDENTIFIABLE_FEATURE"],
            confidence=0.0,
            retrieval_gain_focus="",
            retrieval_gain_terms=[],
            retrieval_gain_queries=[],
            retrieval_gain_mode="rag",
            retrieval_gain_web_queries=[],
            retrieval_gain_reason="",
            retrieval_gain_has_value=False,
            raw_response=raw,
        )

    def _update_thread(self, thread: CoTThread, record: DomainCoTRecord, analysis_round: int) -> int:
        previous_answered = set(thread.answered_questions)
        previous_evidence_count = int(thread.evidence_count)
        previous_decoding_terms = {
            str(item.get("term", "")).strip()
            for item in thread.latest_record.get("domain_decoding", [])
            if isinstance(item, dict) and str(item.get("term", "")).strip()
        }
        previous_mapping_insights = {
            str(item.get("insight", "")).strip()
            for item in thread.latest_record.get("cultural_mapping", [])
            if isinstance(item, dict) and str(item.get("insight", "")).strip()
        }

        thread.attempts += 1
        thread.last_round = analysis_round
        thread.latest_confidence = record.confidence
        thread.evidence_count = len(record.visual_anchoring)
        thread.answered_questions = self._dedupe_text_list(thread.answered_questions + self._answered_questions_from_record(record))
        thread.unresolved_points = record.unresolved_points
        thread.latest_summary = self._record_summary(record)
        thread.latest_record = asdict(record)
        current_decoding_terms = {item.term.strip() for item in record.domain_decoding if item.term.strip()}
        current_mapping_insights = {item.insight.strip() for item in record.cultural_mapping if item.insight.strip()}
        info_gain = (
            len(set(thread.answered_questions) - previous_answered)
            + max(0, thread.evidence_count - previous_evidence_count)
            + len(current_decoding_terms - previous_decoding_terms)
            + len(current_mapping_insights - previous_mapping_insights)
        )
        thread.latest_new_info_gain = info_gain
        thread.stale_rounds = 0 if info_gain > 0 else thread.stale_rounds + 1
        thread.history.append(
            {
                "round": analysis_round,
                "confidence": record.confidence,
                "evidence_count": len(record.visual_anchoring),
                "answered_questions": self._answered_questions_from_record(record),
                "unresolved_points": record.unresolved_points,
                "new_info_gain": info_gain,
                "stale_rounds": thread.stale_rounds,
            }
        )

        focus_resolved = self._thread_focus_resolved(thread, record)
        if focus_resolved:
            thread.status = "ANSWERED"
            thread.pause_reason = ""
        elif record.visual_anchoring and record.confidence >= 0.35:
            thread.status = "PAUSED"
            thread.pause_reason = "evidence_captured"
        elif thread.attempts < thread.max_attempts:
            thread.status = "RETRY"
            thread.pause_reason = ""
        else:
            thread.status = "BLOCKED"
            thread.pause_reason = "attempt_limit_reached"

        return info_gain

    def _thread_focus_resolved(self, thread: CoTThread, record: DomainCoTRecord) -> bool:
        if thread.reason == "slot_overview":
            return bool(record.visual_anchoring or record.domain_decoding or record.cultural_mapping)
        if thread.reason == "retrieval_gain":
            return bool(record.domain_decoding or record.cultural_mapping or (record.visual_anchoring and record.confidence >= 0.35))
        answered_questions = self._answered_questions_from_record(record)
        if any(self._text_similarity(thread.focus, question) >= 0.68 for question in answered_questions):
            return True
        if thread.reason in {"unresolved_point", "missing_visual_anchor", "chronology_conflict"}:
            return bool(record.visual_anchoring and record.confidence >= 0.35)
        return False

    def _cross_validate(
        self,
        outputs: list[DomainCoTRecord],
        slot_schemas: list[SlotSchema],
        meta: dict,
        api_logs: list[dict[str, Any]] | None = None,
    ) -> CrossValidationResult:
        return run_cross_validate(self, outputs, slot_schemas, meta, api_logs)

    def _augment_round_table_review(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        api_logs: list[dict[str, Any]],
    ) -> CrossValidationResult:
        return run_round_table_review(self, outputs, validation, meta, api_logs)

    def _parse_round_table_review(self, raw: str) -> dict[str, Any]:
        return parse_round_table_review_payload(self, raw)

    @staticmethod
    def _round_table_priority(value: object) -> str:
        return normalize_round_table_priority(value)

    def _plan_spawn_tasks(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        threads: list[CoTThread],
        api_logs: list[dict[str, Any]] | None = None,
    ) -> list[SpawnTask]:
        return plan_reflection_spawn_tasks(self, outputs, validation, threads, api_logs)

    def _review_slot_lifecycle(
        self,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        api_logs: list[dict[str, Any]],
        *,
        use_llm: bool = True,
    ) -> CrossValidationResult:
        return review_reflection_slot_lifecycle(self, slot_schemas, outputs, validation, meta, api_logs, use_llm=use_llm)

    def _review_validation_bundle(
        self,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        api_logs: list[dict[str, Any]],
        *,
        use_llm: bool = True,
    ) -> CrossValidationResult:
        return review_reflection_validation_bundle(
            self,
            slot_schemas,
            outputs,
            validation,
            meta,
            api_logs,
            use_llm=use_llm,
        )

    def _task_rag_terms(
        self,
        *,
        focus_text: str,
        fallback_terms: list[str],
        slot_name: str,
        task_reason: str = "",
        api_logs: list[dict[str, Any]] | None = None,
    ) -> list[str]:
        return plan_task_rag_terms(
            self,
            focus_text=focus_text,
            fallback_terms=fallback_terms,
            slot_name=slot_name,
            task_reason=task_reason,
            api_logs=api_logs,
        )

    def _parse_rag_terms_response(self, raw: str) -> list[str]:
        return parse_reflection_rag_terms(self, raw)

    def _normalize_search_queries(self, items: list[str] | object) -> list[str]:
        return normalize_reflection_queries(self, items)

    def _normalize_web_search_queries(self, items: list[str] | object) -> list[str]:
        return normalize_reflection_web_queries(self, items)

    def _fallback_rag_terms(self, *, slot_name: str, fallback_terms: list[str]) -> list[str]:
        return build_fallback_rag_terms(self, slot_name=slot_name, fallback_terms=fallback_terms)

    @staticmethod
    def _clean_search_query(value: object) -> str:
        return clean_reflection_search_query(value)

    def _task_already_resolved(self, task: SpawnTask, threads: list[CoTThread]) -> bool:
        return is_reflection_task_resolved(self, task, threads)

    def _suppress_redundant_tasks(
        self,
        tasks: list[SpawnTask],
        threads: list[CoTThread],
    ) -> tuple[list[SpawnTask], list[str]]:
        return suppress_reflection_tasks(self, tasks, threads)

    def _find_matching_thread(self, task: SpawnTask, threads: list[CoTThread]) -> CoTThread | None:
        return find_reflection_matching_thread(self, task, threads)

    @staticmethod
    def _should_pause_duplicate_task(thread: CoTThread) -> bool:
        return should_pause_reflection_task(thread)

    def _sync_threads_with_tasks(self, threads: list[CoTThread], tasks: list[SpawnTask]) -> list[str]:
        return sync_reflection_threads(self, threads, tasks)

    def _build_routing(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        tasks: list[SpawnTask],
        convergence: dict[str, Any],
    ) -> RoutingDecision:
        return build_reflection_routing(self, outputs, validation, tasks, convergence)

    def _check_convergence(
        self,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        threads: list[CoTThread],
        dialogue_state: DialogueState,
        tasks: list[SpawnTask] | None = None,
    ) -> dict[str, Any]:
        return check_reflection_convergence(self, slot_schemas, outputs, validation, threads, dialogue_state, tasks)

    def _generate_final_appreciation_prompt(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        dialogue_state: DialogueState,
        api_logs: list[dict[str, Any]] | None = None,
    ) -> str:
        return build_final_reflection_prompt(self, outputs, validation, meta, dialogue_state, api_logs=api_logs)

    def _build_round_memory(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        dialogue_state: DialogueState,
        threads: list[CoTThread],
        prior_round_memories: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        previous_memory = self._latest_round_memory(prior_round_memories)
        issues_payload = [
            {
                "issue_type": issue.issue_type,
                "severity": issue.severity,
                "slot_names": issue.slot_names,
                "detail": issue.detail,
                "rag_terms": issue.rag_terms,
            }
            for issue in validation.issues
        ]
        slots_payload = [
            {
                "slot_name": output.slot_name,
                "slot_term": output.slot_term,
                "visual_anchoring": [asdict(item) for item in output.visual_anchoring],
                "domain_decoding": [asdict(item) for item in output.domain_decoding],
                "cultural_mapping": [asdict(item) for item in output.cultural_mapping],
                "question_coverage": [asdict(item) for item in output.question_coverage],
                "unresolved_points": output.unresolved_points,
                "generated_questions": output.generated_questions,
                "retrieval_gain_focus": output.retrieval_gain_focus,
                "retrieval_gain_terms": output.retrieval_gain_terms,
                "retrieval_gain_queries": output.retrieval_gain_queries,
                "retrieval_gain_mode": output.retrieval_gain_mode,
                "retrieval_gain_web_queries": output.retrieval_gain_web_queries,
                "retrieval_gain_reason": output.retrieval_gain_reason,
                "retrieval_gain_has_value": output.retrieval_gain_has_value,
                "confidence": output.confidence,
            }
            for output in outputs
        ]
        threads_payload = [
            {
                "thread_id": thread.thread_id,
                "slot_name": thread.slot_name,
                "focus": thread.focus,
                "reason": thread.reason,
                "status": thread.status,
                "attempts": thread.attempts,
                "latest_new_info_gain": thread.latest_new_info_gain,
                "stale_rounds": thread.stale_rounds,
                "pause_reason": thread.pause_reason,
                "answered_questions": thread.answered_questions,
                "unresolved_points": thread.unresolved_points,
                "latest_summary": thread.latest_summary,
            }
            for thread in threads
        ]

        resolved_questions = self._dedupe_text_list(dialogue_state.resolved_questions)
        unresolved_questions = self._dedupe_text_list(dialogue_state.unresolved_questions)
        issue_details = self._dedupe_text_list([item["detail"] for item in issues_payload])

        previous_resolved = self._memory_text_list(previous_memory, "resolved_questions")
        previous_unresolved = self._memory_text_list(previous_memory, "unresolved_questions")
        previous_issue_details = self._memory_issue_details(previous_memory)

        new_resolved = [item for item in resolved_questions if item not in previous_resolved]
        still_open = [item for item in unresolved_questions if item in previous_unresolved]
        new_open = [item for item in unresolved_questions if item not in previous_unresolved]
        cleared_questions = [item for item in previous_unresolved if item not in unresolved_questions]
        new_issues = [item for item in issue_details if item not in previous_issue_details]
        resolved_issues = [item for item in previous_issue_details if item not in issue_details]
        slot_gains = self._memory_slot_gains(previous_memory, slots_payload)
        total_new_items = (
            len(new_resolved)
            + len(new_open)
            + len(new_issues)
            + sum(
                len(slot_gain.get("new_visual_anchoring", []))
                + len(slot_gain.get("new_domain_decoding", []))
                + len(slot_gain.get("new_cultural_mapping", []))
                + len(slot_gain.get("new_answered_questions", []))
                for slot_gain in slot_gains
            )
        )

        carry_over_issues = [
            issue["detail"]
            for issue in issues_payload
            if str(issue.get("severity", "")).strip() in {"high", "medium"}
        ]

        return {
            "round_index": dialogue_state.final_round_index,
            "converged": dialogue_state.converged,
            "convergence_reason": dialogue_state.convergence_reason,
            "previous_round_index": int(previous_memory.get("round_index", 0) or 0) if previous_memory else 0,
            "resolved_questions": resolved_questions,
            "unresolved_questions": unresolved_questions,
            "removed_questions": dialogue_state.removed_questions,
            "merged_duplicates": dialogue_state.merged_duplicates,
            "issues": issues_payload,
            "cumulative_snapshot": {
                "resolved_questions": resolved_questions,
                "unresolved_questions": unresolved_questions,
                "issues": issue_details,
                "slot_count": len(slots_payload),
                "thread_count": len(threads_payload),
            },
            "info_gain": {
                "new_resolved_questions": new_resolved,
                "new_unresolved_questions": new_open,
                "cleared_questions": cleared_questions,
                "persistent_open_questions": still_open,
                "new_issues": new_issues,
                "resolved_issues": resolved_issues,
                "slot_gains": slot_gains,
                "total_new_items": total_new_items,
            },
            "carry_over": {
                "focus_questions": unresolved_questions[:10],
                "issues": self._dedupe_text_list(carry_over_issues)[:10],
                "focus_slots": self._dedupe_text_list(
                    [slot["slot_name"] for slot in slots_payload if slot.get("unresolved_points")]
                    + [thread["slot_name"] for thread in threads_payload if thread.get("status") in {"OPEN", "RETRY", "BLOCKED"}]
                )[:8],
            },
            "slots": slots_payload,
            "threads": threads_payload,
        }

    @staticmethod
    def _latest_round_memory(prior_round_memories: list[dict[str, Any]] | None) -> dict[str, Any]:
        if not isinstance(prior_round_memories, list):
            return {}
        for item in reversed(prior_round_memories):
            if isinstance(item, dict):
                return item
        return {}

    @staticmethod
    def _memory_text_list(memory: dict[str, Any], key: str) -> list[str]:
        values = memory.get(key, []) if isinstance(memory, dict) else []
        if not isinstance(values, list):
            return []
        return [str(item).strip() for item in values if str(item).strip()]

    @classmethod
    def _memory_issue_details(cls, memory: dict[str, Any]) -> list[str]:
        issues = memory.get("issues", []) if isinstance(memory, dict) else []
        if not isinstance(issues, list):
            return []
        return cls._dedupe_text_list([str(item.get("detail", "")).strip() for item in issues if isinstance(item, dict)])

    @classmethod
    def _memory_slot_gains(cls, previous_memory: dict[str, Any], slots_payload: list[dict[str, Any]]) -> list[dict[str, Any]]:
        previous_slots = previous_memory.get("slots", []) if isinstance(previous_memory, dict) else []
        previous_slot_map = {
            str(item.get("slot_name", "")).strip(): item
            for item in previous_slots
            if isinstance(item, dict) and str(item.get("slot_name", "")).strip()
        }
        slot_gains: list[dict[str, Any]] = []
        for slot in slots_payload:
            slot_name = str(slot.get("slot_name", "")).strip()
            if not slot_name:
                continue
            previous_slot = previous_slot_map.get(slot_name, {})
            gain = {
                "slot_name": slot_name,
                "new_visual_anchoring": cls._memory_structured_diff(
                    previous_slot.get("visual_anchoring", []),
                    slot.get("visual_anchoring", []),
                    "observation",
                ),
                "new_domain_decoding": cls._memory_structured_diff(
                    previous_slot.get("domain_decoding", []),
                    slot.get("domain_decoding", []),
                    "term",
                ),
                "new_cultural_mapping": cls._memory_structured_diff(
                    previous_slot.get("cultural_mapping", []),
                    slot.get("cultural_mapping", []),
                    "insight",
                ),
                "new_answered_questions": cls._memory_answered_question_diff(
                    previous_slot.get("question_coverage", []),
                    slot.get("question_coverage", []),
                ),
            }
            if any(gain[key] for key in ("new_visual_anchoring", "new_domain_decoding", "new_cultural_mapping", "new_answered_questions")):
                slot_gains.append(gain)
        return slot_gains

    @classmethod
    def _memory_structured_diff(cls, previous_items: object, current_items: object, key: str) -> list[str]:
        previous_keys: set[str] = set()
        if isinstance(previous_items, list):
            for item in previous_items:
                if isinstance(item, dict):
                    value = cls._normalize_text(item.get(key, ""))
                    if value:
                        previous_keys.add(value)

        results: list[str] = []
        if not isinstance(current_items, list):
            return results
        for item in current_items:
            if not isinstance(item, dict):
                continue
            raw_value = str(item.get(key, "")).strip()
            normalized = cls._normalize_text(raw_value)
            if not normalized or normalized in previous_keys:
                continue
            previous_keys.add(normalized)
            results.append(raw_value)
        return results

    @classmethod
    def _memory_answered_question_diff(cls, previous_items: object, current_items: object) -> list[str]:
        previous_answered: set[str] = set()
        if isinstance(previous_items, list):
            for item in previous_items:
                if isinstance(item, dict) and bool(item.get("answered")):
                    normalized = cls._normalize_text(item.get("question", ""))
                    if normalized:
                        previous_answered.add(normalized)

        results: list[str] = []
        if not isinstance(current_items, list):
            return results
        for item in current_items:
            if not isinstance(item, dict) or not bool(item.get("answered")):
                continue
            question = str(item.get("question", "")).strip()
            normalized = cls._normalize_text(question)
            if not normalized or normalized in previous_answered:
                continue
            previous_answered.add(normalized)
            results.append(question)
        return results

    def _merge_domain_outputs(
        self,
        base_outputs: list[DomainCoTRecord],
        follow_up_outputs: list[DomainCoTRecord],
    ) -> list[DomainCoTRecord]:
        merged = {item.slot_name: item for item in base_outputs}
        for incoming in follow_up_outputs:
            if incoming.slot_name not in merged:
                merged[incoming.slot_name] = incoming
                continue
            current = merged[incoming.slot_name]
            merged[incoming.slot_name] = DomainCoTRecord(
                slot_name=current.slot_name,
                slot_term=current.slot_term,
                analysis_round=max(current.analysis_round, incoming.analysis_round),
                controlled_vocabulary=self._dedupe_text_list(current.controlled_vocabulary + incoming.controlled_vocabulary),
                visual_anchoring=self._merge_structured_lists(current.visual_anchoring, incoming.visual_anchoring, "observation"),
                domain_decoding=self._merge_structured_lists(current.domain_decoding, incoming.domain_decoding, "term"),
                cultural_mapping=self._merge_structured_lists(current.cultural_mapping, incoming.cultural_mapping, "insight"),
                question_coverage=self._merge_question_coverage(current.question_coverage, incoming.question_coverage),
                unresolved_points=self._dedupe_text_list(current.unresolved_points + incoming.unresolved_points),
                generated_questions=self._dedupe_text_list(current.generated_questions + incoming.generated_questions),
                statuses=self._dedupe_text_list(current.statuses + incoming.statuses),
                confidence=max(current.confidence, incoming.confidence),
                retrieval_gain_focus=incoming.retrieval_gain_focus or current.retrieval_gain_focus,
                retrieval_gain_terms=self._dedupe_text_list(current.retrieval_gain_terms + incoming.retrieval_gain_terms),
                retrieval_gain_queries=self._dedupe_text_list(current.retrieval_gain_queries + incoming.retrieval_gain_queries),
                retrieval_gain_mode=self._merge_retrieval_modes(current.retrieval_gain_mode, incoming.retrieval_gain_mode),
                retrieval_gain_web_queries=self._dedupe_text_list(current.retrieval_gain_web_queries + incoming.retrieval_gain_web_queries),
                retrieval_gain_reason=incoming.retrieval_gain_reason or current.retrieval_gain_reason,
                retrieval_gain_has_value=current.retrieval_gain_has_value or incoming.retrieval_gain_has_value,
                raw_response="\n\n".join(part for part in [current.raw_response, incoming.raw_response] if part),
            )
        return sorted(merged.values(), key=lambda item: item.slot_name)

    @staticmethod
    def _merge_structured_lists(current: list[Any], incoming: list[Any], field_name: str) -> list[Any]:
        merged: list[Any] = []
        seen: set[str] = set()
        for item in current + incoming:
            value = getattr(item, field_name, "")
            key = DynamicAgentPipeline._normalize_text(value)
            if not key or key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    @staticmethod
    def _merge_question_coverage(current: list[QuestionCoverage], incoming: list[QuestionCoverage]) -> list[QuestionCoverage]:
        merged: dict[str, QuestionCoverage] = {item.question: item for item in current}
        for item in incoming:
            existing = merged.get(item.question)
            if not existing:
                merged[item.question] = item
                continue
            merged[item.question] = QuestionCoverage(
                question=item.question,
                answered=existing.answered or item.answered,
                support=item.support or existing.support,
            )
        return list(merged.values())

    def save_result(self, result: PipelineResult, output_dir: str | None = None) -> dict[str, str]:
        root = Path(output_dir or self.config.output_dir or "artifacts")
        root.mkdir(parents=True, exist_ok=True)
        run_dir = root / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        output_files: dict[str, str] = {"run_dir": str(run_dir)}

        memory_json_path = run_dir / "memory.json"
        memory_json_path.write_text(json.dumps(result.round_memory, ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["memory_json"] = str(memory_json_path)

        final_prompt_path = run_dir / "final_appreciation_prompt.md"
        final_prompt_path.write_text(result.final_appreciation_prompt, encoding="utf-8")
        output_files["final_appreciation_prompt"] = str(final_prompt_path)

        domain_path = run_dir / "domain_outputs.json"
        domain_path.write_text(json.dumps([asdict(item) for item in result.domain_outputs], ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["domain_outputs"] = str(domain_path)

        validation_path = run_dir / "cross_validation.json"
        validation_path.write_text(json.dumps(asdict(result.cross_validation), ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["cross_validation"] = str(validation_path)

        dialogue_state_path = run_dir / "dialogue_state.json"
        dialogue_state_payload = asdict(result.dialogue_state)
        dialogue_state_payload["routing"] = asdict(result.routing)
        dialogue_state_payload["execution_log"] = result.execution_log
        dialogue_state_payload["image_path"] = result.image_path
        dialogue_state_path.write_text(json.dumps(dialogue_state_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["dialogue_state"] = str(dialogue_state_path)

        cot_threads_path = run_dir / "cot_threads.json"
        cot_threads_path.write_text(json.dumps([asdict(item) for item in result.cot_threads], ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["cot_threads"] = str(cot_threads_path)

        prepared_path = run_dir / "prepared_image.json"
        prepared_path.write_text(json.dumps(asdict(result.prepared_image), ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["prepared_image"] = str(prepared_path)

        api_calls_path = run_dir / "api_calls.jsonl"
        api_calls_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in result.api_logs), encoding="utf-8")
        output_files["api_calls"] = str(api_calls_path)
        return output_files

    @staticmethod
    def _clone_thread(thread: CoTThread) -> CoTThread:
        return CoTThread(**asdict(thread))

    @staticmethod
    def _extract_json_object(raw: str) -> dict[str, Any]:
        text = raw.strip()
        try:
            parsed = json.loads(text)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            pass
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}

    @staticmethod
    def _safe_confidence(value: object) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return 0.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def _parse_truthy_flag(value: object) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"true", "1", "yes", "y"}
        return bool(value)

    @staticmethod
    def _normalize_text(text: str) -> str:
        return re.sub(r"\s+", "", str(text or "").lower())

    @classmethod
    def _text_similarity(cls, left: str, right: str) -> float:
        return SequenceMatcher(None, cls._normalize_text(left), cls._normalize_text(right)).ratio()

    @classmethod
    def _detect_semantic_duplicates(cls, outputs: list[DomainCoTRecord]) -> list[str]:
        observations: list[tuple[str, str]] = []
        duplicates: list[str] = []
        for output in outputs:
            for item in output.visual_anchoring:
                observations.append((output.slot_name, item.observation))
        for index, (slot_name, text) in enumerate(observations):
            for other_slot, other_text in observations[index + 1 :]:
                if cls._text_similarity(text, other_text) >= 0.84:
                    duplicates.append(f"{slot_name} 与 {other_slot} 对同一视觉特征存在重复描述：{text}")
        return cls._dedupe_text_list(duplicates)

    @staticmethod
    def _extract_dynasties(text: str) -> set[str]:
        found: set[str] = set()
        for dynasty in _DYNASTIES:
            if dynasty in text:
                found.add(dynasty)
        return found

    @classmethod
    def _is_low_value_question(cls, question: str, answered_questions: list[str]) -> bool:
        normalized = cls._normalize_text(question)
        if not normalized:
            return True
        for existing in answered_questions:
            if cls._text_similarity(question, existing) >= 0.75:
                return True
        return False

    @classmethod
    def _dedupe_spawn_tasks(cls, tasks: list[SpawnTask]) -> list[SpawnTask]:
        merged: list[SpawnTask] = []
        seen: set[str] = set()
        for task in tasks:
            key = f"{task.slot_name}:{cls._normalize_text(task.prompt_focus)}"
            if not task.prompt_focus.strip() or key in seen:
                continue
            seen.add(key)

            duplicate_index = next(
                (index for index, existing in enumerate(merged) if cls._tasks_share_topic(existing, task)),
                -1,
            )
            if duplicate_index >= 0:
                merged[duplicate_index] = cls._merge_spawn_task(merged[duplicate_index], task)
                continue
            merged.append(task)
        return merged

    @classmethod
    def _tasks_share_topic(cls, left: SpawnTask, right: SpawnTask) -> bool:
        if left.slot_name != right.slot_name:
            return False
        if cls._text_similarity(left.prompt_focus, right.prompt_focus) >= 0.6:
            return True
        return bool(cls._shared_task_terms(left, right))

    @classmethod
    def _merge_spawn_task(cls, left: SpawnTask, right: SpawnTask) -> SpawnTask:
        preferred = left
        secondary = right
        if (right.priority, len(right.prompt_focus.strip())) > (left.priority, len(left.prompt_focus.strip())):
            preferred = right
            secondary = left
        return SpawnTask(
            slot_name=preferred.slot_name,
            reason=preferred.reason,
            prompt_focus=preferred.prompt_focus,
            rag_terms=cls._dedupe_text_list(preferred.rag_terms + secondary.rag_terms),
            retrieval_mode=cls._merge_retrieval_modes(preferred.retrieval_mode, secondary.retrieval_mode),
            web_queries=cls._dedupe_text_list(preferred.web_queries + secondary.web_queries),
            retrieval_reason=preferred.retrieval_reason or secondary.retrieval_reason,
            priority=max(left.priority, right.priority),
            dispatch_target=preferred.dispatch_target if preferred.dispatch_target else secondary.dispatch_target,
            requested_slot_name=preferred.requested_slot_name or secondary.requested_slot_name,
            source_thread_id=preferred.source_thread_id or secondary.source_thread_id,
        )

    @staticmethod
    def _normalize_retrieval_mode(value: object) -> str:
        text = str(value or "").strip().lower()
        if text == "web":
            return "web"
        if text == "hybrid":
            return "hybrid"
        return "rag"

    @classmethod
    def _merge_retrieval_modes(cls, left: object, right: object) -> str:
        left_mode = cls._normalize_retrieval_mode(left)
        right_mode = cls._normalize_retrieval_mode(right)
        if left_mode == right_mode:
            return left_mode
        if "hybrid" in {left_mode, right_mode}:
            return "hybrid"
        if {left_mode, right_mode} == {"rag", "web"}:
            return "hybrid"
        return left_mode or right_mode or "rag"

    @classmethod
    def _shared_task_terms(cls, left: SpawnTask, right: SpawnTask) -> set[str]:
        left_terms = cls._task_topic_terms(left)
        if not left_terms:
            return set()
        right_terms = cls._task_topic_terms(right)
        if not right_terms:
            return set()
        return left_terms & right_terms

    @classmethod
    def _task_topic_terms(cls, task: SpawnTask) -> set[str]:
        terms = {
            term
            for term in cls._extract_focus_terms(task.prompt_focus)
            if term
        }
        for rag_term in task.rag_terms:
            normalized = cls._normalize_text(rag_term)
            if cls._is_generic_task_term(normalized):
                continue
            if len(normalized) >= 2:
                terms.add(normalized)
        return terms

    @classmethod
    def _extract_focus_terms(cls, text: str) -> set[str]:
        normalized = cls._normalize_text(text)
        if not normalized:
            return set()
        terms: set[str] = set()
        max_len = min(8, len(normalized))
        for size in range(2, max_len + 1):
            for index in range(0, len(normalized) - size + 1):
                piece = normalized[index : index + size]
                if cls._is_generic_task_term(piece):
                    continue
                terms.add(piece)
        return terms

    @staticmethod
    def _is_generic_task_term(term: str) -> bool:
        if not term:
            return True
        generic_terms = {
            "画中",
            "作品中",
            "画面",
            "如何",
            "怎样",
            "哪些",
            "什么",
            "具体",
            "细节",
            "体现",
            "表现",
            "增强",
            "关系",
            "作用",
            "意义",
            "象征",
            "背景",
            "元素",
            "人物",
            "罗汉",
            "南宋",
            "佛教",
            "宗教",
            "形态",
            "色彩",
            "层次",
            "材质",
            "装饰",
            "身份",
            "确认",
        }
        if term in generic_terms:
            return True
        if len(term) <= 1:
            return True
        if len(term) == 2 and term in {"问题", "是否", "有助", "辅助", "画作"}:
            return True
        return False

    @staticmethod
    def _dedupe_text_list(items: list[str] | object) -> list[str]:
        if not isinstance(items, list):
            return []
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text)
            if key in seen:
                continue
            seen.add(key)
            results.append(text)
        return results

    @staticmethod
    def _record_summary(record: DomainCoTRecord) -> str:
        segments: list[str] = []
        for item in record.visual_anchoring[:2]:
            segments.append(item.observation)
        for item in record.domain_decoding[:2]:
            segments.append(item.explanation or item.term)
        for item in record.cultural_mapping[:1]:
            segments.append(item.insight)
        return " | ".join(segment for segment in segments if segment)

    @staticmethod
    def _answered_questions_from_record(record: DomainCoTRecord) -> list[str]:
        return [item.question for item in record.question_coverage if item.answered]

    def _collect_answered_questions(self, outputs: list[DomainCoTRecord]) -> list[str]:
        questions: list[str] = []
        for output in outputs:
            questions.extend(self._answered_questions_from_record(output))
        return self._dedupe_text_list(questions)

    def _collect_unresolved_questions(self, outputs: list[DomainCoTRecord], validation: CrossValidationResult) -> list[str]:
        return self._dedupe_text_list(validation.missing_points)

    def _stable_slots(self, outputs: list[DomainCoTRecord], validation: CrossValidationResult) -> list[str]:
        lifecycle_map = {
            str(item.get("slot_name", "")).strip(): str(item.get("status", "")).strip().upper()
            for item in validation.slot_lifecycle_reviews
            if isinstance(item, dict)
        }
        issue_slots = {slot_name for issue in validation.issues for slot_name in issue.slot_names if issue.severity in {"high", "medium"}}
        stable: list[str] = []
        for output in outputs:
            if lifecycle_map.get(output.slot_name) in {"STABLE", "CLOSED"}:
                stable.append(output.slot_name)
                continue
            unanswered = [item for item in output.question_coverage if not item.answered]
            if output.slot_name in issue_slots:
                continue
            if unanswered:
                continue
            stable.append(output.slot_name)
        return self._dedupe_text_list(stable)

    def _should_run_round_table_review(
        self,
        *,
        round_index: int,
        max_rounds: int,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        new_info_count: int,
    ) -> bool:
        if round_index <= 1 or round_index >= max_rounds or new_info_count == 0:
            return True
        _, remaining_count, _ = self._question_progress(slot_schemas, outputs)
        return remaining_count == 0

    def _should_run_slot_lifecycle_review(
        self,
        *,
        round_index: int,
        max_rounds: int,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        new_info_count: int,
    ) -> bool:
        if round_index <= 1 or round_index >= max_rounds or new_info_count == 0:
            return True
        _, remaining_count, _ = self._question_progress(slot_schemas, outputs)
        return remaining_count == 0

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", str(value or "").strip())
        return slug.strip("-").lower() or "item"
