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
from .prompt_builder import (
    build_domain_cot_prompt,
    build_final_appreciation_prompt,
    build_round_table_prompt,
    build_summary_prompt,
)
from .schema_loader import extract_controlled_vocabulary, load_slot_schemas
from .vlm_runner import VLMRunner


_DYNASTIES = ("先秦", "汉", "魏晋", "隋", "唐", "五代", "北宋", "南宋", "元", "明", "清", "近现代", "当代")
_THREAD_STATUS_ORDER = {"OPEN": 0, "RETRY": 1, "BLOCKED": 2, "PAUSED": 3, "ANSWERED": 4, "MERGED": 5}
_DUPLICATE_STALE_ROUNDS = 2


class DynamicAgentPipeline:
    def __init__(self, config: PipelineConfig | None = None, api_client: NewAPIClient | None = None) -> None:
        self.config = config or PipelineConfig()
        self.api_client = api_client or NewAPIClient()
        self.vlm_runner = VLMRunner(self.api_client)

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
        threads = self._initialize_threads(slot_schemas)
        dialogue_state.threads = [self._clone_thread(thread) for thread in threads]
        self._print_progress(
            stage="init",
            round_index=0,
            max_rounds=max(1, int(self.config.max_dialogue_rounds or 1)),
            threads=threads,
            slot_schemas=slot_schemas,
            outputs=[],
            note=f"prepared_image={prepared_image.path}",
        )

        domain_outputs: list[DomainCoTRecord] = []
        validation = CrossValidationResult(issues=[], semantic_duplicates=[], missing_points=[], rag_terms=[], removed_questions=[])
        routing = RoutingDecision(action="PAUSE_COT", rationale=["初始化完成"], paused_slots=[], spawned_tasks=[], removed_questions=[], merged_duplicates=[])

        max_rounds = max(1, int(self.config.max_dialogue_rounds or 1))
        for round_index in range(1, max_rounds + 1):
            executable_threads = self._select_threads_for_round(threads)
            self._print_progress(
                stage="round_start",
                round_index=round_index,
                max_rounds=max_rounds,
                threads=threads,
                slot_schemas=slot_schemas,
                outputs=domain_outputs,
                executable_threads=executable_threads,
            )
            if not executable_threads:
                convergence = self._check_convergence(slot_schemas, domain_outputs, validation, threads, dialogue_state)
                routing = RoutingDecision(
                    action="PAUSE_COT",
                    rationale=["当前没有可执行线程。"],
                    paused_slots=self._stable_slots(domain_outputs, validation),
                    spawned_tasks=[],
                    removed_questions=validation.removed_questions,
                    merged_duplicates=validation.semantic_duplicates,
                    converged=convergence["converged"],
                    convergence_reason=convergence["reason"],
                    answered_slots=convergence["answered_slots"],
                )
                execution_log.append(
                    {
                        "stage": f"round_{round_index}",
                        "routing": asdict(routing),
                        "note": "no_executable_threads",
                    }
                )
                dialogue_state.turns.append(
                    DialogueTurn(
                        round_index=round_index,
                        active_thread_ids=[],
                        executed_thread_ids=[],
                        spawned_thread_ids=[],
                        answered_thread_ids=[],
                        blocked_thread_ids=[],
                        paused_thread_ids=[],
                        merged_thread_ids=[],
                        routing_action=routing.action,
                        notes=routing.rationale,
                        new_information_count=0,
                        convergence_snapshot=convergence,
                    )
                )
                self._print_progress(
                    stage="round_stop",
                    round_index=round_index,
                    max_rounds=max_rounds,
                    threads=threads,
                    slot_schemas=slot_schemas,
                    outputs=domain_outputs,
                    routing=routing,
                    note="no_executable_threads",
                )
                break

            round_stage_started = perf_counter()
            round_outputs, new_info_count = self._run_thread_round(
                threads=executable_threads,
                prepared_image=prepared_image,
                meta=meta,
                api_logs=api_logs,
                analysis_round=round_index,
            )
            stage_timings = {"domain_cot_parallel_s": round(perf_counter() - round_stage_started, 4)}
            domain_outputs = self._merge_domain_outputs(domain_outputs, round_outputs)
            round_stage_started = perf_counter()
            validation = self._cross_validate(domain_outputs, slot_schemas, meta)
            stage_timings["cross_validate_s"] = round(perf_counter() - round_stage_started, 4)
            round_stage_started = perf_counter()
            validation = self._augment_round_table_review(domain_outputs, validation, meta, api_logs)
            stage_timings["round_table_review_s"] = round(perf_counter() - round_stage_started, 4)

            if new_info_count > 0:
                dialogue_state.no_new_info_rounds = 0
            else:
                dialogue_state.no_new_info_rounds += 1

            round_stage_started = perf_counter()
            candidate_tasks = self._plan_spawn_tasks(domain_outputs, validation, threads)
            stage_timings["plan_spawn_tasks_s"] = round(perf_counter() - round_stage_started, 4)
            round_stage_started = perf_counter()
            candidate_tasks, duplicate_paused_thread_ids = self._suppress_redundant_tasks(candidate_tasks, threads)
            stage_timings["suppress_duplicate_tasks_s"] = round(perf_counter() - round_stage_started, 4)
            round_stage_started = perf_counter()
            spawned_thread_ids = self._sync_threads_with_tasks(threads, candidate_tasks)
            stage_timings["sync_threads_s"] = round(perf_counter() - round_stage_started, 4)
            round_stage_started = perf_counter()
            convergence = self._check_convergence(slot_schemas, domain_outputs, validation, threads, dialogue_state)
            stage_timings["check_convergence_s"] = round(perf_counter() - round_stage_started, 4)
            routing = self._build_routing(domain_outputs, validation, candidate_tasks, convergence)

            answered_thread_ids = [thread.thread_id for thread in executable_threads if thread.status == "ANSWERED"]
            blocked_thread_ids = [thread.thread_id for thread in executable_threads if thread.status == "BLOCKED"]
            paused_thread_ids = self._dedupe_text_list(
                [thread.thread_id for thread in executable_threads if thread.status == "PAUSED"] + duplicate_paused_thread_ids
            )
            merged_thread_ids = [thread.thread_id for thread in threads if thread.status == "MERGED"]

            dialogue_state.resolved_questions = self._collect_answered_questions(domain_outputs)
            dialogue_state.unresolved_questions = self._collect_unresolved_questions(domain_outputs, validation)
            dialogue_state.removed_questions = validation.removed_questions
            dialogue_state.merged_duplicates = validation.semantic_duplicates
            dialogue_state.converged = convergence["converged"]
            dialogue_state.convergence_reason = convergence["reason"]
            dialogue_state.final_round_index = round_index
            dialogue_state.threads = [self._clone_thread(thread) for thread in threads]
            dialogue_state.turns.append(
                DialogueTurn(
                    round_index=round_index,
                    active_thread_ids=[thread.thread_id for thread in threads if thread.status in {"OPEN", "RETRY", "BLOCKED"}],
                    executed_thread_ids=[thread.thread_id for thread in executable_threads],
                    spawned_thread_ids=spawned_thread_ids,
                    answered_thread_ids=answered_thread_ids,
                    blocked_thread_ids=blocked_thread_ids,
                    paused_thread_ids=paused_thread_ids,
                    merged_thread_ids=merged_thread_ids,
                    routing_action=routing.action,
                    notes=routing.rationale,
                    new_information_count=new_info_count,
                    convergence_snapshot=convergence,
                )
            )
            execution_log.append(
                {
                    "stage": f"round_{round_index}",
                    "routing": asdict(routing),
                    "new_information_count": new_info_count,
                    "spawned_thread_ids": spawned_thread_ids,
                    "duplicate_paused_thread_ids": duplicate_paused_thread_ids,
                    "convergence": convergence,
                    "stage_timings": stage_timings,
                }
            )
            self._print_progress(
                stage="round_end",
                round_index=round_index,
                max_rounds=max_rounds,
                threads=threads,
                slot_schemas=slot_schemas,
                outputs=domain_outputs,
                executable_threads=executable_threads,
                routing=routing,
                note=(
                    f"spawned={len(spawned_thread_ids)} duplicate_paused={len(duplicate_paused_thread_ids)} new_info={new_info_count} "
                    f"slowest_substage={self._slowest_stage_label(stage_timings)}"
                ),
            )

            if routing.converged:
                break
            if routing.action == "PAUSE_COT" and dialogue_state.no_new_info_rounds >= max(1, int(self.config.convergence_patience or 1)):
                break

        round_memory = self._build_round_memory(domain_outputs, validation, dialogue_state, threads)
        summary_markdown = self._generate_summary(domain_outputs, validation, prepared_image, meta, api_logs)
        memory_markdown = self._render_memory_markdown(round_memory)
        final_appreciation_prompt = self._generate_final_appreciation_prompt(domain_outputs, validation, meta, dialogue_state)

        return PipelineResult(
            image_path=image_path,
            prepared_image=prepared_image,
            slot_schemas=slot_schemas,
            domain_outputs=domain_outputs,
            cross_validation=validation,
            routing=routing,
            dialogue_state=dialogue_state,
            cot_threads=threads,
            summary_markdown=summary_markdown,
            memory_markdown=memory_markdown,
            round_memory=round_memory,
            final_appreciation_prompt=final_appreciation_prompt,
            api_logs=api_logs,
            execution_log=execution_log,
        )

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

    def _initialize_threads(self, slot_schemas: list[SlotSchema]) -> list[CoTThread]:
        threads: list[CoTThread] = []
        for index, slot in enumerate(slot_schemas, start=1):
            primary_focus = slot.specific_questions[0] if slot.specific_questions else f"围绕 {slot.slot_term or slot.slot_name} 完成整体分析。"
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
        workers = min(max(1, len(threads)), max(1, int(self.config.concurrent_workers or 1)))
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
                for thread in threads
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
            f"cot_running={len(threads)} elapsed_s={round(perf_counter() - started_at, 2):.2f}",
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
        prompt = build_domain_cot_prompt(
            slot=slot,
            meta=meta,
            focus_question=thread.focus,
            analysis_round=analysis_round,
            thread_context={
                "thread_id": thread.thread_id,
                "reason": thread.reason,
                "priority": thread.priority,
                "attempts": thread.attempts,
                "rag_terms": thread.rag_terms,
                "latest_summary": thread.latest_summary,
            },
        )
        raw, api_log = self.vlm_runner.analyze(
            image_path=image_path,
            prompt=prompt,
            system_prompt="你是严谨的中国画多线程分析引擎，只能基于图像证据输出内容。",
            temperature=self.config.domain_temperature,
            model=self.config.domain_model,
            stage=f"domain_cot_round_{analysis_round}",
        )
        api_log["slot_name"] = slot.slot_name
        api_log["analysis_round"] = analysis_round
        api_log["thread_id"] = thread.thread_id
        if raw.strip():
            return self._parse_domain_record(slot, raw, analysis_round), [api_log]
        return self._fallback_domain_record(slot, analysis_round, reason=api_log.get("error") or "empty_response"), [api_log]

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
        generated_questions = self._dedupe_text_list(data.get("generated_questions", []))
        confidence = self._safe_confidence(data.get("confidence", 0.0))
        controlled_used = self._dedupe_text_list(data.get("controlled_vocabulary_used", [])) or slot.controlled_vocabulary

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
            raw_response=raw,
        )

    def _update_thread(self, thread: CoTThread, record: DomainCoTRecord, analysis_round: int) -> int:
        previous_answered = set(thread.answered_questions)
        previous_evidence_count = int(thread.evidence_count)

        thread.attempts += 1
        thread.last_round = analysis_round
        thread.latest_confidence = record.confidence
        thread.evidence_count = len(record.visual_anchoring)
        thread.answered_questions = self._dedupe_text_list(thread.answered_questions + self._answered_questions_from_record(record))
        thread.unresolved_points = record.unresolved_points
        thread.latest_summary = self._record_summary(record)
        thread.latest_record = asdict(record)
        info_gain = len(set(thread.answered_questions) - previous_answered) + max(0, thread.evidence_count - previous_evidence_count)
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
    ) -> CrossValidationResult:
        issues: list[CrossValidationIssue] = []
        missing_points: list[str] = []
        rag_terms: list[str] = []
        removed_questions: list[str] = []
        context_text = " ".join([json.dumps(meta, ensure_ascii=False)] + [schema.description for schema in slot_schemas])
        allowed_dynasties = self._extract_dynasties(context_text)

        for output in outputs:
            if not output.visual_anchoring:
                issues.append(
                    CrossValidationIssue(
                        issue_type="missing_visual_anchor",
                        severity="high",
                        slot_names=[output.slot_name],
                        detail=f"{output.slot_name} 缺少稳定的视觉锚点，暂不足以支撑后续赏析。",
                        rag_terms=output.controlled_vocabulary[:3],
                    )
                )
                missing_points.append(f"{output.slot_name}: 缺少视觉锚点")

            unanswered = [item for item in output.question_coverage if not item.answered]
            for item in unanswered:
                issues.append(
                    CrossValidationIssue(
                        issue_type="question_gap",
                        severity="medium",
                        slot_names=[output.slot_name],
                        detail=f"{output.slot_name} 尚未充分回答问题：{item.question}",
                        evidence=[entry.observation for entry in output.visual_anchoring[:2]],
                        rag_terms=extract_controlled_vocabulary(item.question, item.question),
                    )
                )
                missing_points.append(f"{output.slot_name}: {item.question}")
                rag_terms.extend(extract_controlled_vocabulary(item.question, item.question))

            if output.statuses:
                issues.append(
                    CrossValidationIssue(
                        issue_type="unidentifiable_feature",
                        severity="medium",
                        slot_names=[output.slot_name],
                        detail=f"{output.slot_name} 存在不可辨识特征，需显式保留不确定性。",
                        evidence=[item.reason for item in output.domain_decoding if item.reason],
                        rag_terms=output.controlled_vocabulary[:3],
                    )
                )

            mentioned_dynasties = self._extract_dynasties(
                " ".join([item.term for item in output.domain_decoding] + [item.insight for item in output.cultural_mapping])
            )
            if allowed_dynasties and mentioned_dynasties - allowed_dynasties:
                conflict = sorted(mentioned_dynasties - allowed_dynasties)
                issues.append(
                    CrossValidationIssue(
                        issue_type="chronology_conflict",
                        severity="high",
                        slot_names=[output.slot_name],
                        detail=f"{output.slot_name} 的结论出现潜在时空冲突：{', '.join(conflict)}。",
                        rag_terms=conflict,
                    )
                )

            answered_questions = [item.question for item in output.question_coverage if item.answered]
            for question in output.generated_questions:
                if self._is_low_value_question(question, answered_questions + [item.question for item in output.question_coverage]):
                    removed_questions.append(question)
                else:
                    rag_terms.extend(extract_controlled_vocabulary(question, question))

        semantic_duplicates = self._detect_semantic_duplicates(outputs)
        for duplicate in semantic_duplicates:
            issues.append(
                CrossValidationIssue(
                    issue_type="semantic_duplicate",
                    severity="low",
                    slot_names=[],
                    detail=duplicate,
                )
            )

        return CrossValidationResult(
            issues=issues,
            semantic_duplicates=semantic_duplicates,
            missing_points=self._dedupe_text_list(missing_points),
            rag_terms=self._dedupe_text_list(rag_terms),
            removed_questions=self._dedupe_text_list(removed_questions),
            llm_review="",
        )

    def _augment_round_table_review(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        api_logs: list[dict[str, Any]],
    ) -> CrossValidationResult:
        if not self.api_client.enabled:
            return validation
        prompt = build_round_table_prompt(outputs=outputs, validation=validation, meta=meta)
        raw, api_log = self.vlm_runner.analyze(
            image_path=None,
            prompt=prompt,
            system_prompt="你是苛刻但克制的圆桌审稿人，只补充逻辑复核意见。",
            temperature=self.config.validation_temperature,
            model=self.config.validation_model or self.config.domain_model,
            stage="round_table_validation",
        )
        api_logs.append(api_log)
        validation.llm_review = raw.strip()
        return validation

    def _plan_spawn_tasks(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        threads: list[CoTThread],
    ) -> list[SpawnTask]:
        issue_map: dict[str, list[CrossValidationIssue]] = {}
        for issue in validation.issues:
            for slot_name in issue.slot_names:
                issue_map.setdefault(slot_name, []).append(issue)

        tasks: list[SpawnTask] = []
        for output in outputs:
            slot_issues = issue_map.get(output.slot_name, [])
            for item in output.question_coverage:
                if item.answered:
                    continue
                tasks.append(
                    SpawnTask(
                        slot_name=output.slot_name,
                        reason="specific_question_unanswered",
                        prompt_focus=item.question,
                        rag_terms=self._task_rag_terms(
                            focus_text=item.question,
                            fallback_terms=output.controlled_vocabulary,
                            slot_name=output.slot_name,
                        ),
                        priority=4,
                    )
                )

            for point in output.unresolved_points[:3]:
                tasks.append(
                    SpawnTask(
                        slot_name=output.slot_name,
                        reason="unresolved_point",
                        prompt_focus=point,
                        rag_terms=self._task_rag_terms(
                            focus_text=point,
                            fallback_terms=output.controlled_vocabulary,
                            slot_name=output.slot_name,
                        ),
                        priority=3,
                    )
                )

            for issue in slot_issues:
                if issue.severity != "high":
                    continue
                tasks.append(
                    SpawnTask(
                        slot_name=output.slot_name,
                        reason=issue.issue_type,
                        prompt_focus=f"复核 {output.slot_name} 中与“{issue.detail}”相关的视觉证据与术语对应关系。",
                        rag_terms=self._task_rag_terms(
                            focus_text=issue.detail,
                            fallback_terms=issue.rag_terms + output.controlled_vocabulary,
                            slot_name=output.slot_name,
                        ),
                        priority=5,
                    )
                )

            for question in output.generated_questions:
                if self._is_low_value_question(question, self._answered_questions_from_record(output)):
                    continue
                tasks.append(
                    SpawnTask(
                        slot_name=output.slot_name,
                        reason="generated_question",
                        prompt_focus=question,
                        rag_terms=self._task_rag_terms(
                            focus_text=question,
                            fallback_terms=output.controlled_vocabulary,
                            slot_name=output.slot_name,
                        ),
                        priority=2,
                    )
                )

        tasks = self._dedupe_spawn_tasks(tasks)
        filtered: list[SpawnTask] = []
        for task in tasks:
            if self._task_already_resolved(task, threads):
                continue
            filtered.append(task)
        filtered.sort(key=lambda item: (-item.priority, item.slot_name, item.prompt_focus))
        dynamic_cap = min(max(1, int(self.config.max_threads_per_round or 1)), max(1, len(filtered)))
        return filtered[:dynamic_cap]

    def _task_rag_terms(
        self,
        *,
        focus_text: str,
        fallback_terms: list[str],
        slot_name: str,
    ) -> list[str]:
        terms = extract_controlled_vocabulary(focus_text, focus_text)
        terms.extend(str(item).strip() for item in fallback_terms if str(item).strip())
        if not terms:
            terms.append(slot_name)
        return self._dedupe_text_list(terms)[:5]

    def _task_already_resolved(self, task: SpawnTask, threads: list[CoTThread]) -> bool:
        matched_thread = self._find_matching_thread(task, threads)
        if matched_thread is not None and matched_thread.status == "ANSWERED":
            return True
        return False

    def _suppress_redundant_tasks(
        self,
        tasks: list[SpawnTask],
        threads: list[CoTThread],
    ) -> tuple[list[SpawnTask], list[str]]:
        kept_tasks: list[SpawnTask] = []
        paused_thread_ids: list[str] = []
        for task in tasks:
            matched_thread = self._find_matching_thread(task, threads)
            if matched_thread is None:
                kept_tasks.append(task)
                continue
            if matched_thread.status == "ANSWERED":
                continue
            if self._should_pause_duplicate_task(matched_thread):
                matched_thread.status = "PAUSED"
                matched_thread.pause_reason = "duplicate_stalled"
                paused_thread_ids.append(matched_thread.thread_id)
                continue
            kept_tasks.append(task)
        return kept_tasks, self._dedupe_text_list(paused_thread_ids)

    def _find_matching_thread(self, task: SpawnTask, threads: list[CoTThread]) -> CoTThread | None:
        for thread in threads:
            if thread.slot_name != task.slot_name:
                continue
            if self._text_similarity(thread.focus, task.prompt_focus) < 0.82:
                continue
            return thread
        return None

    @staticmethod
    def _should_pause_duplicate_task(thread: CoTThread) -> bool:
        return (
            thread.status != "ANSWERED"
            and thread.attempts >= 1
            and int(thread.latest_new_info_gain) <= 0
            and int(thread.stale_rounds) >= _DUPLICATE_STALE_ROUNDS
        )

    def _sync_threads_with_tasks(self, threads: list[CoTThread], tasks: list[SpawnTask]) -> list[str]:
        new_thread_ids: list[str] = []
        for task in tasks:
            matched_thread = self._find_matching_thread(task, threads)

            if matched_thread is not None:
                if matched_thread.status in {"PAUSED", "BLOCKED"} and matched_thread.attempts < matched_thread.max_attempts:
                    matched_thread.status = "OPEN"
                    matched_thread.pause_reason = ""
                matched_thread.priority = max(matched_thread.priority, task.priority)
                matched_thread.rag_terms = self._dedupe_text_list(matched_thread.rag_terms + task.rag_terms)
                continue

            thread_id = f"{self._slug(task.slot_name)}-{self._slug(task.reason)}-{len(threads) + 1}"
            threads.append(
                CoTThread(
                    thread_id=thread_id,
                    slot_name=task.slot_name,
                    slot_term=next((thread.slot_term for thread in threads if thread.slot_name == task.slot_name), task.slot_name),
                    focus=task.prompt_focus,
                    reason=task.reason,
                    rag_terms=task.rag_terms,
                    priority=task.priority,
                    status="OPEN",
                    max_attempts=max(1, int(self.config.thread_attempt_limit or 1)),
                    parent_thread_id=task.source_thread_id,
                )
            )
            new_thread_ids.append(thread_id)
        return new_thread_ids

    def _build_routing(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        tasks: list[SpawnTask],
        convergence: dict[str, Any],
    ) -> RoutingDecision:
        stable_slots = self._stable_slots(outputs, validation)
        if tasks and not convergence["converged"]:
            return RoutingDecision(
                action="SPAWN_COT",
                rationale=["存在高价值未解问题，继续动态生成 CoT 线程。"],
                paused_slots=stable_slots,
                spawned_tasks=tasks,
                removed_questions=validation.removed_questions,
                merged_duplicates=validation.semantic_duplicates,
                converged=False,
                convergence_reason=convergence["reason"],
                answered_slots=convergence["answered_slots"],
            )

        rationale = ["当前不再生成新的 CoT 线程。"]
        if convergence["converged"]:
            rationale.append(convergence["reason"])
        else:
            rationale.append("线程池进入停滞或阶段性暂停。")
        return RoutingDecision(
            action="PAUSE_COT",
            rationale=rationale,
            paused_slots=stable_slots,
            spawned_tasks=[],
            removed_questions=validation.removed_questions,
            merged_duplicates=validation.semantic_duplicates,
            converged=convergence["converged"],
            convergence_reason=convergence["reason"],
            answered_slots=convergence["answered_slots"],
        )

    def _check_convergence(
        self,
        slot_schemas: list[SlotSchema],
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        threads: list[CoTThread],
        dialogue_state: DialogueState,
    ) -> dict[str, Any]:
        total_questions = [question for slot in slot_schemas for question in slot.specific_questions]
        answered_questions = self._collect_answered_questions(outputs)
        unanswered_questions = [question for question in total_questions if not any(self._text_similarity(question, answered) >= 0.8 for answered in answered_questions)]
        unresolved_points = self._collect_unresolved_questions(outputs, validation)
        high_issues = [issue.detail for issue in validation.issues if issue.severity == "high"]
        pending_threads = [
            thread.thread_id
            for thread in threads
            if thread.status in {"OPEN", "RETRY"} and thread.attempts < thread.max_attempts
        ]

        fully_answered = not unanswered_questions
        no_new_cot = not pending_threads
        all_threads_terminal = all(thread.status in {"ANSWERED", "PAUSED", "MERGED", "BLOCKED"} or thread.attempts >= thread.max_attempts for thread in threads)
        converged = fully_answered and not unresolved_points and not high_issues and no_new_cot and all_threads_terminal

        reason = "尚未收敛。"
        if converged:
            reason = "所有必答问题均已覆盖，未再产生新的 CoT，线程池已经收敛，可输出最终赏析 prompt。"
        elif dialogue_state.no_new_info_rounds >= max(1, int(self.config.convergence_patience or 1)) and no_new_cot:
            reason = "连续多轮没有新增有效信息，当前已进入停滞状态。"
        elif unanswered_questions:
            reason = "仍存在未完成的问题，需要继续补充。"

        return {
            "converged": converged,
            "reason": reason,
            "answered_slots": self._stable_slots(outputs, validation),
            "answered_questions": answered_questions,
            "unanswered_questions": unanswered_questions,
            "unresolved_points": unresolved_points,
            "high_issues": high_issues,
            "pending_threads": pending_threads,
            "no_new_info_rounds": dialogue_state.no_new_info_rounds,
        }

    def _generate_summary(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        prepared_image: PreparedImage,
        meta: dict,
        api_logs: list[dict[str, Any]],
    ) -> str:
        if self.api_client.enabled:
            prompt = build_summary_prompt(outputs=outputs, validation=validation, meta=meta)
            raw, api_log = self.vlm_runner.analyze(
                image_path=prepared_image.path,
                prompt=prompt,
                system_prompt="你是中国画学术写作者，善于把证据、术语和通俗解释整合成清晰 Markdown。",
                temperature=self.config.summary_temperature,
                model=self.config.summary_model or self.config.domain_model,
                stage="summary_generation",
            )
            api_logs.append(api_log)
            if raw.strip():
                return raw.strip()
        return self._render_summary_locally(outputs, validation)

    def _generate_final_appreciation_prompt(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict,
        dialogue_state: DialogueState,
    ) -> str:
        return build_final_appreciation_prompt(outputs=outputs, validation=validation, meta=meta, dialogue_state=dialogue_state)

    def _render_summary_locally(self, outputs: list[DomainCoTRecord], validation: CrossValidationResult) -> str:
        lines = ["# 国画赏析", "", "## 画面切入", ""]
        overview_bits: list[str] = []
        for output in outputs:
            if output.visual_anchoring:
                overview_bits.append(output.visual_anchoring[0].observation)
        if overview_bits:
            lines.append("这幅作品的关键信息可以先从以下视觉证据进入：")
            for item in overview_bits:
                lines.append(f"- {item}")
        else:
            lines.append("当前批次尚未形成足够稳定的视觉锚点，因此总览仍需谨慎。")

        for output in outputs:
            lines.extend(["", f"## {output.slot_name}", ""])
            if output.visual_anchoring:
                lines.append("**表象观察**")
                for item in output.visual_anchoring:
                    body = item.observation
                    if item.position:
                        body = f"{item.position}：{body}"
                    if item.evidence:
                        body = f"{body}。证据：{item.evidence}"
                    lines.append(f"- {body}")
            if output.domain_decoding:
                lines.append("")
                lines.append("**专业解码**")
                for item in output.domain_decoding:
                    sentence = item.term or "特征"
                    if item.explanation:
                        sentence = f"{sentence}：{item.explanation}"
                    if item.status != "IDENTIFIED" and item.reason:
                        sentence = f"{sentence}（暂不可辨：{item.reason}）"
                    lines.append(f"- {sentence}")
            if output.cultural_mapping:
                lines.append("")
                lines.append("**文化与时代线索**")
                for item in output.cultural_mapping:
                    sentence = item.insight
                    if item.basis:
                        sentence = f"{sentence}。依据：{item.basis}"
                    if item.risk_note:
                        sentence = f"{sentence}。提示：{item.risk_note}"
                    lines.append(f"- {sentence}")

        if validation.issues:
            lines.extend(["", "## 仍需谨慎处", ""])
            for issue in validation.issues:
                if issue.severity == "low":
                    continue
                lines.append(f"- **{issue.issue_type}**：{issue.detail}")
        return "\n".join(lines).strip()

    def _build_round_memory(
        self,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        dialogue_state: DialogueState,
        threads: list[CoTThread],
    ) -> dict[str, Any]:
        return {
            "round_index": dialogue_state.final_round_index,
            "converged": dialogue_state.converged,
            "convergence_reason": dialogue_state.convergence_reason,
            "resolved_questions": dialogue_state.resolved_questions,
            "unresolved_questions": dialogue_state.unresolved_questions,
            "removed_questions": dialogue_state.removed_questions,
            "merged_duplicates": dialogue_state.merged_duplicates,
            "issues": [
                {
                    "issue_type": issue.issue_type,
                    "severity": issue.severity,
                    "slot_names": issue.slot_names,
                    "detail": issue.detail,
                    "rag_terms": issue.rag_terms,
                }
                for issue in validation.issues
            ],
            "slots": [
                {
                    "slot_name": output.slot_name,
                    "slot_term": output.slot_term,
                    "visual_anchoring": [asdict(item) for item in output.visual_anchoring],
                    "domain_decoding": [asdict(item) for item in output.domain_decoding],
                    "cultural_mapping": [asdict(item) for item in output.cultural_mapping],
                    "question_coverage": [asdict(item) for item in output.question_coverage],
                    "unresolved_points": output.unresolved_points,
                    "generated_questions": output.generated_questions,
                    "confidence": output.confidence,
                }
                for output in outputs
            ],
            "threads": [
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
            ],
        }

    def _render_memory_markdown(self, round_memory: dict[str, Any]) -> str:
        lines = ["# Round Memory", ""]
        lines.append(f"- round_index: {round_memory.get('round_index', 0)}")
        lines.append(f"- converged: {str(bool(round_memory.get('converged', False))).lower()}")
        lines.append(f"- convergence_reason: {round_memory.get('convergence_reason', '')}")
        lines.append("")

        lines.extend(["## Resolved Questions", ""])
        resolved = round_memory.get("resolved_questions", []) or []
        if resolved:
            for item in resolved:
                lines.append(f"- {item}")
        else:
            lines.append("- none")
        lines.append("")

        lines.extend(["## Unresolved Questions", ""])
        unresolved = round_memory.get("unresolved_questions", []) or []
        if unresolved:
            for item in unresolved:
                lines.append(f"- {item}")
        else:
            lines.append("- none")
        lines.append("")

        for slot in round_memory.get("slots", []) or []:
            lines.extend([f"## Slot: {slot.get('slot_name', '')}", ""])
            lines.append(f"- slot_term: {slot.get('slot_term', '')}")
            lines.append(f"- confidence: {slot.get('confidence', 0.0)}")
            for title, key in (
                ("visual_anchoring", "visual_anchoring"),
                ("domain_decoding", "domain_decoding"),
                ("cultural_mapping", "cultural_mapping"),
                ("question_coverage", "question_coverage"),
                ("unresolved_points", "unresolved_points"),
                ("generated_questions", "generated_questions"),
            ):
                lines.append(f"- {title}: {json.dumps(slot.get(key, []), ensure_ascii=False)}")
            lines.append("")
        return "\n".join(lines).strip()

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

        memory_path = run_dir / "memory.md"
        memory_path.write_text(result.memory_markdown, encoding="utf-8")
        output_files["memory"] = str(memory_path)

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

        routing_path = run_dir / "routing.json"
        routing_path.write_text(json.dumps(asdict(result.routing), ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["routing"] = str(routing_path)

        dialogue_state_path = run_dir / "dialogue_state.json"
        dialogue_state_path.write_text(json.dumps(asdict(result.dialogue_state), ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["dialogue_state"] = str(dialogue_state_path)

        cot_threads_path = run_dir / "cot_threads.json"
        cot_threads_path.write_text(json.dumps([asdict(item) for item in result.cot_threads], ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["cot_threads"] = str(cot_threads_path)

        prepared_path = run_dir / "prepared_image.json"
        prepared_path.write_text(json.dumps(asdict(result.prepared_image), ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["prepared_image"] = str(prepared_path)

        report_path = run_dir / "report.json"
        report_payload = {
            "image_path": result.image_path,
            "routing_action": result.routing.action,
            "converged": result.routing.converged,
            "convergence_reason": result.routing.convergence_reason,
            "issue_count": len(result.cross_validation.issues),
            "removed_questions": result.routing.removed_questions,
            "execution_log": result.execution_log,
        }
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        output_files["report"] = str(report_path)

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
            merged.append(task)
        return merged

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
        items: list[str] = []
        for output in outputs:
            items.extend(output.unresolved_points)
        items.extend(validation.missing_points)
        return self._dedupe_text_list(items)

    def _stable_slots(self, outputs: list[DomainCoTRecord], validation: CrossValidationResult) -> list[str]:
        issue_slots = {slot_name for issue in validation.issues for slot_name in issue.slot_names if issue.severity in {"high", "medium"}}
        stable: list[str] = []
        for output in outputs:
            unanswered = [item for item in output.question_coverage if not item.answered]
            if output.slot_name in issue_slots:
                continue
            if unanswered or output.unresolved_points:
                continue
            stable.append(output.slot_name)
        return self._dedupe_text_list(stable)

    @staticmethod
    def _slug(value: str) -> str:
        slug = re.sub(r"[^a-zA-Z0-9\u4e00-\u9fff]+", "-", str(value or "").strip())
        return slug.strip("-").lower() or "item"
