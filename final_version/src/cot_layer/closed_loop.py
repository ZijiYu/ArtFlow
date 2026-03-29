from __future__ import annotations

import asyncio
import json
import mimetypes
import re
import sys
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlparse

from .image_utils import prepare_image
from .meta_loader import load_context_meta, merge_meta
from .config_loader import read_text_secret_file
from .models import (
    CrossValidationIssue,
    CrossValidationResult,
    DecodingItem,
    DialogueState,
    DomainCoTRecord,
    EvidenceItem,
    MappingItem,
    PipelineConfig,
    PreparedImage,
    QuestionCoverage,
    SlotSchema,
    SpawnTask,
)
from .new_api_client import NewAPIClient
from .pipeline import DynamicAgentPipeline
from .schema_loader import extract_controlled_vocabulary, extract_slot_terms, load_slot_schemas
from ..common.web_search_client import SerperWebSearchClient, WebSearchHit
from ..common.prompt_utils import build_slot_summary_payload

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PRECEPTION_ROOT = _REPO_ROOT / "preception_layer"


DOWNSTREAM_SYSTEM_PROMPT = """你是 preception_layer 的下游任务扩展器。
请严格基于输入的现有 Slots、文字证据、本体关系和 reflection 任务，输出可直接回灌到下一轮分析的 JSON。

输出结构：
{
  "status": "applied|insufficient_evidence|needs_followup",
  "merge_candidates": [
    {
      "target_slot_name": "已有槽位名",
      "target_slot_term": "已有术语",
      "description_append": "应补到该槽位中的文字证据",
      "additional_questions": ["问题1", "问题2"],
      "reason": "为什么应合并"
    }
  ],
  "new_slots": [
    {
      "slot_name": "具体领域名",
      "slot_term": "具体术语",
      "description": "基于已有证据整理后的描述",
      "specific_questions": ["问题1", "问题2"],
      "metadata": {
        "confidence": 0.0,
        "source_id": "downstream"
      }
    }
  ],
  "ontology_updates": ["新的本体关系或层级说明"],
  "text_evidence_updates": [
    {
      "term": "术语",
      "description": "新的文字证据摘要",
      "text_evidence": ["证据1", "证据2"]
    }
  ],
  "search_queries": [
    {
      "query_text": "建议检索词",
      "intent": "检索目的",
      "expected_evidence": ["期望补到的知识点"]
    }
  ],
  "resolved_questions": ["已解决的问题"],
  "open_questions": ["仍未解决的问题"],
  "notes": ["其他保守说明"]
}

要求：
1. 不要重新凭空发明画面细节。
2. 若证据不足，请返回 status=insufficient_evidence，并把原因写入 notes。
3. 优先补充 cached_documents / rag_documents 中已有的文字证据，并先复用 cached_terms，再决定是否真的需要新的 search_queries。
4. 新增槽位必须具体、稳定、可核验。
5. 如果 cached_documents 已足以支持回答，请不要重复建议相同 query。
6. 如果只是提出下一轮 RAG 建议，请写入 search_queries，不要伪装成已确认事实。
"""


@dataclass(slots=True)
class ClosedLoopConfig:
    output_dir: str = "artifacts_closed_loop"
    max_closed_loop_rounds: int = 3
    max_downstream_tasks_per_round: int = 4
    stall_round_limit: int = 2
    downstream_rag_query_repeat_limit: int = 2
    bootstrap_model: str | None = None
    downstream_model: str | None = None
    embedding_model: str | None = None


@dataclass(slots=True)
class ClosedLoopTaskRecord:
    round_index: int
    task_index: int
    slot_name: str
    task_reason: str
    task_focus: str
    task_path: str
    applied_changes: dict[str, Any]
    status: str


@dataclass(slots=True)
class ClosedLoopResult:
    run_dir: str
    bootstrap_slots_file: str
    bootstrap_context_file: str
    final_slots_file: str
    final_prompt_path: str
    converged: bool
    stop_reason: str
    slot_rounds: list[dict[str, Any]] = field(default_factory=list)
    downstream_tasks: list[ClosedLoopTaskRecord] = field(default_factory=list)


class ClosedLoopCoordinator:
    def __init__(
        self,
        *,
        slots_config: PipelineConfig | None = None,
        closed_loop_config: ClosedLoopConfig | None = None,
        api_client: NewAPIClient | None = None,
        perception_pipeline_factory: Callable[[Any], Any] | None = None,
        downstream_runner_factory: Callable[[Any], Any] | None = None,
        slots_pipeline_factory: Callable[..., Any] = DynamicAgentPipeline,
    ) -> None:
        self.slots_config = slots_config or PipelineConfig()
        self.closed_loop_config = closed_loop_config or ClosedLoopConfig()
        self.api_client = api_client or NewAPIClient()
        self._perception_pipeline_factory = perception_pipeline_factory
        self._downstream_runner_factory = downstream_runner_factory
        self._slots_pipeline_factory = slots_pipeline_factory

    def run(self, *, image_path: str, input_text: str, meta: dict | None = None) -> ClosedLoopResult:
        meta = meta or {}
        run_dir = Path(self.closed_loop_config.output_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir.mkdir(parents=True, exist_ok=True)

        prepared_input = self._prepare_input_image(image_path)
        self._print_closed_loop_progress(
            stage="bootstrap_prepare",
            round_index=0,
            note=f"prepared_image={prepared_input.path}",
        )
        bootstrap = self._run_perception_bootstrap(image_path=prepared_input.path, input_text=input_text, run_dir=run_dir)
        runtime_slots = load_slot_schemas(str(bootstrap["slots_file"]))
        runtime_meta = merge_meta(load_context_meta(str(bootstrap["context_file"])), meta)
        runtime_meta.setdefault("downstream_updates", [])
        runtime_meta.setdefault("closed_loop_notes", [])
        runtime_meta.setdefault("dialogue_turns", [])
        runtime_meta.setdefault("round_memories", [])
        runtime_meta.setdefault("pending_cot_tasks", [])
        runtime_meta.setdefault("retained_facts", [])
        runtime_meta.setdefault("rag_cache", [])
        self._seed_rag_cache_from_post_rag(runtime_meta)
        self._print_closed_loop_progress(
            stage="bootstrap_done",
            round_index=0,
            note=f"slots={len(runtime_slots)} context={bootstrap['context_file']}",
        )

        runtime_state_dir = run_dir / "runtime_state"
        runtime_state_dir.mkdir(parents=True, exist_ok=True)

        slot_rounds: list[dict[str, Any]] = []
        downstream_tasks: list[ClosedLoopTaskRecord] = []
        latest_result = None
        latest_outputs: dict[str, str] = {}
        stop_reason = "max_closed_loop_rounds_reached"
        no_change_rounds = 0
        seen_followup_tasks: list[SpawnTask] = []

        for round_index in range(1, max(1, int(self.closed_loop_config.max_closed_loop_rounds)) + 1):
            self._print_closed_loop_progress(
                stage="round_start",
                round_index=round_index,
                note=f"slots={len(runtime_slots)}",
            )
            slots_path = runtime_state_dir / f"slots_round_{round_index:02d}.jsonl"
            self._write_slots_jsonl(runtime_slots, slots_path)

            slots_pipeline = self._slots_pipeline_factory(
                config=replace(
                    self.slots_config,
                    slots_file=str(slots_path),
                    output_dir=str(run_dir / "slots_rounds" / f"round_{round_index:02d}"),
                ),
                api_client=self.api_client,
            )
            result = slots_pipeline.run(image_path=prepared_input.path, meta=runtime_meta)
            result = slots_pipeline.finalize_result(result, meta=runtime_meta)
            outputs = slots_pipeline.save_result(result, output_dir=str(run_dir / "slots_rounds" / f"round_{round_index:02d}"))
            latest_result = result
            latest_outputs = outputs
            self._append_round_memory(runtime_meta, result.round_memory)

            slot_rounds.append(
                {
                    "round_index": round_index,
                    "slots_run_dir": outputs.get("run_dir", ""),
                    "routing": asdict(result.routing),
                    "issue_count": len(result.cross_validation.issues),
                }
            )
            self._print_closed_loop_progress(
                stage="slots_round_done",
                round_index=round_index,
                result=result,
                note=f"issues={len(result.cross_validation.issues)}",
            )
            runtime_slots = self._apply_slot_lifecycle_reviews(runtime_slots, result.cross_validation)
            runtime_slots, runtime_meta, slot_progression = self._advance_fixed_slots(
                slot_schemas=runtime_slots,
                result=result,
                meta=runtime_meta,
            )
            cot_tasks, downstream_tasks_for_round = self._split_spawn_tasks(result.routing.spawned_tasks)
            cot_tasks = self._compress_cot_tasks(
                self._filter_suppressed_tasks(cot_tasks, slot_progression["suppressed_slots"])
            )
            if not bool(getattr(self.slots_config, "enable_rag_verification", True)):
                downstream_tasks_for_round = []
            downstream_tasks_for_round = self._dedupe_spawn_tasks(
                self._filter_suppressed_tasks(
                    slot_progression["downstream_tasks"] + downstream_tasks_for_round,
                    slot_progression["closed_slots"],
                )
            )
            runtime_meta["pending_cot_tasks"] = [asdict(task) for task in cot_tasks]

            progression_changed = bool(slot_progression["progressed_slots"] or slot_progression["closed_slots"])

            if result.routing.converged and not (cot_tasks or downstream_tasks_for_round):
                stop_reason = result.routing.convergence_reason or "converged"
                self._print_closed_loop_progress(
                    stage="converged",
                    round_index=round_index,
                    result=result,
                    note=stop_reason,
                )
                break

            tasks = downstream_tasks_for_round[: max(1, int(self.closed_loop_config.max_downstream_tasks_per_round))]
            if not tasks and not cot_tasks:
                stop_reason = "no_spawned_tasks"
                self._print_closed_loop_progress(
                    stage="stop",
                    round_index=round_index,
                    result=result,
                    note=stop_reason,
                )
                break

            round_changed = progression_changed
            if tasks:
                self._print_closed_loop_progress(
                    stage="downstream_start",
                    round_index=round_index,
                    result=result,
                    note=f"tasks={len(tasks)} cot_followups={len(cot_tasks)}",
                )
                runner = self._build_downstream_runner(run_dir=run_dir, round_index=round_index)
                for task_index, task in enumerate(tasks, start=1):
                    external_rag = self._run_task_retrieval(
                        task=task,
                        image_path=prepared_input.path,
                        run_dir=run_dir,
                        round_index=round_index,
                        task_index=task_index,
                    )
                    self._merge_external_rag_into_meta(
                        runtime_meta,
                        external_rag=external_rag,
                        task=task,
                    )
                    payload = self._build_downstream_payload(
                        task=task,
                        slot_schemas=runtime_slots,
                        meta=runtime_meta,
                        slots_result=result,
                        external_rag=external_rag,
                    )
                    task_dir = run_dir / "downstream_rounds" / f"round_{round_index:02d}"
                    task_dir.mkdir(parents=True, exist_ok=True)
                    task_name = f"closed_loop_round_{round_index}_task_{task_index}"
                    response = runner.run_json(
                        task_name=task_name,
                        system_prompt=DOWNSTREAM_SYSTEM_PROMPT,
                        user_text=json.dumps(payload, ensure_ascii=False, indent=2),
                        image_file=prepared_input.path,
                    )
                    task_path = task_dir / f"task_{task_index:02d}.json"
                    task_path.write_text(
                        json.dumps(
                            {
                                "task_name": task_name,
                                "slot_name": task.slot_name,
                                "task_reason": task.reason,
                                "task_focus": task.prompt_focus,
                                "image_file": prepared_input.path,
                                "system_prompt": DOWNSTREAM_SYSTEM_PROMPT,
                                "payload": payload,
                                "response": response,
                            },
                            ensure_ascii=False,
                            indent=2,
                        ),
                        encoding="utf-8",
                    )
                    self._cleanup_downstream_task_logs(task_dir)

                    runtime_slots, runtime_meta, applied = self._apply_downstream_response(
                        response=response,
                        slot_schemas=runtime_slots,
                        meta=runtime_meta,
                        task=task,
                    )
                    round_changed = round_changed or bool(applied.get("changed"))
                    downstream_tasks.append(
                        ClosedLoopTaskRecord(
                            round_index=round_index,
                            task_index=task_index,
                            slot_name=task.slot_name,
                            task_reason=task.reason,
                            task_focus=task.prompt_focus,
                            task_path=str(task_path),
                            applied_changes=applied,
                            status=str(response.get("status", "unknown")).strip() or "unknown",
                        )
                    )
                    self._print_closed_loop_progress(
                        stage="downstream_task_done",
                        round_index=round_index,
                        result=result,
                        note=(
                            f"task={task_index}/{len(tasks)} slot={task.slot_name} "
                            f"changed={str(bool(applied.get('changed'))).lower()}"
                        ),
                    )

            current_followup_tasks = self._dedupe_spawn_tasks(cot_tasks + tasks)
            has_new_followups = self._has_unseen_followups(current_followup_tasks, seen_followup_tasks)
            if current_followup_tasks:
                seen_followup_tasks = self._dedupe_spawn_tasks(seen_followup_tasks + current_followup_tasks)

            if round_changed or has_new_followups:
                no_change_rounds = 0
            else:
                no_change_rounds += 1

            note = (
                f"closed_loop round {round_index}: downstream_changed={str(round_changed).lower()} "
                f"downstream_tasks={len(tasks)} cot_followups={len(cot_tasks)} "
                f"slot_progressed={len(slot_progression['progressed_slots'])} slot_closed={len(slot_progression['closed_slots'])}"
            )
            if note not in runtime_meta["closed_loop_notes"]:
                runtime_meta["closed_loop_notes"].append(note)
            self._append_dialogue_turn(runtime_meta, note)
            self._print_closed_loop_progress(
                stage="downstream_round_done",
                round_index=round_index,
                result=result,
                note=(
                    f"changed={str(round_changed).lower()} no_change_rounds={no_change_rounds} "
                    f"cot_followups={len(cot_tasks)} new_followups={str(has_new_followups).lower()}"
                ),
            )

            if no_change_rounds >= max(1, int(self.closed_loop_config.stall_round_limit)):
                stop_reason = "downstream_stalled"
                self._print_closed_loop_progress(
                    stage="stop",
                    round_index=round_index,
                    result=result,
                    note=stop_reason,
                )
                break

        final_slots_file = runtime_state_dir / "slots_final.jsonl"
        self._write_slots_jsonl(runtime_slots, final_slots_file)

        final_prompt_path = run_dir / "final_appreciation_prompt.md"
        final_appreciation = self._build_closed_loop_final_appreciation(
            slot_schemas=runtime_slots,
            meta=runtime_meta,
            latest_result=latest_result,
        )
        final_prompt_path.write_text(final_appreciation, encoding="utf-8")

        report_path = run_dir / "closed_loop_report.json"
        report_payload = {
            "run_dir": str(run_dir),
            "prepared_input": asdict(prepared_input),
            "bootstrap_slots_file": str(bootstrap["slots_file"]),
            "bootstrap_context_file": str(bootstrap["context_file"]),
            "final_slots_file": str(final_slots_file),
            "final_prompt_path": str(final_prompt_path),
            "converged": bool(latest_result.routing.converged) if latest_result else False,
            "stop_reason": stop_reason,
            "slot_rounds": slot_rounds,
            "downstream_tasks": [asdict(item) for item in downstream_tasks],
            "latest_slots_outputs": latest_outputs,
        }
        report_path.write_text(json.dumps(report_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._print_closed_loop_progress(
            stage="finished",
            round_index=len(slot_rounds),
            result=latest_result,
            note=f"stop_reason={stop_reason}",
        )

        return ClosedLoopResult(
            run_dir=str(run_dir),
            bootstrap_slots_file=str(bootstrap["slots_file"]),
            bootstrap_context_file=str(bootstrap["context_file"]),
            final_slots_file=str(final_slots_file),
            final_prompt_path=str(final_prompt_path),
            converged=bool(latest_result.routing.converged) if latest_result else False,
            stop_reason=stop_reason,
            slot_rounds=slot_rounds,
            downstream_tasks=downstream_tasks,
        )

    def _print_closed_loop_progress(
        self,
        *,
        stage: str,
        round_index: int,
        result: Any | None = None,
        note: str = "",
    ) -> None:
        summary = f"[closed_loop] stage={stage} round={round_index}/{max(1, int(self.closed_loop_config.max_closed_loop_rounds))}"
        if result is not None:
            solved_count = len(getattr(result.dialogue_state, "resolved_questions", []) or [])
            remaining_count = len(getattr(result.dialogue_state, "unresolved_questions", []) or [])
            cot_total = len(getattr(result, "cot_threads", []) or [])
            summary += (
                f" cot_total={cot_total} questions_solved={solved_count} "
                f"questions_remaining={remaining_count} routing={result.routing.action}"
            )
        if note:
            summary += f" note={note}"
        print(summary, flush=True)

    def _prepare_input_image(self, image_path: str) -> PreparedImage:
        return prepare_image(
            image_path=image_path,
            max_pixel=max(1, int(self.slots_config.max_pixel or 1)),
            resize=bool(self.slots_config.resize_image),
        )

    def _build_perception_config(
        self,
        *,
        context_path: Path,
        rag_search_record_path: Path,
        llm_chat_record_path: Path,
        output_path: Path,
        judge_model_override: str | None = None,
    ) -> Any:
        perception_config_cls = _load_perception_config()
        base_config = perception_config_cls.from_env()
        base_url = self.api_client.base_url or base_config.base_url
        judge_model = judge_model_override or self.api_client.model or base_config.judge_model
        embedding_model = self.closed_loop_config.embedding_model or base_config.embedding_model
        if base_url.rstrip("/") == "https://api.openai.com/v1" and embedding_model.startswith("baai/"):
            embedding_model = "text-embedding-3-small"
        return replace(
            base_config,
            api_key=self.api_client.api_key or base_config.api_key,
            base_url=base_url,
            judge_model=judge_model,
            embedding_model=embedding_model,
            request_timeout=float(self.api_client.timeout or getattr(base_config, "request_timeout", 180.0)),
            context_path=context_path,
            rag_search_record_path=rag_search_record_path,
            llm_chat_record_path=llm_chat_record_path,
            output_path=output_path,
        )

    def _run_perception_bootstrap(self, *, image_path: str, input_text: str, run_dir: Path) -> dict[str, Path]:
        bootstrap_dir = run_dir / "perception_bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        _, _, perception_pipeline_cls = _load_perception_layer()
        config = self._build_perception_config(
            context_path=bootstrap_dir / "context.md",
            rag_search_record_path=bootstrap_dir / "rag_search_record.md",
            llm_chat_record_path=bootstrap_dir / "llm_chat_record.jsonl",
            output_path=bootstrap_dir / "slots.jsonl",
            judge_model_override=self.closed_loop_config.bootstrap_model,
        )
        pipeline_factory = self._perception_pipeline_factory or perception_pipeline_cls
        pipeline = pipeline_factory(config)
        result = asyncio.run(
            pipeline.run(
                image_file=image_path,
                input_text=input_text,
                output_path=config.output_path,
            )
        )
        return {
            "slots_file": result.output_path,
            "context_file": result.context_path,
        }

    def _build_downstream_runner(self, *, run_dir: Path, round_index: int) -> Any:
        downstream_dir = run_dir / "downstream_rounds" / f"round_{round_index:02d}"
        downstream_dir.mkdir(parents=True, exist_ok=True)
        _, downstream_runner_cls, _ = _load_perception_layer()
        config = self._build_perception_config(
            context_path=downstream_dir / "context.md",
            rag_search_record_path=downstream_dir / "rag_search_record.md",
            llm_chat_record_path=downstream_dir / "llm_chat_record.jsonl",
            output_path=downstream_dir / "slots.jsonl",
            judge_model_override=self.closed_loop_config.downstream_model,
        )
        runner_factory = self._downstream_runner_factory or downstream_runner_cls
        return runner_factory(config)

    @staticmethod
    def _cleanup_downstream_task_logs(task_dir: Path) -> None:
        for name in ("context.md", "llm_chat_record.jsonl"):
            path = task_dir / name
            if path.exists() and path.is_file():
                path.unlink()

    def _build_downstream_payload(
        self,
        *,
        task: SpawnTask,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
        slots_result: Any,
        external_rag: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        external_rag = external_rag or {"queries": [], "documents": []}
        task_slot_term = str(getattr(task, "slot_term", "") or "").strip()
        cached_documents = self._select_cached_documents_for_task(
            task=task,
            slot_schemas=slot_schemas,
            meta=meta,
        )
        cached_terms = self._dedupe_text_list(
            [
                str(item.get("term", "")).strip()
                for item in cached_documents
                if isinstance(item, dict) and str(item.get("term", "")).strip()
            ]
        )
        related_issues = [
            issue.detail
            for issue in slots_result.cross_validation.issues
            if task.slot_name in issue.slot_names
        ]
        retained_facts = self._select_retained_facts_for_task(
            task=task,
            slot_schemas=slot_schemas,
            meta=meta,
        )
        relevant_slots = [
            self._serialize_slot(slot)
            for slot in slot_schemas
            if slot.slot_name == task.slot_name
            or (
                task_slot_term
                and any(
                    self._normalize_text(task_slot_term) == self._normalize_text(term)
                    for term in self._slot_terms(slot)
                )
            )
        ]
        if not relevant_slots:
            relevant_slots = [self._serialize_slot(slot) for slot in slot_schemas]

        extra_constraints = [
            f"reason={task.reason}",
            f"focus={task.prompt_focus}",
            f"priority={task.priority}",
        ]
        if str(getattr(task, "retrieval_reason", "")).strip():
            extra_constraints.append(f"retrieval_reason={str(getattr(task, 'retrieval_reason', '')).strip()}")
        extra_constraints.extend(f"rag_term={term}" for term in task.rag_terms)
        extra_constraints.extend(f"web_query={term}" for term in self._coerce_str_list(getattr(task, "web_queries", [])))
        extra_constraints.extend(f"cached_term={term}" for term in cached_terms[:6])
        extra_constraints.extend(f"retained_fact={item.get('fact', '')}" for item in retained_facts[:4] if str(item.get("fact", "")).strip())
        extra_constraints.extend(f"downstream_rag_query={term}" for term in self._coerce_str_list(external_rag.get("queries")))
        extra_constraints.extend(f"issue={detail}" for detail in related_issues)
        rag_documents = (
            self._coerce_list(meta.get("post_rag_text_extraction"))
            + cached_documents
            + self._coerce_list(external_rag.get("documents"))
        )

        return {
            "painting_profile": meta.get("domain_profile", {}),
            "existing_slots": relevant_slots,
            "ontology_relations": meta.get("ontology_updates", []),
            "cached_terms": cached_terms,
            "cached_documents": cached_documents,
            "retained_facts": retained_facts,
            "rag_documents": self._dedupe_dict_list(rag_documents),
            "external_rag_queries": self._coerce_str_list(external_rag.get("queries")),
            "external_retrieval_queries": self._coerce_str_list(external_rag.get("queries")),
            "task_goal": self._task_goal(task),
            "reflection_context": {
                "routing_action": slots_result.routing.action,
                "convergence_reason": slots_result.routing.convergence_reason,
                "removed_questions": slots_result.routing.removed_questions,
                "merged_duplicates": slots_result.routing.merged_duplicates,
                "resolved_questions": slots_result.dialogue_state.resolved_questions,
                "unresolved_questions": slots_result.dialogue_state.unresolved_questions,
            },
            "extra_constraints": extra_constraints,
        }

    def _run_task_retrieval(
        self,
        *,
        task: SpawnTask,
        image_path: str,
        run_dir: Path,
        round_index: int,
        task_index: int,
    ) -> dict[str, Any]:
        retrieval_mode = DynamicAgentPipeline._normalize_retrieval_mode(getattr(task, "retrieval_mode", "rag"))
        rag_payload = {"queries": [], "documents": []}
        web_payload = {"queries": [], "documents": []}
        if retrieval_mode in {"rag", "hybrid"}:
            rag_payload = self._run_task_rag(
                task=task,
                image_path=image_path,
                run_dir=run_dir,
                round_index=round_index,
                task_index=task_index,
            )
        should_run_web = retrieval_mode in {"web", "hybrid"}
        if not should_run_web:
            fallback_on_empty_rag = bool(getattr(self.slots_config, "web_search_fallback_on_empty_rag", True))
            if (
                bool(getattr(self.slots_config, "enable_web_search", False))
                and fallback_on_empty_rag
                and not rag_payload.get("documents")
                and self._resolve_web_queries(task)
            ):
                should_run_web = True
        if should_run_web:
            web_payload = self._run_task_web_search(
                task=task,
                run_dir=run_dir,
                round_index=round_index,
                task_index=task_index,
            )
        return {
            "queries": self._dedupe_text_list(
                self._coerce_str_list(rag_payload.get("queries")) + self._coerce_str_list(web_payload.get("queries"))
            ),
            "documents": self._dedupe_dict_list(
                self._coerce_list(rag_payload.get("documents")) + self._coerce_list(web_payload.get("documents"))
            ),
        }

    def _run_task_rag(
        self,
        *,
        task: SpawnTask,
        image_path: str,
        run_dir: Path,
        round_index: int,
        task_index: int,
    ) -> dict[str, Any]:
        query_terms = self._normalize_downstream_queries(task.rag_terms, slot_name=task.slot_name)
        if not query_terms:
            return {"queries": [], "documents": []}

        rag_record_path = run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "rag_search_record.md"
        rag_cache_path = self._downstream_rag_cache_path(run_dir)
        config = self._build_perception_config(
            context_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "context.md",
            rag_search_record_path=rag_record_path,
            llm_chat_record_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "llm_chat_record.jsonl",
            output_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "slots.jsonl",
            judge_model_override=self.closed_loop_config.downstream_model,
        )
        rag_client_cls = _load_perception_rag_client()
        repeat_limit = max(1, int(self.closed_loop_config.downstream_rag_query_repeat_limit or 2))
        query_counts = self._read_downstream_rag_query_counts(run_dir)
        query_cache = self._read_downstream_rag_cache(rag_cache_path)

        image_file = Path(image_path)
        image_bytes = image_file.read_bytes() if image_file.exists() and image_file.is_file() else None
        image_mime_type = mimetypes.guess_type(str(image_file))[0] or "image/png"

        documents: list[dict[str, Any]] = []
        used_queries: list[str] = []
        for query in self._dedupe_text_list(query_terms):
            query_key = self._query_signature(query)
            cached_documents = self._cached_documents_for_query(query_cache, query)
            if cached_documents:
                used_queries.append(query)
                documents.extend(cached_documents)
                self._print_closed_loop_progress(
                    stage="downstream_rag_cache_hit",
                    round_index=round_index,
                    note=f"task={task_index} query={query} docs={len(cached_documents)}",
                )
                continue
            if query_counts.get(query_key, 0) >= repeat_limit:
                self._print_closed_loop_progress(
                    stage="downstream_rag_skip",
                    round_index=round_index,
                    note=f"task={task_index} query={query} repeat_limit={repeat_limit}",
                )
                continue
            try:
                rag_client = rag_client_cls(config.rag_endpoint)
                results = rag_client.search(
                    query_text=query,
                    query_image_bytes=image_bytes,
                    query_image_filename=image_file.name if image_bytes is not None else None,
                    query_image_mime_type=image_mime_type if image_bytes is not None else None,
                    top_k=max(1, int(config.rag_top_k)),
                )
            except Exception as exc:  # noqa: BLE001
                self._print_closed_loop_progress(
                    stage="downstream_rag_error",
                    round_index=round_index,
                    note=f"task={task_index} query={query} error={exc}",
                )
                continue
            serialized_results = self._serialize_external_rag_results(query=query, results=results)
            used_queries.append(query)
            query_counts[query_key] = query_counts.get(query_key, 0) + 1
            self._append_downstream_rag_record(
                path=rag_record_path,
                image_path=image_file,
                query=query,
                source_ids=[str(getattr(item, "source_id", "") or "") for item in results],
                scores=[getattr(item, "score", None) for item in results],
                image_attached=image_bytes is not None,
            )
            self._write_downstream_rag_cache_entry(rag_cache_path, query_cache=query_cache, query=query, documents=serialized_results)
            documents.extend(serialized_results)

        if used_queries:
            self._print_closed_loop_progress(
                stage="downstream_rag_done",
                round_index=round_index,
                note=f"task={task_index} queries={len(used_queries)} docs={len(documents)}",
            )
        return {
            "queries": self._dedupe_text_list(used_queries),
            "documents": self._dedupe_dict_list(documents),
        }

    def _run_task_web_search(
        self,
        *,
        task: SpawnTask,
        run_dir: Path,
        round_index: int,
        task_index: int,
    ) -> dict[str, Any]:
        client = self._build_web_search_client()
        if client is None or not client.enabled:
            return {"queries": [], "documents": []}
        query_terms = self._resolve_web_queries(task)
        if not query_terms:
            return {"queries": [], "documents": []}

        record_path = run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "web_search_record.md"
        cache_path = self._downstream_web_cache_path(run_dir)
        repeat_limit = max(1, int(self.closed_loop_config.downstream_rag_query_repeat_limit or 2))
        query_counts = self._read_downstream_web_query_counts(run_dir)
        query_cache = self._read_downstream_web_cache(cache_path)
        top_k = max(1, int(getattr(self.slots_config, "web_search_top_k", 5) or 5))

        documents: list[dict[str, Any]] = []
        used_queries: list[str] = []
        for query in self._dedupe_text_list(query_terms):
            query_key = self._query_signature(query)
            cached_documents = self._cached_documents_for_query(query_cache, query)
            if cached_documents:
                used_queries.append(query)
                documents.extend(cached_documents)
                self._print_closed_loop_progress(
                    stage="downstream_web_cache_hit",
                    round_index=round_index,
                    note=f"task={task_index} query={query} docs={len(cached_documents)}",
                )
                continue
            if query_counts.get(query_key, 0) >= repeat_limit:
                self._print_closed_loop_progress(
                    stage="downstream_web_skip",
                    round_index=round_index,
                    note=f"task={task_index} query={query} repeat_limit={repeat_limit}",
                )
                continue
            hits = client.search(query, top_k=top_k)
            if not hits:
                self._print_closed_loop_progress(
                    stage="downstream_web_empty",
                    round_index=round_index,
                    note=f"task={task_index} query={query}",
                )
                continue
            selected_hits = self._select_web_candidates(task=task, query=query, hits=hits)
            serialized_results = self._serialize_web_documents(client=client, query=query, hits=selected_hits)
            used_queries.append(query)
            query_counts[query_key] = query_counts.get(query_key, 0) + 1
            self._append_downstream_web_record(
                path=record_path,
                query=query,
                hits=hits,
                selected_hits=selected_hits,
            )
            self._write_downstream_web_cache_entry(cache_path, query_cache=query_cache, query=query, documents=serialized_results)
            documents.extend(serialized_results)

        if used_queries:
            self._print_closed_loop_progress(
                stage="downstream_web_done",
                round_index=round_index,
                note=f"task={task_index} queries={len(used_queries)} docs={len(documents)}",
            )
        return {
            "queries": self._dedupe_text_list(used_queries),
            "documents": self._dedupe_dict_list(documents),
        }

    def _select_web_candidates(self, *, task: SpawnTask, query: str, hits: list[WebSearchHit]) -> list[WebSearchHit]:
        if not hits:
            return []
        fetch_top_n = max(1, int(getattr(self.slots_config, "web_search_fetch_top_n", 2) or 2))
        scored_hits = sorted(
            (
                (
                    self._heuristic_web_candidate_score(query=query, task=task, hit=hit),
                    hit,
                )
                for hit in hits
            ),
            key=lambda item: (-item[0], item[1].position or 0, len(item[1].url)),
        )
        heuristic_hits = [hit for _, hit in scored_hits[:fetch_top_n]]
        use_llm_rerank = bool(getattr(self.slots_config, "web_search_use_llm_rerank", True))
        if not use_llm_rerank or not self.api_client.enabled or len(scored_hits) <= 1:
            return heuristic_hits
        skip_if_confident = bool(getattr(self.slots_config, "web_search_skip_llm_if_confident", True))
        if skip_if_confident and self._web_rerank_confident(scored_hits):
            return heuristic_hits
        reranked_hits = self._rerank_web_candidates(
            query=query,
            task=task,
            hits=[hit for _, hit in scored_hits[: max(fetch_top_n + 2, 5)]],
            fetch_top_n=fetch_top_n,
        )
        return reranked_hits or heuristic_hits

    def _rerank_web_candidates(
        self,
        *,
        query: str,
        task: SpawnTask,
        hits: list[WebSearchHit],
        fetch_top_n: int,
    ) -> list[WebSearchHit]:
        if not hits:
            return []
        payload = {
            "slot_name": task.slot_name,
            "task_focus": task.prompt_focus,
            "query": query,
            "candidates": [
                {
                    "index": index,
                    "title": hit.title,
                    "url": hit.url,
                    "domain": urlparse(hit.url).netloc.lower(),
                    "snippet": hit.snippet,
                    "source": hit.source,
                }
                for index, hit in enumerate(hits, start=1)
            ],
        }
        result = self.api_client.chat(
            system_prompt="你是严格的中文网页候选重排序器，只输出 JSON。",
            user_prompt=(
                "请基于 query、任务焦点和候选网页的标题/域名/snippet，选出最值得进入正文抓取的网页。\n"
                '只输出 JSON，结构固定为 {"selected_indices":[1,2],"reason":"原因"}。\n'
                "优先选择：与 query 直接对应、标题和 snippet 明显命中、来源更可靠、较少营销/聚合痕迹的页面。\n\n"
                f"输入: {json.dumps(payload, ensure_ascii=False)}\n"
            ),
            temperature=0.0,
            image_path=None,
            model=self.slots_config.validation_model or self.slots_config.domain_model,
        )
        raw = str(result.content or "").strip()
        if not raw:
            return []
        parsed = DynamicAgentPipeline._extract_json_object(raw)
        if not parsed:
            return []
        selected = []
        for index in parsed.get("selected_indices", []):
            try:
                selected_index = int(index)
            except (TypeError, ValueError):
                continue
            if 1 <= selected_index <= len(hits):
                selected.append(hits[selected_index - 1])
        selected = self._dedupe_web_hits(selected)
        return selected[:fetch_top_n]

    @staticmethod
    def _web_rerank_confident(scored_hits: list[tuple[float, WebSearchHit]]) -> bool:
        if len(scored_hits) <= 1:
            return True
        return (scored_hits[0][0] - scored_hits[1][0]) >= 1.5

    def _heuristic_web_candidate_score(self, *, query: str, task: SpawnTask, hit: WebSearchHit) -> float:
        title = str(hit.title or "").strip().lower()
        snippet = str(hit.snippet or "").strip().lower()
        domain = urlparse(hit.url).netloc.lower()
        query_terms = [
            self._normalize_text(part)
            for part in re.split(r"[\s/|,，;；:：。.!！、()（）]+", str(query or "").strip())
            if self._normalize_text(part)
        ]
        score = 0.0
        text_blob = f"{title} {snippet}"
        normalized_blob = self._normalize_text(text_blob)
        for term in query_terms:
            if not term:
                continue
            if term in normalized_blob:
                score += 1.0
        for token in (task.slot_name, task.prompt_focus, *self._coerce_str_list(task.rag_terms), *self._coerce_str_list(task.web_queries)):
            normalized_token = self._normalize_text(token)
            if normalized_token and normalized_token in normalized_blob:
                score += 0.35
        trusted_domain_markers = ("museum", ".edu", ".gov", ".org", "ac.cn", "cnki", "artron", "dpm.org.cn")
        if any(marker in domain for marker in trusted_domain_markers):
            score += 1.0
        if hit.source in {"answer_box", "knowledge_graph"}:
            score += 0.5
        if hit.position:
            score += max(0.0, 0.6 - (float(hit.position) - 1.0) * 0.1)
        return score

    @staticmethod
    def _dedupe_web_hits(hits: list[WebSearchHit]) -> list[WebSearchHit]:
        deduped: list[WebSearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            key = str(hit.url or "").strip()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(hit)
        return deduped

    def _serialize_web_documents(
        self,
        *,
        client: SerperWebSearchClient,
        query: str,
        hits: list[WebSearchHit],
    ) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for hit in hits:
            page = client.fetch_page(hit.url)
            content = str(page.get("content", "")).strip()
            description = str(page.get("description", "") or hit.snippet).strip()
            evidence = [part for part in [hit.title, hit.snippet, content[:1500] if content else ""] if str(part).strip()]
            if not description and not evidence:
                continue
            documents.append(
                {
                    "term": query,
                    "description": (description or content[:800] or hit.snippet)[:800],
                    "text_evidence": evidence,
                    "source_id": hit.url,
                    "score": None,
                    "metadata": {
                        "title": str(page.get("title", "") or hit.title).strip(),
                        "domain": str(page.get("domain", "") or urlparse(hit.url).netloc.lower()).strip(),
                        "snippet": hit.snippet,
                        "source": hit.source,
                    },
                    "query": query,
                    "origin_stage": "downstream_web",
                }
            )
        return self._dedupe_dict_list(documents)

    def _resolve_web_queries(self, task: SpawnTask) -> list[str]:
        queries = self._normalize_downstream_web_queries(self._coerce_str_list(getattr(task, "web_queries", [])))
        if queries:
            return queries
        seeds = [task.prompt_focus, task.slot_name, *self._coerce_str_list(task.rag_terms)]
        return self._normalize_downstream_web_queries(seeds)

    @staticmethod
    def _split_spawn_tasks(tasks: list[SpawnTask]) -> tuple[list[SpawnTask], list[SpawnTask]]:
        cot_tasks: list[SpawnTask] = []
        downstream_tasks: list[SpawnTask] = []
        for task in tasks:
            target = str(getattr(task, "dispatch_target", "cot")).strip().lower()
            if target == "cot":
                cot_tasks.append(task)
            else:
                downstream_tasks.append(task)
        return cot_tasks, downstream_tasks

    def _advance_fixed_slots(
        self,
        *,
        slot_schemas: list[SlotSchema],
        result: Any,
        meta: dict[str, Any],
    ) -> tuple[list[SlotSchema], dict[str, Any], dict[str, Any]]:
        runtime_meta = json.loads(json.dumps(meta, ensure_ascii=False))
        output_map = {output.slot_name: output for output in getattr(result, "domain_outputs", [])}
        spawned_by_slot: dict[str, list[SpawnTask]] = {}
        for task in getattr(result.routing, "spawned_tasks", []):
            spawned_by_slot.setdefault(task.slot_name, []).append(task)

        updated_slots: list[SlotSchema] = []
        downstream_tasks: list[SpawnTask] = []
        progressed_slots: set[str] = set()
        closed_slots: set[str] = set()

        for slot in slot_schemas:
            if not self._is_fixed_slot(slot):
                updated_slots.append(self._clone_slot(slot))
                continue
            updated_slot, generated_task, event = self._advance_single_fixed_slot(
                slot=slot,
                output=output_map.get(slot.slot_name),
                spawned_tasks=spawned_by_slot.get(slot.slot_name, []),
                meta=runtime_meta,
            )
            updated_slots.append(updated_slot)
            if generated_task is not None:
                downstream_tasks.append(generated_task)
            if event == "progressed":
                progressed_slots.add(slot.slot_name)
                self._append_slot_progress_note(runtime_meta, slot, updated_slot, event)
            elif event == "closed":
                closed_slots.add(slot.slot_name)
                self._append_slot_progress_note(runtime_meta, slot, updated_slot, event)

        self._harvest_retained_facts(
            slot_schemas=updated_slots,
            meta=runtime_meta,
            output_map=output_map,
            source="slot_description",
        )

        return (
            updated_slots,
            runtime_meta,
            {
                "downstream_tasks": downstream_tasks,
                "progressed_slots": progressed_slots,
                "closed_slots": closed_slots,
                "suppressed_slots": progressed_slots | closed_slots,
            },
        )

    def _advance_single_fixed_slot(
        self,
        *,
        slot: SlotSchema,
        output: Any | None,
        spawned_tasks: list[SpawnTask],
        meta: dict[str, Any],
    ) -> tuple[SlotSchema, SpawnTask | None, str | None]:
        base_slot = self._clone_slot(slot)
        metadata = dict(base_slot.metadata)
        lifecycle = str(metadata.get("lifecycle", "ACTIVE")).strip().upper() or "ACTIVE"
        used_terms = self._dedupe_text_list(self._coerce_str_list(metadata.get("used_terms")))
        pending_terms = self._dedupe_text_list(self._coerce_str_list(metadata.get("pending_terms")))
        candidate_terms = self._dedupe_text_list(self._coerce_str_list(metadata.get("candidate_terms")))
        repeat_guard_signatures = self._dedupe_text_list(self._coerce_str_list(metadata.get("repeat_guard_signatures")))
        current_term = str(base_slot.slot_term or "").strip()
        current_signature = self._slot_repeat_signature(base_slot.slot_name, current_term)

        pending_terms = self._merge_slot_term_candidates(
            slot=base_slot,
            output=output,
            pending_terms=pending_terms,
            candidate_terms=candidate_terms,
            spawned_tasks=spawned_tasks,
            used_terms=used_terms,
        )
        metadata["pending_terms"] = pending_terms
        metadata["candidate_terms"] = self._dedupe_text_list(candidate_terms + pending_terms)

        if lifecycle == "CLOSED" and not current_term:
            return (
                SlotSchema(
                    slot_name=base_slot.slot_name,
                    slot_term=base_slot.slot_term,
                    description=base_slot.description,
                    specific_questions=list(base_slot.specific_questions),
                    metadata=metadata,
                    controlled_vocabulary=list(base_slot.controlled_vocabulary),
                ),
                None,
                None,
            )

        if output is None:
            if not current_term and not pending_terms:
                metadata["lifecycle"] = "CLOSED"
                metadata["lifecycle_reason"] = "当前槽位无可用 term。"
                return (
                    SlotSchema(
                        slot_name=base_slot.slot_name,
                        slot_term="",
                        description=base_slot.description,
                        specific_questions=[],
                        metadata=metadata,
                        controlled_vocabulary=[],
                    ),
                    None,
                    "closed",
                )
            return (
                SlotSchema(
                    slot_name=base_slot.slot_name,
                    slot_term=base_slot.slot_term,
                    description=base_slot.description,
                    specific_questions=list(base_slot.specific_questions),
                    metadata=metadata,
                    controlled_vocabulary=list(base_slot.controlled_vocabulary),
                ),
                    None,
                    None,
                )

        if output is not None and current_signature:
            if current_signature in repeat_guard_signatures:
                metadata["repeat_guard_signatures"] = repeat_guard_signatures
                metadata["used_terms"] = self._dedupe_text_list(used_terms + ([current_term] if current_term else []))
                metadata["lifecycle"] = "CLOSED"
                metadata["lifecycle_reason"] = f"当前槽位 term `{current_term}` 已重复出现，触发关停以避免循环。"
                return (
                    SlotSchema(
                        slot_name=base_slot.slot_name,
                        slot_term=current_term,
                        description=base_slot.description,
                        specific_questions=list(base_slot.specific_questions),
                        metadata=metadata,
                        controlled_vocabulary=list(base_slot.controlled_vocabulary),
                    ),
                    None,
                    "closed",
                )
            repeat_guard_signatures = self._dedupe_text_list(repeat_guard_signatures + [current_signature])
            metadata["repeat_guard_signatures"] = repeat_guard_signatures

        next_term = self._select_next_slot_term(
            slot=base_slot,
            current_term=current_term,
            pending_terms=pending_terms,
            used_terms=used_terms,
        )
        if next_term:
            remaining_terms = [
                term
                for term in pending_terms
                if self._normalize_text(term) != self._normalize_text(next_term)
            ]
            updated_used_terms = self._dedupe_text_list(used_terms + ([next_term] if next_term else []))
            updated_slot_terms = self._select_slot_terms_for_term(next_term, metadata)
            metadata["used_terms"] = updated_used_terms
            metadata["pending_terms"] = remaining_terms
            metadata["slot_terms"] = updated_slot_terms
            metadata["candidate_terms"] = self._dedupe_text_list(
                [term for term in metadata.get("candidate_terms", []) if self._normalize_text(str(term)) != self._normalize_text(next_term)]
            )
            metadata["lifecycle"] = "ACTIVE"
            metadata["lifecycle_reason"] = f"当前 term 已分析完成，推进到 `{next_term}`。"
            questions = self._build_slot_questions(
                slot_name=base_slot.slot_name,
                term=next_term,
                slot_terms=updated_slot_terms,
                output=output,
                spawned_tasks=spawned_tasks,
            )
            description = self._build_slot_description(base_slot, next_term, output)
            updated_slot = SlotSchema(
                slot_name=base_slot.slot_name,
                slot_term=next_term,
                description=description,
                specific_questions=questions,
                metadata=metadata,
                controlled_vocabulary=extract_controlled_vocabulary(next_term, description, updated_slot_terms),
            )
            downstream_task = None
            if bool(getattr(self.slots_config, "enable_rag_verification", True)):
                downstream_task = self._build_progression_downstream_task(updated_slot, meta)
            return updated_slot, downstream_task, "progressed"

        metadata["used_terms"] = self._dedupe_text_list(used_terms + ([current_term] if current_term else []))
        if current_term:
            metadata["lifecycle"] = "CLOSED"
            metadata["lifecycle_reason"] = "当前槽位没有更多可推进 term，本轮分析后结束。"
        else:
            metadata["lifecycle"] = "CLOSED"
            metadata["lifecycle_reason"] = "当前槽位无可用 term。"
        return (
            SlotSchema(
                slot_name=base_slot.slot_name,
                slot_term=current_term,
                description=base_slot.description,
                specific_questions=list(base_slot.specific_questions),
                metadata=metadata,
                controlled_vocabulary=list(base_slot.controlled_vocabulary),
            ),
            None,
            "closed",
        )

    def _merge_slot_term_candidates(
        self,
        *,
        slot: SlotSchema,
        output: Any | None,
        pending_terms: list[str],
        candidate_terms: list[str],
        spawned_tasks: list[SpawnTask],
        used_terms: list[str],
    ) -> list[str]:
        current_term = str(slot.slot_term or "").strip()
        combined = list(pending_terms) + list(candidate_terms)
        if output is not None:
            combined.extend(self._coerce_str_list(getattr(output, "retrieval_gain_terms", [])))
            combined.extend(
                str(item.term).strip()
                for item in getattr(output, "domain_decoding", [])
                if str(getattr(item, "term", "")).strip()
            )
        for task in spawned_tasks:
            combined.extend(self._coerce_str_list(getattr(task, "rag_terms", [])))
        results: list[str] = []
        for term in combined:
            normalized = self._normalize_text(term)
            if not normalized:
                continue
            if not self._term_matches_fixed_slot_family(slot, term):
                continue
            if normalized == self._normalize_text(current_term):
                continue
            if normalized in {self._normalize_text(item) for item in used_terms if item.strip()}:
                continue
            if normalized == self._normalize_text(slot.slot_name):
                continue
            if len(normalized) <= 1:
                continue
            if term not in results:
                results.append(term.strip())
        return results

    def _select_next_slot_term(
        self,
        *,
        slot: SlotSchema,
        current_term: str,
        pending_terms: list[str],
        used_terms: list[str],
    ) -> str:
        normalized_used = {self._normalize_text(term) for term in used_terms if term.strip()}
        normalized_current = self._normalize_text(current_term)
        for term in pending_terms:
            normalized = self._normalize_text(term)
            if not normalized or normalized == normalized_current or normalized in normalized_used:
                continue
            if not self._term_matches_fixed_slot_family(slot, term):
                continue
            return term.strip()
        return ""

    def _term_matches_fixed_slot_family(self, slot: SlotSchema, term: str) -> bool:
        if not self._is_fixed_slot(slot):
            return True
        text = str(term or "").strip()
        if not text:
            return False
        if slot.slot_name == "作者时代流派":
            return self._term_matches_author_family(text)
        if slot.slot_name == "尺寸规格/材质形制/收藏地":
            return self._term_matches_material_family(text)
        return True

    def _term_matches_author_family(self, term: str) -> bool:
        text = str(term or "").strip()
        if not text:
            return False
        generic_object_markers = (
            "侍者",
            "主尊",
            "供品",
            "竹林",
            "祥云",
            "圆光",
            "衣纹",
            "袈裟",
            "椅子",
            "背景",
            "画面",
        )
        deny_markers = (
            "皴",
            "描",
            "设色",
            "墨法",
            "墨色",
            "笔法",
            "用笔",
            "笔墨",
            "构图",
            "布局",
            "空间",
            "材质",
            "绢本",
            "纸本",
            "册页",
            "手卷",
            "立轴",
            "博物馆",
            "馆藏",
            "收藏",
            "尺寸",
            "厘米",
            "题跋",
            "印章",
        )
        dynasty_markers = (
            "先秦",
            "汉",
            "魏晋",
            "隋",
            "唐",
            "五代",
            "北宋",
            "南宋",
            "宋",
            "辽",
            "金",
            "元",
            "明",
            "清",
            "民国",
            "近代",
            "现代",
            "当代",
        )
        family_markers = (
            "画家",
            "作者",
            "流派",
            "画派",
            "院体",
            "文人画",
            "人物画",
            "山水画",
            "花鸟画",
            "道释画",
            "罗汉画",
            "佛教人物画",
            "宫廷画",
            "风格",
        )
        if any(marker in text for marker in generic_object_markers):
            return False
        if any(marker in text for marker in dynasty_markers):
            return not any(marker in text for marker in deny_markers)
        if any(marker in text for marker in family_markers):
            return not any(marker in text for marker in deny_markers)
        if any(marker in text for marker in deny_markers):
            return False
        return self._looks_like_artist_name(text)

    def _term_matches_material_family(self, term: str) -> bool:
        text = str(term or "").strip()
        if not text:
            return False
        if text.endswith("画") and not re.search(r"(?:绢本|纸本).{0,6}(?:设色|墨笔)", text):
            return False
        if re.search(r"(?:纵|高)\s*\d+(?:\.\d+)?(?:\s*(?:厘米|cm|毫米|mm))?\s*[×xX＊*]\s*(?:横|宽)?\s*\d+(?:\.\d+)?\s*(?:厘米|cm|毫米|mm)", text, flags=re.IGNORECASE):
            return True
        if re.search(r"(?:绢本|纸本).{0,6}(?:设色|墨笔)", text):
            return True
        if re.search(r"(?:现藏于|馆藏于|馆藏|藏于|收藏于).{0,24}(?:博物馆|美术馆|纪念馆|博物院|文物院|故宫)", text):
            return True
        allow_markers = (
            "绢本",
            "纸本",
            "墨笔",
            "册页",
            "手卷",
            "立轴",
            "长卷",
            "卷轴",
            "条屏",
            "屏风",
            "镜心",
            "装裱",
            "形制",
            "材质",
            "尺寸",
            "厘米",
            "cm",
            "毫米",
            "mm",
            "纵",
            "横",
            "博物馆",
            "美术馆",
            "馆藏",
            "收藏",
            "故宫",
            "博物院",
        )
        deny_markers = (
            "北宋",
            "南宋",
            "五代",
            "元",
            "明",
            "清",
            "唐",
            "人物画",
            "山水画",
            "花鸟画",
            "道释画",
            "流派",
            "画派",
            "皴",
            "描",
            "用笔",
            "笔法",
            "笔墨",
            "构图",
            "布局",
            "空间",
            "意境",
            "象征",
            "题跋",
            "印章",
        )
        if any(marker in text for marker in deny_markers):
            return False
        return any(marker in text for marker in allow_markers)

    @staticmethod
    def _looks_like_artist_name(term: str) -> bool:
        compact = re.sub(r"[\s·•・．.]", "", str(term or "").strip())
        if not re.fullmatch(r"[\u4e00-\u9fff]{2,5}", compact):
            return False
        generic_terms = {
            "作者",
            "画家",
            "人物",
            "侍者",
            "主尊",
            "罗汉",
            "竹林",
            "祥云",
            "绢本",
            "纸本",
            "构图",
            "布局",
            "空间",
            "博物馆",
        }
        return compact not in generic_terms

    def _build_slot_questions(
        self,
        *,
        slot_name: str,
        term: str,
        slot_terms: list[str] | None,
        output: Any | None,
        spawned_tasks: list[SpawnTask],
    ) -> list[str]:
        questions: list[str] = []
        slot_terms = self._dedupe_text_list(slot_terms or ([term] if term else []))
        term_phrase = self._slot_terms_phrase(slot_terms or ([term] if term else []), fallback=term or slot_name)
        for task in spawned_tasks:
            focus = str(getattr(task, "prompt_focus", "")).strip()
            if focus and focus not in questions:
                questions.append(focus)
        if output is not None:
            for item in getattr(output, "question_coverage", []):
                question = str(getattr(item, "question", "")).strip()
                if question and question not in questions:
                    questions.append(question)
        if slot_name == "画作背景":
            templates = [
                f"{term_phrase} 与当前作品主干、卷次或人物身份之间是什么关系？",
                f"{term_phrase} 如何帮助确认当前作品的具体对象与背景信息？",
            ]
        elif slot_name == "作者时代流派":
            templates = [
                f"{term_phrase} 对作者、时代或流派判断提供了什么依据？",
                f"{term_phrase} 如何与当前作品的风格线索互相印证？",
            ]
        elif slot_name == "墨法设色技法":
            templates = [
                f"{term_phrase} 是否真实体现在当前作品中？分别对应哪些画面证据？",
                f"{term_phrase} 如何共同影响当前作品的视觉效果与审美表达？",
            ]
        elif slot_name == "题跋/印章/用笔":
            templates = [
                f"{term_phrase} 在当前作品中是否清晰可辨？分别对应哪些题跋、印章或笔触证据？",
                f"{term_phrase} 如何帮助判断当前作品的作者线索、真伪依据或笔墨特征？",
            ]
        elif slot_name == "构图/空间/布局":
            templates = [
                f"{term_phrase} 如何组织当前作品的空间层次、视觉重心或观看动线？",
                f"{term_phrase} 对当前作品的布局经营和整体节奏产生了什么作用？",
            ]
        elif slot_name == "尺寸规格/材质形制/收藏地":
            templates = [
                f"{term_phrase} 能为当前作品的尺寸、材质、形制或收藏信息提供什么线索？",
                f"{term_phrase} 如何影响当前作品的观看方式、保存状态或设色质感？",
            ]
        elif slot_name == "意境/题材/象征":
            templates = [
                f"{term_phrase} 如何概括当前作品的题材、意境或象征含义？",
                f"{term_phrase} 与当前作品的物象安排和文化语义如何互相印证？",
            ]
        else:
            templates = [
                f"{term_phrase} 在当前作品中是否可见或可证？",
                f"{term_phrase} 如何参与当前作品的审美语言或题旨表达？",
            ]
        for item in templates:
            if item not in questions:
                questions.append(item)
        return questions[:3]

    @staticmethod
    def _build_slot_description(slot: SlotSchema, next_term: str, output: Any | None) -> str:
        output_note = ""
        if output is not None:
            unresolved = [str(item).strip() for item in getattr(output, "unresolved_points", []) if str(item).strip()]
            if unresolved:
                output_note = f" 当前轮仍需结合 `{next_term}` 补充：{unresolved[0]}。"
        base = str(slot.description).strip()
        summary = f"当前推进 term：{next_term}。"
        if summary in base:
            return f"{base}{output_note}".strip()
        if base:
            return f"{base} {summary}{output_note}".strip()
        return f"{summary}{output_note}".strip()

    def _build_progression_downstream_task(self, slot: SlotSchema, meta: dict[str, Any]) -> SpawnTask:
        rag_terms = self._slot_terms(slot)[:3]
        rag_terms = self._dedupe_text_list(rag_terms)
        focus = (
            slot.specific_questions[0]
            if slot.specific_questions
            else f"围绕 {self._slot_terms_phrase(self._slot_terms(slot), fallback=slot.slot_term or slot.slot_name)} 补充证据。"
        )
        return SpawnTask(
            slot_name=slot.slot_name,
            reason="slot_term_progression",
            prompt_focus=focus,
            rag_terms=rag_terms,
            retrieval_mode="rag",
            priority=5,
            dispatch_target="downstream",
        )

    @staticmethod
    def _is_fixed_slot(slot: SlotSchema) -> bool:
        mode = str(slot.metadata.get("slot_mode", "")).strip().lower()
        return mode in {"progressive", "enumerative"}

    @staticmethod
    def _filter_suppressed_tasks(tasks: list[SpawnTask], suppressed_slots: set[str]) -> list[SpawnTask]:
        if not suppressed_slots:
            return list(tasks)
        return [task for task in tasks if task.slot_name not in suppressed_slots]

    def _compress_cot_tasks(self, tasks: list[SpawnTask]) -> list[SpawnTask]:
        grouped: dict[str, SpawnTask] = {}
        for task in tasks:
            existing = grouped.get(task.slot_name)
            if existing is None:
                grouped[task.slot_name] = task
                continue
            preferred = existing
            secondary = task
            if task.priority > existing.priority or (
                task.priority == existing.priority and len(task.prompt_focus) > len(existing.prompt_focus)
            ):
                preferred = task
                secondary = existing
            grouped[task.slot_name] = SpawnTask(
                slot_name=preferred.slot_name,
                reason=preferred.reason,
                prompt_focus=preferred.prompt_focus,
                rag_terms=self._dedupe_text_list(preferred.rag_terms + secondary.rag_terms),
                retrieval_mode=DynamicAgentPipeline._merge_retrieval_modes(preferred.retrieval_mode, secondary.retrieval_mode),
                web_queries=self._dedupe_text_list(preferred.web_queries + secondary.web_queries),
                retrieval_reason=preferred.retrieval_reason or secondary.retrieval_reason,
                priority=max(preferred.priority, secondary.priority),
                dispatch_target=preferred.dispatch_target,
                requested_slot_name=preferred.requested_slot_name or secondary.requested_slot_name,
                source_thread_id=preferred.source_thread_id or secondary.source_thread_id,
            )
        return list(grouped.values())

    def _dedupe_spawn_tasks(self, tasks: list[SpawnTask]) -> list[SpawnTask]:
        deduped: list[SpawnTask] = []
        for task in tasks:
            if not str(task.prompt_focus or "").strip():
                continue
            duplicate_index = next(
                (
                    index
                    for index, existing in enumerate(deduped)
                    if self._tasks_share_topic(existing, task)
                ),
                -1,
            )
            if duplicate_index >= 0:
                deduped[duplicate_index] = self._merge_spawn_tasks(deduped[duplicate_index], task)
                continue
            deduped.append(task)
        return deduped

    def _has_unseen_followups(self, tasks: list[SpawnTask], seen_tasks: list[SpawnTask]) -> bool:
        if not tasks:
            return False
        if not seen_tasks:
            return True
        for task in tasks:
            if any(self._tasks_share_topic(task, seen_task) for seen_task in seen_tasks):
                continue
            return True
        return False

    @staticmethod
    def _tasks_share_topic(left: SpawnTask, right: SpawnTask) -> bool:
        if ClosedLoopCoordinator._normalize_text(getattr(left, "dispatch_target", "cot")) != ClosedLoopCoordinator._normalize_text(
            getattr(right, "dispatch_target", "cot")
        ):
            return False
        similarity = DynamicAgentPipeline._text_similarity(left.prompt_focus, right.prompt_focus)
        if similarity >= 0.82:
            return True
        shared_terms = DynamicAgentPipeline._shared_task_terms(left, right)
        return len(shared_terms) >= 2 or any(len(term) >= 4 for term in shared_terms)

    @staticmethod
    def _merge_spawn_tasks(left: SpawnTask, right: SpawnTask) -> SpawnTask:
        return DynamicAgentPipeline._merge_spawn_task(left, right)

    def _spawn_task_signature(self, task: SpawnTask) -> str:
        return json.dumps(
            {
                "slot_name": task.slot_name,
                "reason": task.reason,
                "prompt_focus": task.prompt_focus,
                "rag_terms": sorted(self._dedupe_text_list(task.rag_terms)),
                "retrieval_mode": DynamicAgentPipeline._normalize_retrieval_mode(task.retrieval_mode),
                "web_queries": sorted(self._dedupe_text_list(task.web_queries)),
                "dispatch_target": task.dispatch_target,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _append_slot_progress_note(self, meta: dict[str, Any], previous: SlotSchema, current: SlotSchema, event: str) -> None:
        if event == "progressed":
            note = f"slot `{previous.slot_name}`: `{previous.slot_term}` -> `{current.slot_term}`"
        else:
            note = f"slot `{previous.slot_name}` closed at `{previous.slot_term or previous.slot_name}`"
        notes = meta.setdefault("closed_loop_notes", [])
        if note not in notes:
            notes.append(note)
        self._append_dialogue_turn(meta, note)

    def _harvest_retained_facts(
        self,
        *,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
        output_map: dict[str, Any] | None = None,
        source: str = "slot_description",
    ) -> int:
        meta.setdefault("retained_facts", [])
        added = 0
        output_map = output_map or {}
        for slot in slot_schemas:
            consumed_texts = self._slot_consumed_texts(output_map.get(slot.slot_name))
            for fact in self._slot_retained_fact_candidates(slot):
                if consumed_texts and self._retained_fact_is_consumed(fact, consumed_texts):
                    continue
                if self._append_retained_fact(meta, slot=slot, fact=fact, source=source):
                    added += 1
        return added

    def _slot_retained_fact_candidates(self, slot: SlotSchema) -> list[str]:
        description = " ".join(str(slot.description or "").strip().split())
        if not description:
            return []
        candidates: list[str] = []
        for chunk in re.split(r"[。！？!?\n]+", description):
            candidate = chunk.strip(" ；;，,")
            if not candidate:
                continue
            candidate = re.sub(r"^(?:补充证据|文字证据|背景线索|辅助信息|补充背景)[:：]\s*", "", candidate)
            if not candidate:
                continue
            if re.match(r"^当前推进\s*term[:：]", candidate, flags=re.IGNORECASE):
                continue
            if candidate.startswith("当前轮仍需结合"):
                continue
            if candidate.startswith(("如需进一步", "若需进一步", "还需查阅", "仍需查阅", "需更高分辨率")):
                continue
            if any(marker in candidate for marker in ("暂无法确认", "无法确认", "待进一步", "是否明确", "是否可见", "尚需结合")):
                continue
            normalized = self._normalize_text(candidate)
            if not normalized or len(candidate) < 6:
                continue
            if normalized in {
                self._normalize_text(slot.slot_name),
                self._normalize_text(slot.slot_term),
            }:
                continue
            if candidate not in candidates:
                candidates.append(candidate)
        return candidates[:8]

    def _slot_consumed_texts(self, output: Any | None) -> list[str]:
        if output is None:
            return []
        texts: list[str] = []
        texts.extend(
            str(item.support).strip()
            for item in getattr(output, "question_coverage", [])
            if str(getattr(item, "support", "")).strip()
        )
        texts.extend(
            " ".join(
                part
                for part in [str(item.observation).strip(), str(item.evidence).strip()]
                if part
            )
            for item in getattr(output, "visual_anchoring", [])
            if str(getattr(item, "observation", "")).strip() or str(getattr(item, "evidence", "")).strip()
        )
        texts.extend(
            " ".join(
                part
                for part in [
                    str(item.term).strip(),
                    str(item.explanation).strip(),
                    str(item.reason).strip(),
                ]
                if part
            )
            for item in getattr(output, "domain_decoding", [])
            if str(getattr(item, "term", "")).strip()
            or str(getattr(item, "explanation", "")).strip()
            or str(getattr(item, "reason", "")).strip()
        )
        texts.extend(
            " ".join(
                part
                for part in [str(item.insight).strip(), str(item.basis).strip()]
                if part
            )
            for item in getattr(output, "cultural_mapping", [])
            if str(getattr(item, "insight", "")).strip() or str(getattr(item, "basis", "")).strip()
        )
        return self._dedupe_text_list(texts)

    def _retained_fact_is_consumed(self, fact: str, consumed_texts: list[str]) -> bool:
        normalized_fact = self._normalize_text(fact)
        if not normalized_fact:
            return False
        for item in consumed_texts:
            normalized_item = self._normalize_text(item)
            if not normalized_item:
                continue
            if normalized_fact in normalized_item or normalized_item in normalized_fact:
                return True
            if len(normalized_fact) >= 10 and len(normalized_item) >= 10:
                if DynamicAgentPipeline._text_similarity(fact, item) >= 0.86:
                    return True
        return False

    def _append_retained_fact(
        self,
        meta: dict[str, Any],
        *,
        slot: SlotSchema,
        fact: str,
        source: str,
    ) -> bool:
        return self._append_retained_fact_entry(
            meta,
            slot_name=slot.slot_name,
            slot_term=slot.slot_term,
            fact=fact,
            source=source,
        )

    def _append_retained_fact_entry(
        self,
        meta: dict[str, Any],
        *,
        slot_name: str,
        slot_term: str,
        fact: str,
        source: str,
    ) -> bool:
        fact_text = " ".join(str(fact or "").strip().split())
        if not fact_text:
            return False
        meta.setdefault("retained_facts", [])
        marker = f"{self._normalize_text(slot_name)}::{self._normalize_text(fact_text)}"
        for item in self._coerce_list(meta.get("retained_facts")):
            if not isinstance(item, dict):
                continue
            existing_marker = (
                f"{self._normalize_text(str(item.get('slot_name', '')).strip())}"
                f"::{self._normalize_text(str(item.get('fact', '')).strip())}"
            )
            if marker == existing_marker:
                return False
        meta["retained_facts"].append(
            {
                "slot_name": str(slot_name or "").strip(),
                "slot_term": str(slot_term or "").strip(),
                "fact": fact_text,
                "source": source or "slot_description",
            }
        )
        return True

    def _select_retained_facts_for_task(
        self,
        *,
        task: SpawnTask,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        slot = next((item for item in slot_schemas if item.slot_name == task.slot_name), None)
        focus_terms = self._dedupe_text_list(
            [
                task.slot_name,
                str(getattr(task, "slot_term", "") or "").strip(),
                task.prompt_focus,
                *self._coerce_str_list(task.rag_terms),
                *self._coerce_str_list(getattr(task, "web_queries", [])),
                *(slot.controlled_vocabulary if slot is not None else []),
            ]
        )
        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reversed(self._coerce_list(meta.get("retained_facts"))):
            if not isinstance(item, dict):
                continue
            fact = str(item.get("fact", "")).strip()
            if not fact:
                continue
            blob = " ".join(
                [
                    str(item.get("slot_name", "")).strip(),
                    str(item.get("slot_term", "")).strip(),
                    fact,
                ]
            )
            normalized_blob = self._normalize_text(blob)
            if not normalized_blob:
                continue
            matched = str(item.get("slot_name", "")).strip() == task.slot_name or any(
                self._normalize_text(token) and self._normalize_text(token) in normalized_blob
                for token in focus_terms
            )
            if not matched:
                continue
            key = f"{self._normalize_text(str(item.get('slot_name', '')).strip())}::{self._normalize_text(fact)}"
            if key in seen:
                continue
            seen.add(key)
            selected.append(
                {
                    "slot_name": str(item.get("slot_name", "")).strip(),
                    "slot_term": str(item.get("slot_term", "")).strip(),
                    "fact": fact,
                    "source": str(item.get("source", "")).strip() or "slot_description",
                }
            )
            if len(selected) >= 6:
                break
        selected.reverse()
        return selected

    def _apply_downstream_response(
        self,
        *,
        response: dict[str, Any],
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
        task: SpawnTask,
    ) -> tuple[list[SlotSchema], dict[str, Any], dict[str, Any]]:
        slots = [self._clone_slot(slot) for slot in slot_schemas]
        runtime_meta = json.loads(json.dumps(meta, ensure_ascii=False))
        runtime_meta.setdefault("post_rag_text_extraction", [])
        runtime_meta.setdefault("rag_cache", [])
        self._seed_rag_cache_from_post_rag(runtime_meta)
        runtime_meta.setdefault("ontology_updates", [])
        runtime_meta.setdefault("downstream_updates", [])
        runtime_meta.setdefault("closed_loop_notes", [])
        runtime_meta.setdefault("dialogue_turns", [])
        runtime_meta.setdefault("round_memories", [])
        runtime_meta.setdefault("retained_facts", [])

        merged_count = 0
        new_slot_count = 0
        text_update_count = 0
        ontology_count = 0
        note_count = 0

        for item in self._coerce_list(response.get("merge_candidates")):
            if not isinstance(item, dict):
                continue
            target_key = str(item.get("target_slot_term") or item.get("target_slot_name") or task.slot_name).strip()
            index = self._match_slot_index(slots, target_key)
            if index < 0:
                continue
            slot_context = slots[index]
            filtered_questions = self._filter_questions_by_theme(
                self._coerce_str_list(item.get("additional_questions") or item.get("specific_questions") or []),
                meta=runtime_meta,
                slot=slot_context,
            )
            update_payload = dict(item)
            update_payload["additional_questions"] = filtered_questions
            updated_slot, changed = self._merge_slot_update(slots[index], update_payload)
            if changed:
                slots[index] = updated_slot
                merged_count += 1

        for item in self._coerce_list(response.get("new_slots")):
            slot = self._coerce_slot(item)
            if slot is None:
                continue
            slot = SlotSchema(
                slot_name=slot.slot_name,
                slot_term=slot.slot_term,
                description=slot.description,
                specific_questions=self._filter_questions_by_theme(slot.specific_questions, meta=runtime_meta, slot=slot),
                metadata=slot.metadata,
                controlled_vocabulary=slot.controlled_vocabulary,
            )
            index = self._match_slot_index(slots, slot.slot_term or slot.slot_name)
            if index >= 0:
                updated_slot, changed = self._merge_slot_update(
                    slots[index],
                    {
                        "description_append": slot.description,
                        "additional_questions": slot.specific_questions,
                    },
                )
                if changed:
                    slots[index] = updated_slot
                    merged_count += 1
                continue
            slots.append(slot)
            new_slot_count += 1

        for item in self._coerce_list(response.get("text_evidence_updates")):
            if not isinstance(item, dict):
                continue
            candidate = {
                "term": str(item.get("term", "")).strip(),
                "description": str(item.get("description", "")).strip(),
                "text_evidence": self._coerce_str_list(item.get("text_evidence")),
            }
            if not candidate["term"] and not candidate["description"]:
                continue
            if self._append_unique_dict(runtime_meta["post_rag_text_extraction"], candidate):
                text_update_count += 1
            self._append_rag_cache_entry(
                runtime_meta,
                {
                    **candidate,
                    "origin_stage": "downstream_response",
                    "source_slot": task.slot_name,
                    "query": "",
                },
            )

        for item in self._coerce_str_list(response.get("ontology_updates")):
            if item not in runtime_meta["ontology_updates"]:
                runtime_meta["ontology_updates"].append(item)
                ontology_count += 1

        downstream_summary = {
            "slot_name": task.slot_name,
            "reason": task.reason,
            "focus": task.prompt_focus,
            "status": str(response.get("status", "")).strip() or "unknown",
            "search_queries": self._coerce_list(response.get("search_queries")),
            "resolved_questions": self._coerce_str_list(response.get("resolved_questions")),
            "open_questions": self._coerce_str_list(response.get("open_questions")),
            "notes": self._coerce_str_list(response.get("notes")),
        }
        if self._append_unique_dict(runtime_meta["downstream_updates"], downstream_summary):
            note_count += 1

        for note in downstream_summary["notes"]:
            if note not in runtime_meta["closed_loop_notes"]:
                runtime_meta["closed_loop_notes"].append(note)
                note_count += 1
        if downstream_summary["notes"]:
            self._append_dialogue_turn(
                runtime_meta,
                f"downstream {task.slot_name}/{task.reason} -> {' | '.join(downstream_summary['notes'])}",
            )

        retained_fact_count = self._harvest_retained_facts(
            slot_schemas=slots,
            meta=runtime_meta,
            output_map=None,
            source="slot_description",
        )

        changed = bool(merged_count or new_slot_count or text_update_count or ontology_count or note_count)
        return (
            slots,
            runtime_meta,
            {
                "changed": changed,
                "merged_slots": merged_count,
                "new_slots": new_slot_count,
                "text_updates": text_update_count,
                "ontology_updates": ontology_count,
                "notes_added": note_count,
                "retained_facts_added": retained_fact_count,
            },
        )

    def _filter_questions_by_theme(
        self,
        questions: list[str],
        *,
        meta: dict[str, Any],
        slot: SlotSchema,
    ) -> list[str]:
        profile = meta.get("domain_profile", {}) if isinstance(meta.get("domain_profile"), dict) else {}
        theme_text = " ".join(
            [
                str(profile.get("name", "")).strip(),
                str(profile.get("category", "")).strip(),
                str(profile.get("subject", "")).strip(),
                str(profile.get("scene_summary", "")).strip(),
                " ".join(str(item).strip() for item in profile.get("knowledge_background", []) if str(item).strip()),
                slot.slot_name,
                slot.slot_term,
                slot.description,
            ]
        )
        return [question for question in questions if not self._question_conflicts_with_theme(question, theme_text)]

    @staticmethod
    def _question_conflicts_with_theme(question: str, theme_text: str) -> bool:
        normalized_theme = ClosedLoopCoordinator._normalize_text(theme_text)
        if not normalized_theme:
            return False

        suspicious_terms: list[str] = []
        suspicious_terms.extend(re.findall(r"《([^》]{2,20})》", question))
        suspicious_terms.extend(re.findall(r"[（(](?:如|例如|比如)?([^）)]{2,20})[）)]", question))
        suspicious_terms.extend(re.findall(r"(?:^|[，。；：（(])(?:例如|比如|如)(?!何)([^，。；！？、]{2,20})", question))
        if "花卉" in question:
            suspicious_terms.append("花卉")

        deduped_terms = ClosedLoopCoordinator._dedupe_text_list([str(item).strip() for item in suspicious_terms if str(item).strip()])
        for term in deduped_terms:
            if ClosedLoopCoordinator._normalize_text(term) in normalized_theme:
                continue
            return True
        return False

    @staticmethod
    def _write_slots_jsonl(slot_schemas: list[SlotSchema], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for slot in slot_schemas:
            payload = {
                "slot_name": slot.slot_name,
                "slot_term": slot.slot_term,
                "description": slot.description,
                "specific_questions": slot.specific_questions,
                "metadata": slot.metadata,
            }
            lines.append(json.dumps(payload, ensure_ascii=False))
        path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

    @staticmethod
    def _serialize_slot(slot: SlotSchema) -> dict[str, Any]:
        return {
            "slot_name": slot.slot_name,
            "slot_term": slot.slot_term,
            "description": slot.description,
            "specific_questions": slot.specific_questions,
            "metadata": slot.metadata,
            "controlled_vocabulary": slot.controlled_vocabulary,
        }

    @staticmethod
    def _clone_slot(slot: SlotSchema) -> SlotSchema:
        return SlotSchema(
            slot_name=slot.slot_name,
            slot_term=slot.slot_term,
            description=slot.description,
            specific_questions=list(slot.specific_questions),
            metadata=dict(slot.metadata),
            controlled_vocabulary=list(slot.controlled_vocabulary),
        )

    def _merge_slot_update(self, slot: SlotSchema, update: dict[str, Any]) -> tuple[SlotSchema, bool]:
        description_append = str(
            update.get("description_append") or update.get("new_evidence_summary") or update.get("description") or ""
        ).strip()
        additional_questions = self._coerce_str_list(
            update.get("additional_questions") or update.get("specific_questions") or []
        )

        description = slot.description
        changed = False
        if description_append and self._normalize_text(description_append) not in self._normalize_text(description):
            description = f"{description} 补充证据：{description_append}".strip() if description else description_append
            changed = True

        specific_questions = self._dedupe_text_list(slot.specific_questions + additional_questions)
        if len(specific_questions) != len(slot.specific_questions):
            changed = True

        if not changed:
            return slot, False
        metadata = dict(slot.metadata)
        metadata["lifecycle"] = "ACTIVE"
        return (
            SlotSchema(
                slot_name=slot.slot_name,
                slot_term=slot.slot_term,
                description=description,
                specific_questions=specific_questions,
                metadata=metadata,
                controlled_vocabulary=extract_controlled_vocabulary(
                    slot.slot_term,
                    description,
                    extract_slot_terms(slot.slot_term, metadata),
                ),
            ),
            True,
        )

    def _coerce_slot(self, item: Any) -> SlotSchema | None:
        if not isinstance(item, dict):
            return None
        slot_name = str(item.get("slot_name", "")).strip()
        slot_term = str(item.get("slot_term", "")).strip()
        description = str(item.get("description", "")).strip()
        if not slot_name and not slot_term:
            return None
        metadata = item.get("metadata") if isinstance(item.get("metadata"), dict) else {}
        metadata = dict(metadata)
        metadata.setdefault("lifecycle", "ACTIVE")
        metadata["slot_terms"] = extract_slot_terms(slot_term or slot_name, metadata)
        return SlotSchema(
            slot_name=slot_name or slot_term,
            slot_term=slot_term or slot_name,
            description=description,
            specific_questions=self._coerce_str_list(item.get("specific_questions")),
            metadata=metadata,
            controlled_vocabulary=extract_controlled_vocabulary(
                slot_term or slot_name,
                description,
                metadata.get("slot_terms", []),
            ),
        )

    def _match_slot_index(self, slots: list[SlotSchema], target: str) -> int:
        normalized_target = self._normalize_text(target)
        if not normalized_target:
            return -1
        for index, slot in enumerate(slots):
            keys = {
                self._normalize_text(slot.slot_name),
                self._normalize_text(slot.slot_term),
            }
            keys.update(self._normalize_text(term) for term in self._slot_terms(slot))
            if normalized_target in keys:
                return index
        return -1

    @staticmethod
    def _task_goal(task: SpawnTask) -> str:
        target = str(getattr(task, "dispatch_target", "cot")).strip().lower()
        if target == "downstream":
            return f"围绕候选槽位“{task.slot_name}”进行 discovery，判断是否值得检索、合并到已有槽位或新建槽位，重点问题：{task.prompt_focus}"
        return f"围绕槽位“{task.slot_name}”处理 {task.reason}，重点回答：{task.prompt_focus}"

    @staticmethod
    def _apply_slot_lifecycle_reviews(slot_schemas: list[SlotSchema], validation: Any) -> list[SlotSchema]:
        review_map = {
            str(item.get("slot_name", "")).strip(): {
                "status": str(item.get("status", "")).strip().upper() or "ACTIVE",
                "reason": str(item.get("reason", "")).strip(),
            }
            for item in getattr(validation, "slot_lifecycle_reviews", [])
            if isinstance(item, dict) and str(item.get("slot_name", "")).strip()
        }
        if not review_map:
            return [ClosedLoopCoordinator._clone_slot(slot) for slot in slot_schemas]

        updated: list[SlotSchema] = []
        for slot in slot_schemas:
            review = review_map.get(slot.slot_name)
            if review is None:
                updated.append(ClosedLoopCoordinator._clone_slot(slot))
                continue
            metadata = dict(slot.metadata)
            metadata["lifecycle"] = review["status"]
            if review["reason"]:
                metadata["lifecycle_reason"] = review["reason"]
            updated.append(
                SlotSchema(
                    slot_name=slot.slot_name,
                    slot_term=slot.slot_term,
                    description=slot.description,
                    specific_questions=list(slot.specific_questions),
                    metadata=metadata,
                    controlled_vocabulary=list(slot.controlled_vocabulary),
                )
            )
        return updated

    @staticmethod
    def _append_unique_dict(items: list[dict[str, Any]], candidate: dict[str, Any]) -> bool:
        marker = json.dumps(candidate, ensure_ascii=False, sort_keys=True)
        existing = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in items if isinstance(item, dict)}
        if marker in existing:
            return False
        items.append(candidate)
        return True

    @staticmethod
    def _dedupe_dict_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in items:
            if not isinstance(item, dict):
                continue
            marker = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if marker in seen:
                continue
            seen.add(marker)
            results.append(item)
        return results

    @staticmethod
    def _append_dialogue_turn(meta: dict[str, Any], content: str) -> None:
        history = meta.setdefault("dialogue_turns", [])
        if not isinstance(history, list):
            meta["dialogue_turns"] = []
            history = meta["dialogue_turns"]
        entry = {"role": "system", "content": content}
        if entry not in history:
            history.append(entry)

    @staticmethod
    def _append_round_memory(meta: dict[str, Any], memory: dict[str, Any]) -> None:
        memories = meta.setdefault("round_memories", [])
        if not isinstance(memories, list):
            meta["round_memories"] = []
            memories = meta["round_memories"]
        marker = json.dumps(memory, ensure_ascii=False, sort_keys=True)
        existing = {json.dumps(item, ensure_ascii=False, sort_keys=True) for item in memories if isinstance(item, dict)}
        if marker in existing:
            return
        memories.append(memory)

    def _build_closed_loop_final_appreciation(
        self,
        *,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
        latest_result: Any | None,
    ) -> str:
        if latest_result is None:
            return ""
        cumulative_outputs = self._collect_cumulative_outputs(
            slot_schemas=slot_schemas,
            meta=meta,
            latest_result=latest_result,
        )
        cumulative_validation = self._collect_cumulative_validation(
            outputs=cumulative_outputs,
            meta=meta,
            latest_result=latest_result,
        )
        cumulative_dialogue_state = self._collect_cumulative_dialogue_state(
            outputs=cumulative_outputs,
            validation=cumulative_validation,
            meta=meta,
            latest_result=latest_result,
        )
        final_pipeline = DynamicAgentPipeline(config=self.slots_config, api_client=self.api_client)
        final_meta = dict(meta)
        final_meta["final_generation_mode"] = "closed_loop_cumulative"
        final_meta["final_slot_summaries"] = build_slot_summary_payload(slot_schemas)
        return final_pipeline._generate_final_appreciation_prompt(
            cumulative_outputs,
            cumulative_validation,
            final_meta,
            cumulative_dialogue_state,
            api_logs=getattr(latest_result, "api_logs", None),
        )

    def _collect_cumulative_outputs(
        self,
        *,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
        latest_result: Any | None,
    ) -> list[DomainCoTRecord]:
        aggregated: dict[str, DomainCoTRecord] = {}
        order: list[str] = []

        def remember(slot_name: str) -> None:
            if slot_name and slot_name not in order:
                order.append(slot_name)

        for memory in self._coerce_list(meta.get("round_memories")):
            if not isinstance(memory, dict):
                continue
            round_index = int(memory.get("round_index", 0) or 0)
            for item in self._coerce_list(memory.get("slots")):
                record = self._coerce_domain_record(item, analysis_round=round_index)
                if record is None:
                    continue
                remember(record.slot_name)
                current = aggregated.get(record.slot_name)
                aggregated[record.slot_name] = self._merge_domain_records(current, record)

        for record in getattr(latest_result, "domain_outputs", []) or []:
            if not isinstance(record, DomainCoTRecord):
                continue
            remember(record.slot_name)
            current = aggregated.get(record.slot_name)
            aggregated[record.slot_name] = self._merge_domain_records(current, self._clone_domain_record(record))

        for slot in slot_schemas:
            remember(slot.slot_name)
            current = aggregated.get(slot.slot_name)
            if current is None:
                aggregated[slot.slot_name] = DomainCoTRecord(
                    slot_name=slot.slot_name,
                    slot_term=slot.slot_term,
                    analysis_round=0,
                    controlled_vocabulary=list(slot.controlled_vocabulary),
                    visual_anchoring=[],
                    domain_decoding=[],
                    cultural_mapping=[],
                    question_coverage=[
                        QuestionCoverage(question=question, answered=False, support="")
                        for question in slot.specific_questions
                    ],
                    unresolved_points=[],
                    generated_questions=[],
                    statuses=[],
                    confidence=float(slot.metadata.get("confidence", 0.0) or 0.0),
                )
                continue
            current.slot_term = str(slot.slot_term or current.slot_term).strip()
            current.controlled_vocabulary = self._dedupe_text_list(
                current.controlled_vocabulary + list(slot.controlled_vocabulary)
            )
            current.question_coverage = self._merge_question_coverage(
                current.question_coverage,
                [QuestionCoverage(question=question, answered=False, support="") for question in slot.specific_questions],
            )
            current.confidence = max(current.confidence, float(slot.metadata.get("confidence", 0.0) or 0.0))

        ranked_names = sorted(
            aggregated,
            key=lambda name: (
                next((index for index, slot in enumerate(slot_schemas) if slot.slot_name == name), len(slot_schemas)),
                order.index(name) if name in order else len(order),
                name,
            ),
        )
        return [aggregated[name] for name in ranked_names]

    def _collect_cumulative_validation(
        self,
        *,
        outputs: list[DomainCoTRecord],
        meta: dict[str, Any],
        latest_result: Any | None,
    ) -> CrossValidationResult:
        latest_validation = getattr(latest_result, "cross_validation", None)
        live_slots = {
            output.slot_name
            for output in outputs
            if output.unresolved_points or any(not item.answered for item in output.question_coverage)
        }
        issues: list[CrossValidationIssue] = []
        seen_issue_keys: set[str] = set()
        for memory in self._coerce_list(meta.get("round_memories")):
            if not isinstance(memory, dict):
                continue
            for item in self._coerce_list(memory.get("issues")):
                issue = self._coerce_issue(item)
                if issue is None:
                    continue
                if issue.slot_names and live_slots and not any(slot_name in live_slots for slot_name in issue.slot_names):
                    continue
                key = f"{issue.issue_type}::{self._normalize_text(issue.detail)}"
                if key in seen_issue_keys:
                    continue
                seen_issue_keys.add(key)
                issues.append(issue)
        if isinstance(latest_validation, CrossValidationResult):
            for issue in latest_validation.issues:
                key = f"{issue.issue_type}::{self._normalize_text(issue.detail)}"
                if key in seen_issue_keys:
                    continue
                seen_issue_keys.add(key)
                issues.append(issue)

        missing_points = self._dedupe_text_list(
            [
                f"{output.slot_name}: {item.question}"
                for output in outputs
                for item in output.question_coverage
                if not item.answered and item.question
            ]
        )
        rag_terms = self._dedupe_text_list(
            [
                term
                for issue in issues
                for term in issue.rag_terms
            ]
            + list(getattr(latest_validation, "rag_terms", []) if isinstance(latest_validation, CrossValidationResult) else [])
        )
        semantic_duplicates = self._dedupe_text_list(
            list(getattr(latest_validation, "semantic_duplicates", []) if isinstance(latest_validation, CrossValidationResult) else [])
        )
        removed_questions = self._dedupe_text_list(
            list(getattr(latest_validation, "removed_questions", []) if isinstance(latest_validation, CrossValidationResult) else [])
        )
        return CrossValidationResult(
            issues=issues,
            semantic_duplicates=semantic_duplicates,
            missing_points=missing_points,
            rag_terms=rag_terms,
            removed_questions=removed_questions,
            llm_review=str(getattr(latest_validation, "llm_review", "") if isinstance(latest_validation, CrossValidationResult) else "").strip(),
            round_table_blind_spots=list(
                getattr(latest_validation, "round_table_blind_spots", [])
                if isinstance(latest_validation, CrossValidationResult)
                else []
            ),
            round_table_follow_up_questions=list(
                getattr(latest_validation, "round_table_follow_up_questions", [])
                if isinstance(latest_validation, CrossValidationResult)
                else []
            ),
            round_table_rag_needs=list(
                getattr(latest_validation, "round_table_rag_needs", [])
                if isinstance(latest_validation, CrossValidationResult)
                else []
            ),
            slot_lifecycle_reviews=list(
                getattr(latest_validation, "slot_lifecycle_reviews", [])
                if isinstance(latest_validation, CrossValidationResult)
                else []
            ),
            follow_up_task_reviews=list(
                getattr(latest_validation, "follow_up_task_reviews", [])
                if isinstance(latest_validation, CrossValidationResult)
                else []
            ),
        )

    def _collect_cumulative_dialogue_state(
        self,
        *,
        outputs: list[DomainCoTRecord],
        validation: CrossValidationResult,
        meta: dict[str, Any],
        latest_result: Any | None,
    ) -> DialogueState:
        latest_state = getattr(latest_result, "dialogue_state", None)
        resolved_questions = self._dedupe_text_list(
            [
                item.question
                for output in outputs
                for item in output.question_coverage
                if item.answered and item.question
            ]
        )
        conversation_history = list(getattr(latest_state, "conversation_history", []) or [])
        for item in self._coerce_list(meta.get("dialogue_turns")):
            if not isinstance(item, dict):
                continue
            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()
            if role and content:
                conversation_history.append(f"{role}: {content}")
        return DialogueState(
            conversation_history=self._dedupe_text_list(conversation_history),
            turns=list(getattr(latest_state, "turns", []) or []),
            threads=list(getattr(latest_state, "threads", []) or []),
            resolved_questions=resolved_questions,
            unresolved_questions=list(validation.missing_points),
            removed_questions=list(validation.removed_questions),
            merged_duplicates=list(validation.semantic_duplicates),
            no_new_info_rounds=int(getattr(latest_state, "no_new_info_rounds", 0) or 0),
            converged=bool(getattr(latest_result, "routing", None).converged) if getattr(latest_result, "routing", None) else False,
            convergence_reason=str(
                getattr(getattr(latest_result, "routing", None), "convergence_reason", "")
                or getattr(latest_state, "convergence_reason", "")
                or ""
            ).strip(),
            final_round_index=int(getattr(latest_state, "final_round_index", 0) or 0),
        )

    def _coerce_domain_record(self, item: Any, *, analysis_round: int = 0) -> DomainCoTRecord | None:
        if not isinstance(item, dict):
            return None
        slot_name = str(item.get("slot_name", "")).strip()
        if not slot_name:
            return None
        slot_term = str(item.get("slot_term", "")).strip()
        visual_anchoring = [
            EvidenceItem(
                observation=str(entry.get("observation", "")).strip(),
                evidence=str(entry.get("evidence", "")).strip(),
                position=str(entry.get("position", "")).strip(),
            )
            for entry in self._coerce_list(item.get("visual_anchoring"))
            if isinstance(entry, dict) and str(entry.get("observation", "")).strip()
        ]
        domain_decoding = [
            DecodingItem(
                term=str(entry.get("term", "")).strip(),
                explanation=str(entry.get("explanation", "")).strip(),
                status=str(entry.get("status", "IDENTIFIED")).strip() or "IDENTIFIED",
                reason=str(entry.get("reason", "")).strip(),
            )
            for entry in self._coerce_list(item.get("domain_decoding"))
            if isinstance(entry, dict) and (str(entry.get("term", "")).strip() or str(entry.get("explanation", "")).strip())
        ]
        cultural_mapping = [
            MappingItem(
                insight=str(entry.get("insight", "")).strip(),
                basis=str(entry.get("basis", "")).strip(),
                risk_note=str(entry.get("risk_note", "")).strip(),
            )
            for entry in self._coerce_list(item.get("cultural_mapping"))
            if isinstance(entry, dict) and str(entry.get("insight", "")).strip()
        ]
        question_coverage = [
            QuestionCoverage(
                question=str(entry.get("question", "")).strip(),
                answered=bool(entry.get("answered", False)),
                support=str(entry.get("support", "")).strip(),
            )
            for entry in self._coerce_list(item.get("question_coverage"))
            if isinstance(entry, dict) and str(entry.get("question", "")).strip()
        ]
        controlled_vocabulary = self._dedupe_text_list(
            self._coerce_str_list(item.get("controlled_vocabulary"))
            + ([slot_term] if slot_term else [])
            + [entry.term for entry in domain_decoding if entry.term]
        )
        return DomainCoTRecord(
            slot_name=slot_name,
            slot_term=slot_term,
            analysis_round=max(0, int(item.get("analysis_round", analysis_round) or analysis_round or 0)),
            controlled_vocabulary=controlled_vocabulary,
            visual_anchoring=visual_anchoring,
            domain_decoding=domain_decoding,
            cultural_mapping=cultural_mapping,
            question_coverage=question_coverage,
            unresolved_points=self._dedupe_text_list(self._coerce_str_list(item.get("unresolved_points"))),
            generated_questions=self._dedupe_text_list(self._coerce_str_list(item.get("generated_questions"))),
            statuses=self._dedupe_text_list(self._coerce_str_list(item.get("statuses"))),
            confidence=float(item.get("confidence", 0.0) or 0.0),
            retrieval_gain_focus=str(item.get("retrieval_gain_focus", "")).strip(),
            retrieval_gain_terms=self._dedupe_text_list(self._coerce_str_list(item.get("retrieval_gain_terms"))),
            retrieval_gain_queries=self._dedupe_text_list(self._coerce_str_list(item.get("retrieval_gain_queries"))),
            retrieval_gain_mode=DynamicAgentPipeline._normalize_retrieval_mode(item.get("retrieval_gain_mode")),
            retrieval_gain_web_queries=self._dedupe_text_list(self._coerce_str_list(item.get("retrieval_gain_web_queries"))),
            retrieval_gain_reason=str(item.get("retrieval_gain_reason", "")).strip(),
            retrieval_gain_has_value=bool(item.get("retrieval_gain_has_value", False)),
            raw_response=str(item.get("raw_response", "")).strip(),
        )

    @staticmethod
    def _clone_domain_record(record: DomainCoTRecord) -> DomainCoTRecord:
        return DomainCoTRecord(
            slot_name=record.slot_name,
            slot_term=record.slot_term,
            analysis_round=record.analysis_round,
            controlled_vocabulary=list(record.controlled_vocabulary),
            visual_anchoring=[
                EvidenceItem(
                    observation=item.observation,
                    evidence=item.evidence,
                    position=item.position,
                )
                for item in record.visual_anchoring
            ],
            domain_decoding=[
                DecodingItem(
                    term=item.term,
                    explanation=item.explanation,
                    status=item.status,
                    reason=item.reason,
                )
                for item in record.domain_decoding
            ],
            cultural_mapping=[
                MappingItem(
                    insight=item.insight,
                    basis=item.basis,
                    risk_note=item.risk_note,
                )
                for item in record.cultural_mapping
            ],
            question_coverage=[
                QuestionCoverage(
                    question=item.question,
                    answered=item.answered,
                    support=item.support,
                )
                for item in record.question_coverage
            ],
            unresolved_points=list(record.unresolved_points),
            generated_questions=list(record.generated_questions),
            statuses=list(record.statuses),
            confidence=record.confidence,
            retrieval_gain_focus=record.retrieval_gain_focus,
            retrieval_gain_terms=list(record.retrieval_gain_terms),
            retrieval_gain_queries=list(record.retrieval_gain_queries),
            retrieval_gain_mode=record.retrieval_gain_mode,
            retrieval_gain_web_queries=list(record.retrieval_gain_web_queries),
            retrieval_gain_reason=record.retrieval_gain_reason,
            retrieval_gain_has_value=record.retrieval_gain_has_value,
            raw_response=record.raw_response,
        )

    def _merge_domain_records(
        self,
        current: DomainCoTRecord | None,
        incoming: DomainCoTRecord,
    ) -> DomainCoTRecord:
        if current is None:
            return incoming
        if incoming.slot_term.strip():
            current.slot_term = incoming.slot_term.strip()
        current.analysis_round = max(current.analysis_round, incoming.analysis_round)
        current.controlled_vocabulary = self._dedupe_text_list(
            current.controlled_vocabulary + incoming.controlled_vocabulary
        )
        current.visual_anchoring = self._merge_evidence_items(current.visual_anchoring, incoming.visual_anchoring)
        current.domain_decoding = self._merge_decoding_items(current.domain_decoding, incoming.domain_decoding)
        current.cultural_mapping = self._merge_mapping_items(current.cultural_mapping, incoming.cultural_mapping)
        current.question_coverage = self._merge_question_coverage(current.question_coverage, incoming.question_coverage)
        current.unresolved_points = self._dedupe_text_list(current.unresolved_points + incoming.unresolved_points)
        current.generated_questions = self._dedupe_text_list(current.generated_questions + incoming.generated_questions)
        current.statuses = self._dedupe_text_list(current.statuses + incoming.statuses)
        current.confidence = max(current.confidence, incoming.confidence)
        if incoming.retrieval_gain_focus.strip():
            current.retrieval_gain_focus = incoming.retrieval_gain_focus.strip()
        current.retrieval_gain_terms = self._dedupe_text_list(
            current.retrieval_gain_terms + incoming.retrieval_gain_terms
        )
        current.retrieval_gain_queries = self._dedupe_text_list(
            current.retrieval_gain_queries + incoming.retrieval_gain_queries
        )
        current.retrieval_gain_mode = DynamicAgentPipeline._merge_retrieval_modes(
            current.retrieval_gain_mode,
            incoming.retrieval_gain_mode,
        )
        current.retrieval_gain_web_queries = self._dedupe_text_list(
            current.retrieval_gain_web_queries + incoming.retrieval_gain_web_queries
        )
        if incoming.retrieval_gain_reason.strip():
            current.retrieval_gain_reason = incoming.retrieval_gain_reason.strip()
        current.retrieval_gain_has_value = current.retrieval_gain_has_value or incoming.retrieval_gain_has_value
        if incoming.raw_response.strip():
            current.raw_response = incoming.raw_response.strip()
        return current

    def _merge_evidence_items(
        self,
        current: list[EvidenceItem],
        incoming: list[EvidenceItem],
    ) -> list[EvidenceItem]:
        merged: dict[str, EvidenceItem] = {}
        order: list[str] = []
        for item in current + incoming:
            key = self._normalize_text(item.observation)
            if not key:
                continue
            if key not in merged:
                merged[key] = EvidenceItem(item.observation, item.evidence, item.position)
                order.append(key)
                continue
            existing = merged[key]
            if len(item.evidence) > len(existing.evidence):
                existing.evidence = item.evidence
            if len(item.position) > len(existing.position):
                existing.position = item.position
        return [merged[key] for key in order]

    def _merge_decoding_items(
        self,
        current: list[DecodingItem],
        incoming: list[DecodingItem],
    ) -> list[DecodingItem]:
        merged: dict[str, DecodingItem] = {}
        order: list[str] = []
        for item in current + incoming:
            key = self._normalize_text(item.term or item.explanation)
            if not key:
                continue
            if key not in merged:
                merged[key] = DecodingItem(item.term, item.explanation, item.status, item.reason)
                order.append(key)
                continue
            existing = merged[key]
            if len(item.explanation) > len(existing.explanation):
                existing.explanation = item.explanation
            if item.status and existing.status != item.status and item.status != "IDENTIFIED":
                existing.status = item.status
            if len(item.reason) > len(existing.reason):
                existing.reason = item.reason
        return [merged[key] for key in order]

    def _merge_mapping_items(
        self,
        current: list[MappingItem],
        incoming: list[MappingItem],
    ) -> list[MappingItem]:
        merged: dict[str, MappingItem] = {}
        order: list[str] = []
        for item in current + incoming:
            key = self._normalize_text(item.insight)
            if not key:
                continue
            if key not in merged:
                merged[key] = MappingItem(item.insight, item.basis, item.risk_note)
                order.append(key)
                continue
            existing = merged[key]
            if len(item.basis) > len(existing.basis):
                existing.basis = item.basis
            if len(item.risk_note) > len(existing.risk_note):
                existing.risk_note = item.risk_note
        return [merged[key] for key in order]

    def _merge_question_coverage(
        self,
        current: list[QuestionCoverage],
        incoming: list[QuestionCoverage],
    ) -> list[QuestionCoverage]:
        merged: dict[str, QuestionCoverage] = {}
        order: list[str] = []
        for item in current + incoming:
            key = self._normalize_text(item.question)
            if not key:
                continue
            if key not in merged:
                merged[key] = QuestionCoverage(item.question, item.answered, item.support)
                order.append(key)
                continue
            existing = merged[key]
            if item.answered and not existing.answered:
                merged[key] = QuestionCoverage(item.question, item.answered, item.support)
                continue
            if item.answered == existing.answered and len(item.support) > len(existing.support):
                existing.support = item.support
        return [merged[key] for key in order]

    def _coerce_issue(self, item: Any) -> CrossValidationIssue | None:
        if not isinstance(item, dict):
            return None
        detail = str(item.get("detail", "")).strip()
        issue_type = str(item.get("issue_type", "")).strip()
        severity = str(item.get("severity", "")).strip()
        if not detail or not issue_type or not severity:
            return None
        slot_names = self._coerce_str_list(item.get("slot_names"))
        if not slot_names and isinstance(item.get("slots"), list):
            slot_names = self._coerce_str_list(item.get("slots"))
        return CrossValidationIssue(
            issue_type=issue_type,
            severity=severity,
            slot_names=slot_names,
            detail=detail,
            evidence=self._coerce_str_list(item.get("evidence")),
            rag_terms=self._coerce_str_list(item.get("rag_terms")),
        )

    def _seed_rag_cache_from_post_rag(self, meta: dict[str, Any]) -> None:
        meta.setdefault("rag_cache", [])
        for item in self._coerce_list(meta.get("post_rag_text_extraction")):
            if not isinstance(item, dict):
                continue
            self._append_rag_cache_entry(
                meta,
                {
                    "term": str(item.get("term", "")).strip(),
                    "description": str(item.get("description", "")).strip(),
                    "text_evidence": self._coerce_str_list(item.get("text_evidence")),
                    "source_id": str(item.get("source_id", "")).strip(),
                    "score": item.get("score"),
                    "metadata": item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
                    "origin_stage": str(item.get("origin_stage", "")).strip() or "post_rag_text_extraction",
                    "source_slot": str(item.get("source_slot", "")).strip(),
                    "query": str(item.get("query", "")).strip(),
                },
            )

    def _merge_external_rag_into_meta(
        self,
        meta: dict[str, Any],
        *,
        external_rag: dict[str, Any],
        task: SpawnTask,
    ) -> None:
        meta.setdefault("rag_cache", [])
        for item in self._coerce_list(external_rag.get("documents")):
            if not isinstance(item, dict):
                continue
            self._append_rag_cache_entry(
                meta,
                {
                    **item,
                    "origin_stage": str(item.get("origin_stage", "")).strip() or "downstream_rag",
                    "source_slot": str(item.get("source_slot", "")).strip() or task.slot_name,
                    "query": str(item.get("query", "")).strip(),
                },
            )

    def _append_rag_cache_entry(self, meta: dict[str, Any], item: dict[str, Any]) -> bool:
        meta.setdefault("rag_cache", [])
        meta.setdefault("retained_facts", [])
        payload = {
            "term": str(item.get("term", "")).strip(),
            "description": str(item.get("description", "")).strip(),
            "text_evidence": self._coerce_str_list(item.get("text_evidence")),
            "source_id": str(item.get("source_id", "")).strip(),
            "score": item.get("score"),
            "metadata": item.get("metadata", {}) if isinstance(item.get("metadata", {}), dict) else {},
            "origin_stage": str(item.get("origin_stage", "")).strip(),
            "source_slot": str(item.get("source_slot", "")).strip(),
            "query": str(item.get("query", "")).strip(),
        }
        if not payload["term"] and not payload["description"] and not payload["text_evidence"]:
            return False
        added = self._append_unique_dict(meta["rag_cache"], payload)
        self._harvest_rag_document_facts(meta, payload)
        return added

    def _harvest_rag_document_facts(self, meta: dict[str, Any], item: dict[str, Any]) -> int:
        slot_name = str(item.get("source_slot", "")).strip()
        slot_term = str(item.get("term", "")).strip()
        added = 0
        for text in self._rag_fact_candidate_texts(item):
            candidate_slots = self._infer_rag_fact_slots(slot_name=slot_name, slot_term=slot_term, text=text)
            for candidate_slot in candidate_slots:
                for fact in self._extract_rag_retained_facts(candidate_slot, text):
                    if self._append_retained_fact_entry(
                        meta,
                        slot_name=candidate_slot,
                        slot_term=slot_term,
                        fact=fact,
                        source="rag_document",
                    ):
                        added += 1
        return added

    def _rag_fact_candidate_texts(self, item: dict[str, Any]) -> list[str]:
        texts: list[str] = []
        description = " ".join(str(item.get("description", "")).strip().split())
        if description:
            texts.append(description)
        for evidence in self._coerce_str_list(item.get("text_evidence"))[:3]:
            compact = " ".join(str(evidence).strip().split())
            if not compact or len(compact) > 160:
                continue
            if "<image>" in compact or "局部" in compact:
                continue
            if self._looks_like_stable_rag_fact_text(compact):
                texts.append(compact)
        return self._dedupe_text_list(texts)

    def _infer_rag_fact_slots(self, *, slot_name: str, slot_term: str, text: str) -> list[str]:
        explicit_slot = str(slot_name or "").strip()
        if explicit_slot in {"作者时代流派", "尺寸规格/材质形制/收藏地"}:
            return [explicit_slot]
        inferred: list[str] = []
        if self._looks_like_author_fact_text(text) or self._term_matches_author_family(slot_term):
            inferred.append("作者时代流派")
        if self._looks_like_material_fact_text(text) or self._term_matches_material_family(slot_term):
            inferred.append("尺寸规格/材质形制/收藏地")
        return self._dedupe_text_list(inferred)

    @staticmethod
    def _looks_like_stable_rag_fact_text(text: str) -> bool:
        return ClosedLoopCoordinator._looks_like_author_fact_text(text) or ClosedLoopCoordinator._looks_like_material_fact_text(text)

    @staticmethod
    def _looks_like_author_fact_text(text: str) -> bool:
        markers = ("画家", "作者", "活跃", "擅长", "流派", "画派", "人物画", "山水画", "花鸟画", "道释画")
        return any(marker in str(text or "") for marker in markers)

    @staticmethod
    def _looks_like_material_fact_text(text: str) -> bool:
        compact = str(text or "")
        if re.search(r"(?:纵|高)\s*\d+(?:\.\d+)?(?:\s*(?:厘米|cm|毫米|mm))?\s*[×xX＊*]\s*(?:横|宽)?\s*\d+(?:\.\d+)?\s*(?:厘米|cm|毫米|mm)", compact, flags=re.IGNORECASE):
            return True
        markers = (
            "现藏于",
            "馆藏",
            "博物馆",
            "美术馆",
            "博物院",
            "故宫",
            "绢本",
            "纸本",
            "墨笔",
            "设色",
            "手卷",
            "立轴",
            "册页",
            "尺寸",
            "厘米",
            "纵",
            "横",
        )
        return any(marker in compact for marker in markers)

    def _extract_rag_retained_facts(self, slot_name: str, text: str) -> list[str]:
        compact = " ".join(str(text or "").strip().split())
        if not compact:
            return []
        if slot_name == "作者时代流派":
            return self._extract_author_facts_from_text(compact)
        if slot_name == "尺寸规格/材质形制/收藏地":
            return self._extract_material_facts_from_text(compact)
        return []

    def _extract_author_facts_from_text(self, text: str) -> list[str]:
        facts: list[str] = []
        for match in re.finditer(r"[^。；]{0,36}(?:活跃|擅长|画家|作者|流派|画派)[^。；]{0,24}", text):
            fact = " ".join(match.group(0).strip(" ，,；;").split())
            if not fact or len(fact) > 80:
                continue
            if self._looks_like_material_fact_text(fact):
                continue
            if fact not in facts:
                facts.append(fact)
        return facts[:4]

    def _extract_material_facts_from_text(self, text: str) -> list[str]:
        facts: list[str] = []
        museum_patterns = (
            r"(现藏于[^，。；]{2,28}(?:博物馆|美术馆|纪念馆|博物院|文物院|故宫))",
            r"([^，。；]{2,28}(?:博物馆|美术馆|纪念馆|博物院|文物院|故宫)馆藏)",
        )
        for pattern in museum_patterns:
            for match in re.finditer(pattern, text):
                fact = " ".join(match.group(1).strip().split())
                if fact and fact not in facts:
                    facts.append(fact)
        for match in re.finditer(
            r"((?:纵|高)\s*\d+(?:\.\d+)?(?:\s*(?:厘米|cm|毫米|mm))?\s*[×xX＊*]\s*(?:横|宽)?\s*\d+(?:\.\d+)?\s*(?:厘米|cm|毫米|mm))",
            text,
            flags=re.IGNORECASE,
        ):
            fact = " ".join(match.group(1).strip().split())
            if fact and fact not in facts:
                facts.append(fact)
        for match in re.finditer(r"([^，。；]{0,18}(?:绢本设色|纸本设色|绢本墨笔|纸本墨笔|绢本|纸本|墨笔|手卷|立轴|册页)[^，。；]{0,18})", text):
            fact = " ".join(match.group(1).strip(" ，,；;").split())
            if not fact or len(fact) > 80:
                continue
            if fact not in facts:
                facts.append(fact)
        return facts[:5]

    def _select_cached_documents_for_task(
        self,
        *,
        task: SpawnTask,
        slot_schemas: list[SlotSchema],
        meta: dict[str, Any],
    ) -> list[dict[str, Any]]:
        slot = next((item for item in slot_schemas if item.slot_name == task.slot_name), None)
        focus_terms = self._dedupe_text_list(
            [
                task.slot_name,
                str(getattr(task, "slot_term", "") or "").strip(),
                *self._coerce_str_list(task.rag_terms),
                *self._coerce_str_list(getattr(task, "web_queries", [])),
                *(slot.controlled_vocabulary if slot is not None else []),
            ]
        )
        results: list[dict[str, Any]] = []
        seen: set[str] = set()
        for item in reversed(self._coerce_list(meta.get("rag_cache"))):
            if not isinstance(item, dict):
                continue
            source_slot = str(item.get("source_slot", "")).strip()
            term = str(item.get("term", "")).strip()
            candidate_text = " ".join(
                [
                    term,
                    str(item.get("description", "")).strip(),
                    " ".join(self._coerce_str_list(item.get("text_evidence"))),
                    source_slot,
                ]
            )
            normalized_candidate = self._normalize_text(candidate_text)
            if not normalized_candidate:
                continue
            matched = source_slot == task.slot_name or any(
                self._normalize_text(token) and self._normalize_text(token) in normalized_candidate
                for token in focus_terms
            )
            if not matched:
                continue
            key = "|".join(
                [
                    self._normalize_text(term),
                    self._normalize_text(str(item.get("description", "")).strip()),
                    self._normalize_text(str(item.get("source_id", "")).strip()),
                ]
            )
            if key in seen:
                continue
            seen.add(key)
            results.append(item)
            if len(results) >= 6:
                break
        results.reverse()
        return results

    @staticmethod
    def _downstream_rag_cache_path(run_dir: Path) -> Path:
        return run_dir / "runtime_state" / "downstream_rag_cache.json"

    @staticmethod
    def _downstream_web_cache_path(run_dir: Path) -> Path:
        return run_dir / "runtime_state" / "downstream_web_cache.json"

    @staticmethod
    def _read_downstream_rag_cache(path: Path) -> list[dict[str, Any]]:
        if not path.exists() or not path.is_file():
            return []
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(payload, list):
            return []
        return [item for item in payload if isinstance(item, dict)]

    def _write_downstream_rag_cache_entry(
        self,
        path: Path,
        *,
        query_cache: list[dict[str, Any]],
        query: str,
        documents: list[dict[str, Any]],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        signature = self._query_signature(query)
        updated = [
            item
            for item in query_cache
            if str(item.get("query_signature", "")).strip() != signature
        ]
        updated.append(
            {
                "query": query,
                "query_signature": signature,
                "documents": self._dedupe_dict_list(documents),
            }
        )
        path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def _read_downstream_web_cache(path: Path) -> list[dict[str, Any]]:
        return ClosedLoopCoordinator._read_downstream_rag_cache(path)

    def _write_downstream_web_cache_entry(
        self,
        path: Path,
        *,
        query_cache: list[dict[str, Any]],
        query: str,
        documents: list[dict[str, Any]],
    ) -> None:
        self._write_downstream_rag_cache_entry(
            path,
            query_cache=query_cache,
            query=query,
            documents=documents,
        )

    def _cached_documents_for_query(self, query_cache: list[dict[str, Any]], query: str) -> list[dict[str, Any]]:
        signature = self._query_signature(query)
        for item in query_cache:
            if str(item.get("query_signature", "")).strip() != signature:
                continue
            documents = item.get("documents", [])
            if isinstance(documents, list):
                return [doc for doc in documents if isinstance(doc, dict)]
        return []

    def _serialize_external_rag_results(self, *, query: str, results: list[Any]) -> list[dict[str, Any]]:
        documents: list[dict[str, Any]] = []
        for item in results:
            content = str(getattr(item, "content", "") or "").strip()
            if not content:
                continue
            documents.append(
                {
                    "term": query,
                    "description": content[:800],
                    "text_evidence": [content],
                    "source_id": str(getattr(item, "source_id", "") or ""),
                    "score": getattr(item, "score", None),
                    "metadata": getattr(item, "metadata", {}) or {},
                    "query": query,
                    "origin_stage": "downstream_rag",
                }
            )
        return self._dedupe_dict_list(documents)

    def _normalize_downstream_queries(self, value: Any, *, slot_name: str = "") -> list[str]:
        max_blocks = max(1, int(getattr(self.slots_config, "rag_query_max_blocks", 1) or 1))
        results: list[str] = []
        seen: set[str] = set()
        for item in self._coerce_str_list(value):
            text = self._clean_downstream_query(item, slot_name=slot_name, max_keyword_blocks=max_blocks)
            if not text:
                continue
            key = re.sub(r"\s+", " ", text).casefold()
            if key in seen:
                continue
            seen.add(key)
            results.append(text)
        return results[:5]

    def _normalize_downstream_web_queries(self, value: Any) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        top_k = max(1, int(getattr(self.slots_config, "web_search_top_k", 5) or 5))
        for item in self._coerce_str_list(value):
            text = self._clean_downstream_web_query(item)
            if not text:
                continue
            key = re.sub(r"\s+", " ", text).casefold()
            if key in seen:
                continue
            seen.add(key)
            results.append(text)
        return results[:top_k]

    @classmethod
    def _clean_downstream_query(cls, value: object, *, slot_name: str = "", max_keyword_blocks: int = 1) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"^[\d\W_]+", "", text)
        text = re.sub(r"[\s\u3000]+", " ", text).strip(" ,，;；:：。.!！、")
        if not text or "?" in text or "？" in text:
            return ""
        lowered = text.casefold()
        question_prefixes = (
            "如何",
            "为什么",
            "为何",
            "怎么",
            "怎样",
            "哪些",
            "有何",
            "何以",
            "是否",
            "能否",
            "可否",
            "what ",
            "why ",
            "how ",
            "which ",
        )
        if any(lowered.startswith(prefix) for prefix in question_prefixes):
            return ""
        parts = [
            part.strip(" ,，;；:：。.!！、")
            for part in re.split(r"[\s/|,，;；]+", text)
            if part.strip(" ,，;；:：。.!！、")
        ]
        if not parts:
            return ""
        if any(("?" in part or "？" in part) for part in parts):
            return ""
        for part in parts:
            lowered_part = part.casefold()
            if any(lowered_part.startswith(prefix) for prefix in question_prefixes):
                return ""
        if len(parts) > max(1, int(max_keyword_blocks)):
            parts = parts[: max(1, int(max_keyword_blocks))]
        compact = " ".join(parts)
        if cls._normalize_text(slot_name) == cls._normalize_text(compact):
            return ""
        if len(compact) < 2 or len(compact) > 24:
            return ""
        if cls._looks_like_over_descriptive_query(compact):
            return ""
        return compact

    @classmethod
    def _clean_downstream_web_query(cls, value: object, *, max_keyword_blocks: int = 6) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        text = re.sub(r"^[\d\W_]+", "", text)
        text = re.sub(r"[\s\u3000]+", " ", text).strip(" ,，;；:：。.!！、")
        if not text:
            return ""
        lowered = text.casefold()
        question_prefixes = (
            "如何",
            "为什么",
            "为何",
            "怎么",
            "怎样",
            "哪些",
            "有何",
            "何以",
            "what ",
            "why ",
            "how ",
            "which ",
        )
        if any(lowered.startswith(prefix) for prefix in question_prefixes):
            return ""
        parts = [part.strip(" ,，;；:：。.!！、") for part in re.split(r"[\s\u3000]+", text) if part.strip(" ,，;；:：。.!！、")]
        if not parts:
            return ""
        if len(parts) > max(1, int(max_keyword_blocks)):
            parts = parts[: max(1, int(max_keyword_blocks))]
        compact = " ".join(parts).strip(" ,，;；:：。.!！、")
        if len(compact) < 2 or len(compact) > 96:
            return ""
        return compact

    def _build_web_search_client(self) -> SerperWebSearchClient | None:
        if not bool(getattr(self.slots_config, "enable_web_search", False)):
            return None
        api_key = str(getattr(self.slots_config, "web_search_api_key", "") or "").strip()
        if not api_key:
            api_key = read_text_secret_file(
                getattr(self.slots_config, "web_search_api_key_file", None),
                line_number=max(1, int(getattr(self.slots_config, "web_search_api_key_line", 1) or 1)),
            )
        return SerperWebSearchClient(
            url=str(getattr(self.slots_config, "web_search_url", "") or "").strip(),
            api_key=api_key,
            timeout=max(1, int(getattr(self.slots_config, "web_search_timeout", 20) or 20)),
        )

    @classmethod
    def _looks_like_over_descriptive_query(cls, text: str) -> bool:
        compact = str(text or "").strip()
        if not compact:
            return True
        if cls._looks_like_stable_title_or_entity(compact):
            return False
        if len(compact) <= 8:
            return False
        descriptive_markers = (
            "通过",
            "表现",
            "突出",
            "采用",
            "整体",
            "画面",
            "本作",
            "当前",
            "主要",
            "形成",
            "增强",
            "强化",
            "结合",
            "呼应",
            "位于",
            "可见",
            "用于",
            "以及",
            "如何",
            "为何",
            "为什么",
            "是否",
            "怎么",
            "怎样",
            "哪些",
            "有何",
            "何以",
        )
        marker_hits = sum(1 for marker in descriptive_markers if marker in compact)
        relation_hits = sum(1 for marker in ("的", "为", "以", "与", "和") if marker in compact)
        if marker_hits >= 1 and len(compact) >= 10:
            return True
        if relation_hits >= 2 and len(compact) >= 10:
            return True
        return False

    @staticmethod
    def _looks_like_stable_title_or_entity(text: str) -> bool:
        compact = str(text or "").strip()
        if not compact:
            return False
        if "《" in compact and "》" in compact:
            return True
        if re.search(r"[图卷轴册屏幛页卷帖记传赋赞铭]", compact) and len(compact) <= 12:
            return True
        if re.fullmatch(r"[A-Za-z0-9\-_]{2,24}", compact):
            return True
        return False

    @staticmethod
    def _coerce_list(value: Any) -> list[Any]:
        return value if isinstance(value, list) else []

    @staticmethod
    def _coerce_str_list(value: Any) -> list[str]:
        results: list[str] = []
        if not isinstance(value, list):
            return results
        for item in value:
            text = str(item).strip()
            if text:
                results.append(text)
        return results

    def _slot_terms(self, slot: SlotSchema) -> list[str]:
        return extract_slot_terms(slot.slot_term, slot.metadata)

    def _slot_terms_phrase(self, slot_terms: list[str], *, fallback: str = "") -> str:
        terms = self._dedupe_text_list([str(item).strip() for item in slot_terms if str(item).strip()])
        if terms:
            return "、".join(terms)
        return fallback.strip()

    def _select_slot_terms_for_term(self, term: str, metadata: dict[str, Any]) -> list[str]:
        normalized_term = self._normalize_text(term)
        if not normalized_term:
            return []
        candidate_term_groups = metadata.get("candidate_term_groups", [])
        if isinstance(candidate_term_groups, list):
            for item in candidate_term_groups:
                if not isinstance(item, dict):
                    continue
                terms = [
                    str(group_term).strip()
                    for group_term in item.get("terms", [])
                    if str(group_term).strip()
                ]
                if normalized_term in {self._normalize_text(group_term) for group_term in terms}:
                    return self._dedupe_text_list(terms)
        current_terms = self._coerce_str_list(metadata.get("slot_terms"))
        if current_terms and normalized_term in {self._normalize_text(item) for item in current_terms}:
            return self._dedupe_text_list(current_terms)
        return [term.strip()]

    @staticmethod
    def _dedupe_text_list(items: list[str]) -> list[str]:
        results: list[str] = []
        seen: set[str] = set()
        for item in items:
            key = ClosedLoopCoordinator._normalize_text(item)
            if not key or key in seen:
                continue
            seen.add(key)
            results.append(item)
        return results

    @staticmethod
    def _normalize_text(text: str) -> str:
        return "".join(str(text or "").strip().lower().split())

    @classmethod
    def _slot_repeat_signature(cls, slot_name: str, slot_term: str) -> str:
        normalized_slot = cls._normalize_text(slot_name)
        normalized_term = cls._normalize_text(slot_term)
        if not normalized_slot or not normalized_term:
            return ""
        return f"{normalized_slot}::{normalized_term}"

    @staticmethod
    def _read_downstream_rag_query_counts(run_dir: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted((run_dir / "downstream_rounds").glob("round_*/rag_search_record.md")):
            if not path.exists() or not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for query in re.findall(r"- query_text: `([^`]+)`", raw):
                key = ClosedLoopCoordinator._query_signature(query)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _read_downstream_web_query_counts(run_dir: Path) -> dict[str, int]:
        counts: dict[str, int] = {}
        for path in sorted((run_dir / "downstream_rounds").glob("round_*/web_search_record.md")):
            if not path.exists() or not path.is_file():
                continue
            try:
                raw = path.read_text(encoding="utf-8")
            except OSError:
                continue
            for query in re.findall(r"- query_text: `([^`]+)`", raw):
                key = ClosedLoopCoordinator._query_signature(query)
                if not key:
                    continue
                counts[key] = counts.get(key, 0) + 1
        return counts

    @staticmethod
    def _query_signature(query: str) -> str:
        parts = [
            part
            for part in re.split(r"[\s/|,，;；:：。.!！、()（）]+", str(query or "").strip())
            if part
        ]
        if not parts:
            return ClosedLoopCoordinator._normalize_text(query)
        normalized_parts = sorted({ClosedLoopCoordinator._normalize_text(part) for part in parts if ClosedLoopCoordinator._normalize_text(part)})
        return "|".join(normalized_parts)

    @staticmethod
    def _append_downstream_rag_record(
        *,
        path: Path,
        image_path: Path,
        query: str,
        source_ids: list[str],
        scores: list[Any],
        image_attached: bool,
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# RAG Search Record\n\n", encoding="utf-8")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        clean_sources = [item for item in source_ids if str(item).strip()]
        clean_scores = [str(item) for item in scores if item is not None]
        lines = [
            f"\n## Search Batch [{timestamp}]\n",
            f"- image: `{image_path}`\n",
            f"- query_text: `{query}`\n",
            f"  - image_attached: `{'true' if image_attached else 'false'}`\n",
            f"  - sources: `{','.join(clean_sources) or 'none'}`\n",
            f"  - alignment_scores: `{','.join(clean_scores) or 'none'}`\n",
        ]
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)

    @staticmethod
    def _append_downstream_web_record(
        *,
        path: Path,
        query: str,
        hits: list[WebSearchHit],
        selected_hits: list[WebSearchHit],
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("# Web Search Record\n\n", encoding="utf-8")
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        selected_urls = [hit.url for hit in selected_hits if str(hit.url).strip()]
        lines = [
            f"\n## Search Batch [{timestamp}]\n",
            f"- query_text: `{query}`\n",
            f"- candidates: `{len(hits)}`\n",
            f"- selected_urls: `{','.join(selected_urls) or 'none'}`\n",
        ]
        for hit in hits:
            marker = "selected" if any(selected.url == hit.url for selected in selected_hits) else "candidate"
            lines.append(
                f"  - [{marker}] title=`{hit.title}` url=`{hit.url}` source=`{hit.source}` position=`{hit.position}` snippet=`{hit.snippet[:180]}`\n"
            )
        with path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)


def _load_perception_layer() -> tuple[Any, Any, Any]:
    if str(_PRECEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(_PRECEPTION_ROOT))
    from perception_layer.config import PipelineConfig as PerceptionConfig  # type: ignore
    from perception_layer.downstream import DownstreamPromptRunner  # type: ignore
    from perception_layer.pipeline import PerceptionPipeline  # type: ignore

    return PerceptionConfig, DownstreamPromptRunner, PerceptionPipeline


def _load_perception_config() -> Any:
    if str(_PRECEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(_PRECEPTION_ROOT))
    from perception_layer.config import PipelineConfig as PerceptionConfig  # type: ignore

    return PerceptionConfig


def _load_perception_rag_client() -> Any:
    if str(_PRECEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(_PRECEPTION_ROOT))
    from perception_layer.clients import HttpRagClient  # type: ignore

    return HttpRagClient
