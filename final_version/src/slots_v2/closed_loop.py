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

from .image_utils import prepare_image
from .meta_loader import load_context_meta, merge_meta
from .models import PipelineConfig, PreparedImage, SlotSchema, SpawnTask
from .new_api_client import NewAPIClient
from .pipeline import DynamicAgentPipeline
from .schema_loader import extract_controlled_vocabulary, load_slot_schemas

_PRECEPTION_ROOT = Path("/Users/ken/MM/Pipeline/preception_layer_1")


DOWNSTREAM_SYSTEM_PROMPT = """你是 preception_layer_1 的下游任务扩展器。
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
3. 优先补充文字证据、已有槽位的描述和问题，而不是重复造新槽位。
4. 新增槽位必须具体、稳定、可核验。
5. 如果只是提出下一轮 RAG 建议，请写入 search_queries，不要伪装成已确认事实。
"""


@dataclass(slots=True)
class ClosedLoopConfig:
    output_dir: str = "artifacts_closed_loop"
    max_closed_loop_rounds: int = 3
    max_downstream_tasks_per_round: int = 4
    stall_round_limit: int = 2
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
    payload_path: str
    response_path: str
    applied_changes: dict[str, Any]
    status: str


@dataclass(slots=True)
class ClosedLoopResult:
    run_dir: str
    bootstrap_slots_file: str
    bootstrap_context_file: str
    final_slots_file: str
    final_context_file: str
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

        for round_index in range(1, max(1, int(self.closed_loop_config.max_closed_loop_rounds)) + 1):
            self._print_closed_loop_progress(
                stage="round_start",
                round_index=round_index,
                note=f"slots={len(runtime_slots)}",
            )
            slots_path = runtime_state_dir / f"slots_round_{round_index:02d}.jsonl"
            context_path = runtime_state_dir / f"context_round_{round_index:02d}.md"
            self._write_slots_jsonl(runtime_slots, slots_path)
            self._write_context_markdown(runtime_meta, context_path)

            slots_pipeline = self._slots_pipeline_factory(
                config=replace(
                    self.slots_config,
                    slots_file=str(slots_path),
                    output_dir=str(run_dir / "slots_rounds" / f"round_{round_index:02d}"),
                ),
                api_client=self.api_client,
            )
            result = slots_pipeline.run(image_path=prepared_input.path, meta=runtime_meta)
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

            if result.routing.converged:
                stop_reason = result.routing.convergence_reason or "converged"
                self._print_closed_loop_progress(
                    stage="converged",
                    round_index=round_index,
                    result=result,
                    note=stop_reason,
                )
                break

            tasks = result.routing.spawned_tasks[: max(1, int(self.closed_loop_config.max_downstream_tasks_per_round))]
            if not tasks:
                stop_reason = "no_spawned_tasks"
                self._print_closed_loop_progress(
                    stage="stop",
                    round_index=round_index,
                    result=result,
                    note=stop_reason,
                )
                break

            round_changed = False
            self._print_closed_loop_progress(
                stage="downstream_start",
                round_index=round_index,
                result=result,
                note=f"tasks={len(tasks)}",
            )
            runner = self._build_downstream_runner(run_dir=run_dir, round_index=round_index)
            for task_index, task in enumerate(tasks, start=1):
                external_rag = self._run_task_rag(
                    task=task,
                    image_path=prepared_input.path,
                    run_dir=run_dir,
                    round_index=round_index,
                    task_index=task_index,
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
                payload_path = task_dir / f"task_{task_index:02d}_payload.json"
                payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

                response = runner.run_json(
                    task_name=f"closed_loop_round_{round_index}_task_{task_index}",
                    system_prompt=DOWNSTREAM_SYSTEM_PROMPT,
                    user_text=json.dumps(payload, ensure_ascii=False, indent=2),
                    image_file=prepared_input.path,
                )
                response_path = task_dir / f"task_{task_index:02d}_response.json"
                response_path.write_text(json.dumps(response, ensure_ascii=False, indent=2), encoding="utf-8")

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
                        payload_path=str(payload_path),
                        response_path=str(response_path),
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

            if round_changed:
                no_change_rounds = 0
            else:
                no_change_rounds += 1

            note = f"closed_loop round {round_index}: downstream_changed={str(round_changed).lower()} tasks={len(tasks)}"
            if note not in runtime_meta["closed_loop_notes"]:
                runtime_meta["closed_loop_notes"].append(note)
            self._append_dialogue_turn(runtime_meta, note)
            self._print_closed_loop_progress(
                stage="downstream_round_done",
                round_index=round_index,
                result=result,
                note=f"changed={str(round_changed).lower()} no_change_rounds={no_change_rounds}",
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
        final_context_file = runtime_state_dir / "context_final.md"
        self._write_slots_jsonl(runtime_slots, final_slots_file)
        self._write_context_markdown(runtime_meta, final_context_file)

        final_prompt_path = run_dir / "final_appreciation_prompt.md"
        final_prompt_path.write_text(
            latest_result.final_appreciation_prompt if latest_result else "",
            encoding="utf-8",
        )

        report_path = run_dir / "closed_loop_report.json"
        report_payload = {
            "run_dir": str(run_dir),
            "prepared_input": asdict(prepared_input),
            "bootstrap_slots_file": str(bootstrap["slots_file"]),
            "bootstrap_context_file": str(bootstrap["context_file"]),
            "final_slots_file": str(final_slots_file),
            "final_context_file": str(final_context_file),
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
            final_context_file=str(final_context_file),
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
        perception_config_cls, _, _ = _load_perception_layer()
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
        related_issues = [
            issue.detail
            for issue in slots_result.cross_validation.issues
            if task.slot_name in issue.slot_names
        ]
        relevant_slots = [
            self._serialize_slot(slot)
            for slot in slot_schemas
            if slot.slot_name == task.slot_name or (task_slot_term and slot.slot_term == task_slot_term)
        ]
        if not relevant_slots:
            relevant_slots = [self._serialize_slot(slot) for slot in slot_schemas]

        extra_constraints = [
            f"reason={task.reason}",
            f"focus={task.prompt_focus}",
            f"priority={task.priority}",
        ]
        extra_constraints.extend(f"rag_term={term}" for term in task.rag_terms)
        extra_constraints.extend(f"downstream_rag_query={term}" for term in self._coerce_str_list(external_rag.get("queries")))
        extra_constraints.extend(f"issue={detail}" for detail in related_issues)
        rag_documents = self._coerce_list(meta.get("post_rag_text_extraction")) + self._coerce_list(external_rag.get("documents"))

        return {
            "painting_profile": meta.get("domain_profile", {}),
            "existing_slots": relevant_slots,
            "ontology_relations": meta.get("ontology_updates", []),
            "rag_documents": self._dedupe_dict_list(rag_documents),
            "external_rag_queries": self._coerce_str_list(external_rag.get("queries")),
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

    def _run_task_rag(
        self,
        *,
        task: SpawnTask,
        image_path: str,
        run_dir: Path,
        round_index: int,
        task_index: int,
    ) -> dict[str, Any]:
        query_terms = self._coerce_str_list(task.rag_terms)
        if not query_terms:
            return {"queries": [], "documents": []}

        config = self._build_perception_config(
            context_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "context.md",
            rag_search_record_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "rag_search_record.md",
            llm_chat_record_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "llm_chat_record.jsonl",
            output_path=run_dir / "downstream_rounds" / f"round_{round_index:02d}" / "slots.jsonl",
            judge_model_override=self.closed_loop_config.downstream_model,
        )
        rag_client_cls = _load_perception_rag_client()
        rag_client = rag_client_cls(config.rag_endpoint)

        image_file = Path(image_path)
        image_bytes = image_file.read_bytes() if image_file.exists() and image_file.is_file() else None
        image_mime_type = mimetypes.guess_type(str(image_file))[0] or "image/png"

        documents: list[dict[str, Any]] = []
        used_queries: list[str] = []
        for query in self._dedupe_text_list(query_terms):
            try:
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
            used_queries.append(query)
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
                    }
                )

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
        runtime_meta.setdefault("ontology_updates", [])
        runtime_meta.setdefault("downstream_updates", [])
        runtime_meta.setdefault("closed_loop_notes", [])
        runtime_meta.setdefault("dialogue_turns", [])
        runtime_meta.setdefault("round_memories", [])

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

    def _write_context_markdown(self, meta: dict[str, Any], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = ["# Closed-loop Context", ""]
        lines.extend(self._render_json_section("Domain Profile", [meta.get("domain_profile", {})]))
        lines.extend(self._render_json_section("Post-RAG Text Extraction", meta.get("post_rag_text_extraction", [])))
        lines.extend(self._render_bullet_section("Ontology Updates", meta.get("ontology_updates", [])))
        lines.extend(self._render_json_section("Downstream Updates", meta.get("downstream_updates", [])))
        lines.extend(self._render_bullet_section("Closed-loop Notes", meta.get("closed_loop_notes", [])))
        lines.extend(self._render_json_section("Round Memories", meta.get("round_memories", [])))
        path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")

    @staticmethod
    def _render_json_section(title: str, items: list[Any]) -> list[str]:
        lines = [f"## {title}", ""]
        if not items:
            lines.append("- none")
        else:
            for item in items:
                lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
        lines.append("")
        return lines

    @staticmethod
    def _render_bullet_section(title: str, items: list[Any]) -> list[str]:
        lines = [f"## {title}", ""]
        if not items:
            lines.append("- none")
        else:
            for item in items:
                lines.append(f"- {item}")
        lines.append("")
        return lines

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
        return (
            SlotSchema(
                slot_name=slot.slot_name,
                slot_term=slot.slot_term,
                description=description,
                specific_questions=specific_questions,
                metadata=dict(slot.metadata),
                controlled_vocabulary=extract_controlled_vocabulary(slot.slot_term, description),
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
        return SlotSchema(
            slot_name=slot_name or slot_term,
            slot_term=slot_term or slot_name,
            description=description,
            specific_questions=self._coerce_str_list(item.get("specific_questions")),
            metadata=metadata,
            controlled_vocabulary=extract_controlled_vocabulary(slot_term or slot_name, description),
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
            if normalized_target in keys:
                return index
        return -1

    @staticmethod
    def _task_goal(task: SpawnTask) -> str:
        return f"围绕槽位“{task.slot_name}”处理 {task.reason}，重点回答：{task.prompt_focus}"

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
        if len(memories) > 3:
            del memories[:-3]

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


def _load_perception_layer() -> tuple[Any, Any, Any]:
    if str(_PRECEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(_PRECEPTION_ROOT))
    from perception_layer.config import PipelineConfig as PerceptionConfig  # type: ignore
    from perception_layer.downstream import DownstreamPromptRunner  # type: ignore
    from perception_layer.pipeline import PerceptionPipeline  # type: ignore

    return PerceptionConfig, DownstreamPromptRunner, PerceptionPipeline


def _load_perception_rag_client() -> Any:
    if str(_PRECEPTION_ROOT) not in sys.path:
        sys.path.insert(0, str(_PRECEPTION_ROOT))
    from perception_layer.clients import HttpRagClient  # type: ignore

    return HttpRagClient
