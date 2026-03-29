from __future__ import annotations

import json
import re
from typing import Any

from .prompt_builder import (
    build_batch_rag_keyword_prompt,
    build_final_answer_request_prompt,
    build_final_appreciation_prompt,
    build_rag_keyword_prompt,
    build_slot_lifecycle_prompt,
    build_validation_review_prompt,
)
from ..cot_layer.config_loader import get_config_value, load_yaml_config
from ..cot_layer.models import CoTThread, CrossValidationResult, DialogueState, DomainCoTRecord, RoutingDecision, SpawnTask


def plan_spawn_tasks(
    pipeline: Any,
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    threads: list[CoTThread],
    api_logs: list[dict[str, Any]] | None = None,
) -> list[SpawnTask]:
    enable_rag_verification = bool(getattr(pipeline.config, "enable_rag_verification", True))
    tasks: list[SpawnTask] = []
    output_by_slot = {output.slot_name: output for output in outputs}
    unanswered_requests: list[dict[str, Any]] = []
    follow_up_review_map = {
        (
            str(item.get("slot_name", "")).strip(),
            str(item.get("question", "")).strip(),
        ): item
        for item in validation.follow_up_task_reviews
        if isinstance(item, dict)
    }
    for output in outputs:
        for item in output.question_coverage:
            if item.answered:
                continue
            unanswered_requests.append(
                {
                    "request_id": f"unanswered::{output.slot_name}::{item.question}",
                    "slot_name": output.slot_name,
                    "focus_text": item.question,
                    "fallback_terms": output.controlled_vocabulary,
                    "task_reason": "specific_question_unanswered",
                }
            )
    unanswered_plans = batch_task_retrieval_plans(pipeline, unanswered_requests, api_logs=api_logs)

    for output in outputs:
        for item in output.question_coverage:
            if item.answered:
                continue
            request_id = f"unanswered::{output.slot_name}::{item.question}"
            retrieval_plan = unanswered_plans.get(
                request_id,
                {
                    "mode": "rag",
                    "rag_queries": [],
                    "web_queries": [],
                    "reason": "",
                },
            )
            retrieval_mode = normalize_retrieval_mode(retrieval_plan.get("mode"))
            dispatch_target = "downstream" if retrieval_mode in {"web", "hybrid"} else "cot"
            tasks.append(
                SpawnTask(
                    slot_name=output.slot_name,
                    reason="specific_question_unanswered",
                    prompt_focus=item.question,
                    rag_terms=retrieval_plan.get("rag_queries", []),
                    retrieval_mode=retrieval_mode,
                    web_queries=retrieval_plan.get("web_queries", []),
                    retrieval_reason=str(retrieval_plan.get("reason", "")).strip(),
                    priority=4,
                    dispatch_target=dispatch_target,
                )
            )
        if enable_rag_verification and _should_promote_retrieval_gain(pipeline, output):
            gain_focus = output.retrieval_gain_focus.strip() or f"继续核验 {output.slot_name} 中与图像相关的新术语。"
            gain_terms = _novel_retrieval_terms(pipeline, output)
            retrieval_plan = _finalize_retrieval_plan(
                pipeline,
                slot_name=output.slot_name,
                focus_text=gain_focus,
                fallback_terms=gain_terms or output.controlled_vocabulary,
                mode=getattr(output, "retrieval_gain_mode", "rag"),
                rag_queries=output.retrieval_gain_queries + gain_terms,
                web_queries=getattr(output, "retrieval_gain_web_queries", []),
                reason=output.retrieval_gain_reason,
            )
            if not retrieval_plan.get("rag_queries") and not retrieval_plan.get("web_queries"):
                retrieval_plan = task_retrieval_plan(
                    pipeline,
                    focus_text=gain_focus,
                    fallback_terms=gain_terms or output.controlled_vocabulary,
                    slot_name=output.slot_name,
                    task_reason="retrieval_gain",
                    api_logs=api_logs,
                )
            retrieval_mode = normalize_retrieval_mode(retrieval_plan.get("mode"))
            tasks.append(
                SpawnTask(
                    slot_name=output.slot_name,
                    reason="retrieval_gain",
                    prompt_focus=gain_focus,
                    rag_terms=retrieval_plan.get("rag_queries", []),
                    retrieval_mode=retrieval_mode,
                    web_queries=retrieval_plan.get("web_queries", []),
                    retrieval_reason=str(retrieval_plan.get("reason", "")).strip(),
                    priority=3,
                    dispatch_target="downstream" if retrieval_mode in {"web", "hybrid"} else "cot",
                )
            )

    round_table_requests: list[dict[str, Any]] = []
    for item in validation.round_table_follow_up_questions:
        if not isinstance(item, dict):
            continue
        if str(item.get("priority", "")).strip().lower() != "high":
            continue
        question = str(item.get("question", "")).strip()
        requested_slot_name = str(item.get("slot_name", "")).strip()
        if not question or not requested_slot_name:
            continue
        review = follow_up_review_map.get((requested_slot_name, question), {})
        action = str(review.get("action", "")).strip().lower()
        if action == "close":
            continue
        slot_name = resolve_follow_up_slot_name(
            pipeline,
            requested_slot_name=requested_slot_name,
            question=question,
            outputs=outputs,
        )
        dispatch_target = "cot"
        if action == "downstream_discovery":
            dispatch_target = "downstream"
            slot_name = requested_slot_name
        elif not slot_name:
            dispatch_target = "downstream"
            slot_name = requested_slot_name
        output = output_by_slot.get(slot_name)
        fallback_terms = output.controlled_vocabulary if output is not None else [slot_name]
        rag_terms = pipeline._normalize_search_queries(item.get("rag_queries", []))
        web_queries = pipeline._normalize_web_search_queries(item.get("web_queries", []))
        retrieval_mode = normalize_retrieval_mode(item.get("retrieval_mode"))
        if not rag_terms and not web_queries:
            round_table_requests.append(
                {
                    "request_id": f"round_table::{requested_slot_name}::{question}",
                    "slot_name": slot_name,
                    "focus_text": question,
                    "fallback_terms": fallback_terms,
                    "task_reason": "round_table_follow_up",
                }
            )
        elif retrieval_mode in {"web", "hybrid"}:
            continue
    round_table_plans = batch_task_retrieval_plans(pipeline, round_table_requests, api_logs=api_logs)

    for item in validation.round_table_follow_up_questions:
        if not isinstance(item, dict):
            continue
        if str(item.get("priority", "")).strip().lower() != "high":
            continue
        question = str(item.get("question", "")).strip()
        requested_slot_name = str(item.get("slot_name", "")).strip()
        if not question or not requested_slot_name:
            continue
        review = follow_up_review_map.get((requested_slot_name, question), {})
        action = str(review.get("action", "")).strip().lower()
        if action == "close":
            continue
        slot_name = resolve_follow_up_slot_name(
            pipeline,
            requested_slot_name=requested_slot_name,
            question=question,
            outputs=outputs,
        )
        dispatch_target = "cot"
        if action == "downstream_discovery":
            dispatch_target = "downstream"
            slot_name = requested_slot_name
        elif not slot_name:
            dispatch_target = "downstream"
            slot_name = requested_slot_name
        output = output_by_slot.get(slot_name)
        fallback_terms = output.controlled_vocabulary if output is not None else [slot_name]
        retrieval_plan = _finalize_retrieval_plan(
            pipeline,
            slot_name=slot_name,
            focus_text=question,
            fallback_terms=fallback_terms,
            mode=item.get("retrieval_mode"),
            rag_queries=item.get("rag_queries", []),
            web_queries=item.get("web_queries", []),
            reason=item.get("retrieval_reason", "") or item.get("why", ""),
        )
        if not retrieval_plan.get("rag_queries") and not retrieval_plan.get("web_queries"):
            retrieval_plan = round_table_plans.get(
                f"round_table::{requested_slot_name}::{question}",
                {
                    "mode": "rag",
                    "rag_queries": [],
                    "web_queries": [],
                    "reason": "",
                },
            )
        retrieval_mode = normalize_retrieval_mode(retrieval_plan.get("mode"))
        if retrieval_mode in {"web", "hybrid"}:
            dispatch_target = "downstream"
        tasks.append(
            SpawnTask(
                slot_name=slot_name,
                reason="round_table_follow_up",
                prompt_focus=question,
                rag_terms=retrieval_plan.get("rag_queries", []),
                retrieval_mode=retrieval_mode,
                web_queries=retrieval_plan.get("web_queries", []),
                retrieval_reason=str(retrieval_plan.get("reason", "")).strip(),
                priority=5,
                dispatch_target=dispatch_target,
                requested_slot_name=requested_slot_name,
            )
        )

    tasks = pipeline._dedupe_spawn_tasks(tasks)
    filtered: list[SpawnTask] = []
    for task in tasks:
        if task_already_resolved(pipeline, task, threads):
            continue
        filtered.append(task)
    filtered.sort(key=lambda item: (-item.priority, item.slot_name, item.prompt_focus))
    dynamic_cap = min(max(1, int(pipeline.config.max_threads_per_round or 1)), max(1, len(filtered)))
    return filtered[:dynamic_cap]


def resolve_follow_up_slot_name(
    pipeline: Any,
    *,
    requested_slot_name: str,
    question: str,
    outputs: list[DomainCoTRecord],
) -> str:
    normalized = requested_slot_name.strip()
    if not normalized or not outputs:
        return ""
    exact = next((output.slot_name for output in outputs if output.slot_name == normalized), "")
    if exact:
        return exact

    best_slot = ""
    best_score = 0.0
    for output in outputs:
        score = follow_up_slot_score(
            pipeline,
            requested_slot_name=normalized,
            question=question,
            output=output,
        )
        if score > best_score:
            best_score = score
            best_slot = output.slot_name
    if best_score >= 0.45:
        return best_slot
    return ""


def review_slot_lifecycle(
    pipeline: Any,
    slot_schemas: list[Any],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    api_logs: list[dict[str, Any]],
    *,
    use_llm: bool = True,
) -> CrossValidationResult:
    if not outputs:
        return validation

    if not pipeline.api_client.enabled or not use_llm:
        validation.slot_lifecycle_reviews = heuristic_slot_lifecycle_reviews(pipeline, slot_schemas, outputs, validation)
        validation.follow_up_task_reviews = heuristic_follow_up_task_reviews(validation)
        return validation

    prompt = build_slot_lifecycle_prompt(
        slot_schemas=slot_schemas,
        outputs=outputs,
        validation=validation,
        meta=meta,
    )
    raw, api_log = pipeline.vlm_runner.analyze(
        image_path=None,
        prompt=prompt,
        system_prompt="你是严格的 slot lifecycle 裁决器，只输出结构化 JSON。",
        temperature=0.0,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
        stage="slot_lifecycle_review",
    )
    api_logs.append(api_log)
    parsed = parse_slot_lifecycle_review(pipeline, raw, validation, slot_schemas=slot_schemas, outputs=outputs)
    validation.slot_lifecycle_reviews = parsed["slot_reviews"]
    validation.follow_up_task_reviews = parsed["follow_up_reviews"]
    return validation


def review_validation_bundle(
    pipeline: Any,
    slot_schemas: list[Any],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    api_logs: list[dict[str, Any]],
    *,
    use_llm: bool = True,
) -> CrossValidationResult:
    if not outputs:
        return validation

    if not pipeline.api_client.enabled or not use_llm:
        validation.slot_lifecycle_reviews = heuristic_slot_lifecycle_reviews(pipeline, slot_schemas, outputs, validation)
        validation.follow_up_task_reviews = heuristic_follow_up_task_reviews(validation)
        return validation

    prompt = build_validation_review_prompt(
        slot_schemas=slot_schemas,
        outputs=outputs,
        validation=validation,
        meta=meta,
        enable_web_search=bool(getattr(pipeline.config, "enable_web_search", False)),
    )
    raw, api_log = pipeline.vlm_runner.analyze(
        image_path=None,
        prompt=prompt,
        system_prompt="你是严格的国画分析复核与路由裁决器，只输出结构化 JSON。",
        temperature=pipeline.config.validation_temperature,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
        stage="validation_review",
    )
    api_logs.append(api_log)
    validation.llm_review = raw.strip()

    parsed_round_table = _parse_round_table_review_payload(pipeline, raw)
    validation.round_table_blind_spots = parsed_round_table["blind_spots"]
    validation.round_table_follow_up_questions = parsed_round_table["follow_up_questions"]
    validation.round_table_rag_needs = parsed_round_table["rag_needs"]
    validation.rag_terms = pipeline._dedupe_text_list(
        validation.rag_terms
        + [query for item in parsed_round_table["follow_up_questions"] for query in item.get("rag_queries", [])]
        + [query for item in parsed_round_table["rag_needs"] for query in item.get("queries", [])]
    )

    parsed_lifecycle = parse_slot_lifecycle_review(
        pipeline,
        raw,
        validation,
        slot_schemas=slot_schemas,
        outputs=outputs,
    )
    validation.slot_lifecycle_reviews = parsed_lifecycle["slot_reviews"]
    validation.follow_up_task_reviews = parsed_lifecycle["follow_up_reviews"]
    return validation


def _parse_round_table_review_payload(pipeline: Any, raw: str) -> dict[str, Any]:
    parsed = pipeline._extract_json_object(str(raw or "").strip())
    if not parsed:
        return {
            "blind_spots": [],
            "follow_up_questions": [],
            "rag_needs": [],
        }

    blind_spots = pipeline._dedupe_text_list(parsed.get("blind_spots", []))
    follow_up_questions: list[dict[str, Any]] = []
    for item in parsed.get("follow_up_questions", []):
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        if not question:
            continue
        follow_up_questions.append(
            {
                "slot_name": str(item.get("slot_name", "")).strip(),
                "question": question,
                "why": str(item.get("why", "")).strip(),
                "priority": "high" if str(item.get("priority", "")).strip().lower() == "high" else "medium",
                "retrieval_mode": str(item.get("retrieval_mode", "")).strip().lower() or "rag",
                "rag_queries": pipeline._normalize_search_queries(item.get("rag_queries", [])),
                "web_queries": pipeline._normalize_web_search_queries(item.get("web_queries", [])),
                "retrieval_reason": str(item.get("retrieval_reason", "")).strip(),
            }
        )

    rag_needs: list[dict[str, Any]] = []
    for item in parsed.get("rag_needs", []):
        if not isinstance(item, dict):
            continue
        topic = str(item.get("topic", "")).strip()
        if not topic:
            continue
        rag_needs.append(
            {
                "topic": topic,
                "reason": str(item.get("reason", "")).strip(),
                "queries": pipeline._normalize_search_queries(item.get("queries", [])),
            }
        )

    return {
        "blind_spots": blind_spots,
        "follow_up_questions": follow_up_questions[:8],
        "rag_needs": rag_needs[:8],
    }


def parse_slot_lifecycle_review(
    pipeline: Any,
    raw: str,
    validation: CrossValidationResult,
    slot_schemas: list[Any] | None = None,
    outputs: list[DomainCoTRecord] | None = None,
) -> dict[str, Any]:
    slot_schemas = slot_schemas or []
    outputs = outputs or []
    parsed = pipeline._extract_json_object(str(raw or "").strip())
    if not parsed:
        return {
            "slot_reviews": heuristic_slot_lifecycle_reviews(pipeline, slot_schemas, outputs, validation),
            "follow_up_reviews": heuristic_follow_up_task_reviews(validation),
        }

    slot_reviews: list[dict[str, Any]] = []
    for item in parsed.get("slot_reviews", []):
        if not isinstance(item, dict):
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        if not slot_name:
            continue
        status = normalize_slot_status(item.get("status"))
        slot_reviews.append(
            {
                "slot_name": slot_name,
                "status": status,
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    follow_up_reviews: list[dict[str, Any]] = []
    for item in parsed.get("follow_up_reviews", []):
        if not isinstance(item, dict):
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        question = str(item.get("question", "")).strip()
        if not slot_name or not question:
            continue
        follow_up_reviews.append(
            {
                "slot_name": slot_name,
                "question": question,
                "action": normalize_follow_up_action(item.get("action")),
                "reason": str(item.get("reason", "")).strip(),
            }
        )

    if not slot_reviews:
        slot_reviews = heuristic_slot_lifecycle_reviews(pipeline, slot_schemas, outputs, validation)
    if not follow_up_reviews:
        follow_up_reviews = heuristic_follow_up_task_reviews(validation)

    return {
        "slot_reviews": slot_reviews[:12],
        "follow_up_reviews": follow_up_reviews[:12],
    }


def heuristic_slot_lifecycle_reviews(
    pipeline: Any,
    slot_schemas: list[Any],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
) -> list[dict[str, Any]]:
    output_map = {output.slot_name: output for output in outputs}
    issue_slots = {
        slot_name
        for issue in validation.issues
        for slot_name in issue.slot_names
        if issue.severity in {"high", "medium"}
    }
    reviews: list[dict[str, Any]] = []
    for slot in slot_schemas:
        output = output_map.get(slot.slot_name)
        unanswered = [item for item in getattr(output, "question_coverage", []) if not item.answered] if output else list(slot.specific_questions)
        status = "ACTIVE"
        reason = "仍存在未覆盖问题或风险。"
        if not unanswered and slot.slot_name not in issue_slots:
            status = "STABLE"
            reason = "当前核心问题已覆盖，暂未发现新的高价值缺口。"
        reviews.append({"slot_name": slot.slot_name, "status": status, "reason": reason})
    return reviews


def _should_promote_retrieval_gain(pipeline: Any, output: DomainCoTRecord) -> bool:
    if not output.retrieval_gain_has_value:
        return False
    if any(not item.answered for item in output.question_coverage):
        return False
    if not (_novel_retrieval_terms(pipeline, output) or output.retrieval_gain_queries):
        return False
    return True


def _novel_retrieval_terms(pipeline: Any, output: DomainCoTRecord) -> list[str]:
    existing_terms = [
        *output.controlled_vocabulary,
        *[item.term for item in output.domain_decoding if item.term],
    ]
    results: list[str] = []
    for term in output.retrieval_gain_terms:
        if not term.strip():
            continue
        if any(pipeline._text_similarity(term, existing) >= 0.82 for existing in existing_terms if existing.strip()):
            continue
        results.append(term.strip())
    return pipeline._dedupe_text_list(results)


def heuristic_follow_up_task_reviews(validation: CrossValidationResult) -> list[dict[str, Any]]:
    reviews: list[dict[str, Any]] = []
    for item in validation.round_table_follow_up_questions:
        if not isinstance(item, dict):
            continue
        slot_name = str(item.get("slot_name", "")).strip()
        question = str(item.get("question", "")).strip()
        if not slot_name or not question:
            continue
        reviews.append(
            {
                "slot_name": slot_name,
                "question": question,
                "action": "cot",
                "reason": "默认保留为已有槽位的继续分析任务。",
            }
        )
    return reviews


def normalize_slot_status(value: object) -> str:
    text = str(value or "").strip().upper()
    if text == "CLOSED":
        return "CLOSED"
    if text == "STABLE":
        return "STABLE"
    return "ACTIVE"


def normalize_follow_up_action(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "downstream_discovery":
        return "downstream_discovery"
    if text == "close":
        return "close"
    return "cot"


def follow_up_slot_score(
    pipeline: Any,
    *,
    requested_slot_name: str,
    question: str,
    output: DomainCoTRecord,
) -> float:
    requested = requested_slot_name.strip()
    focus = question.strip()
    scores = [
        pipeline._text_similarity(requested, output.slot_name),
        pipeline._text_similarity(requested, output.slot_term),
        pipeline._text_similarity(focus, output.slot_name),
        pipeline._text_similarity(focus, output.slot_term),
    ]
    scores.extend(
        pipeline._text_similarity(requested, term)
        for term in getattr(output, "controlled_vocabulary", [])
        if str(term).strip()
    )
    scores.extend(
        pipeline._text_similarity(focus, term)
        for term in getattr(output, "controlled_vocabulary", [])
        if str(term).strip()
    )
    scores.extend(
        pipeline._text_similarity(focus, item.question)
        for item in output.question_coverage
        if item.question
    )
    return max(scores or [0.0])


def task_rag_terms(
    pipeline: Any,
    *,
    focus_text: str,
    fallback_terms: list[str],
    slot_name: str,
    task_reason: str = "",
    api_logs: list[dict[str, Any]] | None = None,
) -> list[str]:
    cache_key = (
        slot_name.strip(),
        focus_text.strip(),
        task_reason.strip(),
    )
    if cache_key in pipeline._rag_term_cache:
        return list(pipeline._rag_term_cache[cache_key])
    if not pipeline.api_client.enabled:
        pipeline._rag_term_cache[cache_key] = []
        return []

    prompt = build_rag_keyword_prompt(
        slot_name=slot_name,
        focus_text=focus_text,
        task_reason=task_reason,
        max_keyword_blocks=max(1, int(getattr(pipeline.config, "rag_query_max_blocks", 2) or 2)),
        enable_web_search=False,
    )
    result = pipeline.api_client.chat(
        system_prompt="你是严谨的中文检索规划器，只输出适合搜索引擎的关键词 JSON。",
        user_prompt=prompt,
        temperature=0.0,
        image_path=None,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
    )
    api_log = {
        "stage": "rag_keyword_planning",
        "ok": bool(result.content),
        "error": result.error,
        "model": result.model,
        "endpoint": result.endpoint,
        "status_code": result.status_code,
        "image_attached": result.image_attached,
        "duration_ms": result.duration_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "slot_name": slot_name,
        "task_reason": task_reason,
        "focus_text": focus_text,
    }
    if api_logs is not None:
        api_logs.append(api_log)
    print(
        f"[slots_v2_request] stage=rag_keyword_planning model={result.model or pipeline.config.validation_model or pipeline.config.domain_model or ''} "
        f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} image=false",
        flush=True,
    )

    terms = parse_rag_terms_response(pipeline, result.content or "", slot_name=slot_name)
    pipeline._rag_term_cache[cache_key] = terms
    return list(terms)


def batch_task_rag_terms(
    pipeline: Any,
    requests: list[dict[str, Any]],
    *,
    api_logs: list[dict[str, Any]] | None = None,
) -> dict[str, list[str]]:
    results: dict[str, list[str]] = {}
    pending: list[tuple[dict[str, Any], tuple[str, str, str]]] = []
    for item in requests:
        request_id = str(item.get("request_id", "")).strip()
        slot_name = str(item.get("slot_name", "")).strip()
        focus_text = str(item.get("focus_text", "")).strip()
        task_reason = str(item.get("task_reason", "")).strip()
        if not request_id or not focus_text:
            continue
        cache_key = (
            slot_name,
            focus_text,
            task_reason,
        )
        if cache_key in pipeline._rag_term_cache:
            results[request_id] = list(pipeline._rag_term_cache[cache_key])
            continue
        if not pipeline.api_client.enabled:
            pipeline._rag_term_cache[cache_key] = []
            results[request_id] = []
            continue
        pending.append((item, cache_key))

    if not pending:
        return results

    if len(pending) == 1:
        item, cache_key = pending[0]
        request_id = str(item.get("request_id", "")).strip()
        results[request_id] = task_rag_terms(
            pipeline,
            focus_text=str(item.get("focus_text", "")).strip(),
            fallback_terms=item.get("fallback_terms", []),
            slot_name=str(item.get("slot_name", "")).strip(),
            task_reason=str(item.get("task_reason", "")).strip(),
            api_logs=api_logs,
        )
        return results

    max_blocks = max(1, int(getattr(pipeline.config, "rag_query_max_blocks", 2) or 2))
    prompt = build_batch_rag_keyword_prompt(
        requests=[
            {
                "request_id": str(item.get("request_id", "")).strip(),
                "slot_name": str(item.get("slot_name", "")).strip(),
                "focus_text": str(item.get("focus_text", "")).strip(),
                "task_reason": str(item.get("task_reason", "")).strip(),
            }
            for item, _ in pending
        ],
        max_keyword_blocks=max_blocks,
        enable_web_search=False,
    )
    result = pipeline.api_client.chat(
        system_prompt="你是严谨的中文检索规划器，只输出适合搜索引擎的批量关键词 JSON。",
        user_prompt=prompt,
        temperature=0.0,
        image_path=None,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
    )
    api_log = {
        "stage": "rag_keyword_planning_batch",
        "ok": bool(result.content),
        "error": result.error,
        "model": result.model,
        "endpoint": result.endpoint,
        "status_code": result.status_code,
        "image_attached": result.image_attached,
        "duration_ms": result.duration_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "batch_size": len(pending),
        "request_ids": [str(item.get("request_id", "")).strip() for item, _ in pending],
    }
    if api_logs is not None:
        api_logs.append(api_log)
    print(
        f"[slots_v2_request] stage=rag_keyword_planning_batch model={result.model or pipeline.config.validation_model or pipeline.config.domain_model or ''} "
        f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} image=false batch={len(pending)}",
        flush=True,
    )

    parsed_items = parse_batch_rag_terms_response(pipeline, result.content or "")
    for item, cache_key in pending:
        request_id = str(item.get("request_id", "")).strip()
        slot_name = str(item.get("slot_name", "")).strip()
        parsed_terms = normalize_search_queries(
            pipeline,
            parsed_items.get(request_id, []),
            slot_name=slot_name,
        )
        pipeline._rag_term_cache[cache_key] = parsed_terms
        results[request_id] = list(parsed_terms)

    return results


def task_retrieval_plan(
    pipeline: Any,
    *,
    focus_text: str,
    fallback_terms: list[str],
    slot_name: str,
    task_reason: str = "",
    api_logs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    cache_key = (
        slot_name.strip(),
        focus_text.strip(),
        task_reason.strip(),
    )
    if not bool(getattr(pipeline.config, "enable_web_search", False)):
        rag_queries = task_rag_terms(
            pipeline,
            focus_text=focus_text,
            fallback_terms=fallback_terms,
            slot_name=slot_name,
            task_reason=task_reason,
            api_logs=api_logs,
        )
        return {
            "mode": "rag",
            "rag_queries": rag_queries,
            "web_queries": [],
            "reason": "",
        }
    if cache_key in getattr(pipeline, "_retrieval_plan_cache", {}):
        cached = getattr(pipeline, "_retrieval_plan_cache", {}).get(cache_key, {})
        return {
            "mode": normalize_retrieval_mode(cached.get("mode")),
            "rag_queries": list(cached.get("rag_queries", [])),
            "web_queries": list(cached.get("web_queries", [])),
            "reason": str(cached.get("reason", "")).strip(),
        }
    if not pipeline.api_client.enabled:
        fallback = {
            "mode": "rag",
            "rag_queries": fallback_rag_terms(pipeline, slot_name=slot_name, fallback_terms=fallback_terms),
            "web_queries": [],
            "reason": "",
        }
        getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = fallback
        return fallback

    prompt = build_rag_keyword_prompt(
        slot_name=slot_name,
        focus_text=focus_text,
        task_reason=task_reason,
        max_keyword_blocks=max(1, int(getattr(pipeline.config, "rag_query_max_blocks", 2) or 2)),
        enable_web_search=True,
    )
    result = pipeline.api_client.chat(
        system_prompt="你是严谨的中文检索路由规划器，只输出结构化 JSON。",
        user_prompt=prompt,
        temperature=0.0,
        image_path=None,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
    )
    api_log = {
        "stage": "retrieval_plan",
        "ok": bool(result.content),
        "error": result.error,
        "model": result.model,
        "endpoint": result.endpoint,
        "status_code": result.status_code,
        "image_attached": result.image_attached,
        "duration_ms": result.duration_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "slot_name": slot_name,
        "task_reason": task_reason,
        "focus_text": focus_text,
    }
    if api_logs is not None:
        api_logs.append(api_log)
    print(
        f"[slots_v2_request] stage=retrieval_plan model={result.model or pipeline.config.validation_model or pipeline.config.domain_model or ''} "
        f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} image=false",
        flush=True,
    )
    plan = parse_retrieval_plan_response(
        pipeline,
        result.content or "",
        slot_name=slot_name,
        focus_text=focus_text,
        fallback_terms=fallback_terms,
    )
    getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = plan
    return {
        "mode": normalize_retrieval_mode(plan.get("mode")),
        "rag_queries": list(plan.get("rag_queries", [])),
        "web_queries": list(plan.get("web_queries", [])),
        "reason": str(plan.get("reason", "")).strip(),
    }


def batch_task_retrieval_plans(
    pipeline: Any,
    requests: list[dict[str, Any]],
    *,
    api_logs: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    results: dict[str, dict[str, Any]] = {}
    pending: list[tuple[dict[str, Any], tuple[str, str, str]]] = []
    for item in requests:
        request_id = str(item.get("request_id", "")).strip()
        slot_name = str(item.get("slot_name", "")).strip()
        focus_text = str(item.get("focus_text", "")).strip()
        task_reason = str(item.get("task_reason", "")).strip()
        if not request_id or not focus_text:
            continue
        cache_key = (
            slot_name,
            focus_text,
            task_reason,
        )
        if cache_key in getattr(pipeline, "_retrieval_plan_cache", {}):
            cached = getattr(pipeline, "_retrieval_plan_cache", {}).get(cache_key, {})
            results[request_id] = {
                "mode": normalize_retrieval_mode(cached.get("mode")),
                "rag_queries": list(cached.get("rag_queries", [])),
                "web_queries": list(cached.get("web_queries", [])),
                "reason": str(cached.get("reason", "")).strip(),
            }
            continue
        pending.append((item, cache_key))

    if not pending:
        return results

    if not bool(getattr(pipeline.config, "enable_web_search", False)):
        rag_results = batch_task_rag_terms(
            pipeline,
            [item for item, _ in pending],
            api_logs=api_logs,
        )
        for item, cache_key in pending:
            request_id = str(item.get("request_id", "")).strip()
            plan = {
                "mode": "rag",
                "rag_queries": list(rag_results.get(request_id, [])),
                "web_queries": [],
                "reason": "",
            }
            getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = plan
            results[request_id] = plan
        return results

    if not pipeline.api_client.enabled:
        for item, cache_key in pending:
            request_id = str(item.get("request_id", "")).strip()
            slot_name = str(item.get("slot_name", "")).strip()
            fallback_terms = item.get("fallback_terms", [])
            plan = {
                "mode": "rag",
                "rag_queries": fallback_rag_terms(pipeline, slot_name=slot_name, fallback_terms=fallback_terms),
                "web_queries": [],
                "reason": "",
            }
            getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = plan
            results[request_id] = plan
        return results

    if len(pending) == 1:
        item, cache_key = pending[0]
        request_id = str(item.get("request_id", "")).strip()
        plan = task_retrieval_plan(
            pipeline,
            focus_text=str(item.get("focus_text", "")).strip(),
            fallback_terms=item.get("fallback_terms", []),
            slot_name=str(item.get("slot_name", "")).strip(),
            task_reason=str(item.get("task_reason", "")).strip(),
            api_logs=api_logs,
        )
        getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = plan
        results[request_id] = plan
        return results

    max_blocks = max(1, int(getattr(pipeline.config, "rag_query_max_blocks", 2) or 2))
    prompt = build_batch_rag_keyword_prompt(
        requests=[
            {
                "request_id": str(item.get("request_id", "")).strip(),
                "slot_name": str(item.get("slot_name", "")).strip(),
                "focus_text": str(item.get("focus_text", "")).strip(),
                "task_reason": str(item.get("task_reason", "")).strip(),
            }
            for item, _ in pending
        ],
        max_keyword_blocks=max_blocks,
        enable_web_search=True,
    )
    result = pipeline.api_client.chat(
        system_prompt="你是严谨的中文检索路由规划器，只输出结构化 JSON。",
        user_prompt=prompt,
        temperature=0.0,
        image_path=None,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
    )
    api_log = {
        "stage": "retrieval_plan_batch",
        "ok": bool(result.content),
        "error": result.error,
        "model": result.model,
        "endpoint": result.endpoint,
        "status_code": result.status_code,
        "image_attached": result.image_attached,
        "duration_ms": result.duration_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "batch_size": len(pending),
        "request_ids": [str(item.get("request_id", "")).strip() for item, _ in pending],
    }
    if api_logs is not None:
        api_logs.append(api_log)
    print(
        f"[slots_v2_request] stage=retrieval_plan_batch model={result.model or pipeline.config.validation_model or pipeline.config.domain_model or ''} "
        f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} image=false batch={len(pending)}",
        flush=True,
    )
    parsed_items = parse_batch_retrieval_plan_response(pipeline, result.content or "")
    for item, cache_key in pending:
        request_id = str(item.get("request_id", "")).strip()
        slot_name = str(item.get("slot_name", "")).strip()
        focus_text = str(item.get("focus_text", "")).strip()
        fallback_terms = item.get("fallback_terms", [])
        parsed_plan = parsed_items.get(request_id, {})
        plan = _finalize_retrieval_plan(
            pipeline,
            slot_name=slot_name,
            focus_text=focus_text,
            fallback_terms=fallback_terms,
            mode=parsed_plan.get("mode"),
            rag_queries=parsed_plan.get("rag_queries", []),
            web_queries=parsed_plan.get("web_queries", []),
            reason=parsed_plan.get("reason", ""),
        )
        getattr(pipeline, "_retrieval_plan_cache", {})[cache_key] = plan
        results[request_id] = plan
    return results


def parse_batch_rag_terms_response(pipeline: Any, raw: str) -> dict[str, list[str]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    parsed = pipeline._extract_json_object(text)
    if not parsed:
        return {}
    results: dict[str, list[str]] = {}
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue
        request_id = str(item.get("request_id", "")).strip()
        if not request_id:
            continue
        queries = item.get("queries", [])
        results[request_id] = queries if isinstance(queries, list) else []
    return results


def parse_batch_retrieval_plan_response(pipeline: Any, raw: str) -> dict[str, dict[str, Any]]:
    text = str(raw or "").strip()
    if not text:
        return {}
    parsed = pipeline._extract_json_object(text)
    if not parsed:
        return {}
    results: dict[str, dict[str, Any]] = {}
    for item in parsed.get("items", []):
        if not isinstance(item, dict):
            continue
        request_id = str(item.get("request_id", "")).strip()
        if not request_id:
            continue
        results[request_id] = {
            "mode": normalize_retrieval_mode(item.get("mode")),
            "rag_queries": item.get("rag_queries", []),
            "web_queries": item.get("web_queries", []),
            "reason": str(item.get("reason", "")).strip(),
        }
    return results


def parse_rag_terms_response(pipeline: Any, raw: str, *, slot_name: str = "") -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return []
    parsed = pipeline._extract_json_object(text)
    if parsed:
        if "queries" not in parsed and "rag_queries" in parsed:
            return normalize_search_queries(pipeline, parsed.get("rag_queries", []), slot_name=slot_name)
        queries = parsed.get("queries", [])
        return normalize_search_queries(pipeline, queries, slot_name=slot_name)
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        loaded = None
    if isinstance(loaded, list):
        return normalize_search_queries(pipeline, loaded, slot_name=slot_name)
    lines = [
        line.strip(" -*\t")
        for line in re.split(r"[\n\r]+", text)
        if line.strip(" -*\t")
    ]
    return normalize_search_queries(pipeline, lines, slot_name=slot_name)


def parse_retrieval_plan_response(
    pipeline: Any,
    raw: str,
    *,
    slot_name: str = "",
    focus_text: str = "",
    fallback_terms: list[str] | None = None,
) -> dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return _finalize_retrieval_plan(
            pipeline,
            slot_name=slot_name,
            focus_text=focus_text,
            fallback_terms=fallback_terms or [],
        )
    parsed = pipeline._extract_json_object(text)
    if not parsed:
        return _finalize_retrieval_plan(
            pipeline,
            slot_name=slot_name,
            focus_text=focus_text,
            fallback_terms=fallback_terms or [],
        )
    return _finalize_retrieval_plan(
        pipeline,
        slot_name=slot_name,
        focus_text=focus_text,
        fallback_terms=fallback_terms or [],
        mode=parsed.get("mode"),
        rag_queries=parsed.get("rag_queries", []),
        web_queries=parsed.get("web_queries", []),
        reason=parsed.get("reason", ""),
    )


def normalize_search_queries(pipeline: Any, items: list[str] | object, *, slot_name: str = "") -> list[str]:
    if not isinstance(items, list):
        return []
    max_blocks = max(1, int(getattr(pipeline.config, "rag_query_max_blocks", 2) or 2))
    results: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = clean_search_query(item, slot_name=slot_name, max_keyword_blocks=max_blocks)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
    return results[:5]


def normalize_web_search_queries(pipeline: Any, items: list[str] | object) -> list[str]:
    if not isinstance(items, list):
        return []
    results: list[str] = []
    seen: set[str] = set()
    for item in items:
        text = clean_web_search_query(item)
        if not text:
            continue
        key = re.sub(r"\s+", " ", text).casefold()
        if key in seen:
            continue
        seen.add(key)
        results.append(text)
    top_k = max(1, int(getattr(pipeline.config, "web_search_top_k", 5) or 5))
    return results[:top_k]


def fallback_rag_terms(pipeline: Any, *, slot_name: str, fallback_terms: list[str]) -> list[str]:
    seed_terms = [str(item).strip() for item in fallback_terms if str(item).strip()]
    if slot_name.strip():
        seed_terms.insert(0, slot_name.strip())
    return normalize_search_queries(pipeline, seed_terms, slot_name=slot_name)


def fallback_web_queries(pipeline: Any, *, focus_text: str, fallback_terms: list[str], slot_name: str) -> list[str]:
    seeds = []
    if focus_text.strip():
        seeds.append(focus_text.strip())
    if slot_name.strip():
        seeds.append(slot_name.strip())
    seeds.extend(str(item).strip() for item in fallback_terms if str(item).strip())
    return normalize_web_search_queries(pipeline, seeds)


def clean_search_query(value: object, *, slot_name: str = "", max_keyword_blocks: int = 2) -> str:
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
    parts = [part.strip(" ,，;；:：。.!！、") for part in re.split(r"[\s/|,，;；]+", text) if part.strip(" ,，;；:：。.!！、")]
    parts = [part for part in parts if part]
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
    if len(compact) < 2 or len(compact) > 24:
        return ""
    if _looks_like_over_descriptive_query(compact):
        return ""
    return compact


def clean_web_search_query(value: object, *, max_keyword_blocks: int = 6) -> str:
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
    compact = " ".join(parts)
    compact = compact.strip(" ,，;；:：。.!！、")
    if len(compact) < 2 or len(compact) > 96:
        return ""
    return compact


def normalize_retrieval_mode(value: object) -> str:
    text = str(value or "").strip().lower()
    if text == "web":
        return "web"
    if text == "hybrid":
        return "hybrid"
    return "rag"


def _finalize_retrieval_plan(
    pipeline: Any,
    *,
    slot_name: str,
    focus_text: str,
    fallback_terms: list[str],
    mode: object = None,
    rag_queries: list[str] | object = None,
    web_queries: list[str] | object = None,
    reason: object = "",
) -> dict[str, Any]:
    finalized_mode = normalize_retrieval_mode(mode)
    rag_results = normalize_search_queries(pipeline, rag_queries or [], slot_name=slot_name)
    web_results = normalize_web_search_queries(pipeline, web_queries or [])
    if finalized_mode in {"rag", "hybrid"} and not rag_results:
        rag_results = fallback_rag_terms(pipeline, slot_name=slot_name, fallback_terms=fallback_terms)
    if finalized_mode in {"web", "hybrid"} and not web_results:
        web_results = fallback_web_queries(
            pipeline,
            focus_text=focus_text,
            fallback_terms=fallback_terms,
            slot_name=slot_name,
        )
    if finalized_mode == "web" and not web_results:
        finalized_mode = "rag"
    if finalized_mode == "hybrid" and not web_results:
        finalized_mode = "rag"
    if finalized_mode in {"rag", "hybrid"} and not rag_results and web_results:
        finalized_mode = "web"
    return {
        "mode": finalized_mode,
        "rag_queries": rag_results[:5],
        "web_queries": web_results[: max(1, int(getattr(pipeline.config, "web_search_top_k", 5) or 5))],
        "reason": str(reason or "").strip(),
    }


def _looks_like_over_descriptive_query(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return True
    if _looks_like_stable_title_or_entity(compact):
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


def _looks_like_stable_title_or_entity(text: str) -> bool:
    compact = str(text or "").strip()
    if not compact:
        return False
    if "《" in compact and "》" in compact:
        return True
    if len(compact) > 18:
        return False
    title_markers = ("图", "画", "卷", "轴", "册", "屏", "页", "赋", "赞", "碑")
    if any(marker in compact for marker in title_markers):
        return True
    return False


def task_already_resolved(pipeline: Any, task: SpawnTask, threads: list[CoTThread]) -> bool:
    matched_thread = find_matching_thread(pipeline, task, threads)
    if matched_thread is not None and matched_thread.status == "ANSWERED":
        return True
    return False


def suppress_redundant_tasks(
    pipeline: Any,
    tasks: list[SpawnTask],
    threads: list[CoTThread],
) -> tuple[list[SpawnTask], list[str]]:
    kept_tasks: list[SpawnTask] = []
    paused_thread_ids: list[str] = []
    for task in tasks:
        matched_thread = find_matching_thread(pipeline, task, threads)
        if matched_thread is None:
            kept_tasks.append(task)
            continue
        if matched_thread.status == "ANSWERED":
            continue
        if should_pause_duplicate_task(matched_thread):
            matched_thread.status = "PAUSED"
            matched_thread.pause_reason = "duplicate_stalled"
            paused_thread_ids.append(matched_thread.thread_id)
            continue
        kept_tasks.append(task)
    return kept_tasks, pipeline._dedupe_text_list(paused_thread_ids)


def find_matching_thread(pipeline: Any, task: SpawnTask, threads: list[CoTThread]) -> CoTThread | None:
    for thread in threads:
        if thread.slot_name != task.slot_name:
            continue
        similarity = pipeline._text_similarity(thread.focus, task.prompt_focus)
        if similarity >= 0.82:
            return thread
        shared_terms = pipeline._shared_task_terms(
            task,
            SpawnTask(
                slot_name=thread.slot_name,
                reason=thread.reason,
                prompt_focus=thread.focus,
                rag_terms=thread.rag_terms,
                priority=thread.priority,
                source_thread_id=thread.thread_id,
            ),
        )
        if len(shared_terms) >= 2 or any(len(term) >= 4 for term in shared_terms):
            return thread
    return None


def should_pause_duplicate_task(thread: CoTThread) -> bool:
    return (
        thread.status != "ANSWERED"
        and thread.attempts >= 1
        and int(thread.latest_new_info_gain) <= 0
        and int(thread.stale_rounds) >= 2
    )


def sync_threads_with_tasks(pipeline: Any, threads: list[CoTThread], tasks: list[SpawnTask]) -> list[str]:
    new_thread_ids: list[str] = []
    for task in tasks:
        if str(getattr(task, "dispatch_target", "cot")).strip().lower() != "cot":
            continue
        matched_thread = find_matching_thread(pipeline, task, threads)

        if matched_thread is not None:
            if matched_thread.status in {"PAUSED", "BLOCKED"} and matched_thread.attempts < matched_thread.max_attempts:
                matched_thread.status = "OPEN"
                matched_thread.pause_reason = ""
            matched_thread.priority = max(matched_thread.priority, task.priority)
            matched_thread.rag_terms = pipeline._dedupe_text_list(matched_thread.rag_terms + task.rag_terms)
            continue
        thread_id = f"{pipeline._slug(task.slot_name)}-{pipeline._slug(task.reason)}-{len(threads) + 1}"
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
                max_attempts=max(1, int(pipeline.config.thread_attempt_limit or 1)),
                parent_thread_id=task.source_thread_id,
            )
        )
        new_thread_ids.append(thread_id)
    return new_thread_ids


def build_routing(
    pipeline: Any,
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    tasks: list[SpawnTask],
    convergence: dict[str, Any],
) -> RoutingDecision:
    stable_slots = pipeline._stable_slots(outputs, validation)
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


def check_convergence(
    pipeline: Any,
    slot_schemas: list[Any],
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    threads: list[CoTThread],
    dialogue_state: DialogueState,
    tasks: list[SpawnTask] | None = None,
) -> dict[str, Any]:
    tasks = tasks or []
    lifecycle_map = {
        str(item.get("slot_name", "")).strip(): normalize_slot_status(item.get("status"))
        for item in validation.slot_lifecycle_reviews
        if isinstance(item, dict)
    }
    active_slot_names = {
        slot.slot_name
        for slot in slot_schemas
        if normalize_slot_status(lifecycle_map.get(slot.slot_name) or slot.metadata.get("lifecycle", "ACTIVE")) == "ACTIVE"
    }
    total_questions = [
        question
        for slot in slot_schemas
        if slot.slot_name in active_slot_names
        for question in slot.specific_questions
    ]
    answered_questions = pipeline._collect_answered_questions(outputs)
    unanswered_questions = [
        question for question in total_questions if not any(pipeline._text_similarity(question, answered) >= 0.8 for answered in answered_questions)
    ]
    unresolved_points = pipeline._collect_unresolved_questions(outputs, validation)
    high_issues = [issue.detail for issue in validation.issues if issue.severity == "high"]
    pending_threads = [
        thread.thread_id
        for thread in threads
        if thread.status in {"OPEN", "RETRY"} and thread.attempts < thread.max_attempts
    ]

    fully_answered = not unanswered_questions
    no_new_cot = not pending_threads
    all_threads_terminal = all(
        thread.status in {"ANSWERED", "PAUSED", "MERGED", "BLOCKED"} or thread.attempts >= thread.max_attempts for thread in threads
    )
    pending_discovery_tasks = [
        {
            "slot_name": task.slot_name,
            "reason": task.reason,
            "focus": task.prompt_focus,
            "dispatch_target": getattr(task, "dispatch_target", "cot"),
        }
        for task in tasks
    ]
    converged = fully_answered and not unresolved_points and not high_issues and no_new_cot and all_threads_terminal and not pending_discovery_tasks

    reason = "尚未收敛。"
    if converged:
        reason = "所有必答问题均已覆盖，未再产生新的 CoT，线程池已经收敛，可输出最终赏析 prompt。"
    elif pending_discovery_tasks:
        reason = "仍存在待处理的 follow-up 或 downstream discovery 任务。"
    elif dialogue_state.no_new_info_rounds >= max(1, int(pipeline.config.convergence_patience or 1)) and no_new_cot:
        reason = "连续多轮没有新增有效信息，当前已进入停滞状态。"
    elif unanswered_questions:
        reason = "仍存在未完成的问题，需要继续补充。"

    return {
        "converged": converged,
        "reason": reason,
        "answered_slots": pipeline._stable_slots(outputs, validation),
        "answered_questions": answered_questions,
        "unanswered_questions": unanswered_questions,
        "unresolved_points": unresolved_points,
        "high_issues": high_issues,
        "pending_threads": pending_threads,
        "pending_tasks": pending_discovery_tasks,
        "no_new_info_rounds": dialogue_state.no_new_info_rounds,
    }


def generate_final_appreciation_prompt(
    pipeline: Any,
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    dialogue_state: DialogueState,
    api_logs: list[dict[str, Any]] | None = None,
) -> str:
    fallback_prompt = build_final_appreciation_prompt(
        outputs=outputs,
        validation=validation,
        meta=meta,
        dialogue_state=dialogue_state,
    )
    if not pipeline.api_client.enabled:
        return fallback_prompt

    question = (
        str(meta.get("final_user_question", "")).strip()
        or str(meta.get("input_text", "")).strip()
        or str(meta.get("latest_user_message", "")).strip()
    )
    if not question:
        config_path = getattr(pipeline.api_client, "config_path", None)
        config = load_yaml_config(config_path) if config_path else load_yaml_config()
        question = str(
            get_config_value(
                config,
                "image",
                "final_question",
                default=get_config_value(config, "image", "initial_question", default=""),
            )
        ).strip()
    if not question:
        return fallback_prompt

    prompt = build_final_answer_request_prompt(
        question=question,
        outputs=outputs,
        validation=validation,
        meta=meta,
        dialogue_state=dialogue_state,
    )
    result = pipeline.api_client.chat(
        system_prompt="你是严谨的中国画赏析写作者。请直接回答用户问题，优先基于现有证据，保持克制，不要编造，也不要遗漏输入中标记为 must_cover 的槽位事实。",
        user_prompt=prompt,
        temperature=0.1,
        image_path=None,
        model=getattr(pipeline.config, "final_answer_model", None)
        or getattr(pipeline.config, "final_prompt_model", None)
        or getattr(pipeline.config, "validation_model", None)
        or getattr(pipeline.config, "domain_model", None),
    )
    api_log = {
        "stage": "final_appreciation_generation",
        "ok": bool(result.content),
        "error": result.error,
        "model": result.model,
        "endpoint": result.endpoint,
        "status_code": result.status_code,
        "image_attached": result.image_attached,
        "duration_ms": result.duration_ms,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "total_tokens": result.total_tokens,
        "question": question,
    }
    if api_logs is not None:
        api_logs.append(api_log)
    print(
        f"[slots_v2_request] stage=final_appreciation_generation model={result.model or getattr(pipeline.config, 'final_answer_model', '') or getattr(pipeline.config, 'final_prompt_model', '') or ''} "
        f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} image=false",
        flush=True,
    )
    content = str(result.content or "").strip()
    if not content:
        return fallback_prompt
    if "## " not in content:
        return "## 赏析\n" + content
    return content
