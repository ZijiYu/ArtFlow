from __future__ import annotations

from typing import Any

from .prompt_builder import build_round_table_prompt
from ..cot_layer.models import CrossValidationIssue, CrossValidationResult, CoTThread, DomainCoTRecord, SlotSchema
from ..reflection_layer.service import batch_task_rag_terms


def cross_validate(
    pipeline: Any,
    outputs: list[DomainCoTRecord],
    slot_schemas: list[SlotSchema],
    meta: dict,
    api_logs: list[dict[str, Any]] | None = None,
) -> CrossValidationResult:
    issues: list[CrossValidationIssue] = []
    missing_points: list[str] = []
    rag_terms: list[str] = []
    removed_questions: list[str] = []
    context_text = " ".join([__import__("json").dumps(meta, ensure_ascii=False)] + [schema.description for schema in slot_schemas])
    allowed_dynasties = pipeline._extract_dynasties(context_text)
    question_gap_requests: list[dict[str, Any]] = []

    for output in outputs:
        unanswered = [item for item in output.question_coverage if not item.answered]
        for item in unanswered:
            question_gap_requests.append(
                {
                    "request_id": f"question_gap::{output.slot_name}::{item.question}",
                    "slot_name": output.slot_name,
                    "focus_text": item.question,
                    "fallback_terms": output.controlled_vocabulary,
                    "task_reason": "question_gap",
                }
            )
    question_gap_terms = batch_task_rag_terms(pipeline, question_gap_requests, api_logs=api_logs)

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
            question_rag_terms = question_gap_terms.get(
                f"question_gap::{output.slot_name}::{item.question}",
                [],
            )
            issues.append(
                CrossValidationIssue(
                    issue_type="question_gap",
                    severity="medium",
                    slot_names=[output.slot_name],
                    detail=f"{output.slot_name} 尚未充分回答问题：{item.question}",
                    evidence=[entry.observation for entry in output.visual_anchoring[:2]],
                    rag_terms=question_rag_terms,
                )
            )
            missing_points.append(f"{output.slot_name}: {item.question}")
            rag_terms.extend(question_rag_terms)

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

        mentioned_dynasties = pipeline._extract_dynasties(
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

    semantic_duplicates = pipeline._detect_semantic_duplicates(outputs)
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
        missing_points=pipeline._dedupe_text_list(missing_points),
        rag_terms=pipeline._dedupe_text_list(rag_terms),
        removed_questions=pipeline._dedupe_text_list(removed_questions),
        llm_review="",
    )


def augment_round_table_review(
    pipeline: Any,
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    api_logs: list[dict[str, Any]],
) -> CrossValidationResult:
    if not pipeline.api_client.enabled:
        return validation
    prompt = build_round_table_prompt(
        outputs=outputs,
        validation=validation,
        meta=meta,
        enable_web_search=bool(getattr(pipeline.config, "enable_web_search", False)),
    )
    raw, api_log = pipeline.vlm_runner.analyze(
        image_path=None,
        prompt=prompt,
        system_prompt="你是严格的细节复核员，只识别细节缺口、追问点和需要补检索的信息。",
        temperature=pipeline.config.validation_temperature,
        model=pipeline.config.validation_model or pipeline.config.domain_model,
        stage="round_table_validation",
    )
    api_logs.append(api_log)
    validation.llm_review = raw.strip()
    parsed = parse_round_table_review(pipeline, raw)
    validation.round_table_blind_spots = parsed["blind_spots"]
    validation.round_table_follow_up_questions = parsed["follow_up_questions"]
    validation.round_table_rag_needs = parsed["rag_needs"]
    validation.rag_terms = pipeline._dedupe_text_list(
        validation.rag_terms
        + [query for item in parsed["follow_up_questions"] for query in item.get("rag_queries", [])]
        + [query for item in parsed["rag_needs"] for query in item.get("queries", [])]
    )
    return validation


def parse_round_table_review(pipeline: Any, raw: str) -> dict[str, Any]:
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
                "priority": round_table_priority(item.get("priority")),
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


def round_table_priority(value: object) -> str:
    text = str(value or "").strip().lower()
    return "high" if text == "high" else "medium"
