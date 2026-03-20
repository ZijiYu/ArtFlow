from __future__ import annotations

import json
import math
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .agents import SlotAgentRunner
from .models import PipelineResult
from .new_api_client import NewAPIClient
from .prompt_builder import (
    build_baseline_prompt,
    build_communal_guest1_prompt,
    build_communal_guest_next_prompt,
    build_enhanced_prompt,
    build_slot_pipe_agent_prompt,
    build_slot_pipe_final_prompt,
    build_slot_pipe_layer_judge_prompt,
    build_slot_pipe_slot_judge_prompt,
    build_slot_pipe_v4_checker_prompt,
    build_slot_pipe_v4_content_prompt,
    build_slot_pipe_v4_expression_summary_prompt,
    build_slot_pipe_v4_final_prompt,
    build_slot_pipe_v4_guest_prompt,
    build_slot_pipe_v4_reviewer_prompt,
    build_solitary_first_prompt,
    build_solitary_reflection_prompt,
)
from .slots import normalize_slots, slot_label
from .token_tracker import TokenTracker
from .vlm_runner import VLMRunner


@dataclass(slots=True)
class PipelineConfig:
    mode: str = "slot"
    slots: list[str] = field(default_factory=list)
    agent_temperature: float = 0.7
    vlm_temperature: float = 0.2
    solitary_model: str | None = None
    solitary_rounds: int = 3
    guest_num: int = 3
    guest_model: str | None = None
    judge_model: str | None = None
    slot_pipe_agents_per_slot: int = 2
    slot_pipe_max_retries: int = 3
    slot_pipe_version: int = 4
    slot_pipe_slots_file: str | None = "artifacts/slots.jsonl"
    slot_pipe_content_agents: int = 2
    slot_pipe_expression_guests: int = 3
    agent_model: str | None = None
    embedding_model: str | None = None
    baseline_model: str | None = None
    enhanced_model: str | None = None
    final_appreciation_model: str | None = None
    checker_model: str | None = None
    reviewer_model: str | None = None
    content_model: str | None = None
    expression_guest_model: str | None = None
    expression_summary_model: str | None = None


class TcpPromptPipeline:
    """Simplified v1 pipeline: one-round agents + two prompts + two VLM outputs."""

    def __init__(self, config: PipelineConfig | None = None, api_client: NewAPIClient | None = None) -> None:
        self.config = config or PipelineConfig()
        self.api_client = api_client or NewAPIClient()
        self.tracker = TokenTracker()
        self._tracker_lock = threading.Lock()
        self._embedding_logs_lock = threading.Lock()
        self._embedding_api_logs: list[dict] = []
        self.agent_runner = SlotAgentRunner(
            self.api_client,
            temperature=self.config.agent_temperature,
            agent_model=self.config.agent_model,
        )
        self.vlm_runner = VLMRunner(self.api_client)

    def run(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        mode = (self.config.mode or "slot").lower()
        if mode == "solitary":
            return self._run_solitary(image_path=image_path, meta=meta)
        if mode == "communal":
            return self._run_communal(image_path=image_path, meta=meta)
        if mode == "slot_pipe":
            return self._run_slot_pipe(image_path=image_path, meta=meta)
        return self._run_slot(image_path=image_path, meta=meta)

    def _run_slot(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        meta = meta or {}
        selected_slots = normalize_slots(self.config.slots)
        logs: list[dict] = []

        # v1 keeps the stage placeholder for later visual extractor module extension.
        visual_stub = {"image_path": image_path, "meta": meta, "status": "not_implemented_v1"}
        self.tracker.add_text("visual_stub", json.dumps(visual_stub, ensure_ascii=False))

        outputs, agent_api_logs = self.agent_runner.run_parallel(image_path=image_path, meta=meta, slots=selected_slots)
        slot_context: dict[str, str] = {}
        api_logs: list[dict] = []
        api_logs.extend(agent_api_logs)
        for api_log in agent_api_logs:
            self.tracker.add_api_usage("slot_agent", int(api_log.get("total_tokens", 0) or 0))

        for output in outputs:
            slot_context[output.agent_name] = output.context_text
            self.tracker.add_text("slot_context", output.context_text)

        baseline_prompt = build_baseline_prompt(image_path=image_path, meta=meta, slots=selected_slots)
        self.tracker.add_text("baseline_prompt", baseline_prompt)

        enhanced_prompt = build_enhanced_prompt(image_path=image_path, meta=meta, slot_context=slot_context)
        self.tracker.add_text("enhanced_prompt", enhanced_prompt)

        baseline_analysis, baseline_api_log = self.vlm_runner.analyze(
            image_path=image_path,
            prompt=baseline_prompt,
            temperature=self.config.vlm_temperature,
            model=self.config.baseline_model,
            inference_kind="baseline",
        )
        api_logs.append(baseline_api_log)
        self.tracker.add_api_usage("vlm_baseline", int(baseline_api_log.get("total_tokens", 0) or 0))
        self.tracker.add_text("baseline_analysis", baseline_analysis)

        enhanced_analysis, enhanced_api_log = self.vlm_runner.analyze(
            image_path=image_path,
            prompt=enhanced_prompt,
            temperature=self.config.vlm_temperature,
            model=self.config.enhanced_model,
            inference_kind="enhanced",
        )
        api_logs.append(enhanced_api_log)
        self.tracker.add_api_usage("vlm_enhanced", int(enhanced_api_log.get("total_tokens", 0) or 0))
        self.tracker.add_text("enhanced_analysis", enhanced_analysis)

        logs.append(
            {
                "selected_slots": selected_slots,
                "api_enabled": self.api_client.enabled,
                "api_failures": [x for x in api_logs if not x.get("ok")],
                "token_usage": self.tracker.snapshot(),
            }
        )

        return PipelineResult(
            mode="slot",
            selected_slots=selected_slots,
            slot_context=slot_context,
            baseline_prompt=baseline_prompt,
            enhanced_prompt=enhanced_prompt,
            baseline_analysis=baseline_analysis,
            enhanced_analysis=enhanced_analysis,
            solitary_rounds=[],
            communal_rounds=[],
            slot_pipe_layers=[],
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
        )

    def _run_solitary(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        meta = meta or {}
        selected_slots = normalize_slots(self.config.slots)
        logs: list[dict] = []
        api_logs: list[dict] = []
        solitary_rounds: list[dict[str, str]] = []
        solitary_model = self.config.solitary_model or self.config.baseline_model or self.config.enhanced_model
        total_rounds = max(1, int(self.config.solitary_rounds or 1))

        previous_analysis = ""
        for round_index in range(1, total_rounds + 1):
            if round_index == 1:
                round_prompt = build_solitary_first_prompt()
            else:
                previous_round_analysis_only = self._extract_solitary_analysis_for_next_round(previous_analysis)
                previous_for_reflection = self._sanitize_image_unavailable_text(previous_round_analysis_only)
                round_prompt = build_solitary_reflection_prompt(
                    previous_analysis=previous_for_reflection,
                    round_index=round_index,
                )

            self.tracker.add_text(f"solitary_round{round_index}_prompt", round_prompt)
            round_analysis, round_api_log = self.vlm_runner.analyze(
                image_path=image_path,
                prompt=round_prompt,
                temperature=self.config.vlm_temperature,
                model=solitary_model,
                inference_kind=f"solitary_round_{round_index}",
            )
            api_logs.append(round_api_log)
            self.tracker.add_api_usage(f"solitary_round_{round_index}", int(round_api_log.get("total_tokens", 0) or 0))
            self.tracker.add_text(f"solitary_round{round_index}_analysis", round_analysis)
            solitary_rounds.append({"round": str(round_index), "prompt": round_prompt, "analysis": round_analysis})
            previous_analysis = round_analysis

        first_prompt = solitary_rounds[0]["prompt"] if solitary_rounds else ""
        first_analysis = solitary_rounds[0]["analysis"] if solitary_rounds else ""
        final_prompt = solitary_rounds[-1]["prompt"] if solitary_rounds else ""
        final_analysis = solitary_rounds[-1]["analysis"] if solitary_rounds else ""

        logs.append(
            {
                "selected_slots": selected_slots,
                "api_enabled": self.api_client.enabled,
                "api_failures": [x for x in api_logs if not x.get("ok")],
                "token_usage": self.tracker.snapshot(),
            }
        )

        return PipelineResult(
            mode="solitary",
            selected_slots=selected_slots,
            slot_context={},
            baseline_prompt=first_prompt,
            enhanced_prompt=final_prompt,
            baseline_analysis=first_analysis,
            enhanced_analysis=final_analysis,
            solitary_rounds=solitary_rounds,
            communal_rounds=[],
            slot_pipe_layers=[],
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
        )

    def _run_communal(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        _ = meta or {}
        selected_slots: list[str] = []
        logs: list[dict] = []
        api_logs: list[dict] = []
        communal_rounds: list[dict[str, str]] = []

        guest_num = max(1, int(self.config.guest_num or 1))
        guest_model = self.config.guest_model or self.config.baseline_model or self.config.enhanced_model
        previous_guest_texts: list[str] = []

        for guest_index in range(1, guest_num + 1):
            if guest_index == 1:
                guest_prompt = build_communal_guest1_prompt()
            else:
                guest_prompt = build_communal_guest_next_prompt(
                    previous_guest_texts=previous_guest_texts,
                    guest_index=guest_index,
                )

            self.tracker.add_text(f"communal_guest_{guest_index}_prompt", guest_prompt)

            result = self.api_client.chat(
                system_prompt="你是群赏中的客人，请基于图像并结合前序观点，给出个人化且有证据的鉴赏。",
                user_prompt=guest_prompt,
                temperature=self.config.vlm_temperature,
                image_path=image_path,
                model=guest_model,
            )
            guest_analysis = (result.content or "").strip() or "API不可用或调用失败，未获得本轮客人鉴赏。"

            api_log = {
                "stage": "communal_guest",
                "guest_index": guest_index,
                "slot": None,
                "inference_kind": f"communal_guest_{guest_index}",
                "model": result.model,
                "ok": bool(result.content),
                "error": result.error,
                "image_attached": result.image_attached,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
                "status_code": result.status_code,
                "endpoint": result.endpoint,
                "request_summary": result.request_summary,
                "response_summary": result.response_summary,
                "prompt_text": guest_prompt,
                "response_text": guest_analysis,
            }
            api_logs.append(api_log)
            self.tracker.add_api_usage(f"communal_guest_{guest_index}", int(result.total_tokens or 0))
            self.tracker.add_text(f"communal_guest_{guest_index}_analysis", guest_analysis)

            communal_rounds.append(
                {
                    "round": str(guest_index),
                    "guest": str(guest_index),
                    "prompt": guest_prompt,
                    "analysis": guest_analysis,
                }
            )
            previous_guest_texts.append(guest_analysis)

        final_analysis = "\n\n".join(
            f"### 客人{i + 1}\n{text}" for i, text in enumerate(previous_guest_texts)
        ).strip()
        first_prompt = communal_rounds[0]["prompt"] if communal_rounds else ""
        first_analysis = communal_rounds[0]["analysis"] if communal_rounds else ""

        logs.append(
            {
                "mode": "communal",
                "selected_slots": selected_slots,
                "slot_constraint_disabled": True,
                "guest_num": guest_num,
                "guest_model": guest_model,
                "api_enabled": self.api_client.enabled,
                "api_failures": [x for x in api_logs if not x.get("ok")],
                "token_usage": self.tracker.snapshot(),
            }
        )

        return PipelineResult(
            mode="communal",
            selected_slots=selected_slots,
            slot_context={},
            baseline_prompt=first_prompt,
            enhanced_prompt=final_analysis,
            baseline_analysis=first_analysis,
            enhanced_analysis=final_analysis,
            solitary_rounds=[],
            communal_rounds=communal_rounds,
            slot_pipe_layers=[],
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
        )

    def _run_slot_pipe_v4(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        meta = meta or {}
        slots_data = self._load_slot_pipe_v4_slots()
        if self.config.slots:
            wanted = {str(x).strip() for x in self.config.slots if str(x).strip()}
            slots_data = [x for x in slots_data if str(x.get("slot_name", "")).strip() in wanted]

        checker_model = self.config.checker_model or self.config.judge_model or self.config.agent_model
        reviewer_model = self.config.reviewer_model or checker_model
        content_model = self.config.content_model or self.config.agent_model or checker_model
        guest_model = self.config.expression_guest_model or self.config.agent_model or content_model
        summary_model = self.config.expression_summary_model or self.config.judge_model or guest_model
        final_appreciation_model = (
            self.config.final_appreciation_model
            or self.config.enhanced_model
            or self.config.baseline_model
            or guest_model
        )

        content_agents = max(1, int(self.config.slot_pipe_content_agents or 1))
        guest_count = max(1, int(self.config.slot_pipe_expression_guests or 1))
        max_workers = max(1, min(8, len(slots_data) if slots_data else 1))

        api_logs: list[dict] = []
        logs: list[dict] = []
        slot_pipe_layers: list[dict] = []
        slot_order = [str(x.get("slot_name", "")).strip() for x in slots_data if str(x.get("slot_name", "")).strip()]
        slot_rank = {name: idx for idx, name in enumerate(slot_order)}

        # Layer 1: 要素确认（checker -> reviewer）
        layer1: dict = {
            "layer_index": 1,
            "layer_name": "要素确认",
            "layer_goal": "核验slot是否真实出现并值得进入鉴赏。",
            "slots_before": [str(x.get("slot_name", "")).strip() for x in slots_data],
            "slots": [],
        }
        verify_by_slot: dict[str, dict] = {}
        slot_data_by_name: dict[str, dict] = {
            str(x.get("slot_name", "")).strip(): x
            for x in slots_data
            if str(x.get("slot_name", "")).strip()
        }
        verified_slots: list[dict] = []

        def _as_score_0_5(value: object) -> float:
            try:
                num = float(value)
            except Exception:
                return 0.0
            if num < 0:
                return 0.0
            if num > 5:
                return 5.0
            return round(num, 2)

        def _run_layer1_slot(slot: dict) -> tuple[str, dict, dict, list[dict], bool]:
            slot_name = str(slot.get("slot_name", "")).strip()
            if not slot_name:
                return "", {}, {}, [], False
            checker_prompt = build_slot_pipe_v4_checker_prompt(slot=slot, meta=meta)
            checker_result = self.api_client.chat(
                system_prompt="你是国画要素核验checker，请只返回JSON。",
                user_prompt=checker_prompt,
                temperature=0.1,
                image_path=image_path,
                model=checker_model,
            )
            checker_raw = (checker_result.content or "").strip()
            checker_json = self._parse_json_object(checker_raw) or {}

            checker_log = {
                "stage": "slot_pipe_v4_layer1_checker",
                "layer_index": 1,
                "layer_name": "要素确认",
                "slot": slot_name,
                "agent_index": 1,
                "attempt_index": 1,
                "inference_kind": f"slot_pipe_v4_l1_{self._slug(slot_name)}_checker",
                "model": checker_result.model,
                "ok": bool(checker_result.content),
                "error": checker_result.error,
                "image_attached": checker_result.image_attached,
                "prompt_tokens": checker_result.prompt_tokens,
                "completion_tokens": checker_result.completion_tokens,
                "total_tokens": checker_result.total_tokens,
                "status_code": checker_result.status_code,
                "endpoint": checker_result.endpoint,
                "request_summary": checker_result.request_summary,
                "response_summary": checker_result.response_summary,
                "prompt_text": checker_prompt,
                "response_text": checker_raw,
                "parsed": checker_json,
            }
            self._track_text(f"slot_pipe_v4_l1_{slot_name}_checker_prompt", checker_prompt)
            self._track_text(f"slot_pipe_v4_l1_{slot_name}_checker_analysis", checker_raw)
            self._track_api_usage("slot_pipe_v4_layer1_checker", int(checker_result.total_tokens or 0))

            reviewer_prompt = build_slot_pipe_v4_reviewer_prompt(slot=slot, checker_result=checker_json, meta=meta)
            reviewer_result = self.api_client.chat(
                system_prompt="你是国画领域专家reviewer，请只返回JSON。",
                user_prompt=reviewer_prompt,
                temperature=0.1,
                image_path=image_path,
                model=reviewer_model,
            )
            reviewer_raw = (reviewer_result.content or "").strip()
            reviewer_json = self._parse_json_object(reviewer_raw) or {}

            reviewer_log = {
                "stage": "slot_pipe_v4_layer1_reviewer",
                "layer_index": 1,
                "layer_name": "要素确认",
                "slot": slot_name,
                "agent_index": 2,
                "attempt_index": 1,
                "inference_kind": f"slot_pipe_v4_l1_{self._slug(slot_name)}_reviewer",
                "model": reviewer_result.model,
                "ok": bool(reviewer_result.content),
                "error": reviewer_result.error,
                "image_attached": reviewer_result.image_attached,
                "prompt_tokens": reviewer_result.prompt_tokens,
                "completion_tokens": reviewer_result.completion_tokens,
                "total_tokens": reviewer_result.total_tokens,
                "status_code": reviewer_result.status_code,
                "endpoint": reviewer_result.endpoint,
                "request_summary": reviewer_result.request_summary,
                "response_summary": reviewer_result.response_summary,
                "prompt_text": reviewer_prompt,
                "response_text": reviewer_raw,
                "parsed": reviewer_json,
            }
            self._track_text(f"slot_pipe_v4_l1_{slot_name}_reviewer_prompt", reviewer_prompt)
            self._track_text(f"slot_pipe_v4_l1_{slot_name}_reviewer_analysis", reviewer_raw)
            self._track_api_usage("slot_pipe_v4_layer1_reviewer", int(reviewer_result.total_tokens or 0))

            checker_score = _as_score_0_5(checker_json.get("score"))
            checker_reason = str(checker_json.get("reason", "")).strip()
            reviewer_confidence = _as_score_0_5(reviewer_json.get("confidence"))
            final_reason = str(reviewer_json.get("reason", "")).strip() or checker_reason
            final_verified = bool(checker_score > 0 or reviewer_confidence > 0)

            verify_result: dict = {
                "score": checker_score,
                "confidence": reviewer_confidence,
                "reason": final_reason,
                "checker": checker_json,
                "reviewer": reviewer_json,
            }

            layer_slot_record = {
                "slot": slot_name,
                "agent_attempts": [checker_log, reviewer_log],
                "judge_rounds": [],
                "final_points": [
                    {
                        "point": f"score={checker_score}; confidence={reviewer_confidence}; reason={final_reason}",
                        "sources": [1, 2],
                        "support": 2,
                    }
                ],
                "verify_result": verify_result,
            }
            return slot_name, verify_result, layer_slot_record, [checker_log, reviewer_log], final_verified

        layer1_results: list[tuple[str, dict, dict, list[dict], bool]] = []
        layer1_slots = [x for x in slots_data if str(x.get("slot_name", "")).strip()]
        if layer1_slots:
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_run_layer1_slot, slot) for slot in layer1_slots]
                for future in as_completed(futures):
                    layer1_results.append(future.result())
        layer1_results.sort(key=lambda x: slot_rank.get(x[0], 10**9))
        for slot_name, verify_result, layer_slot_record, slot_api_logs, final_verified in layer1_results:
            verify_by_slot[slot_name] = verify_result
            layer1["slots"].append(layer_slot_record)
            api_logs.extend(slot_api_logs)
            if final_verified and slot_name in slot_data_by_name:
                verified_slots.append(slot_data_by_name[slot_name])

        # v4 keeps the slot set stable across layers; verification is a signal, not a filter.
        layer1["slots_after"] = [str(x.get("slot_name", "")).strip() for x in slots_data]
        slot_pipe_layers.append(layer1)

        # Layer 2: 内容要点与问题生成
        layer2: dict = {
            "layer_index": 2,
            "layer_name": "内容要点",
            "layer_goal": "为每个slot提炼内容要点并转为问题。",
            "slots_before": [str(x.get("slot_name", "")).strip() for x in slots_data],
            "slots": [],
        }
        content_by_slot: dict[str, dict] = {}

        def _run_layer2_slot(slot: dict) -> tuple[str, dict, dict, list[dict]]:
            slot_name = str(slot.get("slot_name", "")).strip()
            if not slot_name:
                return "", {}, {}, []
            slot_attempts: list[dict] = []
            merged_points: list[str] = []
            merged_questions: list[str] = []

            for agent_index in range(1, content_agents + 1):
                prompt = build_slot_pipe_v4_content_prompt(
                    slot=slot,
                    verify_result=verify_by_slot.get(slot_name, {}),
                    meta=meta,
                    agent_index=agent_index,
                )
                result = self.api_client.chat(
                    system_prompt="你是内容要点层agent，请只返回JSON。",
                    user_prompt=prompt,
                    temperature=self.config.agent_temperature,
                    image_path=image_path,
                    model=content_model,
                )
                raw = (result.content or "").strip()
                parsed = self._parse_json_object(raw) or {}

                points = parsed.get("content_points") if isinstance(parsed.get("content_points"), list) else []
                questions = parsed.get("questions") if isinstance(parsed.get("questions"), list) else []

                for p in points:
                    p_text = str(p).strip()
                    if p_text:
                        merged_points.append(p_text)
                for q in questions:
                    q_text = str(q).strip()
                    if q_text:
                        merged_questions.append(q_text)

                attempt_log = {
                    "stage": "slot_pipe_v4_layer2_content",
                    "layer_index": 2,
                    "layer_name": "内容要点",
                    "slot": slot_name,
                    "agent_index": agent_index,
                    "attempt_index": 1,
                    "inference_kind": f"slot_pipe_v4_l2_{self._slug(slot_name)}_a{agent_index}",
                    "model": result.model,
                    "ok": bool(result.content),
                    "error": result.error,
                    "image_attached": result.image_attached,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "status_code": result.status_code,
                    "endpoint": result.endpoint,
                    "request_summary": result.request_summary,
                    "response_summary": result.response_summary,
                    "prompt_text": prompt,
                    "response_text": raw,
                    "parsed": parsed,
                }
                slot_attempts.append(attempt_log)
                self._track_text(f"slot_pipe_v4_l2_{slot_name}_a{agent_index}_prompt", prompt)
                self._track_text(f"slot_pipe_v4_l2_{slot_name}_a{agent_index}_analysis", raw)
                self._track_api_usage("slot_pipe_v4_layer2_content", int(result.total_tokens or 0))

            default_questions = slot.get("specific_questions") if isinstance(slot.get("specific_questions"), list) else []
            for q in default_questions:
                q_text = str(q).strip()
                if q_text:
                    merged_questions.append(q_text)

            dedup_points = self._dedupe_texts(merged_points, limit=8)
            dedup_questions = self._dedupe_texts(merged_questions, limit=6)
            if not dedup_questions and dedup_points:
                dedup_questions = [f"{p}如何在画面中成立？" for p in dedup_points[:3]]

            slot_content = {
                "content_points": dedup_points,
                "questions": dedup_questions,
            }

            layer_slot_record = {
                "slot": slot_name,
                "agent_attempts": slot_attempts,
                "judge_rounds": [],
                "final_points": [
                    {"point": q, "sources": [x + 1 for x in range(len(slot_attempts))], "support": 1}
                    for q in dedup_questions
                ],
                "content_points": dedup_points,
                "questions": dedup_questions,
            }
            return slot_name, slot_content, layer_slot_record, slot_attempts

        layer2_results: list[tuple[str, dict, dict, list[dict]]] = []
        if slots_data:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(slots_data)))) as pool:
                futures = [pool.submit(_run_layer2_slot, slot) for slot in slots_data]
                for future in as_completed(futures):
                    layer2_results.append(future.result())
        layer2_results.sort(key=lambda x: slot_rank.get(x[0], 10**9))
        for slot_name, slot_content, layer_slot_record, slot_api_logs in layer2_results:
            if not slot_name:
                continue
            content_by_slot[slot_name] = slot_content
            layer2["slots"].append(layer_slot_record)
            api_logs.extend(slot_api_logs)

        layer2["slots_after"] = list(content_by_slot.keys())
        slot_pipe_layers.append(layer2)

        # Layer 3: 群赏表达（按问题串行客人，再汇总）
        layer3: dict = {
            "layer_index": 3,
            "layer_name": "鉴赏表达",
            "layer_goal": "围绕问题做群赏回答与心得池化，形成可复用表达提示。",
            "slots_before": [str(x.get("slot_name", "")).strip() for x in slots_data],
            "slots": [],
        }
        final_slots: list[dict] = []

        def _run_layer3_slot(slot: dict) -> tuple[str, dict, dict, list[dict]]:
            slot_name = str(slot.get("slot_name", "")).strip()
            if slot_name not in content_by_slot:
                return "", {}, {}, []
            questions = content_by_slot.get(slot_name, {}).get("questions") or []
            slot_attempts: list[dict] = []
            question_results: list[dict] = []
            expression_points: list[str] = []

            for q_idx, question in enumerate(questions, start=1):
                guest_rounds: list[dict] = []
                for guest_index in range(1, guest_count + 1):
                    guest_prompt = build_slot_pipe_v4_guest_prompt(
                        slot=slot,
                        question=question,
                        previous_guest_rounds=guest_rounds,
                        guest_index=guest_index,
                    )
                    guest_result = self.api_client.chat(
                        system_prompt="你是群赏客人，请只返回JSON。",
                        user_prompt=guest_prompt,
                        temperature=self.config.vlm_temperature,
                        image_path=image_path,
                        model=guest_model,
                    )
                    guest_raw = (guest_result.content or "").strip()
                    guest_json = self._parse_json_object(guest_raw) or {}
                    answer = str(guest_json.get("answer", "")).strip()
                    insight = str(guest_json.get("insight", "")).strip()

                    round_payload = {
                        "guest_index": guest_index,
                        "answer": answer,
                        "insight": insight,
                    }
                    guest_rounds.append(round_payload)

                    guest_log = {
                        "stage": "slot_pipe_v4_layer3_guest",
                        "layer_index": 3,
                        "layer_name": "鉴赏表达",
                        "slot": slot_name,
                        "question_index": q_idx,
                        "question": question,
                        "agent_index": guest_index,
                        "attempt_index": 1,
                        "inference_kind": f"slot_pipe_v4_l3_{self._slug(slot_name)}_q{q_idx}_g{guest_index}",
                        "model": guest_result.model,
                        "ok": bool(guest_result.content),
                        "error": guest_result.error,
                        "image_attached": guest_result.image_attached,
                        "prompt_tokens": guest_result.prompt_tokens,
                        "completion_tokens": guest_result.completion_tokens,
                        "total_tokens": guest_result.total_tokens,
                        "status_code": guest_result.status_code,
                        "endpoint": guest_result.endpoint,
                        "request_summary": guest_result.request_summary,
                        "response_summary": guest_result.response_summary,
                        "prompt_text": guest_prompt,
                        "response_text": guest_raw,
                        "parsed": guest_json,
                    }
                    slot_attempts.append(guest_log)
                    self._track_text(f"slot_pipe_v4_l3_{slot_name}_q{q_idx}_g{guest_index}_prompt", guest_prompt)
                    self._track_text(f"slot_pipe_v4_l3_{slot_name}_q{q_idx}_g{guest_index}_analysis", guest_raw)
                    self._track_api_usage("slot_pipe_v4_layer3_guest", int(guest_result.total_tokens or 0))

                summary_prompt = build_slot_pipe_v4_expression_summary_prompt(
                    slot=slot,
                    question=question,
                    guest_rounds=guest_rounds,
                )
                summary_result = self.api_client.chat(
                    system_prompt="你是群赏汇总者，请只返回JSON。",
                    user_prompt=summary_prompt,
                    temperature=0.1,
                    image_path=image_path,
                    model=summary_model,
                )
                summary_raw = (summary_result.content or "").strip()
                summary_json = self._parse_json_object(summary_raw) or {}
                pooled_answer = str(summary_json.get("answer", "")).strip()
                insights = self._dedupe_texts(summary_json.get("insights") if isinstance(summary_json.get("insights"), list) else [], limit=5)
                tips = self._dedupe_texts(summary_json.get("tips") if isinstance(summary_json.get("tips"), list) else [], limit=5)
                expression_points.extend(tips)

                summary_log = {
                    "stage": "slot_pipe_v4_layer3_summary",
                    "layer_index": 3,
                    "layer_name": "鉴赏表达",
                    "slot": slot_name,
                    "question_index": q_idx,
                    "question": question,
                    "agent_index": guest_count + 1,
                    "attempt_index": 1,
                    "inference_kind": f"slot_pipe_v4_l3_{self._slug(slot_name)}_q{q_idx}_summary",
                    "model": summary_result.model,
                    "ok": bool(summary_result.content),
                    "error": summary_result.error,
                    "image_attached": summary_result.image_attached,
                    "prompt_tokens": summary_result.prompt_tokens,
                    "completion_tokens": summary_result.completion_tokens,
                    "total_tokens": summary_result.total_tokens,
                    "status_code": summary_result.status_code,
                    "endpoint": summary_result.endpoint,
                    "request_summary": summary_result.request_summary,
                    "response_summary": summary_result.response_summary,
                    "prompt_text": summary_prompt,
                    "response_text": summary_raw,
                    "parsed": summary_json,
                }
                slot_attempts.append(summary_log)
                self._track_text(f"slot_pipe_v4_l3_{slot_name}_q{q_idx}_summary_prompt", summary_prompt)
                self._track_text(f"slot_pipe_v4_l3_{slot_name}_q{q_idx}_summary_analysis", summary_raw)
                self._track_api_usage("slot_pipe_v4_layer3_summary", int(summary_result.total_tokens or 0))

                question_results.append(
                    {
                        "question": question,
                        "pooled_answer": pooled_answer,
                        "insights": insights,
                        "tips": tips,
                        "guest_rounds": guest_rounds,
                    }
                )

            verify_info = verify_by_slot.get(slot_name, {})
            slot_expression_points = self._dedupe_texts(expression_points, limit=8)
            final_slot_item = {
                "slot_name": slot_name,
                "slot_term": str(slot.get("slot_term", "")).strip(),
                "slot_score": float(verify_info.get("score", 0.0) or 0.0),
                "slot_confidence": float(verify_info.get("confidence", 0.0) or 0.0),
                "questions": [str(x).strip() for x in (content_by_slot.get(slot_name, {}).get("questions", []) or []) if str(x).strip()],
                "expression_points": slot_expression_points,
            }

            layer_slot_record = {
                "slot": slot_name,
                "agent_attempts": slot_attempts,
                "judge_rounds": [],
                "final_points": [
                    {"point": t, "sources": [1], "support": 1}
                    for t in slot_expression_points
                ],
                "question_results": question_results,
            }
            return slot_name, final_slot_item, layer_slot_record, slot_attempts

        layer3_results: list[tuple[str, dict, dict, list[dict]]] = []
        if slots_data:
            with ThreadPoolExecutor(max_workers=max(1, min(max_workers, len(slots_data)))) as pool:
                futures = [pool.submit(_run_layer3_slot, slot) for slot in slots_data]
                for future in as_completed(futures):
                    layer3_results.append(future.result())
        layer3_results.sort(key=lambda x: slot_rank.get(x[0], 10**9))
        for slot_name, final_slot_item, layer_slot_record, slot_api_logs in layer3_results:
            if not slot_name:
                continue
            final_slots.append(final_slot_item)
            layer3["slots"].append(layer_slot_record)
            api_logs.extend(slot_api_logs)

        layer3["slots_after"] = [str(x.get("slot_name", "")).strip() for x in final_slots]
        slot_pipe_layers.append(layer3)

        final_prompt = build_slot_pipe_v4_final_prompt(final_slots)
        self._track_text("slot_pipe_v4_final_appreciation_prompt", final_prompt)
        final_analysis, final_appreciation_log = self.vlm_runner.analyze(
            image_path=image_path,
            prompt=final_prompt,
            temperature=self.config.vlm_temperature,
            model=final_appreciation_model,
            inference_kind="slot_pipe_v4_final_appreciation",
        )
        api_logs.append(final_appreciation_log)
        self._track_text("slot_pipe_v4_final_appreciation_analysis", final_analysis)
        self._track_api_usage("slot_pipe_v4_final_appreciation", int(final_appreciation_log.get("total_tokens", 0) or 0))

        logs.append(
            {
                "mode": "slot_pipe",
                "slot_pipe_version": 4,
                "slots_file": self.config.slot_pipe_slots_file,
                "slots_loaded": len(slots_data),
                "slots_verified": len(verified_slots),
                "content_agents": content_agents,
                "expression_guests": guest_count,
                "checker_model": checker_model,
                "reviewer_model": reviewer_model,
                "content_model": content_model,
                "expression_guest_model": guest_model,
                "expression_summary_model": summary_model,
                "final_appreciation_model": final_appreciation_model,
                "api_enabled": self.api_client.enabled,
                "api_failures": [x for x in api_logs if not x.get("ok")],
                "token_usage": self.tracker.snapshot(),
                "v4_final_slots": final_slots,
            }
        )

        final_slot_names = [str(x.get("slot_name", "")).strip() for x in final_slots if str(x.get("slot_name", "")).strip()]
        return PipelineResult(
            mode="slot_pipe",
            selected_slots=final_slot_names,
            slot_context={
                str(x.get("slot_name", "")).strip(): "\n".join(f"- {t}" for t in (x.get("expression_points") or []) if str(t).strip())
                for x in final_slots
                if str(x.get("slot_name", "")).strip()
            },
            baseline_prompt="slot_pipe_v4_mode",
            enhanced_prompt=final_prompt,
            baseline_analysis="",
            enhanced_analysis=final_analysis,
            solitary_rounds=[],
            communal_rounds=[],
            slot_pipe_layers=slot_pipe_layers,
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
            slot_pipe_v4={
                "final_slots": final_slots,
                "slots_file": self.config.slot_pipe_slots_file,
            },
        )

    def _load_slot_pipe_v4_slots(self) -> list[dict]:
        path = str(self.config.slot_pipe_slots_file or "artifacts/slots.jsonl").strip()
        if not path:
            return []
        file_path = Path(path)
        if not file_path.exists() or not file_path.is_file():
            return []

        slots: list[dict] = []
        for raw_line in file_path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            slot_name = str(item.get("slot_name", "")).strip()
            if not slot_name:
                continue
            slots.append(item)
        return slots

    @staticmethod
    def _dedupe_texts(items: object, limit: int = 8) -> list[str]:
        if not isinstance(items, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = re.sub(r"\s+", "", text.lower())
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
            if len(out) >= max(1, int(limit or 1)):
                break
        return out

    def _run_slot_pipe(self, image_path: str, meta: dict | None = None) -> PipelineResult:
        if int(self.config.slot_pipe_version or 4) >= 4:
            return self._run_slot_pipe_v4(image_path=image_path, meta=meta)

        meta = meta or {}
        self._reset_embedding_api_logs()
        selected_slots = [slot_label(slot) for slot in normalize_slots(self.config.slots)]
        agents_per_slot = max(1, int(self.config.slot_pipe_agents_per_slot or 1))
        max_retries = max(0, int(self.config.slot_pipe_max_retries or 0))
        judge_model = self.config.judge_model or self.config.enhanced_model or self.config.baseline_model
        final_appreciation_model = (
            self.config.final_appreciation_model
            or self.config.enhanced_model
            or self.config.baseline_model
        )

        layers = [
            (1, "要素整理", "提炼该slot在图像中的视觉要素与证据。"),
            (2, "内容要点", "整理可用于最终鉴赏写作的内容要点与论述方向。"),
            (3, "专业表达", "将要点提升为更专业、准确、可落地的鉴赏表达建议。"),
        ]

        api_logs: list[dict] = []
        logs: list[dict] = []
        slot_pipe_layers: list[dict] = []

        current_slots = list(selected_slots)
        previous_layer_points: dict[str, list[dict]] = {}
        slot_timeline: list[dict] = []

        for layer_index, layer_name, layer_goal in layers:
            layer_record: dict = {
                "layer_index": layer_index,
                "layer_name": layer_name,
                "layer_goal": layer_goal,
                "slots_before": list(current_slots),
                "slots": [],
            }
            slot_timeline.append(
                {
                    "layer_index": layer_index,
                    "layer_name": layer_name,
                    "slots": list(current_slots),
                }
            )

            slot_states: dict[str, dict] = {
                slot: {
                    "latest_text_by_agent": {},
                    "agent_attempts": [],
                    "judge_rounds": [],
                }
                for slot in current_slots
            }

            initial_tasks: list[dict] = []
            for slot in current_slots:
                for agent_index in range(1, agents_per_slot + 1):
                    initial_tasks.append(
                        {
                            "slot": slot,
                            "agent_index": agent_index,
                            "attempt_index": 1,
                            "feedback": "",
                        }
                    )

            initial_logs = self._run_slot_pipe_agent_tasks(
                image_path=image_path,
                meta=meta,
                layer_index=layer_index,
                layer_name=layer_name,
                layer_goal=layer_goal,
                previous_layer_points=previous_layer_points,
                tasks=initial_tasks,
                slot_states=slot_states,
            )
            api_logs.extend(initial_logs)

            layer_judge_rounds: list[dict] = []
            last_layer_judge_result: dict = {
                "layer_ok": True,
                "summary": "default_no_judge",
                "slot_updates": [],
                "next_slots": current_slots,
                "slot_decisions": [],
                "retry_tasks": [],
            }
            last_layer_judge_log: dict = {}

            retry_round = 0
            while True:
                slot_point_map_for_judge = {
                    slot: self._pool_slot_points(slot_states[slot]["latest_text_by_agent"])
                    for slot in current_slots
                }
                slot_global_payload = {
                    slot: {
                        "pooled_points": slot_point_map_for_judge.get(slot, []),
                        "latest_agent_outputs": [
                            {"agent_index": idx, "text": text}
                            for idx, text in sorted(slot_states[slot]["latest_text_by_agent"].items())
                        ],
                        "agent_attempts": slot_states[slot]["agent_attempts"],
                    }
                    for slot in current_slots
                }

                layer_judge_result, layer_judge_log = self._run_slot_pipe_layer_judge(
                    image_path=image_path,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_goal=layer_goal,
                    current_slots=current_slots,
                    slot_global_payload=slot_global_payload,
                    max_retry=max_retries,
                    retry_round=retry_round,
                    judge_model=judge_model,
                )
                layer_judge_rounds.append(layer_judge_log)
                api_logs.append(layer_judge_log)
                last_layer_judge_result = layer_judge_result
                last_layer_judge_log = layer_judge_log

                slot_decisions = layer_judge_result.get("slot_decisions")
                if isinstance(slot_decisions, list):
                    decision_map = {
                        str(item.get("slot", "")).strip(): item
                        for item in slot_decisions
                        if isinstance(item, dict) and str(item.get("slot", "")).strip()
                    }
                    for slot in current_slots:
                        if slot in decision_map:
                            slot_states[slot]["judge_rounds"].append(decision_map[slot])

                retry_tasks = self._normalize_retry_tasks(
                    layer_judge_result.get("retry_tasks"),
                    current_slots=current_slots,
                    max_agent=agents_per_slot,
                )
                layer_ok = bool(layer_judge_result.get("layer_ok", True))

                if layer_ok or not retry_tasks or retry_round >= max_retries:
                    break

                for task in retry_tasks:
                    slot = str(task.get("slot", "")).strip()
                    agent_index = int(task.get("agent_index", 0) or 0)
                    task["attempt_index"] = self._count_agent_attempts(
                        slot_states.get(slot, {}).get("agent_attempts", []),
                        agent_index,
                    ) + 1

                retry_round += 1
                retry_logs = self._run_slot_pipe_agent_tasks(
                    image_path=image_path,
                    meta=meta,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_goal=layer_goal,
                    previous_layer_points=previous_layer_points,
                    tasks=retry_tasks,
                    slot_states=slot_states,
                )
                api_logs.extend(retry_logs)

            slot_points_for_layer: dict[str, list[dict]] = {
                slot: self._pool_slot_points(slot_states[slot]["latest_text_by_agent"])
                for slot in current_slots
            }

            for slot in current_slots:
                layer_record["slots"].append(
                    {
                        "slot": slot,
                        "agent_attempts": slot_states[slot]["agent_attempts"],
                        "judge_rounds": slot_states[slot]["judge_rounds"],
                        "final_points": slot_points_for_layer.get(slot, []),
                        "latest_agent_outputs": [
                            {"agent_index": idx, "text": text}
                            for idx, text in sorted(slot_states[slot]["latest_text_by_agent"].items())
                        ],
                    }
                )

            layer_record["layer_judge"] = last_layer_judge_log
            layer_record["layer_judge_rounds"] = layer_judge_rounds

            next_slots = self._normalize_slot_list(
                last_layer_judge_result.get("next_slots"),
                require_cn_theory=True,
                avoid_overlap_with=current_slots,
            )
            next_slots = self._stabilize_next_slots(current_slots, next_slots)

            layer_record["slots_after"] = list(next_slots)
            layer_record["slot_updates"] = last_layer_judge_result.get("slot_updates") or []

            current_slots = next_slots
            previous_layer_points = self._remap_slot_points_by_updates(
                slot_points_for_layer,
                last_layer_judge_result.get("slot_updates") or [],
                fallback_slots=current_slots,
            )
            slot_pipe_layers.append(layer_record)

        final_prompt = build_slot_pipe_final_prompt(slot_pipe_layers)
        self._track_text("slot_pipe_final_appreciation_prompt", final_prompt)
        final_analysis, final_appreciation_log = self.vlm_runner.analyze(
            image_path=image_path,
            prompt=final_prompt,
            temperature=self.config.vlm_temperature,
            model=final_appreciation_model,
            inference_kind="slot_pipe_final_appreciation",
        )
        api_logs.append(final_appreciation_log)
        self._track_text("slot_pipe_final_appreciation_analysis", final_analysis)
        self._track_api_usage("slot_pipe_final_appreciation", int(final_appreciation_log.get("total_tokens", 0) or 0))

        api_logs.extend(self._consume_embedding_api_logs())

        logs.append(
            {
                "mode": "slot_pipe",
                "selected_slots": selected_slots,
                "final_slots": current_slots,
                "initial_slot_limit": "all",
                "slot_timeline": slot_timeline,
                "agents_per_slot": agents_per_slot,
                "max_retries": max_retries,
                "judge_model": judge_model,
                "final_appreciation_model": final_appreciation_model,
                "embedding_summary": self._build_embedding_summary(api_logs),
                "api_enabled": self.api_client.enabled,
                "api_failures": [x for x in api_logs if not x.get("ok")],
                "token_usage": self.tracker.snapshot(),
            }
        )

        return PipelineResult(
            mode="slot_pipe",
            selected_slots=current_slots,
            slot_context={
                slot: "\n".join(f"- {p.get('point', '')}" for p in points if str(p.get("point", "")).strip())
                for slot, points in previous_layer_points.items()
            },
            baseline_prompt="slot_pipe_mode",
            enhanced_prompt=final_prompt,
            baseline_analysis="",
            enhanced_analysis=final_analysis,
            solitary_rounds=[],
            communal_rounds=[],
            slot_pipe_layers=slot_pipe_layers,
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
        )

    def _run_slot_pipe_agent_tasks(
        self,
        image_path: str,
        meta: dict,
        layer_index: int,
        layer_name: str,
        layer_goal: str,
        previous_layer_points: dict[str, list[dict]],
        tasks: list[dict],
        slot_states: dict[str, dict],
    ) -> list[dict]:
        if not tasks:
            return []

        logs: list[dict] = []
        max_workers = max(1, min(len(tasks), 16))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    self._run_slot_pipe_agent_call,
                    image_path=image_path,
                    meta=meta,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_goal=layer_goal,
                    slot=str(task.get("slot", "")).strip(),
                    previous_layer_points=previous_layer_points.get(str(task.get("slot", "")).strip()) or [],
                    judge_feedback=str(task.get("feedback", "")).strip(),
                    agent_index=int(task.get("agent_index", 0) or 0),
                    attempt_index=int(task.get("attempt_index", 0) or 0),
                ): task
                for task in tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                slot = str(task.get("slot", "")).strip()
                agent_index = int(task.get("agent_index", 0) or 0)
                try:
                    text, log = future.result()
                except Exception as exc:
                    log = {
                        "stage": "slot_pipe_agent",
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "slot": slot,
                        "agent_index": agent_index,
                        "attempt_index": int(task.get("attempt_index", 0) or 0),
                        "inference_kind": f"slot_pipe_l{layer_index}_{self._slug(slot)}_a{agent_index}_task_error",
                        "model": self.config.agent_model,
                        "ok": False,
                        "error": f"slot_pipe_agent_task_exception: {exc}",
                        "image_attached": False,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "status_code": None,
                        "endpoint": "",
                        "request_summary": {},
                        "response_summary": {},
                        "prompt_text": "",
                        "response_text": "",
                    }
                    text = ""

                logs.append(log)
                state = slot_states.get(slot)
                if not state:
                    continue
                if text:
                    state["latest_text_by_agent"][agent_index] = text
                state["agent_attempts"].append(log)

        for slot, state in slot_states.items():
            state["agent_attempts"].sort(
                key=lambda item: (
                    int(item.get("agent_index", 0) or 0),
                    int(item.get("attempt_index", 0) or 0),
                )
            )
        return logs

    def _normalize_retry_tasks(
        self,
        raw: object,
        current_slots: list[str],
        max_agent: int,
    ) -> list[dict]:
        if not isinstance(raw, list):
            return []
        valid_slots = set(current_slots)
        normalized: list[dict] = []
        seen: set[tuple[str, int]] = set()
        for item in raw:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot", "")).strip()
            if slot not in valid_slots:
                continue
            try:
                agent_index = int(item.get("agent_index", 0) or 0)
            except Exception:
                continue
            if agent_index < 1 or agent_index > max_agent:
                continue
            key = (slot, agent_index)
            if key in seen:
                continue
            seen.add(key)
            feedback = str(item.get("feedback", "")).strip()
            normalized.append(
                {
                    "slot": slot,
                    "agent_index": agent_index,
                    "feedback": feedback,
                    "attempt_index": 0,
                }
            )
        return normalized

    def _run_slot_pipe_one_slot(
        self,
        image_path: str,
        meta: dict,
        layer_index: int,
        layer_name: str,
        layer_goal: str,
        slot: str,
        previous_layer_points: list[dict],
        agents_per_slot: int,
        max_retries: int,
        judge_model: str | None,
    ) -> dict:
        latest_text_by_agent: dict[int, str] = {}
        slot_record: dict = {
            "slot": slot,
            "agent_attempts": [],
            "judge_rounds": [],
        }
        slot_api_logs: list[dict] = []

        with ThreadPoolExecutor(max_workers=agents_per_slot) as agent_pool:
            initial_futures = {
                agent_pool.submit(
                    self._run_slot_pipe_agent_call,
                    image_path=image_path,
                    meta=meta,
                    layer_index=layer_index,
                    layer_name=layer_name,
                    layer_goal=layer_goal,
                    slot=slot,
                    previous_layer_points=previous_layer_points,
                    judge_feedback="",
                    agent_index=agent_index,
                    attempt_index=1,
                ): agent_index
                for agent_index in range(1, agents_per_slot + 1)
            }
            initial_attempts: list[dict] = []
            for future in as_completed(initial_futures):
                agent_index = initial_futures[future]
                attempt_text, attempt_log = future.result()
                latest_text_by_agent[agent_index] = attempt_text
                initial_attempts.append(attempt_log)
                slot_api_logs.append(attempt_log)
            initial_attempts.sort(key=lambda x: int(x.get("agent_index", 0) or 0))
            slot_record["agent_attempts"].extend(initial_attempts)

        judge_round = 1
        retry_count = 0
        while True:
            pooled_points = self._pool_slot_points(latest_text_by_agent)
            judge_result, judge_log = self._run_slot_pipe_slot_judge(
                image_path=image_path,
                layer_index=layer_index,
                layer_name=layer_name,
                layer_goal=layer_goal,
                slot=slot,
                max_retry=max_retries,
                retry_round=judge_round,
                pooled_points=pooled_points,
                latest_text_by_agent=latest_text_by_agent,
                judge_model=judge_model,
            )
            slot_record["judge_rounds"].append(judge_log)
            slot_api_logs.append(judge_log)

            if judge_result.get("slot_status") != "needs_improve":
                break
            if retry_count >= max_retries:
                break

            agents_to_retry = self._normalize_agent_indexes(
                judge_result.get("agents_to_retry"),
                max_agent=agents_per_slot,
            )
            if not agents_to_retry:
                break

            feedback_by_agent = judge_result.get("feedback_by_agent") or {}
            next_attempt_index_map = {
                agent_index: self._count_agent_attempts(slot_record["agent_attempts"], agent_index) + 1
                for agent_index in agents_to_retry
            }

            with ThreadPoolExecutor(max_workers=max(1, len(agents_to_retry))) as retry_pool:
                retry_futures = {
                    retry_pool.submit(
                        self._run_slot_pipe_agent_call,
                        image_path=image_path,
                        meta=meta,
                        layer_index=layer_index,
                        layer_name=layer_name,
                        layer_goal=layer_goal,
                        slot=slot,
                        previous_layer_points=previous_layer_points,
                        judge_feedback=str(feedback_by_agent.get(str(agent_index), "")).strip(),
                        agent_index=agent_index,
                        attempt_index=next_attempt_index_map[agent_index],
                    ): agent_index
                    for agent_index in agents_to_retry
                }
                retry_attempts: list[dict] = []
                for future in as_completed(retry_futures):
                    agent_index = retry_futures[future]
                    retry_text, retry_log = future.result()
                    latest_text_by_agent[agent_index] = retry_text
                    retry_attempts.append(retry_log)
                    slot_api_logs.append(retry_log)
                retry_attempts.sort(
                    key=lambda x: (
                        int(x.get("agent_index", 0) or 0),
                        int(x.get("attempt_index", 0) or 0),
                    )
                )
                slot_record["agent_attempts"].extend(retry_attempts)

            retry_count += 1
            judge_round += 1

        final_points = self._pool_slot_points(latest_text_by_agent)
        slot_record["final_points"] = final_points
        slot_record["latest_agent_outputs"] = [
            {"agent_index": idx, "text": text}
            for idx, text in sorted(latest_text_by_agent.items())
        ]
        return {
            "slot": slot,
            "slot_record": slot_record,
            "final_points": final_points,
            "api_logs": slot_api_logs,
        }

    def _run_slot_pipe_agent_call(
        self,
        image_path: str,
        meta: dict,
        layer_index: int,
        layer_name: str,
        layer_goal: str,
        slot: str,
        previous_layer_points: list[dict],
        judge_feedback: str,
        agent_index: int,
        attempt_index: int,
    ) -> tuple[str, dict]:
        prompt = build_slot_pipe_agent_prompt(
            layer_index=layer_index,
            layer_name=layer_name,
            layer_goal=layer_goal,
            slot=slot,
            meta=meta,
            previous_layer_slot_points=previous_layer_points,
            judge_feedback=judge_feedback,
            agent_index=agent_index,
            attempt_index=attempt_index,
        )
        result = self.api_client.chat(
            system_prompt="你是国画领域的分工agent，请按任务要求做图像鉴赏。",
            user_prompt=prompt,
            temperature=self.config.agent_temperature,
            image_path=image_path,
            model=self.config.agent_model,
        )
        content = (result.content or "").strip() or "API不可用或调用失败，未获得slot_pipe agent输出。"
        self._track_text(
            f"slot_pipe_l{layer_index}_{slot}_agent{agent_index}_attempt{attempt_index}_prompt",
            prompt,
        )
        self._track_text(
            f"slot_pipe_l{layer_index}_{slot}_agent{agent_index}_attempt{attempt_index}_analysis",
            content,
        )
        self._track_api_usage(
            f"slot_pipe_l{layer_index}_{slot}_agent{agent_index}_attempt{attempt_index}",
            int(result.total_tokens or 0),
        )
        log = {
            "stage": "slot_pipe_agent",
            "layer_index": layer_index,
            "layer_name": layer_name,
            "slot": slot,
            "agent_index": agent_index,
            "attempt_index": attempt_index,
            "inference_kind": f"slot_pipe_l{layer_index}_{self._slug(slot)}_a{agent_index}_t{attempt_index}",
            "model": result.model,
            "ok": bool(result.content),
            "error": result.error,
            "image_attached": result.image_attached,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "status_code": result.status_code,
            "endpoint": result.endpoint,
            "request_summary": result.request_summary,
            "response_summary": result.response_summary,
            "prompt_text": prompt,
            "response_text": content,
        }
        return content, log

    def _run_slot_pipe_slot_judge(
        self,
        image_path: str,
        layer_index: int,
        layer_name: str,
        layer_goal: str,
        slot: str,
        max_retry: int,
        retry_round: int,
        pooled_points: list[dict],
        latest_text_by_agent: dict[int, str],
        judge_model: str | None,
    ) -> tuple[dict, dict]:
        latest_agent_outputs = [
            {"agent_index": idx, "text": text}
            for idx, text in sorted(latest_text_by_agent.items())
        ]
        prompt = build_slot_pipe_slot_judge_prompt(
            layer_index=layer_index,
            layer_name=layer_name,
            layer_goal=layer_goal,
            slot=slot,
            max_retry=max_retry,
            retry_round=retry_round,
            pooled_points=pooled_points,
            latest_agent_outputs=latest_agent_outputs,
        )
        result = self.api_client.chat(
            system_prompt="你是严格JSON输出的质量评审员。",
            user_prompt=prompt,
            temperature=0.1,
            image_path=image_path,
            model=judge_model,
        )
        raw = (result.content or "").strip()
        parsed = self._parse_json_object(raw)
        if not parsed:
            parsed = {
                "slot_status": "good",
                "reason": "judge_parse_failed_fallback_good",
                "agents_to_retry": [],
                "feedback_by_agent": {},
            }
        self._track_text(f"slot_pipe_l{layer_index}_{slot}_judge_r{retry_round}_prompt", prompt)
        self._track_text(f"slot_pipe_l{layer_index}_{slot}_judge_r{retry_round}_analysis", raw)
        self._track_api_usage(
            f"slot_pipe_l{layer_index}_{slot}_judge_r{retry_round}",
            int(result.total_tokens or 0),
        )
        log = {
            "stage": "slot_pipe_slot_judge",
            "layer_index": layer_index,
            "layer_name": layer_name,
            "slot": slot,
            "judge_round": retry_round,
            "inference_kind": f"slot_pipe_l{layer_index}_{self._slug(slot)}_judge_r{retry_round}",
            "model": result.model,
            "ok": bool(result.content),
            "error": result.error,
            "image_attached": result.image_attached,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "status_code": result.status_code,
            "endpoint": result.endpoint,
            "request_summary": result.request_summary,
            "response_summary": result.response_summary,
            "prompt_text": prompt,
            "response_text": raw,
            "judge_result": parsed,
        }
        return parsed, log

    def _run_slot_pipe_layer_judge(
        self,
        image_path: str,
        layer_index: int,
        layer_name: str,
        layer_goal: str,
        current_slots: list[str],
        slot_global_payload: dict[str, dict],
        max_retry: int,
        retry_round: int,
        judge_model: str | None,
    ) -> tuple[dict, dict]:
        prompt = build_slot_pipe_layer_judge_prompt(
            layer_index=layer_index,
            layer_name=layer_name,
            layer_goal=layer_goal,
            current_slots=current_slots,
            slot_global_payload=slot_global_payload,
            max_retry=max_retry,
            retry_round=retry_round,
        )
        result = self.api_client.chat(
            system_prompt="你是层级规划judge，请只返回JSON。",
            user_prompt=prompt,
            temperature=0.1,
            image_path=image_path,
            model=judge_model,
        )
        raw = (result.content or "").strip()
        parsed = self._parse_json_object(raw)
        if not parsed:
            parsed = {
                "layer_ok": True,
                "summary": "judge_parse_failed_keep_slots",
                "slot_decisions": [],
                "retry_tasks": [],
                "slot_updates": [],
                "next_slots": current_slots,
            }
        self._track_text(f"slot_pipe_l{layer_index}_layer_judge_prompt", prompt)
        self._track_text(f"slot_pipe_l{layer_index}_layer_judge_analysis", raw)
        self._track_api_usage(f"slot_pipe_l{layer_index}_layer_judge", int(result.total_tokens or 0))
        log = {
            "stage": "slot_pipe_layer_judge",
            "layer_index": layer_index,
            "layer_name": layer_name,
            "slot": None,
            "inference_kind": f"slot_pipe_l{layer_index}_layer_judge",
            "model": result.model,
            "ok": bool(result.content),
            "error": result.error,
            "image_attached": result.image_attached,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.total_tokens,
            "status_code": result.status_code,
            "endpoint": result.endpoint,
            "request_summary": result.request_summary,
            "response_summary": result.response_summary,
            "prompt_text": prompt,
            "response_text": raw,
            "judge_result": parsed,
        }
        return parsed, log

    def _track_text(self, stage: str, text: str) -> None:
        with self._tracker_lock:
            self.tracker.add_text(stage, text)

    def _track_api_usage(self, stage: str, total_tokens: int) -> None:
        with self._tracker_lock:
            self.tracker.add_api_usage(stage, total_tokens)

    @staticmethod
    def _build_embedding_summary(api_logs: list[dict]) -> dict:
        embedding_logs = [x for x in api_logs if x.get("stage") == "slot_pipe_embedding"]
        total = len(embedding_logs)
        parse_ok = sum(1 for x in embedding_logs if bool(x.get("embedding_parse_ok")))
        fallback = sum(1 for x in embedding_logs if bool(x.get("fallback_used")))
        api_ok = sum(1 for x in embedding_logs if bool(x.get("ok")))
        return {
            "total": total,
            "api_ok": api_ok,
            "embedding_parse_ok": parse_ok,
            "fallback_used": fallback,
        }

    def _reset_embedding_api_logs(self) -> None:
        with self._embedding_logs_lock:
            self._embedding_api_logs = []

    def _append_embedding_api_log(self, log: dict) -> None:
        with self._embedding_logs_lock:
            self._embedding_api_logs.append(log)

    def _consume_embedding_api_logs(self) -> list[dict]:
        with self._embedding_logs_lock:
            return list(self._embedding_api_logs)

    def _pool_slot_points(self, latest_text_by_agent: dict[int, str]) -> list[dict]:
        merged = self._collect_and_merge_points(latest_text_by_agent)
        if not merged:
            return []

        if len(merged) <= 5:
            ranked = sorted(
                merged,
                key=lambda x: (-int(x.get("support", 0)), -len(str(x.get("point", "")))),
            )
            return ranked

        target_k = self._target_cluster_count(len(merged))
        texts = [str(item.get("point", "")) for item in merged]
        embeddings = self._embed_texts(texts)
        clusters = self._kmeans_cluster(embeddings, k=target_k, iterations=8)

        representatives: list[dict] = []
        for cluster_indexes in clusters:
            if not cluster_indexes:
                continue
            centroid = self._mean_vector([embeddings[i] for i in cluster_indexes])
            best_idx = cluster_indexes[0]
            best_score = -10.0
            cluster_sources: set[int] = set()
            cluster_support = 0

            for idx in cluster_indexes:
                item = merged[idx]
                support = int(item.get("support", 0) or 0)
                cluster_support += support
                cluster_sources.update(int(x) for x in (item.get("sources") or []))
                sim = self._cosine_similarity(embeddings[idx], centroid)
                score = sim + (support * 0.02)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            rep_item = dict(merged[best_idx])
            rep_item["sources"] = sorted(cluster_sources)
            rep_item["support"] = cluster_support
            representatives.append(rep_item)

        ranked_reps = sorted(
            representatives,
            key=lambda x: (-int(x.get("support", 0)), -len(str(x.get("point", "")))),
        )
        return ranked_reps[:5]

    @staticmethod
    def _collect_and_merge_points(latest_text_by_agent: dict[int, str]) -> list[dict]:
        bucket: dict[str, dict] = {}
        for agent_index, text in sorted(latest_text_by_agent.items()):
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            for line in lines:
                clean = line.lstrip("-*• ").strip()
                if not clean:
                    continue
                key = TcpPromptPipeline._semantic_key(clean)
                item = bucket.get(key)
                if not item:
                    bucket[key] = {
                        "point": clean,
                        "sources": [agent_index],
                        "support": 1,
                    }
                else:
                    item["support"] = int(item.get("support", 0) or 0) + 1
                    sources = list(item.get("sources") or [])
                    if agent_index not in sources:
                        sources.append(agent_index)
                    item["sources"] = sorted(sources)
                    if len(clean) > len(str(item.get("point", ""))):
                        item["point"] = clean
        return list(bucket.values())

    @staticmethod
    def _target_cluster_count(n_points: int) -> int:
        if n_points <= 2:
            return n_points
        if n_points <= 5:
            return n_points
        approx = int(round(math.sqrt(n_points) + 1))
        return min(5, max(3, approx))

    def _embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        model_name = self.config.embedding_model or os.getenv("EMBEDDING_MODEL_NAME") or os.getenv("BGE_MODEL_NAME", "baai/bge-m3")

        emb = self.api_client.embeddings(texts, model=model_name)
        self._track_api_usage("slot_pipe_embedding", int(emb.get("total_tokens", 0) or 0))
        raw_vectors = emb.get("vectors") if isinstance(emb.get("vectors"), list) else []
        emb_usable = bool(emb.get("ok")) and len(raw_vectors) == len(texts)

        if emb_usable:
            vectors: list[list[float]] = []
            for idx, text in enumerate(texts):
                candidate = raw_vectors[idx]
                vector: list[float] = []
                if isinstance(candidate, list):
                    for x in candidate:
                        try:
                            vector.append(float(x))
                        except Exception:
                            continue

                if vector:
                    norm = math.sqrt(sum(v * v for v in vector))
                    if norm > 0:
                        vector = [v / norm for v in vector]
                    vectors.append(vector)
                    self._append_embedding_api_log(
                        {
                            "stage": "slot_pipe_embedding",
                            "layer_index": None,
                            "layer_name": "pooling",
                            "slot": None,
                            "inference_kind": f"slot_pipe_embedding_t{idx + 1}",
                            "model": emb.get("model") or model_name,
                            "ok": bool(emb.get("ok")),
                            "error": emb.get("error"),
                            "image_attached": False,
                            "prompt_tokens": emb.get("prompt_tokens", 0),
                            "completion_tokens": 0,
                            "total_tokens": emb.get("total_tokens", 0),
                            "status_code": emb.get("status_code"),
                            "endpoint": emb.get("endpoint", ""),
                            "request_summary": emb.get("request_summary", {}),
                            "response_summary": emb.get("response_summary", {}),
                            "prompt_text": text,
                            "response_text": "",
                            "embedding_parse_ok": True,
                            "fallback_used": False,
                        }
                    )
                else:
                    fallback_vector = self._text_to_embedding(text)
                    vectors.append(fallback_vector)
                    self._append_embedding_api_log(
                        {
                            "stage": "slot_pipe_embedding",
                            "layer_index": None,
                            "layer_name": "pooling",
                            "slot": None,
                            "inference_kind": f"slot_pipe_embedding_t{idx + 1}",
                            "model": emb.get("model") or model_name,
                            "ok": bool(emb.get("ok")),
                            "error": "embedding_vector_empty_fallback",
                            "image_attached": False,
                            "prompt_tokens": emb.get("prompt_tokens", 0),
                            "completion_tokens": 0,
                            "total_tokens": emb.get("total_tokens", 0),
                            "status_code": emb.get("status_code"),
                            "endpoint": emb.get("endpoint", ""),
                            "request_summary": emb.get("request_summary", {}),
                            "response_summary": emb.get("response_summary", {}),
                            "prompt_text": text,
                            "response_text": "",
                            "embedding_parse_ok": False,
                            "fallback_used": True,
                        }
                    )
            return self._pad_vectors(vectors, texts)

        self._append_embedding_api_log(
            {
                "stage": "slot_pipe_embedding",
                "layer_index": None,
                "layer_name": "pooling",
                "slot": None,
                "inference_kind": "slot_pipe_embedding_batch",
                "model": emb.get("model") or model_name,
                "ok": bool(emb.get("ok")),
                "error": emb.get("error") or "embedding_batch_unavailable",
                "image_attached": False,
                "prompt_tokens": emb.get("prompt_tokens", 0),
                "completion_tokens": 0,
                "total_tokens": emb.get("total_tokens", 0),
                "status_code": emb.get("status_code"),
                "endpoint": emb.get("endpoint", ""),
                "request_summary": emb.get("request_summary", {}),
                "response_summary": emb.get("response_summary", {}),
                "prompt_text": "",
                "response_text": "",
                "embedding_parse_ok": False,
                "fallback_used": True,
            }
        )

        def embed_one(text: str, text_index: int) -> list[float]:
            prompt = (
                "请为下面句子生成语义向量并只输出JSON。"
                "返回格式必须是 {\"embedding\": [float, ...]}，不要输出任何额外文字。\n\n"
                f"句子：{text}"
            )
            result = self.api_client.chat(
                system_prompt="你是embedding服务返回器，只输出JSON。",
                user_prompt=prompt,
                temperature=0.0,
                image_path=None,
                model=model_name,
            )
            self._track_api_usage("slot_pipe_embedding", int(result.total_tokens or 0))
            raw = (result.content or "").strip()
            parsed = self._parse_json_object(raw)
            vector: list[float] = []
            parsed_ok = False
            if parsed and isinstance(parsed.get("embedding"), list):
                for item in parsed.get("embedding", []):
                    try:
                        vector.append(float(item))
                    except Exception:
                        continue
                if vector:
                    parsed_ok = True
                    norm = math.sqrt(sum(v * v for v in vector))
                    normalized = vector
                    if norm > 0:
                        normalized = [v / norm for v in vector]
                    self._append_embedding_api_log(
                        {
                            "stage": "slot_pipe_embedding",
                            "layer_index": None,
                            "layer_name": "pooling",
                            "slot": None,
                            "inference_kind": f"slot_pipe_embedding_t{text_index + 1}",
                            "model": result.model,
                            "ok": bool(result.content),
                            "error": result.error,
                            "image_attached": result.image_attached,
                            "prompt_tokens": result.prompt_tokens,
                            "completion_tokens": result.completion_tokens,
                            "total_tokens": result.total_tokens,
                            "status_code": result.status_code,
                            "endpoint": result.endpoint,
                            "request_summary": result.request_summary,
                            "response_summary": result.response_summary,
                            "prompt_text": prompt,
                            "response_text": raw,
                            "embedding_parse_ok": parsed_ok,
                            "fallback_used": False,
                        }
                    )
                    return normalized

            fallback_vector = self._text_to_embedding(text)
            self._append_embedding_api_log(
                {
                    "stage": "slot_pipe_embedding",
                    "layer_index": None,
                    "layer_name": "pooling",
                    "slot": None,
                    "inference_kind": f"slot_pipe_embedding_t{text_index + 1}",
                    "model": result.model,
                    "ok": bool(result.content),
                    "error": result.error,
                    "image_attached": result.image_attached,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "total_tokens": result.total_tokens,
                    "status_code": result.status_code,
                    "endpoint": result.endpoint,
                    "request_summary": result.request_summary,
                    "response_summary": result.response_summary,
                    "prompt_text": prompt,
                    "response_text": raw,
                    "embedding_parse_ok": False,
                    "fallback_used": True,
                }
            )
            return fallback_vector

        vectors: list[list[float]] = []
        max_workers = max(1, min(len(texts), 8))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(embed_one, text, idx): idx for idx, text in enumerate(texts)}
            temp: dict[int, list[float]] = {}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    temp[idx] = future.result()
                except Exception:
                    self._append_embedding_api_log(
                        {
                            "stage": "slot_pipe_embedding",
                            "layer_index": None,
                            "layer_name": "pooling",
                            "slot": None,
                            "inference_kind": f"slot_pipe_embedding_t{idx + 1}",
                            "model": model_name,
                            "ok": False,
                            "error": "embedding_task_exception",
                            "image_attached": False,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "total_tokens": 0,
                            "status_code": None,
                            "endpoint": "",
                            "request_summary": {},
                            "response_summary": {},
                            "prompt_text": "",
                            "response_text": "",
                            "embedding_parse_ok": False,
                            "fallback_used": True,
                        }
                    )
                    temp[idx] = self._text_to_embedding(texts[idx])
            for idx in range(len(texts)):
                vectors.append(temp.get(idx, self._text_to_embedding(texts[idx])))

        return self._pad_vectors(vectors, texts)

    def _pad_vectors(self, vectors: list[list[float]], texts: list[str]) -> list[list[float]]:
        target_dim = max((len(v) for v in vectors), default=0)
        if target_dim <= 0:
            return [self._text_to_embedding(text) for text in texts]

        padded: list[list[float]] = []
        for v in vectors:
            if len(v) < target_dim:
                padded.append(v + [0.0] * (target_dim - len(v)))
            else:
                padded.append(v)
        return padded

    @staticmethod
    def _text_to_embedding(text: str, dim: int = 256) -> list[float]:
        vector = [0.0] * dim
        normalized = TcpPromptPipeline._semantic_key(text)
        if not normalized:
            return vector

        # Use character bigrams as a lightweight local embedding representation.
        chars = list(normalized)
        if len(chars) == 1:
            idx = ord(chars[0]) % dim
            vector[idx] += 1.0
        else:
            for i in range(len(chars) - 1):
                bigram = chars[i] + chars[i + 1]
                idx = sum(ord(c) for c in bigram) % dim
                vector[idx] += 1.0

        norm = math.sqrt(sum(v * v for v in vector))
        if norm <= 0:
            return vector
        return [v / norm for v in vector]

    @staticmethod
    def _kmeans_cluster(embeddings: list[list[float]], k: int, iterations: int = 8) -> list[list[int]]:
        n = len(embeddings)
        if n == 0:
            return []
        k = max(1, min(k, n))

        centroids: list[list[float]] = [list(embeddings[0])]
        while len(centroids) < k:
            best_idx = 0
            best_dist = -1.0
            for idx, emb in enumerate(embeddings):
                nearest = max(TcpPromptPipeline._cosine_similarity(emb, c) for c in centroids)
                dist = 1.0 - nearest
                if dist > best_dist:
                    best_dist = dist
                    best_idx = idx
            centroids.append(list(embeddings[best_idx]))

        assignments = [0] * n
        for _ in range(max(1, iterations)):
            changed = False
            for i, emb in enumerate(embeddings):
                best_cluster = 0
                best_sim = -2.0
                for c_idx, centroid in enumerate(centroids):
                    sim = TcpPromptPipeline._cosine_similarity(emb, centroid)
                    if sim > best_sim:
                        best_sim = sim
                        best_cluster = c_idx
                if assignments[i] != best_cluster:
                    assignments[i] = best_cluster
                    changed = True

            clusters = [[] for _ in range(k)]
            for idx, c_idx in enumerate(assignments):
                clusters[c_idx].append(idx)

            new_centroids: list[list[float]] = []
            for c_idx, members in enumerate(clusters):
                if not members:
                    new_centroids.append(list(centroids[c_idx]))
                    continue
                new_centroids.append(TcpPromptPipeline._mean_vector([embeddings[m] for m in members]))
            centroids = new_centroids

            if not changed:
                break

        final_clusters = [[] for _ in range(k)]
        for idx, c_idx in enumerate(assignments):
            final_clusters[c_idx].append(idx)
        return [c for c in final_clusters if c]

    @staticmethod
    def _mean_vector(vectors: list[list[float]]) -> list[float]:
        if not vectors:
            return []
        dim = len(vectors[0])
        acc = [0.0] * dim
        for v in vectors:
            for i, value in enumerate(v):
                acc[i] += value
        size = float(len(vectors))
        mean = [x / size for x in acc]
        norm = math.sqrt(sum(v * v for v in mean))
        if norm <= 0:
            return mean
        return [v / norm for v in mean]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        if not a or not b or len(a) != len(b):
            return 0.0
        return float(sum(x * y for x, y in zip(a, b)))

    @staticmethod
    def _semantic_key(text: str) -> str:
        letters = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]", text.lower())
        return "".join(letters[:48]) or text.lower()[:48]

    @staticmethod
    def _parse_json_object(text: str) -> dict | None:
        if not text.strip():
            return None
        candidate = text.strip()
        try:
            parsed = json.loads(candidate)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start >= 0 and end > start:
            block = candidate[start : end + 1]
            try:
                parsed = json.loads(block)
                return parsed if isinstance(parsed, dict) else None
            except json.JSONDecodeError:
                return None
        return None

    @staticmethod
    def _normalize_agent_indexes(raw: object, max_agent: int) -> list[int]:
        if not isinstance(raw, list):
            return []
        out: list[int] = []
        seen: set[int] = set()
        for item in raw:
            try:
                idx = int(item)
            except Exception:
                continue
            if idx < 1 or idx > max_agent or idx in seen:
                continue
            seen.add(idx)
            out.append(idx)
        return out

    @staticmethod
    def _count_agent_attempts(agent_attempts: list[dict], agent_index: int) -> int:
        return sum(1 for item in agent_attempts if int(item.get("agent_index", 0) or 0) == agent_index)

    @staticmethod
    def _normalize_slot_list(
        raw: object,
        require_cn_theory: bool = False,
        avoid_overlap_with: list[str] | None = None,
    ) -> list[str]:
        if not isinstance(raw, list):
            return []
        out: list[str] = []
        seen: set[str] = set()
        overlap_refs = [str(x).strip() for x in (avoid_overlap_with or []) if str(x).strip()]
        for item in raw:
            slot = str(item).strip()
            if not slot:
                continue
            if require_cn_theory and not TcpPromptPipeline._is_valid_chinese_theory_slot(slot):
                continue
            if overlap_refs and TcpPromptPipeline._is_semantically_overlapping_slot_name(slot, overlap_refs):
                continue
            key = slot.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(slot)
        return out

    @staticmethod
    def _stabilize_next_slots(current_slots: list[str], proposed_slots: list[str]) -> list[str]:
        if not current_slots:
            return list(proposed_slots)
        if not proposed_slots:
            return list(current_slots)

        # 演化策略保守化：每层最多减少1个slot，避免过快收缩。
        min_count = max(1, len(current_slots) - 1)
        out: list[str] = []
        seen: set[str] = set()
        for slot in proposed_slots:
            key = str(slot).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(slot).strip())

        if len(out) >= min_count:
            return out

        for slot in current_slots:
            key = str(slot).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(slot).strip())
            if len(out) >= min_count:
                break
        return out

    @staticmethod
    def _is_semantically_overlapping_slot_name(slot: str, reference_slots: list[str]) -> bool:
        candidate = str(slot).strip()
        if not candidate:
            return True
        refs = [str(x).strip() for x in reference_slots if str(x).strip()]
        included_refs = [ref for ref in refs if ref != candidate and ref in candidate]

        # 避免把多个既有维度合并为一个新slot，例如“笔墨气韵”。
        if len(included_refs) >= 2:
            return True

        # 对明显拼接命名做保守过滤。
        if included_refs and re.search(r"[、/及与和兼并]|以及", candidate):
            return True

        return False

    @staticmethod
    def _is_valid_chinese_theory_slot(slot: str) -> bool:
        text = str(slot).strip()
        if not text:
            return False
        if re.search(r"[A-Za-z]", text):
            return False
        if re.search(r"slot", text, flags=re.IGNORECASE):
            return False
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", text)
        if len(chinese_chars) < 2:
            return False

        # 限制中间层slot使用国画鉴赏理论相关概念，避免泛化命名。
        theory_terms = [
            "笔墨", "笔法", "墨法", "皴法", "线条", "设色", "用笔", "用墨", "浓淡", "干湿",
            "构图", "章法", "经营位置", "虚实", "留白", "层次", "开合", "疏密", "取势",
            "气韵", "气势", "神采", "格调", "意境", "境界", "韵致", "骨法", "形神",
            "题款", "题跋", "印章", "款识", "诗书画印", "媒材", "纸墨", "绢本",
        ]
        return any(term in text for term in theory_terms)

    @staticmethod
    def _remap_slot_points_by_updates(
        slot_points_for_layer: dict[str, list[dict]],
        slot_updates: list[dict],
        fallback_slots: list[str],
    ) -> dict[str, list[dict]]:
        if not slot_updates:
            return {k: list(v) for k, v in slot_points_for_layer.items() if k in fallback_slots}

        out: dict[str, list[dict]] = {}
        touched: set[str] = set()

        def _append_points(dst: str, points: list[dict]) -> None:
            key = str(dst).strip()
            if not key:
                return
            out.setdefault(key, [])
            out[key].extend(list(points))

        for item in slot_updates:
            if not isinstance(item, dict):
                continue
            slot = str(item.get("slot", "")).strip()
            action = str(item.get("action", "keep")).strip().lower()
            new_slot = str(item.get("new_slot", "")).strip()
            raw_new_slots = item.get("new_slots")
            new_slots = [
                str(x).strip()
                for x in (raw_new_slots if isinstance(raw_new_slots, list) else [])
                if str(x).strip()
            ]
            if not slot:
                continue
            touched.add(slot)
            points = list(slot_points_for_layer.get(slot) or [])

            if (
                action == "rename"
                and new_slot
                and TcpPromptPipeline._is_valid_chinese_theory_slot(new_slot)
                and not TcpPromptPipeline._is_semantically_overlapping_slot_name(new_slot, fallback_slots)
            ):
                # rename采取“复制而非迁移”：保留原slot语义，避免新slot命名波动导致要点漂移。
                _append_points(slot, points)
                _append_points(new_slot, points)
                continue

            if (
                action == "reduce"
                and new_slot
                and TcpPromptPipeline._is_valid_chinese_theory_slot(new_slot)
                and not TcpPromptPipeline._is_semantically_overlapping_slot_name(new_slot, fallback_slots)
            ):
                # reduce: 多个来源slot可聚合到同一个new_slot。
                _append_points(new_slot, points)
                continue

            if action == "split" and new_slots:
                valid_targets = [
                    s
                    for s in new_slots
                    if TcpPromptPipeline._is_valid_chinese_theory_slot(s)
                    and not TcpPromptPipeline._is_semantically_overlapping_slot_name(s, fallback_slots)
                ]
                if len(valid_targets) >= 2:
                    for target in valid_targets:
                        _append_points(target, points)
                    continue

            _append_points(slot, points)

        for slot, points in slot_points_for_layer.items():
            if slot in touched:
                continue
            _append_points(slot, points)

        for slot in fallback_slots:
            out.setdefault(slot, list(slot_points_for_layer.get(slot) or []))
        return out

    @staticmethod
    def _slug(text: str) -> str:
        lowered = text.lower().strip()
        normalized = re.sub(r"[^a-z0-9\u4e00-\u9fff_-]+", "_", lowered)
        normalized = re.sub(r"_+", "_", normalized).strip("_")
        return normalized or "slot"

    def save_result(self, result: PipelineResult, output_dir: str = "outputs") -> dict[str, str]:
        root_dir = Path(output_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = root_dir / stamp
        run_dir.mkdir(parents=True, exist_ok=True)

        api_calls_path = run_dir / "api_calls.jsonl"
        image_delivery_checks_path = run_dir / "image_delivery_checks.json"
        report_path = run_dir / "report.json"
        output_files: dict[str, str] = {
            "run_dir": str(run_dir),
            "mode": result.mode,
        }

        is_slot_like_mode = result.mode == "slot"

        if is_slot_like_mode:
            baseline_prompt_path = run_dir / "prompt_baseline.md"
            enhanced_prompt_path = run_dir / "prompt_enhanced.md"
            baseline_analysis_path = run_dir / "analysis_baseline.md"
            enhanced_analysis_path = run_dir / "analysis_enhanced.md"
            baseline_prompt_path.write_text(result.baseline_prompt, encoding="utf-8")
            enhanced_prompt_path.write_text(result.enhanced_prompt, encoding="utf-8")
            baseline_analysis_path.write_text(result.baseline_analysis, encoding="utf-8")
            enhanced_analysis_path.write_text(result.enhanced_analysis, encoding="utf-8")

            output_files["baseline_prompt"] = str(baseline_prompt_path)
            output_files["enhanced_prompt"] = str(enhanced_prompt_path)
            output_files["baseline_analysis"] = str(baseline_analysis_path)
            output_files["enhanced_analysis"] = str(enhanced_analysis_path)

        for item in result.solitary_rounds:
            round_id = item.get("round", "unknown")
            round_prompt_path = run_dir / f"solitary_round{round_id}_prompt.md"
            round_analysis_path = run_dir / f"solitary_round{round_id}_analysis.md"
            round_prompt_path.write_text(item.get("prompt", ""), encoding="utf-8")
            round_analysis_path.write_text(item.get("analysis", ""), encoding="utf-8")
            output_files[f"solitary_round{round_id}_prompt"] = str(round_prompt_path)
            output_files[f"solitary_round{round_id}_analysis"] = str(round_analysis_path)

        for item in result.communal_rounds:
            round_id = item.get("round", "unknown")
            round_prompt_path = run_dir / f"communal_guest{round_id}_prompt.md"
            round_analysis_path = run_dir / f"communal_guest{round_id}_analysis.md"
            round_prompt_path.write_text(item.get("prompt", ""), encoding="utf-8")
            round_analysis_path.write_text(item.get("analysis", ""), encoding="utf-8")
            output_files[f"communal_guest{round_id}_prompt"] = str(round_prompt_path)
            output_files[f"communal_guest{round_id}_analysis"] = str(round_analysis_path)

        if result.mode == "communal":
            communal_summary_path = run_dir / "communal_summary.md"
            communal_summary_path.write_text(result.enhanced_analysis, encoding="utf-8")
            output_files["communal_summary"] = str(communal_summary_path)

        if result.mode == "slot_pipe":
            slot_pipe_dir = run_dir / "slot_pipe"
            slot_pipe_dir.mkdir(parents=True, exist_ok=True)
            output_files["slot_pipe_dir"] = str(slot_pipe_dir)

            slots_timeline: list[dict] = []
            for layer in result.slot_pipe_layers:
                layer_index = int(layer.get("layer_index", 0) or 0)
                layer_name = str(layer.get("layer_name", ""))
                layer_prefix = f"layer{layer_index}_{self._slug(layer_name or str(layer_index))}"

                slots_timeline.append(
                    {
                        "layer_index": layer_index,
                        "layer_name": layer_name,
                        "slots_before": layer.get("slots_before", []),
                        "slots_after": layer.get("slots_after", []),
                    }
                )

                layer_judge_path = slot_pipe_dir / f"{layer_prefix}_judge.json"
                layer_judge_path.write_text(
                    json.dumps(layer.get("layer_judge", {}), ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output_files[f"{layer_prefix}_judge"] = str(layer_judge_path)

                layer_judge_prompt_path = slot_pipe_dir / f"{layer_prefix}_judge_prompt.md"
                layer_judge_analysis_path = slot_pipe_dir / f"{layer_prefix}_judge_analysis.md"
                layer_judge_prompt_path.write_text(str((layer.get("layer_judge", {}) or {}).get("prompt_text", "")), encoding="utf-8")
                layer_judge_analysis_path.write_text(str((layer.get("layer_judge", {}) or {}).get("response_text", "")), encoding="utf-8")
                output_files[f"{layer_prefix}_judge_prompt"] = str(layer_judge_prompt_path)
                output_files[f"{layer_prefix}_judge_analysis"] = str(layer_judge_analysis_path)

                for slot_info in layer.get("slots", []):
                    slot = str(slot_info.get("slot", ""))
                    slot_slug = self._slug(slot)
                    for attempt in slot_info.get("agent_attempts", []):
                        agent_index = int(attempt.get("agent_index", 0) or 0)
                        attempt_index = int(attempt.get("attempt_index", 0) or 0)
                        base = f"{layer_prefix}_slot_{slot_slug}_agent{agent_index}_attempt{attempt_index}"
                        prompt_path = slot_pipe_dir / f"{base}_prompt.md"
                        analysis_path = slot_pipe_dir / f"{base}_analysis.md"
                        prompt_path.write_text(str(attempt.get("prompt_text", "")), encoding="utf-8")
                        analysis_path.write_text(str(attempt.get("response_text", "")), encoding="utf-8")
                        output_files[f"{base}_prompt"] = str(prompt_path)
                        output_files[f"{base}_analysis"] = str(analysis_path)

                    pooled_path = slot_pipe_dir / f"{layer_prefix}_slot_{slot_slug}_pooled.json"
                    pooled_path.write_text(
                        json.dumps(slot_info.get("final_points", []), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    output_files[f"{layer_prefix}_slot_{slot_slug}_pooled"] = str(pooled_path)

                    judge_rounds_path = slot_pipe_dir / f"{layer_prefix}_slot_{slot_slug}_judge_rounds.json"
                    judge_rounds_path.write_text(
                        json.dumps(slot_info.get("judge_rounds", []), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    output_files[f"{layer_prefix}_slot_{slot_slug}_judge_rounds"] = str(judge_rounds_path)

                    for round_index, judge_item in enumerate(slot_info.get("judge_rounds", []), start=1):
                        if not isinstance(judge_item, dict):
                            continue
                        if str(judge_item.get("stage", "")) != "slot_pipe_slot_judge":
                            continue
                        judge_prompt_path = slot_pipe_dir / f"{layer_prefix}_slot_{slot_slug}_judge_round{round_index}_prompt.md"
                        judge_analysis_path = slot_pipe_dir / f"{layer_prefix}_slot_{slot_slug}_judge_round{round_index}_analysis.md"
                        judge_prompt_path.write_text(str(judge_item.get("prompt_text", "")), encoding="utf-8")
                        judge_analysis_path.write_text(str(judge_item.get("response_text", "")), encoding="utf-8")
                        output_files[f"{layer_prefix}_slot_{slot_slug}_judge_round{round_index}_prompt"] = str(judge_prompt_path)
                        output_files[f"{layer_prefix}_slot_{slot_slug}_judge_round{round_index}_analysis"] = str(judge_analysis_path)

            slots_timeline_path = slot_pipe_dir / "slots_timeline.json"
            slots_timeline_path.write_text(
                json.dumps(slots_timeline, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_files["slot_pipe_slots_timeline"] = str(slots_timeline_path)

            final_prompt_path = slot_pipe_dir / "final_prompt_supplement.md"
            final_prompt_path.write_text(result.enhanced_prompt, encoding="utf-8")
            output_files["slot_pipe_final_prompt"] = str(final_prompt_path)

            final_appreciation_path = slot_pipe_dir / "final_appreciation.md"
            final_appreciation_path.write_text(result.enhanced_analysis, encoding="utf-8")
            output_files["slot_pipe_final_appreciation"] = str(final_appreciation_path)

            final_points_path = slot_pipe_dir / "final_slot_points.json"
            final_points_payload: dict[str, list[dict]] = {}
            if result.slot_pipe_layers:
                last_layer = result.slot_pipe_layers[-1]
                for slot_info in last_layer.get("slots", []):
                    slot = str(slot_info.get("slot", "")).strip()
                    if not slot:
                        continue
                    points = []
                    for item in slot_info.get("final_points", []):
                        if not isinstance(item, dict):
                            continue
                        points.append(
                            {
                                "point": item.get("point", ""),
                                "sources": item.get("sources", []),
                                "support": item.get("support", 0),
                            }
                        )
                    final_points_payload[slot] = points
            final_points_path.write_text(
                json.dumps(final_points_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            output_files["slot_pipe_final_points"] = str(final_points_path)

            if result.slot_pipe_v4:
                v4_payload_path = slot_pipe_dir / "v4_final_slots.json"
                v4_payload_path.write_text(
                    json.dumps(result.slot_pipe_v4, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                output_files["slot_pipe_v4_final_slots"] = str(v4_payload_path)

        api_calls_path.write_text(
            "\n".join(
                [json.dumps({"type": "run_meta", "mode": result.mode}, ensure_ascii=False)]
                + [json.dumps(item, ensure_ascii=False) for item in result.api_logs]
            ),
            encoding="utf-8",
        )

        image_delivery_checks = self._build_image_delivery_checks(result.api_logs)
        image_delivery_checks_path.write_text(
            json.dumps(
                {
                    "mode": result.mode,
                    "summary": {
                        "total_calls": len(image_delivery_checks),
                        "request_has_image_true": sum(1 for x in image_delivery_checks if x.get("request_has_image")),
                        "image_attached_true": sum(1 for x in image_delivery_checks if x.get("image_attached")),
                        "model_ack_heuristic_true": sum(1 for x in image_delivery_checks if x.get("model_ack_heuristic")),
                    },
                    "checks": image_delivery_checks,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        report_path.write_text(
            json.dumps(
                {
                    "mode": result.mode,
                    "run_dir": str(run_dir),
                    "selected_slots": result.selected_slots,
                    "token_usage": result.token_usage,
                    "api_logs": result.api_logs,
                    "solitary_rounds": result.solitary_rounds,
                    "communal_rounds": result.communal_rounds,
                    "slot_pipe_layers": result.slot_pipe_layers,
                    "logs": result.logs,
                    "output_files": output_files,
                    "api_calls_file": str(api_calls_path),
                    "image_delivery_checks_file": str(image_delivery_checks_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        output_files["api_calls"] = str(api_calls_path)
        output_files["image_delivery_checks"] = str(image_delivery_checks_path)
        output_files["report"] = str(report_path)
        return output_files

    @staticmethod
    def _build_image_delivery_checks(api_logs: list[dict]) -> list[dict]:
        checks: list[dict] = []
        visual_keywords = [
            "图像",
            "图片",
            "画面",
            "可见",
            "可观察",
            "从图中",
            "从画中",
            "image",
            "visual",
        ]

        for item in api_logs:
            request_summary = item.get("request_summary") or {}
            response_summary = item.get("response_summary") or {}
            content_preview = str(response_summary.get("content_preview") or "")
            lowered_preview = content_preview.lower()

            matched_keyword = ""
            for kw in visual_keywords:
                if (kw in content_preview) or (kw in lowered_preview):
                    matched_keyword = kw
                    break

            request_has_image = bool(request_summary.get("has_image"))
            image_attached = bool(item.get("image_attached"))
            checks.append(
                {
                    "stage": item.get("stage"),
                    "slot": item.get("slot"),
                    "inference_kind": item.get("inference_kind"),
                    "model": item.get("model"),
                    "status_code": item.get("status_code"),
                    "ok": item.get("ok"),
                    "error": item.get("error"),
                    "request_has_image": request_has_image,
                    "image_attached": image_attached,
                    "transport_ok": request_has_image and image_attached,
                    "model_ack_heuristic": bool(matched_keyword),
                    "model_ack_keyword": matched_keyword or None,
                    "response_content_preview": content_preview,
                }
            )

        return checks

    @staticmethod
    def _sanitize_image_unavailable_text(text: str) -> str:
        if not text.strip():
            return text

        blocked_phrases = [
            "无法读取图像",
            "无法直接读取图像",
            "未读到图像",
            "没有读到图像",
            "看不到图像",
            "看不到图片",
            "未看到图像",
            "没有看到图像",
            "无法查看图像",
            "cannot read the image",
            "cannot access the image",
            "cannot see the image",
        ]

        kept_lines: list[str] = []
        for line in text.splitlines():
            lowered_line = line.strip().lower()
            if not lowered_line:
                kept_lines.append(line)
                continue
            if any(phrase in line or phrase in lowered_line for phrase in blocked_phrases):
                continue
            kept_lines.append(line)

        cleaned = "\n".join(kept_lines).strip()
        return cleaned or text

    @staticmethod
    def _extract_solitary_analysis_for_next_round(text: str) -> str:
        if not text.strip():
            return text

        marker = "【本轮赏析】"
        if marker in text:
            after = text.split(marker, 1)[1].strip()
            return after or text

        fallback_markers = ["本轮赏析", "重新鉴赏", "赏析正文", "最终赏析"]
        lowered = text.lower()
        for m in fallback_markers:
            idx = lowered.find(m.lower())
            if idx >= 0:
                sliced = text[idx + len(m):].lstrip("：: \n")
                if sliced.strip():
                    return sliced.strip()

        return text
