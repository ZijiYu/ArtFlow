from __future__ import annotations

import json
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
    build_solitary_first_prompt,
    build_solitary_reflection_prompt,
)
from .slots import normalize_slots
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
    agent_model: str | None = None
    baseline_model: str | None = None
    enhanced_model: str | None = None


class TcpPromptPipeline:
    """Simplified v1 pipeline: one-round agents + two prompts + two VLM outputs."""

    def __init__(self, config: PipelineConfig | None = None, api_client: NewAPIClient | None = None) -> None:
        self.config = config or PipelineConfig()
        self.api_client = api_client or NewAPIClient()
        self.tracker = TokenTracker()
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
            token_usage=self.tracker.snapshot(),
            api_logs=api_logs,
            logs=logs,
        )

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
