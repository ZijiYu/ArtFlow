from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from .agents import SlotAgentRunner
from .models import PipelineResult
from .new_api_client import NewAPIClient
from .prompt_builder import build_baseline_prompt, build_enhanced_prompt
from .slots import normalize_slots
from .token_tracker import TokenTracker
from .vlm_runner import VLMRunner


@dataclass(slots=True)
class PipelineConfig:
    slots: list[str] = field(default_factory=list)
    agent_temperature: float = 0.7
    vlm_temperature: float = 0.2
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
            selected_slots=selected_slots,
            slot_context=slot_context,
            baseline_prompt=baseline_prompt,
            enhanced_prompt=enhanced_prompt,
            baseline_analysis=baseline_analysis,
            enhanced_analysis=enhanced_analysis,
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

        slot_path = run_dir / "slot_context.md"
        baseline_prompt_path = run_dir / "prompt_baseline.md"
        enhanced_prompt_path = run_dir / "prompt_enhanced.md"
        baseline_analysis_path = run_dir / "analysis_baseline.md"
        enhanced_analysis_path = run_dir / "analysis_enhanced.md"
        api_calls_path = run_dir / "api_calls.jsonl"
        report_path = run_dir / "report.json"

        slot_sections = ["# Slot Context (Human Readable)"]
        for slot in result.selected_slots:
            text = result.slot_context.get(slot, "").strip()
            if not text:
                continue
            slot_sections.append(f"\n## {slot}\n{text}\n")
        slot_path.write_text("\n".join(slot_sections), encoding="utf-8")
        baseline_prompt_path.write_text(result.baseline_prompt, encoding="utf-8")
        enhanced_prompt_path.write_text(result.enhanced_prompt, encoding="utf-8")
        baseline_analysis_path.write_text(result.baseline_analysis, encoding="utf-8")
        enhanced_analysis_path.write_text(result.enhanced_analysis, encoding="utf-8")
        api_calls_path.write_text(
            "\n".join(json.dumps(item, ensure_ascii=False) for item in result.api_logs),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "selected_slots": result.selected_slots,
                    "token_usage": result.token_usage,
                    "api_logs": result.api_logs,
                    "logs": result.logs,
                    "slot_context_file": str(slot_path),
                    "baseline_prompt_file": str(baseline_prompt_path),
                    "enhanced_prompt_file": str(enhanced_prompt_path),
                    "baseline_analysis_file": str(baseline_analysis_path),
                    "enhanced_analysis_file": str(enhanced_analysis_path),
                    "api_calls_file": str(api_calls_path),
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        return {
            "run_dir": str(run_dir),
            "slot_context": str(slot_path),
            "baseline_prompt": str(baseline_prompt_path),
            "enhanced_prompt": str(enhanced_prompt_path),
            "baseline_analysis": str(baseline_analysis_path),
            "enhanced_analysis": str(enhanced_analysis_path),
            "api_calls": str(api_calls_path),
            "report": str(report_path),
        }
