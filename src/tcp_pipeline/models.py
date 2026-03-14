from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class AgentOutput:
    agent_name: str
    context_text: str


@dataclass(slots=True)
class PipelineResult:
    selected_slots: list[str]
    slot_context: dict[str, str]
    baseline_prompt: str
    enhanced_prompt: str
    baseline_analysis: str
    enhanced_analysis: str
    token_usage: dict[str, int]
    api_logs: list[dict[str, Any]]
    logs: list[dict[str, Any]]
