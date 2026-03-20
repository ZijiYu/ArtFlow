from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AgentOutput:
    agent_name: str
    context_text: str


@dataclass(slots=True)
class PipelineResult:
    mode: str
    selected_slots: list[str]
    slot_context: dict[str, str]
    baseline_prompt: str
    enhanced_prompt: str
    baseline_analysis: str
    enhanced_analysis: str
    solitary_rounds: list[dict[str, str]]
    communal_rounds: list[dict[str, str]]
    slot_pipe_layers: list[dict[str, Any]]
    token_usage: dict[str, int]
    api_logs: list[dict[str, Any]]
    logs: list[dict[str, Any]]
    slot_pipe_v4: dict[str, Any] = field(default_factory=dict)
