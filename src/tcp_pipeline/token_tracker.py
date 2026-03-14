from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class TokenTracker:
    """Approximate token usage by stage without external dependencies."""

    stage_tokens: dict[str, int] = field(default_factory=dict)
    stage_api_tokens: dict[str, int] = field(default_factory=dict)
    total_tokens: int = 0
    api_total_tokens: int = 0

    def add_text(self, stage: str, text: str) -> int:
        tokens = max(1, len(text) // 4)
        self.stage_tokens[stage] = self.stage_tokens.get(stage, 0) + tokens
        self.total_tokens += tokens
        return tokens

    def snapshot(self) -> dict[str, int]:
        payload = dict(self.stage_tokens)
        for stage, value in self.stage_api_tokens.items():
            payload[f"api_{stage}"] = value
        payload["total"] = self.total_tokens
        payload["api_total"] = self.api_total_tokens
        return payload

    def add_api_usage(self, stage: str, total_tokens: int) -> int:
        used = max(0, int(total_tokens))
        self.stage_api_tokens[stage] = self.stage_api_tokens.get(stage, 0) + used
        self.api_total_tokens += used
        self.total_tokens += used
        return used
