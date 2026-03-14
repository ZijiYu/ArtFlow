from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from .models import AgentOutput
from .new_api_client import ChatResult, NewAPIClient
from .slots import slot_label


class SlotAgentRunner:
    def __init__(self, api_client: NewAPIClient, temperature: float = 0.7, agent_model: str | None = None) -> None:
        self.api_client = api_client
        self.temperature = temperature
        self.agent_model = agent_model

    def run_parallel(self, image_path: str, meta: dict, slots: list[str]) -> tuple[list[AgentOutput], list[dict[str, Any]]]:
        outputs: list[AgentOutput] = []
        api_logs: list[dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, len(slots))) as pool:
            future_map = {pool.submit(self._run_one, slot, image_path, meta): slot for slot in slots}
            for future, slot in future_map.items():
                try:
                    output, api_log = future.result()
                except Exception as exc:
                    output = AgentOutput(agent_name=slot, context_text=self._fallback_context(slot))
                    api_log = {
                        "stage": "slot_agent",
                        "slot": slot,
                        "ok": False,
                        "error": f"runner_exception: {exc}",
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "status_code": None,
                    }
                outputs.append(output)
                api_logs.append(api_log)
        return outputs, api_logs

    def _run_one(self, slot: str, image_path: str, meta: dict) -> tuple[AgentOutput, dict[str, Any]]:
        slot_zh = slot_label(slot)
        fallback = self._fallback_context(slot)

        if not self.api_client.enabled:
            return (
                AgentOutput(agent_name=slot, context_text=fallback),
                {
                    "stage": "slot_agent",
                    "slot": slot,
                    "ok": False,
                    "error": "api_not_enabled",
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                    "status_code": None,
                },
            )

        result: ChatResult = self.api_client.chat(
            system_prompt="你是国画分领域鉴赏助手。你的任务是只围绕一个slot产出可读的鉴赏要点。",
            user_prompt=(
                f"槽位={slot_zh}\n"
                "槽位关注点=请仅围绕该槽位提供鉴赏要点，不要扩散到其他槽位。\n"
                f"image_path={image_path}\n"
                f"meta={json.dumps(meta, ensure_ascii=False)}\n"
                "请输出人类可读的“鉴赏要点”，共3到5条，每条一行，以“- ”开头。"
                "文字尽量自然，不要模板化。"
                "不要输出JSON，不要解释你自己。"
            ),
            temperature=self.temperature,
            image_path=image_path,
            model=self.agent_model,
        )
        api_log = {
            "stage": "slot_agent",
            "slot": slot,
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
        }
        if not result.content:
            return AgentOutput(agent_name=slot, context_text=fallback), api_log

        cleaned = self._normalize_context_text(result.content)
        if not cleaned:
            api_log["ok"] = False
            api_log["error"] = "empty_context_after_normalize"
            return AgentOutput(agent_name=slot, context_text=fallback), api_log

        return AgentOutput(agent_name=slot, context_text=cleaned), api_log

    @staticmethod
    def _fallback_context(slot: str) -> str:
        templates = {
            "brush-and-ink": "- 山石与树木的笔墨处理层次清楚，能看出用笔的节奏变化。",
            "composition": "- 画面前后景关系明确，视线会自然被引向纵深处。",
            "color": "- 色彩与墨色层次衔接自然，冷暖关系形成了稳定的画面节奏。",
            "spirit": "- 整体气息沉静，留白和实景之间形成了舒展的呼吸感。",
            "school": "- 风格特征具有明确的流派取向，笔墨语言呈现出稳定的传统谱系。",
            "history": "- 作品在时代语境中可见典型审美取向，与历史山水传统形成呼应。",
            "literary": "- 作品呈现出文人观看自然时的抒情意味，画外有诗意联想空间。",
            "medium": "- 画面墨色渗化与肌理变化体现了传统水墨媒介的特征。",
        }
        return templates.get(slot, "- 该slot暂无稳定观点，建议补充图像细节后再做鉴赏。")

    @staticmethod
    def _normalize_context_text(text: str) -> str:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        normalized: list[str] = []
        for line in lines:
            if line.startswith(("-", "*", "•")):
                normalized.append(f"- {line.lstrip('-*• ').strip()}")
            else:
                normalized.append(line)
        return "\n".join(normalized[:6]).strip()
