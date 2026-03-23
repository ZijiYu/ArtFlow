from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .clients import LLMClient, LlmChatLogger, OpenAIJsonClient, RecordingLLMClient
from .config import PipelineConfig
from .pipeline import ContextLogger, _prepare_image_payload


class DownstreamPromptRunner:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        llm_client: LLMClient | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or PipelineConfig.from_env()
        base_llm = llm_client or OpenAIJsonClient(self._config, client=client)
        self._llm = RecordingLLMClient(base_llm, LlmChatLogger(self._config.llm_chat_record_path))
        self._context = ContextLogger(self._config.context_path)

    def run_json(
        self,
        *,
        task_name: str,
        system_prompt: str,
        user_text: str,
        image_file: str | Path | None = None,
        reset_llm_chat_record: bool = False,
    ) -> dict[str, Any]:
        if reset_llm_chat_record:
            self._llm.reset_log()

        image_path = Path(image_file) if image_file else None
        image_base64: str | None = None
        image_mime_type = "image/png"
        if image_path is not None:
            image_base64, _, image_mime_type, image_size = _prepare_image_payload(
                image_path,
                max_pixels=self._config.max_image_pixels,
            )
            image_summary = f"image=`{image_path}` size={image_size[0]}x{image_size[1]}"
        else:
            image_summary = "image=`none`"

        try:
            response = self._llm.complete_json(
                system_prompt=system_prompt,
                user_text=user_text,
                image_base64=image_base64,
                image_mime_type=image_mime_type,
            )
        except Exception as exc:  # noqa: BLE001
            response = {
                "status": "needs_followup",
                "merge_candidates": [],
                "new_slots": [],
                "ontology_updates": [],
                "text_evidence_updates": [],
                "search_queries": [],
                "resolved_questions": [],
                "open_questions": [f"downstream_json_parse_failed: {exc}"],
                "notes": [f"downstream JSON parse failed; no updates were applied for task `{task_name}`."],
            }

        self._context.append(
            "Downstream Prompt",
            [
                f"task=`{task_name}`",
                image_summary,
                f"system_prompt={json.dumps(system_prompt, ensure_ascii=False)}",
                f"user_text={json.dumps(user_text, ensure_ascii=False)}",
                f"response={json.dumps(response, ensure_ascii=False)}",
            ],
        )
        return response
