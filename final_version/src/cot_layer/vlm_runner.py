from __future__ import annotations

from typing import Any

from .new_api_client import NewAPIClient


class VLMRunner:
    def __init__(self, api_client: NewAPIClient) -> None:
        self.api_client = api_client

    def analyze(
        self,
        *,
        image_path: str | None,
        prompt: str,
        system_prompt: str,
        temperature: float,
        model: str | None,
        stage: str,
    ) -> tuple[str, dict[str, Any]]:
        user_prompt = (
            "请严格依据图像和任务提示执行分析。若图像局部无法辨认，请直接说明不确定，不要猜测。\n\n"
            f"任务提示:\n{prompt}"
        )
        result = self.api_client.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            temperature=temperature,
            image_path=image_path,
            model=model,
        )
        api_log = {
            "stage": stage,
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
        }
        print(
            f"[slots_v2_request] stage={stage} model={result.model or model or ''} "
            f"duration_ms={result.duration_ms:.2f} ok={str(bool(result.content)).lower()} "
            f"image={str(result.image_attached).lower()}",
            flush=True,
        )
        return result.content or "", api_log
