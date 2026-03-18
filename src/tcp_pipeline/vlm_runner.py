from __future__ import annotations

from typing import Any

from .new_api_client import NewAPIClient


class VLMRunner:
    def __init__(self, api_client: NewAPIClient, model: str | None = None) -> None:
        self.api_client = api_client
        self.model = model

    def analyze(
        self,
        image_path: str,
        prompt: str,
        temperature: float = 0.2,
        model: str | None = None,
        inference_kind: str = "unknown",
    ) -> tuple[str, dict[str, Any]]:
        user_prompt = (
            f"图像路径: {image_path}\n"
            "请基于该图像进行国画鉴赏。默认你已接收到图像，请直接依据图像内容分析；仅当请求中确实没有图像时，才说明无法读取并谨慎给出结论。\n\n"
            f"任务提示: {prompt}"
        )
        result = self.api_client.chat(
            system_prompt="你是视觉语言模型鉴赏助手。",
            user_prompt=user_prompt,
            temperature=temperature,
            image_path=image_path,
            model=model or self.model,
        )
        api_log = {
            "stage": "vlm_analyze",
            "inference_kind": inference_kind,
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
        if not result.image_attached and not api_log["error"]:
            api_log["error"] = "image_not_attached: check --image path or use http(s) image url"
        if result.content:
            return result.content, api_log
        return "API不可用或调用失败，未获得VLM赏析结果。", api_log
