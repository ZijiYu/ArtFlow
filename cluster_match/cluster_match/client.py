from __future__ import annotations

import time
import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from time import perf_counter
from typing import Any


@dataclass(slots=True)
class ChatResult:
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str
    endpoint: str
    status_code: int | None
    duration_ms: float
    error: str | None = None


class ChatCompletionsClient:
    def __init__(self, api_key: str, base_url: str, timeout: int = 180, max_retries: int = 2) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, max_retries)

    @property
    def endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @staticmethod
    def _collect_text(value: Any) -> list[str]:
        if isinstance(value, str):
            return [value] if value.strip() else []
        if isinstance(value, list):
            chunks: list[str] = []
            for item in value:
                chunks.extend(ChatCompletionsClient._collect_text(item))
            return chunks
        if isinstance(value, dict):
            chunks: list[str] = []
            for key in ("text", "content", "output_text", "reasoning_content"):
                if key in value:
                    chunks.extend(ChatCompletionsClient._collect_text(value[key]))
            return chunks
        return []

    @classmethod
    def _extract_content(cls, parsed: dict[str, Any]) -> str | None:
        try:
            message = parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        content = "\n".join(cls._collect_text(message)).strip()
        return content or None

    @staticmethod
    def _extract_usage(parsed: dict[str, Any], prompt_text: str, content: str | None) -> tuple[int, int, int]:
        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        if isinstance(usage, dict) and int(usage.get("total_tokens", 0) or 0) > 0:
            return (
                int(usage.get("prompt_tokens", 0) or 0),
                int(usage.get("completion_tokens", 0) or 0),
                int(usage.get("total_tokens", 0) or 0),
            )
        prompt_tokens = max(1, len(prompt_text) // 4)
        completion_tokens = max(1, len(content or "") // 4)
        return prompt_tokens, completion_tokens, prompt_tokens + completion_tokens

    def chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        model: str,
        temperature: float = 0.0,
    ) -> ChatResult:
        payload = {
            "model": model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        attempts = self.max_retries + 1

        for attempt in range(1, attempts + 1):
            request = urllib.request.Request(
                self.endpoint,
                data=body,
                headers=self._headers(),
                method="POST",
            )
            started_at = perf_counter()
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    parsed = json.loads(response.read().decode("utf-8"))
                    content = self._extract_content(parsed)
                    prompt_tokens, completion_tokens, total_tokens = self._extract_usage(parsed, user_prompt, content)
                    return ChatResult(
                        content=content,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=total_tokens,
                        model=str(parsed.get("model", model)),
                        endpoint=self.endpoint,
                        status_code=int(getattr(response, "status", 200) or 200),
                        duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
                    )
            except urllib.error.HTTPError as exc:
                if exc.code in {408, 409, 425, 429} or exc.code >= 500:
                    if attempt < attempts:
                        time.sleep(min(5.0, 0.8 * attempt))
                        continue
                return ChatResult(
                    content=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=model,
                    endpoint=self.endpoint,
                    status_code=exc.code,
                    duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
                    error=f"http_error:{exc.code}",
                )
            except Exception as exc:  # noqa: BLE001
                if attempt < attempts:
                    time.sleep(min(5.0, 0.8 * attempt))
                    continue
                return ChatResult(
                    content=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model=model,
                    endpoint=self.endpoint,
                    status_code=None,
                    duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
                    error=str(exc),
                )

        return ChatResult(
            content=None,
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            model=model,
            endpoint=self.endpoint,
            status_code=None,
            duration_ms=0.0,
            error="unknown_retry_exhausted",
        )
