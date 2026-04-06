from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

from .config_loader import DEFAULT_CONFIG_PATH, get_config_value, load_yaml_config, read_text_secret_file


@dataclass(slots=True)
class ChatResult:
    content: str | None
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str = ""
    error: str | None = None
    endpoint: str = ""
    status_code: int | None = None
    image_attached: bool = False
    duration_ms: float = 0.0


@dataclass(slots=True)
class NewAPIClient:
    api_key: str | None = None
    api_key_file: str | None = None
    api_key_line: int | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int | None = None
    config_path: str | None = None
    allow_remote_image_url: bool | None = None
    local_image_mode: str | None = None

    def __post_init__(self) -> None:
        config = load_yaml_config(self.config_path or DEFAULT_CONFIG_PATH)
        api_key_value = str(get_config_value(config, "api", "key", default="")).strip()
        api_key_file = str(self.api_key_file or get_config_value(config, "api", "key_file", default="")).strip()
        api_key_line = max(1, int(get_config_value(config, "api", "key_line", default=1) if self.api_key_line is None else self.api_key_line))
        api_key_from_file = read_text_secret_file(api_key_file, line_number=api_key_line)
        api_key_env = str(get_config_value(config, "api", "key_env", default="OPENAI_API_KEY")).strip() or "OPENAI_API_KEY"
        shared_model = str(get_config_value(config, "models", "default", default="")).strip()
        self.api_key = (
            self.api_key
            or api_key_value
            or api_key_from_file
            or os.getenv(api_key_env)
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("NEW_API_KEY")
        )
        self.base_url = (
            self.base_url
            or os.getenv("NEW_API_BASE_URL")
            or str(get_config_value(config, "api", "base_url", default="")).strip()
        ).rstrip("/")
        self.model = (
            self.model
            or os.getenv("NEW_API_MODEL")
            or str(get_config_value(config, "api", "model", default="")).strip()
            or shared_model
        )
        self.timeout = max(1, int(self.timeout or get_config_value(config, "api", "timeout", default=60) or 60))
        self.allow_remote_image_url = bool(
            get_config_value(config, "image", "allow_remote_url", default=True)
            if self.allow_remote_image_url is None
            else self.allow_remote_image_url
        )
        self.local_image_mode = (
            self.local_image_mode
            or str(get_config_value(config, "image", "local_input_mode", default="data_url")).strip()
            or "data_url"
        )

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _endpoint(self) -> str:
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _make_user_content(self, user_prompt: str, image_path: str | None) -> tuple[Any, bool]:
        if not image_path:
            return user_prompt, False
        if self.allow_remote_image_url and image_path.startswith(("http://", "https://")):
            return [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_path}},
            ], True
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return user_prompt, False
        mime_type, _ = mimetypes.guess_type(str(path))
        mime_type = mime_type or "image/png"
        if (self.local_image_mode or "data_url").lower() != "data_url":
            return user_prompt, False
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{data}"}},
        ], True

    @staticmethod
    def _extract_content(parsed: dict[str, Any]) -> str | None:
        def collect(value: Any) -> list[str]:
            if isinstance(value, str):
                return [value] if value.strip() else []
            if isinstance(value, dict):
                chunks: list[str] = []
                for key in ("text", "content", "output_text", "reasoning_content"):
                    if key in value:
                        chunks.extend(collect(value[key]))
                return chunks
            if isinstance(value, list):
                chunks: list[str] = []
                for item in value:
                    chunks.extend(collect(item))
                return chunks
            return []

        try:
            message = parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None
        merged = "\n".join(collect(message)).strip()
        return merged or None

    @staticmethod
    def _extract_usage(parsed: dict[str, Any], prompt_text: str, content: str | None) -> tuple[int, int, int]:
        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        if isinstance(usage, dict) and int(usage.get("total_tokens", 0) or 0) > 0:
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            return prompt, completion, int(usage.get("total_tokens", 0) or 0)
        prompt = max(1, len(prompt_text) // 4)
        completion = max(1, len(content or "") // 4)
        return prompt, completion, prompt + completion

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.7,
        image_path: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        if not self.enabled:
            return ChatResult(
                content=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model or self.model or "",
                error="api_not_configured",
            )

        endpoint = self._endpoint()
        user_content, image_attached = self._make_user_content(user_prompt=user_prompt, image_path=image_path)
        payload = {
            "model": model or self.model,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(endpoint, data=body, headers=self._headers(), method="POST")
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
                    model=str(parsed.get("model", model or self.model or "")),
                    endpoint=endpoint,
                    status_code=int(getattr(response, "status", 200) or 200),
                    image_attached=image_attached,
                    duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
                )
        except urllib.error.HTTPError as exc:
            return ChatResult(
                content=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model or self.model or "",
                error=f"http_error:{exc.code}",
                endpoint=endpoint,
                status_code=exc.code,
                image_attached=image_attached,
                duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
        except Exception as exc:  # noqa: BLE001
            return ChatResult(
                content=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model or self.model or "",
                error=str(exc),
                endpoint=endpoint,
                image_attached=image_attached,
                duration_ms=round((perf_counter() - started_at) * 1000.0, 2),
            )
