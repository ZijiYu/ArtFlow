from __future__ import annotations

import base64
import json
import mimetypes
import os
import socket
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any


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
    request_summary: dict[str, Any] | None = None
    response_summary: dict[str, Any] | None = None


@dataclass(slots=True)
class NewAPIClient:
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None
    timeout: int = 60

    def __post_init__(self) -> None:
        self.api_key = self.api_key or os.getenv("NEW_API_KEY")
        self.base_url = (self.base_url or os.getenv("NEW_API_BASE_URL") or "").rstrip("/")
        self.model = self.model or os.getenv("NEW_API_MODEL") or "qwen/qwen-2.5-vl-7b-instruct"
        timeout_from_env = os.getenv("NEW_API_TIMEOUT")
        if timeout_from_env:
            try:
                self.timeout = max(1, int(timeout_from_env))
            except ValueError:
                pass

    def _request_headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    @property
    def enabled(self) -> bool:
        return bool(self.api_key and self.base_url)

    def _endpoint(self) -> str:
        # Supports both forms:
        # 1) base_url=https://host/v1  -> /chat/completions appended
        # 2) base_url=https://host/v1/chat/completions -> used directly
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        return f"{self.base_url}/chat/completions"

    def _embeddings_endpoint(self) -> str:
        # Supports both forms:
        # 1) base_url=https://host/v1  -> /embeddings appended
        # 2) base_url=https://host/v1/embeddings -> used directly
        # 3) base_url=https://host/v1/chat/completions -> replaced with /embeddings
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        if self.base_url.endswith("/chat/completions"):
            return self.base_url[: -len("/chat/completions")] + "/embeddings"
        return f"{self.base_url}/embeddings"

    @staticmethod
    def _extract_content(parsed: dict[str, Any]) -> str | None:
        def collect_text(value: Any) -> list[str]:
            chunks: list[str] = []
            if isinstance(value, str):
                if value.strip():
                    chunks.append(value)
                return chunks
            if isinstance(value, dict):
                for key in ("text", "content", "output_text", "reasoning_content"):
                    if key in value:
                        chunks.extend(collect_text(value[key]))
                return chunks
            if isinstance(value, list):
                for item in value:
                    chunks.extend(collect_text(item))
                return chunks
            return chunks

        try:
            message = parsed["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return None

        merged = "\n".join(collect_text(message)).strip()
        if merged:
            return merged

        # Some providers put text at top-level fields.
        merged_top = "\n".join(collect_text(parsed)).strip()
        return merged_top or None

    @staticmethod
    def _make_user_content(user_prompt: str, image_path: str | None) -> tuple[Any, bool]:
        if not image_path:
            return user_prompt, False

        # Allow remote images directly for providers that accept image_url.
        if image_path.startswith("http://") or image_path.startswith("https://"):
            return [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": image_path}},
            ], True

        path = Path(image_path)
        if not path.exists() or not path.is_file():
            # Keep compatibility when user passes a non-local image identifier.
            return user_prompt, False

        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/jpeg"

        raw = path.read_bytes()
        data_b64 = base64.b64encode(raw).decode("ascii")
        data_url = f"data:{mime_type};base64,{data_b64}"

        return [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": data_url}},
        ], True

    @staticmethod
    def _extract_usage(parsed: dict[str, Any], prompt_text: str, output_text: str | None) -> tuple[int, int, int]:
        usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens", 0) or 0)
            completion = int(usage.get("completion_tokens", 0) or 0)
            total = int(usage.get("total_tokens", 0) or 0)
            if total > 0:
                return prompt, completion, total

        # Fallback approximation when provider omits usage.
        prompt = max(1, len(prompt_text) // 4)
        completion = max(1, len(output_text or "") // 4)
        return prompt, completion, prompt + completion

    @staticmethod
    def _brief(text: str, limit: int = 220) -> str:
        text = " ".join(text.split())
        if len(text) <= limit:
            return text
        return text[: limit - 3] + "..."

    def chat(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
        image_path: str | None = None,
        model: str | None = None,
    ) -> ChatResult:
        model_used = model or self.model or ""
        request_summary = {
            "model": model_used,
            "temperature": temperature,
            "has_image": bool(image_path),
            "image_path": image_path,
            "system_prompt_preview": self._brief(system_prompt),
            "user_prompt_preview": self._brief(user_prompt),
        }

        if not self.enabled:
            return ChatResult(
                content=None,
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                model=model_used,
                error="api_not_enabled: missing NEW_API_KEY or NEW_API_BASE_URL",
                endpoint=self._endpoint() if self.base_url else "",
                request_summary=request_summary,
                response_summary={"error": "api_not_enabled"},
            )

        user_content, image_attached = self._make_user_content(user_prompt, image_path)
        request_summary = {
            "model": model_used,
            "temperature": temperature,
            "has_image": bool(image_attached),
            "image_path": image_path,
            "system_prompt_preview": self._brief(system_prompt),
            "user_prompt_preview": self._brief(user_prompt),
        }

        payload = {
            "model": model_used,
            "temperature": temperature,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=self._endpoint(),
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status_code = getattr(resp, "status", None)
            parsed = json.loads(raw)
            content = self._extract_content(parsed)
            prompt_tk, completion_tk, total_tk = self._extract_usage(
                parsed,
                prompt_text=f"{system_prompt}\n{user_prompt}",
                output_text=content,
            )
            if not content:
                return ChatResult(
                    content=None,
                    prompt_tokens=prompt_tk,
                    completion_tokens=completion_tk,
                    total_tokens=total_tk,
                    model=model_used,
                    error="empty_content_in_response",
                    endpoint=self._endpoint(),
                    status_code=status_code,
                    image_attached=image_attached,
                    request_summary=request_summary,
                    response_summary={
                        "raw_preview": self._brief(raw),
                        "has_choices": isinstance(parsed.get("choices"), list),
                    },
                )
            return ChatResult(
                content=content,
                prompt_tokens=prompt_tk,
                completion_tokens=completion_tk,
                total_tokens=total_tk,
                model=model_used,
                endpoint=self._endpoint(),
                status_code=status_code,
                image_attached=image_attached,
                request_summary=request_summary,
                response_summary={
                    "content_preview": self._brief(content),
                    "has_choices": isinstance(parsed.get("choices"), list),
                },
            )
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            reason = f"http_error {exc.code}: {err_body[:300]}".strip()
            return ChatResult(
                content=None,
                prompt_tokens=max(1, len(system_prompt + user_prompt) // 4),
                completion_tokens=0,
                total_tokens=max(1, len(system_prompt + user_prompt) // 4),
                model=model_used,
                error=reason,
                endpoint=self._endpoint(),
                status_code=exc.code,
                image_attached=image_attached,
                request_summary=request_summary,
                response_summary={"error_body_preview": self._brief(err_body)},
            )
        except urllib.error.URLError as exc:
            return ChatResult(
                content=None,
                prompt_tokens=max(1, len(system_prompt + user_prompt) // 4),
                completion_tokens=0,
                total_tokens=max(1, len(system_prompt + user_prompt) // 4),
                model=model_used,
                error=f"url_error: {exc.reason}",
                endpoint=self._endpoint(),
                image_attached=image_attached,
                request_summary=request_summary,
                response_summary={"error": f"url_error: {exc.reason}"},
            )
        except (TimeoutError, socket.timeout):
            return ChatResult(
                content=None,
                prompt_tokens=max(1, len(system_prompt + user_prompt) // 4),
                completion_tokens=0,
                total_tokens=max(1, len(system_prompt + user_prompt) // 4),
                model=model_used,
                error=f"timeout_error: exceeded {self.timeout}s",
                endpoint=self._endpoint(),
                image_attached=image_attached,
                request_summary=request_summary,
                response_summary={"error": f"timeout_error: exceeded {self.timeout}s"},
            )
        except json.JSONDecodeError as exc:
            return ChatResult(
                content=None,
                prompt_tokens=max(1, len(system_prompt + user_prompt) // 4),
                completion_tokens=0,
                total_tokens=max(1, len(system_prompt + user_prompt) // 4),
                model=model_used,
                error=f"json_decode_error: {exc}",
                endpoint=self._endpoint(),
                image_attached=image_attached,
                request_summary=request_summary,
                response_summary={"error": f"json_decode_error: {exc}"},
            )

    def embeddings(self, inputs: list[str], model: str | None = None) -> dict[str, Any]:
        clean_inputs = [str(x) for x in inputs if str(x).strip()]
        model_used = model or self.model or ""
        request_summary = {
            "model": model_used,
            "input_size": len(clean_inputs),
        }

        if not clean_inputs:
            return {
                "ok": True,
                "vectors": [],
                "model": model_used,
                "error": None,
                "endpoint": self._embeddings_endpoint() if self.base_url else "",
                "status_code": None,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"note": "empty_input"},
            }

        if not self.enabled:
            return {
                "ok": False,
                "vectors": [],
                "model": model_used,
                "error": "api_not_enabled: missing NEW_API_KEY or NEW_API_BASE_URL",
                "endpoint": self._embeddings_endpoint() if self.base_url else "",
                "status_code": None,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"error": "api_not_enabled"},
            }

        payload = {
            "model": model_used,
            "input": clean_inputs,
            "encoding_format": "float",
        }
        body = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url=self._embeddings_endpoint(),
            data=body,
            headers=self._request_headers(),
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                raw = resp.read().decode("utf-8")
                status_code = getattr(resp, "status", None)
            parsed = json.loads(raw)

            data = parsed.get("data", []) if isinstance(parsed, dict) else []
            vectors: list[list[float]] = []
            if isinstance(data, list):
                for item in data:
                    if not isinstance(item, dict):
                        vectors.append([])
                        continue
                    emb = item.get("embedding")
                    if not isinstance(emb, list):
                        vectors.append([])
                        continue
                    vec: list[float] = []
                    for x in emb:
                        try:
                            vec.append(float(x))
                        except Exception:
                            continue
                    vectors.append(vec)

            usage = parsed.get("usage", {}) if isinstance(parsed, dict) else {}
            prompt_tokens = int(usage.get("prompt_tokens", 0) or 0) if isinstance(usage, dict) else 0
            total_tokens = int(usage.get("total_tokens", 0) or 0) if isinstance(usage, dict) else prompt_tokens

            ok = len(vectors) == len(clean_inputs) and all(len(v) > 0 for v in vectors)
            return {
                "ok": ok,
                "vectors": vectors,
                "model": model_used,
                "error": None if ok else "embedding_response_incomplete",
                "endpoint": self._embeddings_endpoint(),
                "status_code": status_code,
                "prompt_tokens": prompt_tokens,
                "total_tokens": total_tokens,
                "request_summary": request_summary,
                "response_summary": {
                    "vector_count": len(vectors),
                    "input_size": len(clean_inputs),
                },
            }
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8", errors="replace")
            except Exception:
                err_body = ""
            reason = f"http_error {exc.code}: {err_body[:300]}".strip()
            return {
                "ok": False,
                "vectors": [],
                "model": model_used,
                "error": reason,
                "endpoint": self._embeddings_endpoint(),
                "status_code": exc.code,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"error_body_preview": self._brief(err_body)},
            }
        except urllib.error.URLError as exc:
            return {
                "ok": False,
                "vectors": [],
                "model": model_used,
                "error": f"url_error: {exc.reason}",
                "endpoint": self._embeddings_endpoint(),
                "status_code": None,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"error": f"url_error: {exc.reason}"},
            }
        except (TimeoutError, socket.timeout):
            return {
                "ok": False,
                "vectors": [],
                "model": model_used,
                "error": f"timeout_error: exceeded {self.timeout}s",
                "endpoint": self._embeddings_endpoint(),
                "status_code": None,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"error": f"timeout_error: exceeded {self.timeout}s"},
            }
        except json.JSONDecodeError as exc:
            return {
                "ok": False,
                "vectors": [],
                "model": model_used,
                "error": f"json_decode_error: {exc}",
                "endpoint": self._embeddings_endpoint(),
                "status_code": None,
                "prompt_tokens": 0,
                "total_tokens": 0,
                "request_summary": request_summary,
                "response_summary": {"error": f"json_decode_error: {exc}"},
            }
