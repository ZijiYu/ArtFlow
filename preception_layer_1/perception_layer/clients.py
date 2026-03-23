from __future__ import annotations

import difflib
import json
import math
import re
import sys
import threading
import uuid
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from time import perf_counter
from typing import Any, Protocol
from urllib import error, request

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]

from .config import PipelineConfig
from .models import RagDocument


class LLMClient(Protocol):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        ...


class RagClient(Protocol):
    def search(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        query_image_filename: str | None,
        query_image_mime_type: str | None,
        top_k: int,
    ) -> list[RagDocument]:
        ...


class SimilarityBackend(Protocol):
    def similarity(self, left: str, right: str) -> float:
        ...


class LlmChatLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._path.write_text("[]\n", encoding="utf-8")

    def append(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None,
        image_mime_type: str,
        response_payload: dict[str, Any],
    ) -> None:
        record = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "system_prompt": system_prompt,
            "user_text": user_text,
            "image_attached": image_base64 is not None,
            "image_mime_type": image_mime_type if image_base64 is not None else None,
            "image_base64_length": len(image_base64) if image_base64 is not None else 0,
            "response": response_payload,
        }
        with self._lock:
            records = self._load_records()
            records.append(record)
            self._path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _load_records(self) -> list[dict[str, Any]]:
        if not self._path.exists():
            return []
        raw = self._path.read_text(encoding="utf-8").strip()
        if not raw:
            return []
        payload = json.loads(raw)
        if isinstance(payload, list):
            return [item for item in payload if isinstance(item, dict)]
        raise RuntimeError(f"Unexpected llm chat record format in {self._path}")


class RecordingLLMClient:
    def __init__(self, base_client: LLMClient, logger: LlmChatLogger) -> None:
        self._base_client = base_client
        self._logger = logger

    def reset_log(self) -> None:
        self._logger.reset()

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        response_payload = self._base_client.complete_json(
            system_prompt=system_prompt,
            user_text=user_text,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        self._logger.append(
            system_prompt=system_prompt,
            user_text=user_text,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            response_payload=response_payload,
        )
        return response_payload


class OpenAIJsonClient:
    def __init__(self, config: PipelineConfig, client: Any | None = None) -> None:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required to use OpenAIJsonClient.")
        if not config.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            max_retries=2,
        )
        self._model = config.judge_model

    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict[str, Any]:
        content: str | list[dict[str, Any]]
        if image_base64:
            content = [
                {"type": "text", "text": user_text},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image_mime_type};base64,{image_base64}"},
                },
            ]
        else:
            content = user_text

        started_at = perf_counter()
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        print(
            f"[perception_request] type=llm model={self._model} "
            f"duration_ms={round((perf_counter() - started_at) * 1000.0, 2):.2f} "
            f"image={str(image_base64 is not None).lower()}",
            flush=True,
        )
        payload = response.choices[0].message.content
        if not isinstance(payload, str):
            raise RuntimeError("LLM did not return a JSON string.")
        return _parse_json_payload(payload)


class HttpRagClient:
    def __init__(self, endpoint: str) -> None:
        self._endpoint = endpoint

    def search(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        query_image_filename: str | None,
        query_image_mime_type: str | None,
        top_k: int,
    ) -> list[RagDocument]:
        body, content_type = _build_multipart_form_data(
            fields={
                "query_text": query_text or "",
                "top_k": str(top_k),
            },
            file_field_name="query_image",
            file_bytes=query_image_bytes,
            file_name=query_image_filename,
            file_mime_type=query_image_mime_type,
        )
        req = request.Request(
            self._endpoint,
            data=body,
            headers={"Content-Type": content_type},
            method="POST",
        )
        started_at = perf_counter()
        try:
            with request.urlopen(req, timeout=30) as response:  # noqa: S310
                raw = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(
                "RAG request failed with HTTP "
                f"{exc.code} at {self._endpoint}. "
                "encoding=multipart/form-data. "
                f"payload_summary={_summarize_rag_payload(query_text, query_image_bytes, query_image_filename, top_k)}. "
                f"server_response={detail or exc.reason}. "
                "This usually means the endpoint rejected the multipart field names or the uploaded image format."
            ) from exc
        print(
            f"[perception_request] type=rag query={query_text or ''} "
            f"duration_ms={round((perf_counter() - started_at) * 1000.0, 2):.2f}",
            flush=True,
        )
        items = raw.get("results") or raw.get("data") or raw.get("items") or []
        return [self._normalize_item(item, index) for index, item in enumerate(items)]

    @staticmethod
    def _normalize_item(item: dict[str, Any], index: int) -> RagDocument:
        source_id = str(item.get("source_id") or item.get("id") or index)
        content = (
            item.get("content")
            or item.get("text")
            or item.get("chunk_text")
            or item.get("document")
            or json.dumps(item, ensure_ascii=False)
        )
        score = item.get("score") or item.get("similarity")
        metadata = {key: value for key, value in item.items() if key not in {"source_id", "id", "content", "text"}}
        return RagDocument(source_id=source_id, content=str(content), score=float(score) if score is not None else None, metadata=metadata)


class OpenAIEmbeddingSimilarityBackend:
    def __init__(self, config: PipelineConfig, client: Any | None = None) -> None:
        if OpenAI is None:  # pragma: no cover
            raise RuntimeError("openai package is required to use OpenAIEmbeddingSimilarityBackend.")
        if not config.api_key:
            raise RuntimeError("OPENAI_API_KEY is missing.")
        self._client = client or OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.request_timeout,
            max_retries=2,
        )
        self._model = config.embedding_model
        self._fallback = LexicalSimilarityBackend()
        self._use_fallback_only = False
        self._fallback_notice_emitted = False

    def similarity(self, left: str, right: str) -> float:
        if self._use_fallback_only:
            return self._fallback.similarity(left, right)
        try:
            left_vec = self._encode(left)
            right_vec = self._encode(right)
            return _cosine_similarity(left_vec, right_vec)
        except Exception as exc:  # pragma: no cover - exercised via fake backend tests
            self._use_fallback_only = True
            if not self._fallback_notice_emitted:
                print(
                    f"[perception_layer] embedding model `{self._model}` unavailable, "
                    f"falling back to lexical similarity: {exc}",
                    file=sys.stderr,
                )
                self._fallback_notice_emitted = True
            return self._fallback.similarity(left, right)

    @lru_cache(maxsize=512)
    def _encode(self, text: str) -> tuple[float, ...]:
        response = self._client.embeddings.create(model=self._model, input=[text])
        return tuple(float(value) for value in response.data[0].embedding)


class LexicalSimilarityBackend:
    def similarity(self, left: str, right: str) -> float:
        normalized_left = _normalize_similarity_text(left)
        normalized_right = _normalize_similarity_text(right)
        if not normalized_left or not normalized_right:
            return 0.0
        if normalized_left == normalized_right:
            return 1.0

        sequence_score = difflib.SequenceMatcher(None, normalized_left, normalized_right).ratio()
        left_ngrams = _char_ngrams(normalized_left)
        right_ngrams = _char_ngrams(normalized_right)
        if left_ngrams or right_ngrams:
            overlap = len(left_ngrams & right_ngrams)
            union = len(left_ngrams | right_ngrams)
            jaccard_score = overlap / union if union else 0.0
        else:
            jaccard_score = 0.0
        return max(sequence_score, jaccard_score)


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if not left or not right:
        return 0.0
    numerator = sum(left_item * right_item for left_item, right_item in zip(left, right))
    left_norm = math.sqrt(sum(item * item for item in left))
    right_norm = math.sqrt(sum(item * item for item in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return numerator / (left_norm * right_norm)


def _normalize_similarity_text(text: str) -> str:
    return re.sub(r"\s+", "", text).strip().lower()


def _char_ngrams(text: str, n: int = 2) -> set[str]:
    if len(text) < n:
        return {text} if text else set()
    return {text[index : index + n] for index in range(len(text) - n + 1)}


def _summarize_rag_payload(
    query_text: str | None,
    query_image_bytes: bytes | None,
    query_image_filename: str | None,
    top_k: int,
) -> str:
    query_text_preview = (query_text or "")[:80]
    image_length = len(query_image_bytes) if query_image_bytes else 0
    return json.dumps(
        {
            "query_text": query_text_preview,
            "query_image_present": bool(query_image_bytes),
            "query_image_length": image_length,
            "query_image_filename": query_image_filename or "",
            "top_k": top_k,
        },
        ensure_ascii=False,
    )


def _parse_json_payload(payload: str) -> dict[str, Any]:
    candidates = _json_parse_candidates(payload)
    last_error: Exception | None = None
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            continue
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed and isinstance(parsed[0], dict):
            return parsed[0]
    snippet = _summarize_json_payload(payload)
    if last_error is not None:
        raise RuntimeError(f"LLM returned invalid JSON. last_error={last_error}. payload_snippet={snippet}")
    raise RuntimeError(f"LLM returned non-object JSON. payload_snippet={snippet}")


def _json_parse_candidates(payload: str) -> list[str]:
    raw = str(payload or "").strip()
    if not raw:
        return []

    candidates = [raw]
    unfenced = re.sub(r"^```(?:json)?\s*", "", raw, flags=re.I)
    unfenced = re.sub(r"\s*```$", "", unfenced)
    if unfenced != raw:
        candidates.append(unfenced.strip())

    extracted = _extract_balanced_json_object(unfenced)
    if extracted:
        candidates.append(extracted)

    deduped: list[str] = []
    seen: set[str] = set()
    for item in candidates:
        text = item.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        deduped.append(text)
    return deduped


def _extract_balanced_json_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        return ""
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return ""


def _summarize_json_payload(payload: str, *, limit: int = 400) -> str:
    text = " ".join(str(payload or "").split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _build_multipart_form_data(
    *,
    fields: dict[str, str],
    file_field_name: str,
    file_bytes: bytes | None,
    file_name: str | None,
    file_mime_type: str | None,
) -> tuple[bytes, str]:
    boundary = f"----CodexBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for field_name, field_value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{field_name}"\r\n\r\n'.encode("utf-8"))
        body.extend(field_value.encode("utf-8"))
        body.extend(b"\r\n")

    if file_bytes is not None:
        file_upload_name = file_name or "query_image.png"
        mime_type = file_mime_type or "application/octet-stream"
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(
            (
                f'Content-Disposition: form-data; name="{file_field_name}"; '
                f'filename="{file_upload_name}"\r\n'
            ).encode("utf-8")
        )
        body.extend(f"Content-Type: {mime_type}\r\n\r\n".encode("utf-8"))
        body.extend(file_bytes)
        body.extend(b"\r\n")

    body.extend(f"--{boundary}--\r\n".encode("utf-8"))
    return bytes(body), f"multipart/form-data; boundary={boundary}"
