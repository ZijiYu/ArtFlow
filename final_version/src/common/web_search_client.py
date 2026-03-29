from __future__ import annotations

import json
import re
from dataclasses import dataclass
from html import unescape
from html.parser import HTMLParser
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


@dataclass(slots=True)
class WebSearchHit:
    title: str
    url: str
    snippet: str
    source: str = "organic"
    position: int = 0


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._chunks: list[str] = []

    def handle_data(self, data: str) -> None:
        text = str(data or "").strip()
        if text:
            self._chunks.append(text)

    def text(self) -> str:
        return " ".join(self._chunks)


class SerperWebSearchClient:
    def __init__(self, *, url: str, api_key: str, timeout: int = 20) -> None:
        self.url = str(url or "").strip()
        self.api_key = str(api_key or "").strip()
        self.timeout = max(1, int(timeout or 20))

    @property
    def enabled(self) -> bool:
        return bool(self.url and self.api_key)

    def search(self, query: str, *, top_k: int = 5) -> list[WebSearchHit]:
        if not self.enabled or not str(query or "").strip():
            return []
        payload = json.dumps({"q": query, "num": max(1, int(top_k or 5))}, ensure_ascii=False).encode("utf-8")
        request = Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "X-API-KEY": self.api_key,
                "User-Agent": "Mozilla/5.0",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                raw = response.read().decode("utf-8", errors="ignore")
        except (HTTPError, URLError, TimeoutError):
            return []
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return self._parse_hits(parsed, top_k=max(1, int(top_k or 5)))

    def fetch_page(self, url: str, *, max_chars: int = 6_000) -> dict[str, Any]:
        if not str(url or "").strip():
            return {}
        request = Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0",
                "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
            },
            method="GET",
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                content_type = str(response.headers.get("Content-Type", "")).lower()
                raw_bytes = response.read(2_000_000)
        except (HTTPError, URLError, TimeoutError):
            return {}
        if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
            return {}
        html = raw_bytes.decode("utf-8", errors="ignore")
        text = self._extract_text(html)[: max(512, int(max_chars or 6_000))]
        if not text:
            return {}
        return {
            "url": url,
            "domain": urlparse(url).netloc.lower(),
            "title": self._extract_title(html),
            "description": self._extract_meta_description(html),
            "content": text,
        }

    @staticmethod
    def _parse_hits(payload: dict[str, Any], *, top_k: int) -> list[WebSearchHit]:
        hits: list[WebSearchHit] = []
        seen_urls: set[str] = set()

        def append_hit(*, title: str, url: str, snippet: str, source: str, position: int) -> None:
            clean_url = str(url or "").strip()
            if not clean_url or clean_url in seen_urls:
                return
            seen_urls.add(clean_url)
            hits.append(
                WebSearchHit(
                    title=str(title or "").strip(),
                    url=clean_url,
                    snippet=str(snippet or "").strip(),
                    source=source,
                    position=position,
                )
            )

        answer_box = payload.get("answerBox", {})
        if isinstance(answer_box, dict):
            append_hit(
                title=str(answer_box.get("title") or answer_box.get("answer") or "").strip(),
                url=str(answer_box.get("link") or answer_box.get("url") or "").strip(),
                snippet=str(answer_box.get("snippet") or answer_box.get("answer") or "").strip(),
                source="answer_box",
                position=0,
            )

        knowledge_graph = payload.get("knowledgeGraph", {})
        if isinstance(knowledge_graph, dict):
            append_hit(
                title=str(knowledge_graph.get("title") or knowledge_graph.get("type") or "").strip(),
                url=str(knowledge_graph.get("website") or knowledge_graph.get("url") or "").strip(),
                snippet=str(knowledge_graph.get("description") or "").strip(),
                source="knowledge_graph",
                position=0,
            )

        organic = payload.get("organic", [])
        if isinstance(organic, list):
            for index, item in enumerate(organic, start=1):
                if not isinstance(item, dict):
                    continue
                append_hit(
                    title=str(item.get("title") or "").strip(),
                    url=str(item.get("link") or "").strip(),
                    snippet=str(item.get("snippet") or "").strip(),
                    source="organic",
                    position=int(item.get("position", index) or index),
                )
                if len(hits) >= top_k:
                    break

        return hits[:top_k]

    @staticmethod
    def _extract_title(html: str) -> str:
        match = re.search(r"<title[^>]*>(.*?)</title>", html, flags=re.IGNORECASE | re.DOTALL)
        if not match:
            return ""
        return re.sub(r"\s+", " ", unescape(match.group(1))).strip()

    @staticmethod
    def _extract_meta_description(html: str) -> str:
        patterns = (
            r'<meta[^>]+name=["\']description["\'][^>]+content=["\'](.*?)["\']',
            r'<meta[^>]+property=["\']og:description["\'][^>]+content=["\'](.*?)["\']',
        )
        for pattern in patterns:
            match = re.search(pattern, html, flags=re.IGNORECASE | re.DOTALL)
            if not match:
                continue
            return re.sub(r"\s+", " ", unescape(match.group(1))).strip()
        return ""

    @staticmethod
    def _extract_text(html: str) -> str:
        cleaned = re.sub(r"(?is)<(script|style|noscript).*?>.*?</\\1>", " ", html)
        cleaned = re.sub(r"(?is)<svg.*?>.*?</svg>", " ", cleaned)
        extractor = _HTMLTextExtractor()
        extractor.feed(cleaned)
        text = extractor.text()
        text = unescape(text)
        text = re.sub(r"\s+", " ", text).strip()
        return text
