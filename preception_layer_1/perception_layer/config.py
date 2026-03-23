from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

DEFAULT_BASE_URL = "https://api.zjuqx.cn/v1"
DEFAULT_JUDGE_MODEL = "gemini-3pro"
DEFAULT_EMBEDDING_MODEL = "baai/bge-m3"
DEFAULT_RAG_ENDPOINT = "http://221.12.22.162:8888/test/8002/api/search"
DEFAULT_RAG_TOP_K = 5
DEFAULT_RAG_SIMILARITY_THRESHOLD = 0.35
DEFAULT_DUPLICATE_THRESHOLD = 0.83
DEFAULT_MAX_IMAGE_PIXELS = 1003520
DEFAULT_CONTEXT_PATH = Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/context.md")
DEFAULT_RAG_SEARCH_RECORD_PATH = Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/rag_search_record.md")
DEFAULT_LLM_CHAT_RECORD_PATH = Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/llm_chat_record.jsonl")
DEFAULT_OUTPUT_PATH = Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/slots.jsonl")
DEFAULT_REQUEST_TIMEOUT = 180.0


@dataclass(slots=True)
class PipelineConfig:
    api_key: str | None
    base_url: str = DEFAULT_BASE_URL
    judge_model: str = DEFAULT_JUDGE_MODEL
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    rag_endpoint: str = DEFAULT_RAG_ENDPOINT
    rag_top_k: int = DEFAULT_RAG_TOP_K
    rag_similarity_threshold: float = DEFAULT_RAG_SIMILARITY_THRESHOLD
    duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS
    context_path: Path = DEFAULT_CONTEXT_PATH
    rag_search_record_path: Path = DEFAULT_RAG_SEARCH_RECORD_PATH
    llm_chat_record_path: Path = DEFAULT_LLM_CHAT_RECORD_PATH
    output_path: Path = DEFAULT_OUTPUT_PATH
    request_timeout: float = DEFAULT_REQUEST_TIMEOUT

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        api_key = os.getenv("OPENAI_API_KEY") or os.getenv("PERCEPTION_LLM_API_KEY")
        return cls(
            api_key=api_key,
            base_url=os.getenv("PERCEPTION_BASE_URL", os.getenv("PERCEPTION_LLM_BASE_URL", DEFAULT_BASE_URL)),
            judge_model=os.getenv("PERCEPTION_JUDGE_MODEL", os.getenv("PERCEPTION_LLM_MODEL", DEFAULT_JUDGE_MODEL)),
            embedding_model=os.getenv(
                "PERCEPTION_EMBEDDING_MODEL",
                os.getenv("BGE_M3_MODEL_NAME", DEFAULT_EMBEDDING_MODEL),
            ),
            rag_endpoint=os.getenv("PERCEPTION_RAG_ENDPOINT", DEFAULT_RAG_ENDPOINT),
            rag_top_k=int(os.getenv("PERCEPTION_RAG_TOP_K", str(DEFAULT_RAG_TOP_K))),
            rag_similarity_threshold=float(
                os.getenv("PERCEPTION_RAG_SCORE_THRESHOLD", str(DEFAULT_RAG_SIMILARITY_THRESHOLD))
            ),
            duplicate_threshold=float(
                os.getenv("PERCEPTION_DUPLICATE_THRESHOLD", os.getenv("PERCEPTION_DEDUP_THRESHOLD", str(DEFAULT_DUPLICATE_THRESHOLD)))
            ),
            max_image_pixels=int(os.getenv("PERCEPTION_MAX_IMAGE_PIXELS", str(DEFAULT_MAX_IMAGE_PIXELS))),
            context_path=Path(os.getenv("PERCEPTION_CONTEXT_PATH", str(DEFAULT_CONTEXT_PATH))),
            rag_search_record_path=Path(
                os.getenv("PERCEPTION_RAG_SEARCH_RECORD_PATH", str(DEFAULT_RAG_SEARCH_RECORD_PATH))
            ),
            llm_chat_record_path=Path(
                os.getenv("PERCEPTION_LLM_CHAT_RECORD_PATH", str(DEFAULT_LLM_CHAT_RECORD_PATH))
            ),
            output_path=Path(os.getenv("PERCEPTION_OUTPUT_PATH", str(DEFAULT_OUTPUT_PATH))),
            request_timeout=float(
                os.getenv("PERCEPTION_API_TIMEOUT", os.getenv("OPENAI_API_TIMEOUT", str(DEFAULT_REQUEST_TIMEOUT)))
            ),
        )

    @property
    def llm_api_key(self) -> str | None:
        return self.api_key

    @property
    def llm_base_url(self) -> str:
        return self.base_url

    @property
    def llm_model(self) -> str:
        return self.judge_model

    @property
    def dedup_similarity_threshold(self) -> float:
        return self.duplicate_threshold
