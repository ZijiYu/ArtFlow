from __future__ import annotations

import argparse
import asyncio
import io
import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from .config import (
    DEFAULT_BASE_URL,
    DEFAULT_CONTEXT_PATH,
    DEFAULT_DUPLICATE_THRESHOLD,
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_JUDGE_MODEL,
    DEFAULT_MAX_IMAGE_PIXELS,
    DEFAULT_OUTPUT_PATH,
    DEFAULT_RAG_ENDPOINT,
    DEFAULT_RAG_SIMILARITY_THRESHOLD,
    DEFAULT_RAG_TOP_K,
    PipelineConfig,
)
from .pipeline import PerceptionPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Chinese painting perception layer")
    parser.add_argument("--image", required=True, help="Path to the input image file")
    parser.add_argument("--text", required=True, help="Base text information")
    parser.add_argument("--output", help="Optional JSONL output path")
    parser.add_argument("--terminal-log", help="Path to save terminal stdout/stderr")
    parser.add_argument("--api-key", help="API key, overrides OPENAI_API_KEY when provided")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="LLM and embedding base URL")
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL, help="Embedding model name")
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL, help="Judge model name")
    parser.add_argument("--rag-endpoint", default=DEFAULT_RAG_ENDPOINT, help="RAG search endpoint")
    parser.add_argument("--rag-top-k", type=int, default=DEFAULT_RAG_TOP_K, help="RAG top-k")
    parser.add_argument(
        "--rag-score-threshold",
        type=float,
        default=DEFAULT_RAG_SIMILARITY_THRESHOLD,
        help="Similarity threshold for grounding",
    )
    parser.add_argument(
        "--duplicate-threshold",
        type=float,
        default=DEFAULT_DUPLICATE_THRESHOLD,
        help="Similarity threshold for slot deduplication",
    )
    parser.add_argument(
        "--max-image-pixels",
        type=int,
        default=DEFAULT_MAX_IMAGE_PIXELS,
        help="Maximum total pixels before resizing",
    )
    return parser


def build_config_from_args(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        api_key=args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("PERCEPTION_LLM_API_KEY"),
        base_url=args.base_url,
        judge_model=args.judge_model,
        embedding_model=args.embedding_model,
        rag_endpoint=args.rag_endpoint,
        rag_top_k=args.rag_top_k,
        rag_similarity_threshold=args.rag_score_threshold,
        duplicate_threshold=args.duplicate_threshold,
        max_image_pixels=args.max_image_pixels,
        context_path=DEFAULT_CONTEXT_PATH,
        output_path=Path(args.output) if args.output else DEFAULT_OUTPUT_PATH,
    )


def resolve_terminal_log_path(args: argparse.Namespace, config: PipelineConfig) -> Path:
    if args.terminal_log:
        return Path(args.terminal_log)
    return config.output_path.parent / "terminal_output.log"


class _TeeStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()

    @property
    def encoding(self) -> str:
        return getattr(self._streams[0], "encoding", "utf-8")


@contextmanager
def tee_terminal_output(log_path: Path) -> object:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    with log_path.open("a", encoding="utf-8") as handle:
        sys.stdout = _TeeStream(original_stdout, handle)
        sys.stderr = _TeeStream(original_stderr, handle)
        try:
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            sys.stdout = original_stdout
            sys.stderr = original_stderr


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config_from_args(args)
    terminal_log_path = resolve_terminal_log_path(args, config)
    with tee_terminal_output(terminal_log_path):
        print(f"[perception_layer] terminal output is being saved to {terminal_log_path}")
        pipeline = PerceptionPipeline(config)
        result = asyncio.run(
            pipeline.run(
                image_file=Path(args.image),
                input_text=args.text,
                output_path=config.output_path,
            )
        )
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
