from __future__ import annotations

import argparse
import json
import os
import sys
import time

from openai import OpenAI


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether an embedding model is callable.")
    parser.add_argument("--api-key", help="API key. Falls back to OPENAI_API_KEY.")
    parser.add_argument("--base-url", required=True, help="OpenAI-compatible base URL.")
    parser.add_argument("--model", required=True, help="Embedding model name.")
    parser.add_argument(
        "--text",
        default="测试国画术语 embedding 可用性：斧劈皴、绢本、题跋。",
        help="Text to embed.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="Request timeout in seconds.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY is missing. Pass --api-key or export OPENAI_API_KEY.", file=sys.stderr)
        return 2

    client = OpenAI(
        api_key=api_key,
        base_url=args.base_url,
        timeout=args.timeout,
    )

    started_at = time.perf_counter()
    try:
        response = client.embeddings.create(
            model=args.model,
            input=[args.text],
        )
    except Exception as exc:
        elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
        print(
            json.dumps(
                {
                    "ok": False,
                    "base_url": args.base_url,
                    "model": args.model,
                    "elapsed_ms": elapsed_ms,
                    "error_type": exc.__class__.__name__,
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 1

    elapsed_ms = round((time.perf_counter() - started_at) * 1000, 2)
    embedding = response.data[0].embedding
    usage = getattr(response, "usage", None)
    total_tokens = getattr(usage, "total_tokens", None) if usage is not None else None
    print(
        json.dumps(
            {
                "ok": True,
                "base_url": args.base_url,
                "model": args.model,
                "elapsed_ms": elapsed_ms,
                "embedding_dimensions": len(embedding),
                "preview": _preview_vector(embedding),
                "total_tokens": total_tokens,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _preview_vector(values: list[float], limit: int = 8) -> list[float]:
    return [round(float(value), 6) for value in values[:limit]]


if __name__ == "__main__":
    raise SystemExit(main())
