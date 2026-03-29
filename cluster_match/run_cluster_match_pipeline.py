#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import evaluate_cluster_match_results
from cluster_match.categories import SCHEMA_PROFILES
from cluster_match.config import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_ENHANCED_DIR,
    DEFAULT_FINAL_CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
)
from cluster_match.runner import main as run_main


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run cluster_match extraction and optional evaluation.")
    parser.add_argument("--config", default=str(DEFAULT_FINAL_CONFIG_PATH))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--enhanced-dir", default=str(DEFAULT_ENHANCED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--schema-profile", choices=SCHEMA_PROFILES, default="simple_v1")
    parser.add_argument("--sources", nargs="+", choices=["baseline", "enhanced"], default=["baseline", "enhanced"])
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--judge-mode", choices=("exact", "llm"), default="exact")
    parser.add_argument("--judge-model", default="")
    return parser


def resolve_results_root(output_dir: str, schema_profile: str) -> Path:
    base = Path(output_dir).expanduser()
    if schema_profile != "legacy" and base == DEFAULT_OUTPUT_DIR:
        return base / schema_profile
    return base


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    extraction_argv = [
        "--config",
        args.config,
        "--dataset-jsonl",
        args.dataset_jsonl,
        "--baseline-dir",
        args.baseline_dir,
        "--enhanced-dir",
        args.enhanced_dir,
        "--output-dir",
        args.output_dir,
        "--model",
        args.model,
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
        "--schema-profile",
        args.schema_profile,
        "--sources",
        *args.sources,
        "--overwrite",
        "--sleep-seconds",
        str(args.sleep_seconds),
        "--temperature",
        str(args.temperature),
    ]

    if args.limit is not None:
        extraction_argv.extend(["--limit", str(args.limit)])
    for sample_id in args.sample_ids or []:
        extraction_argv.extend(["--sample-id", sample_id])

    extraction_status = run_main(extraction_argv)
    if extraction_status != 0:
        return extraction_status

    if not {"baseline", "enhanced"}.issubset(set(args.sources)):
        print(
            f"[cluster_match][step 2/2] evaluation skipped sources={','.join(args.sources)} reason=single_source_mode",
            flush=True,
        )
        return 0

    results_root = resolve_results_root(args.output_dir, args.schema_profile)
    evaluation_argv = [
        "--results-root",
        str(results_root),
        "--output-root",
        str(results_root / "evaluation"),
        "--dataset-jsonl",
        args.dataset_jsonl,
        "--judge-mode",
        args.judge_mode,
        "--timeout",
        str(args.timeout),
        "--max-retries",
        str(args.max_retries),
    ]

    if args.judge_mode == "llm":
        evaluation_argv.extend(["--config", args.config])
        if args.judge_model:
            evaluation_argv.extend(["--model", args.judge_model])

    for sample_id in args.sample_ids or []:
        evaluation_argv.extend(["--sample-id", sample_id])

    return evaluate_cluster_match_results.main(evaluation_argv)


if __name__ == "__main__":
    raise SystemExit(main())
