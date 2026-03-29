from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from .client import ChatCompletionsClient
from .config import (
    DEFAULT_BASELINE_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_ENHANCED_DIR,
    DEFAULT_FINAL_CONFIG_PATH,
    DEFAULT_MODEL,
    DEFAULT_OUTPUT_DIR,
    RuntimeConfig,
)
from .categories import SCHEMA_PROFILES
from .dataset import SOURCE_DIR_NAMES, build_jobs, locate_text
from .json_utils import normalize_result, parse_json_object
from .prompting import SYSTEM_PROMPT, build_user_prompt


@dataclass(slots=True)
class RunStats:
    total: int = 0
    ok: int = 0
    failed: int = 0
    skipped_existing: int = 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract question-aware factual clusters from pilot30 appreciation texts.")
    parser.add_argument("--config", default=str(DEFAULT_FINAL_CONFIG_PATH))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--baseline-dir", default=str(DEFAULT_BASELINE_DIR))
    parser.add_argument("--enhanced-dir", default=str(DEFAULT_ENHANCED_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--api-key")
    parser.add_argument("--base-url")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--schema-profile", choices=SCHEMA_PROFILES, default="legacy")
    parser.add_argument("--sources", nargs="+", choices=["baseline", "enhanced"], default=["baseline", "enhanced"])
    parser.add_argument("--sample-id", action="append", dest="sample_ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[cluster_match][{now}] {message}", flush=True)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _mask_secret(value: str) -> str:
    if not value:
        return "<missing>"
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def _processed_count(stats: RunStats) -> int:
    return stats.ok + stats.failed + stats.skipped_existing


def _estimate_eta(stats: RunStats, started_at: float) -> tuple[float, float, float]:
    processed = _processed_count(stats)
    if processed <= 0:
        return 0.0, 0.0, 0.0
    elapsed = max(0.0, time.perf_counter() - started_at)
    average = elapsed / processed
    remaining = max(0, stats.total - processed)
    eta = average * remaining
    return elapsed, average, eta


def _count_source_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for item in path.glob("*.md") if item.is_file())


def _build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    return RuntimeConfig.from_sources(
        config_path=args.config,
        dataset_path=args.dataset_jsonl,
        baseline_dir=args.baseline_dir,
        enhanced_dir=args.enhanced_dir,
        output_dir=args.output_dir,
        model=args.model,
        timeout=args.timeout,
        api_key=args.api_key,
        base_url=args.base_url,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    runtime = _build_runtime_config(args)
    output_dir = runtime.output_dir
    if args.schema_profile != "legacy" and Path(args.output_dir).expanduser() == DEFAULT_OUTPUT_DIR:
        output_dir = output_dir / args.schema_profile
    output_dir = _ensure_dir(output_dir)
    run_started_at = time.perf_counter()

    run_root = _ensure_dir(output_dir / "runs" / _timestamp())
    raw_root = _ensure_dir(run_root / "raw_responses")
    prompt_root = _ensure_dir(run_root / "prompts")
    run_index_path = run_root / "run_index.jsonl"

    sample_ids = set(args.sample_ids or [])
    jobs = build_jobs(
        dataset_path=runtime.dataset_path,
        baseline_dir=runtime.baseline_dir,
        enhanced_dir=runtime.enhanced_dir,
        sources=args.sources,
        sample_ids=sample_ids or None,
        limit=args.limit,
    )

    for source in args.sources:
        _ensure_dir(output_dir / SOURCE_DIR_NAMES[source])

    client = ChatCompletionsClient(
        api_key=runtime.api_key,
        base_url=runtime.base_url,
        timeout=runtime.timeout,
        max_retries=args.max_retries,
    )
    stats = RunStats(total=len(jobs))

    _log("start")
    _log(
        "params "
        f"model={runtime.model} base_url={runtime.base_url} timeout={runtime.timeout}s "
        f"max_retries={args.max_retries} temperature={args.temperature} overwrite={args.overwrite} "
        f"schema_profile={args.schema_profile}"
    )
    _log(
        "paths "
        f"config={runtime.config_path} dataset={runtime.dataset_path} output_dir={output_dir} run_root={run_root}"
    )
    _log(
        "inputs "
        f"baseline_dir={runtime.baseline_dir} enhanced_dir={runtime.enhanced_dir} "
        f"sources={','.join(args.sources)} sample_filter={','.join(sorted(sample_ids)) if sample_ids else '<all>'} "
        f"limit={args.limit if args.limit is not None else '<none>'}"
    )
    _log(
        "auth "
        f"api_key={_mask_secret(runtime.api_key)} "
        f"sleep_seconds={args.sleep_seconds}"
    )
    _log(f"jobs total={stats.total}")

    if stats.total == 0:
        baseline_file_count = _count_source_files(runtime.baseline_dir)
        enhanced_file_count = _count_source_files(runtime.enhanced_dir)
        _log(
            "warning "
            f"no jobs were built; baseline_md_count={baseline_file_count} "
            f"enhanced_md_count={enhanced_file_count}"
        )
        _log(
            "hint "
            f"check whether dataset sample_id values in {runtime.dataset_path} match the md filenames "
            f"under the selected source directories"
        )

    with run_index_path.open("w", encoding="utf-8") as index_handle:
        for index, job in enumerate(jobs, start=1):
            source_root = runtime.baseline_dir if job.source == "baseline" else runtime.enhanced_dir
            input_path = job.input_path
            if not input_path.exists():
                relocated = locate_text(source_root, job.sample_id)
                if relocated is not None:
                    _log(
                        f"[{index}/{stats.total}] relocate source={job.source} sample_id={job.sample_id} "
                        f"old_input_path={input_path} new_input_path={relocated}"
                    )
                    input_path = relocated

            output_path = output_dir / job.source_dir_name / f"{job.sample_id}.json"
            prompt_path = _ensure_dir(prompt_root / job.source_dir_name) / f"{job.sample_id}.md"
            raw_path = _ensure_dir(raw_root / job.source_dir_name) / f"{job.sample_id}.txt"
            job_started_at = time.perf_counter()

            elapsed_before, average_before, eta_before = _estimate_eta(stats, run_started_at)
            _log(
                f"[{index}/{stats.total}] start source={job.source} sample_id={job.sample_id} "
                f"elapsed={_format_duration(elapsed_before)} avg={_format_duration(average_before)} "
                f"eta={_format_duration(eta_before)}"
            )
            _log(
                f"[{index}/{stats.total}] detail question={job.question} "
                f"input_path={input_path} output_path={output_path}"
            )

            if output_path.exists() and not args.overwrite:
                stats.skipped_existing += 1
                entry = {
                    "sample_id": job.sample_id,
                    "source": job.source,
                    "status": "skipped_existing",
                    "question": job.question,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                }
                index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                elapsed_after, average_after, eta_after = _estimate_eta(stats, run_started_at)
                _log(
                    f"[{index}/{stats.total}] skip status=skipped_existing "
                    f"elapsed={_format_duration(elapsed_after)} avg={_format_duration(average_after)} "
                    f"eta={_format_duration(eta_after)} output_path={output_path}"
                )
                continue

            if not input_path.exists():
                stats.failed += 1
                entry = {
                    "sample_id": job.sample_id,
                    "source": job.source,
                    "status": "failed",
                    "question": job.question,
                    "input_path": str(input_path),
                    "output_path": str(output_path),
                    "error": "input_not_found",
                }
                index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                elapsed_after, average_after, eta_after = _estimate_eta(stats, run_started_at)
                job_seconds = time.perf_counter() - job_started_at
                _log(
                    f"[{index}/{stats.total}] done status=failed sample_id={job.sample_id} "
                    f"job_time={_format_duration(job_seconds)} elapsed={_format_duration(elapsed_after)} "
                    f"avg={_format_duration(average_after)} eta={_format_duration(eta_after)} "
                    f"error=input_not_found"
                )
                continue

            appreciation_text = _read_text(input_path)
            user_prompt = build_user_prompt(job.question, appreciation_text, schema_profile=args.schema_profile)
            prompt_path.write_text(user_prompt, encoding="utf-8")
            _log(
                f"[{index}/{stats.total}] prompt "
                f"input_chars={len(appreciation_text)} prompt_chars={len(user_prompt)} prompt_path={prompt_path}"
            )

            result = client.chat(
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                model=runtime.model,
                temperature=args.temperature,
            )
            raw_path.write_text(result.content or "", encoding="utf-8")

            entry = {
                "sample_id": job.sample_id,
                "source": job.source,
                "question": job.question,
                "input_path": str(input_path),
                "output_path": str(output_path),
                "raw_response_path": str(raw_path),
                "prompt_path": str(prompt_path),
                "model": result.model,
                "endpoint": result.endpoint,
                "status_code": result.status_code,
                "duration_ms": result.duration_ms,
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "total_tokens": result.total_tokens,
            }

            if result.error:
                stats.failed += 1
                entry["status"] = "failed"
                entry["error"] = result.error
                index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
                elapsed_after, average_after, eta_after = _estimate_eta(stats, run_started_at)
                job_seconds = time.perf_counter() - job_started_at
                _log(
                    f"[{index}/{stats.total}] done status=failed sample_id={job.sample_id} "
                    f"job_time={_format_duration(job_seconds)} elapsed={_format_duration(elapsed_after)} "
                    f"avg={_format_duration(average_after)} eta={_format_duration(eta_after)} "
                    f"error={result.error}"
                )
                continue

            try:
                parsed = parse_json_object(result.content or "")
                normalized = normalize_result(parsed, schema_profile=args.schema_profile)
                _write_json(output_path, normalized)
                stats.ok += 1
                entry["status"] = "ok"
            except Exception as exc:  # noqa: BLE001
                stats.failed += 1
                entry["status"] = "failed"
                entry["error"] = str(exc)

            index_handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            elapsed_after, average_after, eta_after = _estimate_eta(stats, run_started_at)
            job_seconds = time.perf_counter() - job_started_at
            _log(
                f"[{index}/{stats.total}] done status={entry['status']} sample_id={job.sample_id} "
                f"job_time={_format_duration(job_seconds)} elapsed={_format_duration(elapsed_after)} "
                f"avg={_format_duration(average_after)} eta={_format_duration(eta_after)} "
                f"tokens={result.total_tokens} status_code={result.status_code} output_path={output_path}"
            )

            if args.sleep_seconds > 0:
                _log(f"[{index}/{stats.total}] sleep seconds={args.sleep_seconds}")
                time.sleep(args.sleep_seconds)

    summary = {
        "run_root": str(run_root),
        "config_path": str(runtime.config_path),
        "dataset_path": str(runtime.dataset_path),
        "baseline_dir": str(runtime.baseline_dir),
        "enhanced_dir": str(runtime.enhanced_dir),
        "output_dir": str(output_dir),
        "model": runtime.model,
        "base_url": runtime.base_url,
        "schema_profile": args.schema_profile,
        "sources": args.sources,
        "stats": asdict(stats),
    }
    _write_json(run_root / "summary.json", summary)
    elapsed_total, average_total, eta_total = _estimate_eta(stats, run_started_at)
    _log(
        "finished "
        f"ok={stats.ok} failed={stats.failed} skipped={stats.skipped_existing} total={stats.total} "
        f"elapsed={_format_duration(elapsed_total)} avg={_format_duration(average_total)} "
        f"eta={_format_duration(eta_total)} summary={run_root / 'summary.json'}"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if stats.failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
