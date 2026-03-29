#!/usr/bin/env bash
set -euo pipefail

CONFIG_PATH="/Users/ken/MM/Pipeline/final_version/config.yaml"
SCRIPT_DIR="/Users/ken/MM/Pipeline/final_version"
EVAL_V3_ENHANCED_DIR="/Users/ken/MM/Pipeline/eval_v3/enhanced"
RUN_LOG_DIR="/Users/ken/MM/Pipeline/eval_v3/artifacts/pilot30_run_logs"

mkdir -p "$EVAL_V3_ENHANCED_DIR"
mkdir -p "$RUN_LOG_DIR"

python - <<'PY'
import json
import shutil
import subprocess
import sys
from time import perf_counter
from pathlib import Path

import yaml

config_path = Path("/Users/ken/MM/Pipeline/final_version/config.yaml")
config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
eval_cfg = config.get("eval_v3", {}) if isinstance(config, dict) else {}
dataset_path = Path(str(eval_cfg.get("pilot_dataset_jsonl", "/Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl")))
enhanced_dir = Path(str(eval_cfg.get("pilot_enhanced_dir", "/Users/ken/MM/Pipeline/eval_v3/enhanced")))
run_log_dir = Path("/Users/ken/MM/Pipeline/eval_v3/artifacts/pilot30_run_logs")
closed_loop_root = Path(str(config.get("closed_loop", {}).get("output_dir", "/Users/ken/MM/Pipeline/final_version/artifacts_closed_loop/manual_test")))

enhanced_dir.mkdir(parents=True, exist_ok=True)
run_log_dir.mkdir(parents=True, exist_ok=True)

samples = [json.loads(line) for line in dataset_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summary = []
durations = []


def format_seconds(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    minutes, secs = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

for index, sample in enumerate(samples, start=1):
    sample_id = sample["sample_id"]
    image_path = sample["image_path"]
    question = sample["question"]
    stdout_log = run_log_dir / f"{sample_id}.stdout.log"
    stderr_log = run_log_dir / f"{sample_id}.stderr.log"
    completed_count = len(durations)
    avg_duration = (sum(durations) / completed_count) if completed_count else 0.0
    eta_seconds = avg_duration * (len(samples) - completed_count)

    print(
        f"[pilot30_enhanced] sample={index}/{len(samples)} sample_id={sample_id} "
        f"image={Path(image_path).name} eta={format_seconds(eta_seconds) if completed_count else 'estimating'}",
        flush=True,
    )

    cmd = [
        sys.executable,
        "/Users/ken/MM/Pipeline/final_version/pics/closed_loop.py",
        "--config",
        str(config_path),
        "--image",
        image_path,
        "--text",
        question,
    ]
    before = {p for p in closed_loop_root.iterdir() if p.is_dir()} if closed_loop_root.exists() else set()
    started_at = perf_counter()
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    merged_lines = []
    assert process.stdout is not None
    for raw_line in process.stdout:
        merged_lines.append(raw_line)
        line = raw_line.rstrip("\n")
        print(f"[pilot30_enhanced][{index}/{len(samples)}][{sample_id}] {line}", flush=True)
    returncode = process.wait()
    duration_s = perf_counter() - started_at
    durations.append(duration_s)
    stdout_log.write_text("".join(merged_lines), encoding="utf-8")
    stderr_log.write_text("stderr merged into stdout log\n", encoding="utf-8")

    after = {p for p in closed_loop_root.iterdir() if p.is_dir()} if closed_loop_root.exists() else set()
    new_dirs = sorted(after - before, key=lambda item: item.name)
    run_dir = new_dirs[-1] if new_dirs else None
    final_prompt = run_dir / "final_appreciation_prompt.md" if run_dir else None
    target_path = enhanced_dir / f"{sample_id}.md"

    status = "failed"
    if returncode == 0 and final_prompt and final_prompt.exists():
        shutil.copy2(final_prompt, target_path)
        status = "ok"
    avg_duration = sum(durations) / len(durations)
    remaining = len(samples) - index
    eta_seconds = avg_duration * remaining

    item = {
        "index": index,
        "sample_id": sample_id,
        "image_path": image_path,
        "question": question,
        "status": status,
        "returncode": returncode,
        "run_dir": str(run_dir) if run_dir else "",
        "final_prompt_path": str(target_path) if target_path.exists() else "",
        "duration_s": round(duration_s, 2),
    }
    summary.append(item)
    print(
        f"[pilot30_enhanced] completed={index}/{len(samples)} sample_id={sample_id} "
        f"status={status} duration={format_seconds(duration_s)} avg={format_seconds(avg_duration)} "
        f"eta={format_seconds(eta_seconds)}",
        flush=True,
    )
    print(json.dumps(item, ensure_ascii=False), flush=True)

summary_path = run_log_dir / "pilot30_enhanced_summary.json"
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"samples": len(samples), "ok": sum(1 for item in summary if item["status"] == "ok"), "summary_path": str(summary_path)}, ensure_ascii=False, indent=2))
PY
