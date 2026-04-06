#!/usr/bin/env bash

set -euo pipefail

ROOT="/Users/ken/MM/Pipeline"
FINAL_VERSION_ROOT="$ROOT/final_version"
OUTPUT_ROOT="$ROOT/ablation/step_5_gemini_3.1"
CONFIG_PATH="$FINAL_VERSION_ROOT/config.yaml"
DATASET_JSONL="$ROOT/ablation/test_image.jsonl"
LOG_PATH="$OUTPUT_ROOT/model_replacement_baseline20.log"
PID_PATH="$OUTPUT_ROOT/model_replacement_baseline20.pid"
AGENT_PY="/opt/anaconda3/envs/agent/bin/python"

mkdir -p "$OUTPUT_ROOT"

MODE="${1:-foreground}"

run_cmd=(
  "$AGENT_PY"
  "$ROOT/ablation/run_ablation_dataset.py"
  --config "$CONFIG_PATH"
  --dataset-jsonl "$DATASET_JSONL"
  --output-root "$OUTPUT_ROOT"
  --variant baseline
  --limit 20
  --runner-cmd "uv run --with pillow --with openai python"
)

if [[ "$MODE" == "--background" ]]; then
  "$AGENT_PY" - <<'PY'
import json
import subprocess
from pathlib import Path

root = Path("/Users/ken/MM/Pipeline")
output_root = root / "ablation" / "step_5_gemini_3.1"
log_path = output_root / "model_replacement_baseline20.log"
pid_path = output_root / "model_replacement_baseline20.pid"
cmd = [
    "/opt/anaconda3/envs/agent/bin/python",
    str(root / "ablation" / "run_ablation_dataset.py"),
    "--config", str(root / "final_version" / "config.yaml"),
    "--dataset-jsonl", str(root / "ablation" / "test_image.jsonl"),
    "--output-root", str(output_root),
    "--variant", "baseline",
    "--limit", "20",
    "--runner-cmd", "uv run --with pillow --with openai python",
]
log_handle = log_path.open("ab", buffering=0)
proc = subprocess.Popen(
    cmd,
    cwd=str(root / "final_version"),
    stdin=subprocess.DEVNULL,
    stdout=log_handle,
    stderr=subprocess.STDOUT,
    start_new_session=True,
)
pid_path.write_text(str(proc.pid), encoding="utf-8")
print(json.dumps({
    "pid": proc.pid,
    "log_path": str(log_path),
    "pid_path": str(pid_path),
    "mode": "background",
    "variant": "baseline",
    "limit": 20,
}, ensure_ascii=False))
PY
  exit 0
fi

cd "$FINAL_VERSION_ROOT"
exec "${run_cmd[@]}"
