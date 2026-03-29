#!/usr/bin/env bash

set -euo pipefail

python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py \
  --config /Users/ken/MM/Pipeline/final_version/config.yaml \
  --dataset-jsonl /Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl \
  --baseline-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_baseline_md \
  --enhanced-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_enhanced_md \
  --output-dir /Users/ken/MM/Pipeline/cluster_match/artifacts \
  --model google/gemini-3.1-pro-preview \
  --timeout 180 \
  --max-retries 2 \
  --schema-profile simple_v1 \
  --sources baseline enhanced \
  --overwrite \
  --sample-id "${1:-s030_01_618710a8_4070_11ed_9adc_c934f75048ef}"

python /Users/ken/MM/Pipeline/cluster_match/evaluate_cluster_match_results.py \
  --results-root /Users/ken/MM/Pipeline/cluster_match/artifacts/simple_v1 \
  --output-root /Users/ken/MM/Pipeline/cluster_match/artifacts/simple_v1/evaluation \
  --dataset-jsonl /Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl \
  --judge-mode exact \
  --sample-id "${1:-s030_01_618710a8_4070_11ed_9adc_c934f75048ef}"
