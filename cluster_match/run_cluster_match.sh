python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match_pipeline.py \
  --config /Users/ken/MM/Pipeline/final_version/config.yaml \
  --dataset-jsonl /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot_16_dataset.jsonl \
  --baseline-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot_16_baseline_md \
  --enhanced-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_enhanced_md \
  --output-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/pilot16_simple_v1 \
  --model google/gemini-3.1-pro-preview \
  --timeout 180 \
  --max-retries 2 \
  --schema-profile simple_v1 \
  --sources baseline \
  "$@"
