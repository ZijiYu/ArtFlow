export OPENAI_API_KEY="sk-jGKJtrju4HnIdttWD902Ad017d2d484b93F0CcEc08CcA9A6"
cd /Users/ken/MM/Pipeline/eval_v2
/opt/anaconda3/envs/agent/bin/python  run_eval_v2.py \
  --context-baseline-file /Users/ken/MM/Pipeline/eval_v2/inputs/context_baseline.txt \
  --context-enhanced-file /Users/ken/MM/Pipeline/eval_v2/inputs/context_enhanced.txt \
  --slots-file /Users/ken/MM/Pipeline/eval_v2/inputs/slots.txt \
  --image-context-v-file /Users/ken/MM/Pipeline/eval_v2/inputs/image_context_v.txt \
  --embedding-model text-embedding-3-large \
  --judge-model openai/gpt-4.1 \
  --output-dir /Users/ken/MM/Pipeline/eval_v2/artifacts
