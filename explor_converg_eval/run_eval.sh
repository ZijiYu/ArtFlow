cd /Users/ken/MM/Pipeline/explor_converg_eval
bash scripts/run_comparison.sh all intersection \
  --source "ground_truth=/Users/ken/MM/eval_data/gt_data.jsonl:type=jsonl:id_field=image:text_field=assistant:display_name=GT" \
  --source "pipeline=/Users/ken/MM/eval_data/pipeline:type=txt_dir:display_name=Pipeline" \
  --source "zhihua=/Users/ken/MM/eval_data/zhihua:type=txt_dir:display_name=Zhihua" \
  --source "gpt_5.4=/Users/ken/MM/eval_data/gpt_5.4:type=txt_dir:display_name=GPT-5.4" \
  --source "gemini_3.1=/Users/ken/MM/eval_data/gemini_3.1:type=txt_dir:display_name=Gemini-3.1" \
  --source "qwen3.5-397b=/Users/ken/MM/eval_data/qwen3.5-397b:type=txt_dir:display_name=Qwen3.5-397B" \
  --source "kimi_2.5=/Users/ken/MM/eval_data/kimi-2.5:type=txt_dir:display_name=Kimi-2.5" \
  --source "gpt_4.1=/Users/ken/MM/eval_data/gpt-4.1:type=txt_dir:display_name=GPT-4.1" \
  --gt-source ground_truth
