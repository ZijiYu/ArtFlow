export OPENAI_API_KEY="sk-jGKJtrju4HnIdttWD902Ad017d2d484b93F0CcEc08CcA9A6"
cd /Users/ken/MM/Pipeline/final_version/preception_layer
/opt/anaconda3/envs/agent/bin/python -m perception_layer.cli \
  --image /Users/ken/MM/Pipeline/final_version/preception_layer/images/10340357965d9de3.png \
  --text "请对这幅画进行赏析。" \
  --base-url https://api.zjuqx.cn/v1 \
  --embedding-model text-embedding-3-large \
  --judge-model google/gemini-3.1-pro-preview \
  --output /Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/slots.jsonl


# export OPENAI_API_KEY="sk-jGKJtrju4HnIdttWD902Ad017d2d484b93F0CcEc08CcA9A6"
# /opt/anaconda3/envs/agent/bin/python /Users/ken/MM/Pipeline/final_version/preception_layer/test_embedding_model.py \
#   --base-url https://api.zjuqx.cn/v1 \
#   --model text-embedding-3-large