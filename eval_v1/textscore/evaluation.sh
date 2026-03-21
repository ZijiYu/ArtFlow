export OPENAI_API_KEY="sk-jGKJtrju4HnIdttWD902Ad017d2d484b93F0CcEc08CcA9A6"
CONDA_PATH=$(conda info --base)
source "$CONDA_PATH/etc/profile.d/conda.sh"
conda activate agent
/opt/anaconda3/envs/agent/bin/python /Users/ken/MM/Pipeline/run_textscore.py \
    --slots 技法 结构 文学 美学 材料 \
    --base-url "https://api.zjuqx.cn/v1" \
    --model "openai/gpt-4o-mini" \
    --output artifacts/result.json \
    --visualization artifacts/view.html