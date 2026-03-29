# Eval V2

`eval_v2` 实现了你要求的整套国画赏析对比评测流水线：

- 对 `context_baseline` / `context_enhanced` 先做 sentence index
- 两个 context 并行调用 `baai/bge-m3` 做 semantic embedding
- 基于 embedding 做 semantic duplicate check，输出重复语义簇 JSONL
- 基于语义聚类统计 `similar_semantic_num`、`duplicate_sentence_num`、`unique_semantic_num`
- 调用 `openai/gpt-4.1` 抽取中国画专业术语，输出 `term_num` 和术语 JSONL
- 调用 `openai/gpt-4.1` 基于自然语言 `slots` 做 slot match，输出 JSONL 和 `slots_match`
- 调用 `openai/gpt-4.1` 基于 `image_context_v` 做术语 Fidelity 判定，输出 JSONL 和 `accuracy`
- 最后按你给的 prompt 形式做 Win/Loss/Tie 总评，并生成 Textual Loss

## 安装

```bash
cd /Users/ken/MM/Pipeline/eval_v2
python3 -m pip install -e .
```

需要环境变量：

```bash
export OPENAI_API_KEY=
```

默认模型与接口：

- `base_url`: `https://api.zjuqx.cn/v1`
- `embedding_model`: `baai/bge-m3`
- `judge_model`: `openai/gpt-4.1`

## 运行

```bash
cd /Users/ken/MM/Pipeline/eval_v2
python3 run_eval_v2.py \
  --context-baseline-file ./inputs/context_baseline.txt \
  --context-enhanced-file ./inputs/context_enhanced.txt \
  --slots-file ./inputs/slots.txt \
  --image-context-v-file ./inputs/image_context_v.txt \
  --output-dir ./artifacts
```

也可以直接传文本参数：

```bash
python3 run_eval_v2.py \
  --context-baseline "文本A" \
  --context-enhanced "文本B" \
  --slots-text "重点关注馆藏、题跋、印章、技法、设色等术语。" \
  --image-context-v "画面可见绢本设色、浅绛山水、右上角题跋与多方印章。"
```

## 输出

运行后会写入：

- `artifacts/baseline_duplicate_clusters.jsonl`
- `artifacts/baseline_terms.jsonl`
- `artifacts/baseline_slot_matches.jsonl`
- `artifacts/baseline_fidelity.jsonl`
- `artifacts/enhanced_duplicate_clusters.jsonl`
- `artifacts/enhanced_terms.jsonl`
- `artifacts/enhanced_slot_matches.jsonl`
- `artifacts/enhanced_fidelity.jsonl`
- `artifacts/eval_v2_result.json`

`eval_v2_result.json` 中包含：

- 结构化 `slots_spec`
- 两个 context 的全部统计指标
- 最终 `winner`
- `textual_loss_for`
- `textual_loss`
- 全部 API token 汇总 `llm_tokens`
