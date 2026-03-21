# TextScore

这是一个基于 `TextGrad / TextLoss` 思路改成“只打分、不优化”的实验脚本。

它做的事：

- 固定读取 `textscore/context_1.txt` 和 `textscore/context_2.txt`
- 先做 sentence index
- 按 `slots` 做覆盖度和质量评分
- 分别对两个文本做单文本 TextLoss 风格句子评分
- 不再做逐句双文本对比，不假设两个文本相同 `sentence_id` 的句子语义可对齐
- 根据句子级结果统计哪一侧“值得优化”的句子更多
- 生成 HTML 可视化，并在需要优化的句子下用可折叠区域展示 `reasoning`

## 安装

```bash
python3 -m pip install -e .
```

需要设置环境变量：

```bash
export OPENAI_API_KEY=your_key
```

当前默认配置：

- `base_url`: `https://api.zjuqx.cn/v1`
- `model`: `openai/gpt-4o-mini`

## 运行

```bash
python3 run_textscore.py \
  --slots 癌症 靶向治疗 分子分型 \
  --base-url "https://api.zjuqx.cn/v1" \
  --model "openai/gpt-4o-mini" \
  --output artifacts/result.json \
  --visualization artifacts/view.html
```

默认读取：

- [`textscore/context_1.txt`](/Users/ken/MM/Pipeline/textscore/context_1.txt)
- [`textscore/context_2.txt`](/Users/ken/MM/Pipeline/textscore/context_2.txt)

输出：

- JSON: `artifacts/result.json`
- HTML: `artifacts/view.html`

## 输出结构

核心输出模型如下：

```python
from pydantic import BaseModel
from typing import List

class SlotAnalysis(BaseModel):
    sentence_ids: List[int]
    sentences: List[str]
    slot_name: str
    slot_term: str
    score: int

class SentenceScore(BaseModel):
    sentence_id: int
    sentence_text: str
    score: int
    loss: int
    logic_score: int
    slot_relevance_score: int
    redundancy_score: int
    worth_optimizing: bool
    reasoning: str
    improvement_suggestion: str
    matched_slots: List[str]
    matched_terms: List[str]

class ContextOptimizationSummary(BaseModel):
    need_optimization_count: int
    worth_optimizing_sentence_ids: List[int]
    total_loss: int

class TextScoreResult(BaseModel):
    context_1_score: int
    context_2_score: int
    context_1_slots_score: int
    context_2_slots_score: int
    context_1_slots_analysis: List[SlotAnalysis]
    context_2_slots_analysis: List[SlotAnalysis]
    context_1_sentence_scores: List[SentenceScore]
    context_2_sentence_scores: List[SentenceScore]
    context_1_optimization_summary: ContextOptimizationSummary
    context_2_optimization_summary: ContextOptimizationSummary
    context_more_to_optimize: str
    tokens: int
```

## 设计说明

- 两次单文本 TextLoss 风格分析分别生成句子级 `score/loss`
- 每个文本内部再拆成两条独立 prompt：一条只做 `slots_analysis`，一条只做句子级优化判定
- 句子级结果额外考虑逻辑、slot 相关性、重复语义和是否值得优化
- 总分仍只基于 `slots_analysis` 的分数求和
- HTML 只强调“值得优化”的句子，并把 `reasoning` / `improvement_suggestion` 放进可折叠区域
