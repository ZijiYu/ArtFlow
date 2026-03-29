# cluster_match

基于本地 `md` 目录的批量事实要素抽取工具。

它会：

- 读取 pilot30 数据集中的 `sample_id -> question` 映射
- 分别处理 `pilot30_enhanced_md` 与 `pilot30_baseline_md` 两组赏析文本
- 调用与 `final_version` 相同的 API 配置
- 默认使用 `google/gemini-3.1-pro-preview`
- 支持 `legacy` 与 `academic_v2` 两种 schema profile
- 输出符合指定结构的 JSON 结果

## 默认输入

- 数据集：`/Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl`
- baseline：`/Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_baseline_md`
- enhanced：`/Users/ken/MM/Pipeline/cluster_match/artifacts/pilot30_enhanced_md`
- API 配置：`/Users/ken/MM/Pipeline/final_version/config.yaml`

## 输出目录

- `artifacts/pilot30_baseline/*.json`
- `artifacts/enhanced/*.json`
- `artifacts/runs/<timestamp>/run_index.jsonl`
- `artifacts/runs/<timestamp>/summary.json`
- `artifacts/runs/<timestamp>/raw_responses/...`

当 `schema_profile=academic_v2` 且未显式指定 `--output-dir` 时，结果会自动落到：

- `artifacts/academic_v2/pilot30_baseline/*.json`
- `artifacts/academic_v2/enhanced/*.json`

## 用法

处理全部可用样本：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py
```

或直接用 shell 脚本启动：

```bash
bash /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh
```

说明：当前 launcher 默认复用已有 `baseline` 结果，只重跑 `enhanced(v_2)`，并在抽取后自动执行评测。

只跑一个样本：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py \
  --sample-id s030_01_618710a8_4070_11ed_9adc_c934f75048ef
```

```bash
bash /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh \
  --sample-id s030_01_618710a8_4070_11ed_9adc_c934f75048ef
```

如果要单图直接跑 LLM 裁判评测：

```bash
bash /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh \
  --sample-id s030_01_618710a8_4070_11ed_9adc_c934f75048ef \
  --judge-mode llm \
  --judge-model google/gemini-3.1-pro-preview
```

只跑 enhanced：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py \
  --sources enhanced
```

只跑 baseline：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py \
  --sources baseline
```

对应 shell 用法：

```bash
bash /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh --sources baseline
```

```bash
bash /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh --sources enhanced
```

说明：

- `--sources baseline` 只重跑 baseline 抽取
- `--sources enhanced` 只重跑 enhanced 抽取
- 只跑单边时，launcher 会跳过评测，避免误用旧结果
- 同时传入 `--sources baseline enhanced` 时，才会在抽取后自动评测

显式指定新分层 schema：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py \
  --schema-profile academic_v2 \
  --output-dir /Users/ken/MM/Pipeline/cluster_match/artifacts/academic_v2
```

覆盖已有输出重新生成：

```bash
python /Users/ken/MM/Pipeline/cluster_match/run_cluster_match.py --overwrite
```

生成统计汇总与论文风格图表：

```bash
python /Users/ken/MM/Pipeline/cluster_match/analyze_cluster_match_results.py
```

## 常用环境变量

```bash
PYTHON_BIN=python3
MODEL_NAME=google/gemini-3.1-pro-preview
OUTPUT_DIR=/Users/ken/MM/Pipeline/cluster_match/artifacts
TIMEOUT_SECONDS=180
MAX_RETRIES=2
SCHEMA_PROFILE=academic_v2
```

## 说明

- 原始输入目录不会被移动或改写，新工具只读取它们。
- `legacy` 中如果某一类和问题相关但文本未涉及，会输出 `原句: "未涉及"`。
- `legacy` 中如果某一类与问题不相关，会输出 `相关性: "不相关"` 和 `原句: "不相关"`。
- `legacy` schema 使用原来的平铺 15 类结构，保留 `关键词 / 相关性 / 原句`。
- `academic_v2` schema 使用分层结构，叶子节点为 `相关性 + 要素列表`，更适合论文式要素体系分析。
- `academic_v2` 会优先把要素拆成最小语义单元，例如 `使用了淡墨披麻皴描绘远山` 会尽量规整为 `["淡墨", "披麻皴", "远山"]`。
- `academic_v2` 中如果文本没有明确提到某个叶子层级，该节点会归一化为 `相关性: "不相关"` 和 `要素列表: []`。

## 加权评测

默认口径：

- `baseline = GT`
- `enhanced = Gen`
- 遍历 GT 中每个有效要素，在同层级的 Gen 要素里寻找匹配
- 只要 `enhanced` 中有一个要素与某个 GT 要素匹配，就算该 GT 要素命中
- 如果 `enhanced` 中还有 `baseline` 未覆盖的额外细节，这些额外细节按成功项计分
- 强匹配记 `1.0`，弱匹配记 `0.5`，未命中记 `0.0`
- `Matched_TP_w = N_strong + 0.5 * N_weak`
- `TP_w = Matched_TP_w + N_extra_gen_success`
- `ACC = TP_w / (N_GT + N_extra_gen_success)`
- `Precision = TP_w / N_Gen`
- `F1 = 2 * Precision * ACC / (Precision + ACC)`

快速跑严格字符评测：

```bash
python /Users/ken/MM/Pipeline/cluster_match/evaluate_cluster_match_results.py \
  --judge-mode exact
```

跑 LLM 语义裁判评测：

```bash
python /Users/ken/MM/Pipeline/cluster_match/evaluate_cluster_match_results.py \
  --judge-mode llm
```

说明：

- `llm` 模式会优先走规则匹配，规则无法判断时再调用裁判模型
- 如果单次裁判调用超时或返回异常，该要素对会自动降级为未命中，不会中断整批评测
- 降级次数会写入 `summary.json` 和 `evaluation_report.md`

只评一个样本：

```bash
python /Users/ken/MM/Pipeline/cluster_match/evaluate_cluster_match_results.py \
  --judge-mode llm \
  --sample-id s030_01_618710a8_4070_11ed_9adc_c934f75048ef
```

## 单图模式方案

建议直接复用现有两段式链路，不另外维护第二套脚本：

- 在抽取阶段继续使用 `run_cluster_match.py --sample-id <sample_id>`
- 在评测阶段继续使用 `evaluate_cluster_match_results.py --sample-id <sample_id>`
- 由 [run_cluster_match.sh](/Users/ken/MM/Pipeline/cluster_match/run_cluster_match.sh) 统一接收 `--sample-id`，并同步传给抽取和评测
- 终端仍保留当前的 `[当前/总数]`、耗时、ETA、样本问题、`ACC / Precision / F1`

这样做的好处是：

- 不新增并行维护的单图代码路径
- 单图与批量共用同一套 prompt、schema、归一化、评测公式
- 单图调试完以后，直接切回批量命令即可

输出内容包括：

- `summary.json`
- `evaluation_report.md`
- `tables/sample_scores.csv`
- `tables/category_scores_summary.csv`
- `tables/match_details.csv`
- `figures/overall_metrics.png`
- `figures/sample_metric_distribution.png`
