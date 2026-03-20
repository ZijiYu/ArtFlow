# TCP 国画鉴赏 Prompt Framework

这是一个可本地运行的国画鉴赏框架，支持四种模式：
- `slot`
- `solitary`
- `communal`
- `slot_pipe`（V3/V4）

## 1. 运行前准备

建议使用 `uv`：

```bash
uv sync
```

配置 API：

```bash
export NEW_API_KEY="your_key"
export NEW_API_BASE_URL="your_api_base_url"
export NEW_API_MODEL="gpt-4o-mini". # 这个可以不用
```

说明：
- 未配置 API 时，流程仍可执行，但会进入回退文本。
- 控制台会打印 API 调用开始/结束日志，便于观察是否卡住。

## 2. 快速命令（整合 command）

### slot

```bash
uv run main.py \
  --mode slot \
  --image "./pics/测试1.png" \
  --slots all \
  --agent-model gpt-4o-mini \
  --baseline-model gpt-4o-mini \
  --enhanced-model gpt-4o-mini
```

### communal

```bash
uv run main.py \
  --mode communal \
  --image "./pics/测试1.png" \
  --guest-num 4 \
  --guest-model gpt-4o-mini
```

### solitary

```bash
uv run main.py \
  --mode solitary \
  --image "./pics/测试1.png" \
  --slots all \
  --solitary-model gpt-4o-mini \
  --solitary-rounds 4
```

### slot_pipe V3

```bash
uv run main.py \
  --mode slot_pipe \
  --slot-pipe-version 3 \
  --image "./pics/测试1.png" \
  --slot-pipe-agents-per-slot 1 \
  --slot-pipe-max-retries 1 \
  --agent-model gpt-4o-mini \
  --judge-model gpt-4o-mini \
  --output-dir outputs/v3
```

### slot_pipe V4

```bash
uv run main.py \
  --mode slot_pipe \
  --slot-pipe-version 4 \
  --image "./pics/测试2.jpeg" \
  --slot-pipe-slots-file artifacts/slots.jsonl \
  --checker-model gpt-4o-mini \
  --reviewer-model gpt-4o-mini \
  --content-model gpt-4o-mini \
  --expression-guest-model gpt-4o-mini \
  --expression-summary-model gpt-4o-mini \
  --final-appreciation-model gpt-4o-mini \
  --output-dir outputs/v4
```

## 3. 参数总览（全部模式）

### 3.1 通用参数

- `--mode`
  - 可选：`slot` / `solitary` / `communal` / `slot_pipe`
- `--image`
  - 图像路径或图像 URL（必填）
- `--meta`
  - JSON 字符串，默认 `{}`
- `--slots`
  - slot 列表，逗号分隔；留空或 `all` 使用默认全量
- `--agent-temperature`
  - agent 生成温度，默认 `0.7`
- `--vlm-temperature`
  - VLM 鉴赏温度，默认 `0.2`
- `--agent-model`
  - agent 通用模型
- `--baseline-model`
  - baseline 鉴赏模型
- `--enhanced-model`
  - enhanced 鉴赏模型
- `--api-timeout`
  - 单次 API 超时秒数，默认 `60`
- `--output-dir`
  - 输出目录，默认 `outputs`

### 3.2 solitary 模式参数

- `--solitary-model`
  - solitary 专用模型
- `--solitary-rounds`
  - 反思轮数，最少 1，默认 3

### 3.3 communal 模式参数

- `--guest-num`
  - 客人数量，默认 3
- `--guest-model`
  - 客人模型

### 3.4 slot_pipe 通用参数（V3/V4）

- `--judge-model`
  - judge 模型（V3 主要使用）
- `--slot-pipe-version`
  - 版本号，默认 `4`；设为 `3` 可回退旧流程
- `--final-appreciation-model`
  - 最终鉴赏模型

### 3.5 slot_pipe V3 参数

- `--slot-pipe-agents-per-slot`
  - 每个 slot 的并行 agent 数，默认 2
- `--slot-pipe-max-retries`
  - 每个 slot 最大返工轮数，默认 3
- `--embedding-model`
  - 池化降重 embedding 模型

### 3.6 slot_pipe V4 参数

- `--slot-pipe-slots-file`
  - V4 slot 输入 JSONL 文件，默认 `artifacts/slots.jsonl`
- `--slot-pipe-content-agents`
  - 内容层每个 slot 的调用次数，默认 2
- `--slot-pipe-expression-guests`
  - 表达层每个问题的群赏客人数，默认 3
- `--checker-model`
  - Layer1 checker 模型
- `--reviewer-model`
  - Layer1 reviewer 模型
- `--content-model`
  - Layer2 内容模型
- `--expression-guest-model`
  - Layer3 guest 模型
- `--expression-summary-model`
  - Layer3 summary 模型

## 4. 模式说明

### slot

- 一轮并行 slot agent 生成上下文。
- 生成 baseline / enhanced 两套 prompt。
- 各跑一次 VLM，比较结果。

### solitary

- 单模型多轮反思迭代。
- 每轮读取上一轮结果并重写。

### communal

- 多客人串行接力。
- 后序客人参考前序客人观点，强调互补与增量观察。

### slot_pipe V3

- 三层迭代 + 全局 judge + 返工机制 + embedding 池化压缩。

### slot_pipe V4

- 三层固定流程：
  - Layer1 要素确认：checker 给 `score(0-5)`，reviewer 给 `confidence(0-5)`
  - Layer2 内容要点：输出内容问题集合
  - Layer3 鉴赏表达：沉淀表达规范要点
- 最终 slot 结构强调：
  - 分/置信度
  - 内容问题
  - 鉴赏表达要点

## 5. 输出文件

每次运行会生成 `outputs/<timestamp>/` 目录。

根目录常见文件：
- `api_calls.jsonl`
- `image_delivery_checks.json`
- `report.json`

slot/solitary/communal 常见：
- `prompt_baseline.md`
- `prompt_enhanced.md`
- `analysis_baseline.md`
- `analysis_enhanced.md`
- `slot_context.md`（slot 模式）

slot_pipe 常见：
- `slot_pipe/slots_timeline.json`
- `slot_pipe/final_prompt_supplement.md`
- `slot_pipe/final_appreciation.md`
- `slot_pipe/v4_final_slots.json`（V4）

## 6. 目录结构

- `main.py`：CLI 入口
- `src/tcp_pipeline/pipeline.py`：主编排
- `src/tcp_pipeline/prompt_builder.py`：prompt 构造
- `src/tcp_pipeline/new_api_client.py`：API 客户端
- `src/tcp_pipeline/vlm_runner.py`：VLM 调用封装
- `src/tcp_pipeline/agents.py`：slot agent 并发执行
- `src/tcp_pipeline/slots.py`：slot 定义与归一化
- `src/tcp_pipeline/token_tracker.py`：token 统计
- `src/tcp_pipeline/models.py`：数据结构
- `docs/代码运行逻辑_v1_4.md`：运行逻辑文档（v1.4）
