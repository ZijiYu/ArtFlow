# slots_v2

`slots_v2` 现在包含两层能力：

1. `slots_v2` 自身的多轮 CoT 收敛引擎
2. 与 `preception_layer_1` 联动的闭环 coordinator

闭环入口在 [closed_loop.py](/Users/ken/MM/Pipeline/final_version/pics/closed_loop.py)，会按下面的顺序工作：

1. 用 `preception_layer_1` bootstrap 生成初始 `slots.jsonl` 与 `context.md`
2. 用 `slots_v2` 跑多轮 `Domain CoT -> Round-table -> Reflection`
3. 把 Reflection 产生的 `spawned_tasks` 转成 `preception_layer_1` downstream task
4. 把 downstream 结果回灌为新的 runtime `slots.jsonl` 与 `context.md`
5. 继续下一轮，直到收敛或达到轮数上限

## 核心入口

- `slots_v2` 单独运行：
  [main.py](/Users/ken/MM/Pipeline/final_version/pics/main.py)
- 闭环运行：
  [closed_loop.py](/Users/ken/MM/Pipeline/final_version/pics/closed_loop.py)
- 闭环 bridge：
  [closed_loop.py](/Users/ken/MM/Pipeline/final_version/src/slots_v2/closed_loop.py)

## 快速开始

建议使用你当前的 Anaconda 环境：

```bash
export OPENAI_API_KEY="your_key"
```

只跑 `slots_v2`：

```bash
/opt/anaconda3/envs/agent/bin/python /Users/ken/MM/Pipeline/final_version/pics/main.py \
  --image /Users/ken/MM/Pipeline/slots_v1/pics/测试1.png \
  --slots-file /Users/ken/MM/Pipeline/preception_layer/artifacts/slots.jsonl \
  --meta-context-file /Users/ken/MM/Pipeline/preception_layer/artifacts/context.md \
  --output-dir /Users/ken/MM/Pipeline/final_version/artifacts
```

跑完整闭环：

```bash
/opt/anaconda3/envs/agent/bin/python /Users/ken/MM/Pipeline/final_version/pics/closed_loop.py \
  --image /Users/ken/MM/Pipeline/slots_v1/pics/测试1.png \
  --text "请对这幅国画做严谨赏析，优先看皴法、题跋和时代线索。" \
  --meta-context-file /Users/ken/MM/Pipeline/preception_layer/artifacts/context.md \
  --output-dir /Users/ken/MM/Pipeline/final_version/artifacts_closed_loop
```

## 输出

`slots_v2` 单轮或多轮输出：

- `summary.md`
- `final_appreciation_prompt.md`
- `domain_outputs.json`
- `cross_validation.json`
- `routing.json`
- `dialogue_state.json`
- `cot_threads.json`

闭环额外输出：

- `perception_bootstrap/slots.jsonl`
- `perception_bootstrap/context.md`
- `runtime_state/slots_final.jsonl`
- `runtime_state/context_final.md`
- `downstream_rounds/.../task_*_payload.json`
- `downstream_rounds/.../task_*_response.json`
- `closed_loop_report.json`

## 测试

单元测试：

```bash
cd /Users/ken/MM/Pipeline/final_version
PYTHONPYCACHEPREFIX=/tmp/slots_v2_pycache /opt/anaconda3/envs/agent/bin/python -m unittest discover -s tests
```

编译检查：

```bash
PYTHONPYCACHEPREFIX=/tmp/slots_v2_pycache /opt/anaconda3/envs/agent/bin/python -m compileall /Users/ken/MM/Pipeline/final_version/src /Users/ken/MM/Pipeline/final_version/pics
```

更完整的技术说明、各环节输入输出、Quickstart 和 Mermaid 图见 [TECHNICAL_REPORT.md](/Users/ken/MM/Pipeline/final_version/TECHNICAL_REPORT.md)。
