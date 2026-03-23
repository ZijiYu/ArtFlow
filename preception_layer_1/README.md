# Guohua Perception Layer

领域感知模块，负责将国画图像与基础文字信息转成可落库的 Slots，并完成 RAG 对齐、语义去重与动态本体维护。

## 能力范围

- Step 0: 先调用模型分析这幅画是什么画、主题是什么，以及可用于后续抽取的国画知识背景。
- Step 1: 并行执行视觉线索提取和文本信号分析，再做多模态 anchor 三元组抽取。
- Step 2: 调用 RAG 检索，对证据与视觉线索做相似度过滤。RAG 图片输入使用内存中的 `multipart/form-data` 文件上传。
- Step 2.5: RAG 完成后进入 text-only 阶段，后续 Slot 生成和本体推理只读取文字证据，不再携带图像或视觉证据。
- Step 3: 推理 `is-a` 层级关系，持续更新 `context.md`。
- Dedup: 使用 embedding 语义相似度做强制合并，默认阈值为 `0.83`。

## 环境变量

- `OPENAI_API_KEY`: 必填。LLM 与 embedding 共用；若 CLI 不传 `--api-key`，这里作为兜底。

## CLI

```bash
export OPENAI_API_KEY=your_key

python -m perception_layer.cli \
  --image /path/to/image.png \
  --text "基础文字信息" \
  --api-key your_key \
  --base-url https://api.zjuqx.cn/v1 \
  --embedding-model baai/bge-m3 \
  --judge-model gemini-3pro \
  --rag-endpoint http://221.12.22.162:8888/test/8002/api/search \
  --terminal-log /Users/ken/MM/Pipeline/preception_layer/artifacts/terminal_output.log \
  --output /Users/ken/MM/Pipeline/preception_layer/artifacts/slots.jsonl
```

模型与路由参数以 CLI 输入为准，不再从环境变量里挑选模型。embedding 和判定模型都通过 OpenAI 兼容接口调用，不再依赖本地 `FlagEmbedding`。LLM 侧图片继续以 base64 传入，多模态 RAG 侧图片改为内存流 `multipart/form-data` 上传。

若不显式传 `--terminal-log`，终端中的标准输出和标准错误会默认保存到输出目录下的 `terminal_output.log`。

默认 `context.md` 会保存到 `/Users/ken/MM/Pipeline/preception_layer/artifacts/context.md`。

默认会额外生成 `/Users/ken/MM/Pipeline/preception_layer/artifacts/rag_search_record.md`，记录每次 RAG 搜索的查询词、是否带图、返回来源和对齐分数。

默认还会生成 `/Users/ken/MM/Pipeline/preception_layer/artifacts/llm_chat_record.jsonl`，内容为格式化 JSON 数组，记录每次 LLM 调用的 system prompt、user 内容、是否附图和模型返回，便于直接阅读。

主流程 `PerceptionPipeline.run()` 默认会刷新这份聊天记录；下游扩展 `DownstreamPromptRunner.run_json()` 默认会在原文件后追加，不覆盖主流程历史记录。

## Prompt 扩展

下游如果还要继续调用 RAG、补充新的 Slots、细化问题或补本体关系，可以参考 `/Users/ken/MM/Pipeline/preception_layer/downstream_prompt_guidelines.md`。这个文档整理了下游 prompt 的推荐输入结构、限制模板、扩展场景和常见失败模式，避免和当前主流程的约束冲突。

## 技术报告

完整技术报告见 `/Users/ken/MM/Pipeline/preception_layer/TECHNICAL_REPORT.md`，其中包含快速使用、处理逻辑、Mermaid 流程图、输入输出结构以及下游扩展调用说明。
