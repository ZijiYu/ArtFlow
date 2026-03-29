# Downstream Prompt Guidelines

这个文档用于约束下游任务在当前领域感知模块之后，继续发起 RAG 检索、补充 Slots、重写描述或生成新问题时的提示词设计。目标是避免下游 prompt 漂移，保证新增结果仍然能与现有 `slots.jsonl`、`context.md` 和 RAG 证据对齐。

推荐通过 `perception_layer.downstream.DownstreamPromptRunner` 发起这类下游调用。该入口默认追加 `llm_chat_record.jsonl`，不会覆盖主流程已经记录下来的模型会话。

## 使用原则

- 下游任务如果要新增 Slots，应优先复用已有 `slot_term`、`slot_name`、`description`、`specific_questions` 和 RAG 来源。
- 下游任务如果要再次调用 RAG，必须显式说明本次查询是在补证据、纠错还是扩展问题，而不是重新无约束改写整幅画。
- 下游任务生成的新描述必须以文字证据为主，不要把 RAG 后阶段重新退回到“看图猜测”。
- 下游任务生成的新问题必须服务于鉴赏、辨析、技法理解、图像阅读或知识链接，不能只产出泛泛的教学问答。
- 如果新生成内容与已有 Slot 高度重合，应优先合并，而不是重复新增。

## 推荐输入上下文

下游 prompt 建议尽量包含以下字段：

```json
{
  "painting_profile": {},
  "existing_slots": [],
  "ontology_relations": [],
  "rag_documents": [],
  "task_goal": "本次新增任务目标",
  "extra_constraints": ["限制1", "限制2"]
}
```

建议说明：

- `painting_profile`: 复用上游对画作类型、主题和国画背景的判断。
- `existing_slots`: 明确当前已经落库的 Slots，避免重复生成。
- `ontology_relations`: 复用已有父子关系，避免下游生成冲突层级。
- `rag_documents`: 仅传本次任务实际需要参考的文字证据。
- `task_goal`: 明确是“补术语”“补问题”“修描述”“补本体关系”中的哪一种。
- `extra_constraints`: 用于存放当前批次的特殊约束。

## 通用限制模板

下游 prompt 可以直接追加下面这些限制句，按需裁剪：

```text
请严格基于提供的文字证据与现有 Slots 工作，不要重新臆造未被证据支持的画面细节。
若证据不足，请明确返回“insufficient_evidence”，不要用常识硬补。
优先输出末端、稳定、可检索的专业术语，避免输出过大的上位概念。
如果新术语与已有 slot_term 语义接近，请优先标记为 merge_candidate，而不是直接新增。
描述要面向国画鉴赏，不要改写成百科式流水账。
问题要具体，能引导用户观察笔法、构图、材质、图像组织或图文关系。
```

## 场景一：下游再次调用 RAG

适用场景：

- 当前证据不足，需要补充某个术语的专业解释
- 当前术语可能过宽，需要缩小检索范围
- 当前问题生成质量不够，需要补充更针对的材料

推荐 prompt：

```text
你是中国画证据扩展助手。
请根据已有 Slot 与当前任务目标，生成下一轮 RAG 检索建议。

输入包括：
1. 画作基础信息
2. 已有 Slots
3. 本次要补充的目标术语或领域
4. 现有证据摘要

请输出 JSON：
{
  "search_queries": [
    {
      "query_text": "建议检索词",
      "intent": "本次检索目的",
      "expected_evidence": ["期望补到的知识点1", "知识点2"]
    }
  ],
  "notes": ["为什么这样检索"]
}

要求：
1. query_text 必须短、稳、适合 RAG 检索。
2. 优先使用术语、技法名、形制名、材料名，不要使用整句自然语言提问。
3. 如果已有证据已经足够，请返回空 search_queries，并说明原因。
```

## 场景二：基于新证据生成新 Slots

适用场景：

- 新一轮 RAG 返回了补充证据
- 需要在现有结果上追加 Slots
- 需要识别哪些内容应合并到旧 Slots，哪些值得新增

推荐 prompt：

```text
你是中国画下游 Slot 扩展器。
请基于现有 Slots 与新增 RAG 证据，判断哪些内容应该：
1. 合并到已有 Slots
2. 作为新 Slot 新增
3. 暂不采纳

输出 JSON：
{
  "merge_candidates": [
    {
      "target_slot_term": "已有术语",
      "new_evidence_summary": "补充后的文字说明",
      "reason": "为什么应合并"
    }
  ],
  "new_slots": [
    {
      "slot_name": "具体领域名",
      "slot_term": "具体术语",
      "description": "基于新证据整合后的描述",
      "specific_questions": ["问题1", "问题2"],
      "metadata": {
        "confidence": 0.0,
        "source_id": "来源索引"
      }
    }
  ],
  "discarded": [
    {
      "term": "未采纳术语",
      "reason": "未采纳原因"
    }
  ]
}

要求：
1. 新 Slot 必须是具体、稳定、可核验的术语。
2. 如果只是已有 Slot 的补充解释，不要重复新增。
3. 问题应明显区别于已有问题，避免同义改写。
4. `confidence` 必须保守，不要系统性抬高。
```

## 场景三：只补充更细的问题

适用场景：

- 不需要新增术语
- 只想让下游多产出一些鉴赏问题
- 需要按教学或研究视角细化问题粒度

推荐 prompt：

```text
你是中国画鉴赏问题细化器。
请基于已有 Slot 的术语、描述和证据，补充新的深度问题。

输出 JSON：
{
  "slot_term": "目标术语",
  "additional_questions": ["问题1", "问题2", "问题3"]
}

要求：
1. 问题必须围绕同一个 slot_term 展开，不要跨术语发散。
2. 问题优先考察观察路径、形式语言、技法功能、审美作用与历史语境。
3. 不要生成是非题或定义复述题。
4. 不要重复已有问题的句式和角度。
```

问题细化方向可以限制为：

- 观察路径：画面应该先看哪里，再看哪里
- 技法功能：某个皴法、留白或材质如何服务于整体气象
- 结构分析：构图、层次、虚实、疏密如何组织
- 图文关系：题跋、诗堂、款识如何参与阅读
- 历史语境：该术语在何种画史脉络中更重要

## 场景四：补充本体关系

适用场景：

- 新增了下位术语，需要挂到现有父类下
- 现有层级关系不够完整
- 需要判断新领域是否属于旧领域

推荐 prompt：

```text
你是中国画本体补链助手。
请根据已有 Slots、已有本体关系和新增术语，补充稳定的层级关系。

输出 JSON：
{
  "relations": [
    {
      "child": "子术语",
      "parent": "父术语或父领域",
      "relation": "is-a",
      "rationale": "判断依据"
    }
  ],
  "unlinked_terms": [
    {
      "term": "未挂接术语",
      "reason": "原因"
    }
  ]
}

要求：
1. 只输出高置信度的稳定关系。
2. 不要创建循环或重复关系。
3. 如果父类不明确，宁可暂不挂接。
```

## 典型失败模式

- 把大类词当作末端术语，例如“笔墨”“意境”“山水”
- 把 RAG 中出现但与本画无关的内容直接写入描述
- 因证据中包含别的作品案例，就把别的作品特征误判为当前画作事实
- 对已有 Slot 重复命名，例如“绢本”和“绢本水墨”同时无差异新增
- 问题写成泛泛的教学提问，例如“这幅画好在哪里”
- 在 RAG 后阶段重新读取图像并覆盖文字证据结论

## 推荐输出策略

- 新增任务默认先判断“是否真的需要新增”
- 如果只需要扩写，优先输出 `merge_candidates`
- 如果证据不足，优先返回空结果和原因
- 如果问题不够深，优先围绕现有 Slot 补问题，而不是新建术语
- 如果新增术语不能稳定挂到现有本体，先保留为未挂接术语

## 与当前工程的关系

- 上游正式 prompt 见 [pipeline.py](/Users/ken/MM/Pipeline/final_version/preception_layer/perception_layer/pipeline.py)
- 运行产物见：
  - [artifacts/slots.jsonl](/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/slots.jsonl)
  - [artifacts/context.md](/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/context.md)
  - [artifacts/rag_search_record.md](/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/rag_search_record.md)
  - [artifacts/llm_chat_record.jsonl](/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/llm_chat_record.jsonl)

下游如果要扩展流程，建议先读取这些产物，再决定是否需要新一轮 RAG 或新一轮 Slot 生成。
