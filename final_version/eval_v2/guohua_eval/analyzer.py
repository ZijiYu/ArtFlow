from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Sequence

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - 允许在无 openai 包时用 fake client 做离线测试
    OpenAI = None  # type: ignore[assignment]

try:
    from openai import APIStatusError
except ImportError:  # pragma: no cover
    APIStatusError = Exception  # type: ignore[assignment]

from .models import (
    ContextMetrics,
    DuplicateCluster,
    EvalV2Result,
    FidelityOutput,
    FidelityRecord,
    FinalJudgment,
    FinalJudgmentOutput,
    SemanticTerm,
    SlotDefinition,
    SlotMatchOutput,
    SlotMatchRecord,
    SlotsSpec,
    TermExtractionOutput,
)
from .sentence_indexer import build_indexed_text, estimate_token_count


SLOT_PARSE_SYSTEM_PROMPT = """你是一个严格的任务配置解析器。
你的任务是把用户输入的 slots 自然语言说明，解析成结构化的专业评测 slot 列表。

必须遵守：
1. 只能依据用户输入内容做解析，禁止补充用户未暗示的新评测方向。
2. 每个 slot 需要包含 `slot_name`、`description`、`covered_terms`。
3. `slot_name` 要简洁稳定，适合作为统计维度。
4. `covered_terms` 填与该 slot 明确相关的术语、实体、细节类别。
5. 输出必须是合法 JSON，字段名固定为 `slots`。
"""


TERM_EXTRACTION_SYSTEM_PROMPT = """你是一个严格的中国画术语抽取器。
你的任务是从给定文本中抽取“文本中明确出现、且可被专业核验”的中国画术语。

必须遵守：
1. 只能依据输入文本，禁止引入外部知识。
2. 只允许抽取“末端、具体、可核验”的术语，不要抽大类标签、评价词、相关描述词。
3. 允许的术语类型包括：具体技法名称、具体设色法、具体皴法/描法/墨法、材质、装裱形制、尺寸、朝代、馆藏机构、馆藏地点、题跋作者、题跋原文、印章释文、印主、款识、定名、登录号等。
4. 严禁输出宽泛词或描述性短语，例如：笔墨、线条、笔触、笔法、构图、色彩、气韵、意境、中国山水画、天人合一、高低错落的构图、蓝色的山峦、暖黄色的背景。
5. 如果一个表达还能继续细分，则不要抽上位词，要抽最具体的下位术语。
6. 反例：
   - “笔法细腻” -> 不输出
   - “构图完整” -> 不输出
   - “蓝色与暖黄色对比” -> 不输出
   - “题跋署石涛” -> 可抽 `石涛` / 类别 `题跋作者`
   - “现藏于国立故宫博物院” -> 可抽 `国立故宫博物院` / 类别 `馆藏机构`
   - “雨点皴” -> 可抽 `雨点皴` / 类别 `皴法`
   - “绢本设色” -> 可抽 `绢本` / 类别 `材质`，以及 `设色` 只有在出现具体形式如 `浅绛设色`、`青绿设色` 时才可抽
7. 每条术语必须给出 `term`、`category`、`detail`、`sentence_ids`、`evidence_sentences`。
8. 如果文本未提供足够证据，不要抽取该术语。
9. 输出必须是合法 JSON，字段名固定为 `terms`。
"""


SLOT_MATCH_SYSTEM_PROMPT = """你是一个严谨的中国画专业术语【Slot 命中判定器】。
你的任务是将抽取的术语与预设的评测 Slots 进行精准对齐，并剔除无实际技法价值的通用词。

必须严格遵守以下判定准则：

1. **专业域隔离（去通用化）**：
   - 严禁匹配通用领域术语。例如：[线条、笔法、构图、题跋、印章、材质、水墨、色彩] 均视为噪声，不予命中。
   - 仅允许匹配具备明确画论定义的专业技法。例如：[雨点皴、高古游丝描、积墨法、三远法] 等。

2. **最低颗粒度原则（去粗取精）**：
   - 禁止命中宽泛的“大类标签”。如果一个术语只是类别名称（如：皴法、描法、墨法），则不予命中。
   - 必须命中具体的、不可再分的“末端技法”。
   - 示例：若文中提到“雨点皴”，应命中“雨点皴”对应的 Slot；若文中仅提到“使用了某种皴法”，则不判定为命中任何 Slot。

3. **证据闭环**：
   - 只能依据输入的结构化 Slots 和 Terms 做判断，禁止脑补。
   - 一个 Slot 被激活的前提是：抽取的术语在语义上与 Slot 预设的含义完全对齐。

4. **输出格式规范**：
   - 仅当至少有一个 Slot 命中时，才输出结果。
   - 输出必须是合法 JSON，根字段名为 `matches`。
   - 每个 match 条目字段固定为：`slot_name`、`matched_terms`、`matched_categories`、`sentence_ids`、`reason`。
   - 在 `reason` 中需明确说明：为什么该术语满足“专业域”且达到了“最低颗粒度”。
"""


FIDELITY_SYSTEM_PROMPT = """你是一个严格的视觉线索一致性判定器。
你的任务是判断文本术语是否被参考视觉线索基准支持。

必须遵守：
1. 只能依据 `image_context_v` 与给定术语做判断，禁止依赖外部知识。
2. 输入中如果混入了宽泛词、评价词、画面描述词，应将其视为非术语，并判为 `supported_by_ground_truth=false`，reason 中说明“该项不是可核验术语”。
3. 每个术语都必须输出一条结果。
4. 输出字段为 `fidelity`，每条包含 `term`、`category`、`supported_by_ground_truth`、`reason`。
5. 只有当视觉线索基准能直接支持该术语时，`supported_by_ground_truth` 才能为 true。
6. 输出必须是合法 JSON。
"""


FINAL_JUDGMENT_SYSTEM_PROMPT = """你是一个严格的专业中国画赏析评测裁判。
你的任务是根据两个文本的统计指标、重复语义簇和术语/fidelity 结果，判断哪篇赏析更符合专业水准。

必须遵守：
1. 输出必须是合法 JSON。
2. `winner` 只能是 `context_baseline`、`context_enhanced` 或 `tie`。
3. `textual_loss_for` 只能是 `context_baseline`、`context_enhanced` 或 `tie`。
4. `textual_loss` 必须明确指出差的一方哪些语义簇造成冗余、忽略了哪些视觉线索。
5. `reasoning` 必须是高层判断依据，不能只重复数值。
"""


class _UsageCounter:
    def __init__(self) -> None:
        self._total = 0
        self._lock = Lock()

    def add_from_response(self, response: Any) -> None:
        usage = getattr(response, "usage", None)
        total_tokens = getattr(usage, "total_tokens", 0) if usage is not None else 0
        if total_tokens:
            with self._lock:
                self._total += int(total_tokens)

    @property
    def total(self) -> int:
        return self._total


class _UnionFind:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, left: int, right: int) -> None:
        root_left = self.find(left)
        root_right = self.find(right)
        if root_left != root_right:
            self.parent[root_right] = root_left


GENERIC_TERM_BLACKLIST = {
    "笔墨",
    "线条",
    "笔触",
    "笔法",
    "构图",
    "色彩",
    "设色",
    "气韵",
    "意境",
    "艺术理念",
    "文化思想",
    "中国山水画",
    "天人合一",
    "线条与轮廓",
}

GENERIC_TERM_PATTERNS = [
    re.compile(r"^[高低远近上下左右前后].*构图$"),
    re.compile(r"^.+的构图$"),
    re.compile(r"^.+的山[川峦石水林木树].*$"),
    re.compile(r"^.+的背景$"),
    re.compile(r"^.+对比$"),
]

GENERIC_CATEGORY_BLACKLIST = {
    "艺术理念",
    "文化思想",
    "流派",
}

CONCRETE_TERM_PATTERNS = [
    re.compile(r".+(皴|描|擦|染|点苔|积墨|破墨|泼墨)$"),
    re.compile(r".+(设色|青绿|浅绛|没骨|工笔|写意)$"),
    re.compile(r".+(绢本|纸本|绫本|册页|手卷|立轴|横披|镜心|扇面)$"),
    re.compile(r".+(题跋|题款|款识|印|印文|朱文印|白文印)$"),
    re.compile(r".+(博物院|美术馆|故宫|纪念馆|图书馆)$"),
    re.compile(r".+(厘米|公分|cm)$"),
    re.compile(r".+(御览之宝|石渠宝笈)$"),
]

CONCRETE_CATEGORY_HINTS = {
    "技法名称",
    "皴法",
    "描法",
    "墨法",
    "设色法",
    "设色",
    "材质",
    "尺寸",
    "朝代",
    "馆藏机构",
    "馆藏地点",
    "馆藏地址",
    "收藏机构",
    "题跋作者",
    "题跋内容",
    "题款",
    "款识",
    "印章内容",
    "印章释文",
    "印主"
}


def _normalize_term_text(text: str) -> str:
    return re.sub(r"\s+", "", text.strip())


def _looks_like_generic_term(term: str, category: str) -> bool:
    normalized_term = _normalize_term_text(term)
    normalized_category = _normalize_term_text(category)
    if not normalized_term:
        return True
    if normalized_term in GENERIC_TERM_BLACKLIST:
        return True
    if normalized_category in GENERIC_CATEGORY_BLACKLIST and normalized_term not in CONCRETE_CATEGORY_HINTS:
        return True
    return any(pattern.match(normalized_term) for pattern in GENERIC_TERM_PATTERNS)


def _has_concrete_anchor(term: str, category: str, detail: str, evidence_sentences: Sequence[str]) -> bool:
    normalized_term = _normalize_term_text(term)
    normalized_category = _normalize_term_text(category)
    evidence_text = " ".join(evidence_sentences)
    full_text = " ".join([normalized_term, normalized_category, detail, evidence_text])
    if normalized_category in CONCRETE_CATEGORY_HINTS:
        return True
    if any(pattern.search(normalized_term) for pattern in CONCRETE_TERM_PATTERNS):
        return True
    if re.search(r"(题跋|题款|款识|印章|印文|馆藏|现藏|收藏于|登录号|厘米|公分|cm|北宋|南宋|元代|明代|清代)", full_text):
        return True
    return False


def _filter_terms(terms: Sequence[SemanticTerm]) -> list[SemanticTerm]:
    filtered: list[SemanticTerm] = []
    seen: set[tuple[str, str]] = set()
    for item in terms:
        term = item.term.strip()
        category = item.category.strip()
        if _looks_like_generic_term(term, category):
            continue
        if not _has_concrete_anchor(term, category, item.detail, item.evidence_sentences):
            continue
        key = (_normalize_term_text(term), _normalize_term_text(category))
        if key in seen:
            continue
        seen.add(key)
        filtered.append(
            item.model_copy(
                update={
                    "term": term,
                    "category": category,
                    "detail": item.detail.strip(),
                    "evidence_sentences": [sentence.strip() for sentence in item.evidence_sentences if sentence.strip()],
                }
            )
        )
    return filtered


def _build_slot_parse_prompt(slots_text: str) -> str:
    return f"""请将下面的 slots 自然语言说明解析成 JSON。

slots_input:
{slots_text}

请输出：
{{
  "slots": [
    {{
      "slot_name": "技法",
      "description": "关注国画中的技法、笔法、皴法、设色等",
      "covered_terms": ["工笔", "没骨", "披麻皴"]
    }}
  ]
}}
"""


def _build_term_extraction_prompt(context_name: str, indexed_sentences: Sequence[dict]) -> str:
    context_text = "\n".join(f"[{item['sentence_id']}] {item['text']}" for item in indexed_sentences)
    return f"""请抽取 {context_name} 中明确出现的中国画专业术语。

context_name: {context_name}
context:
{context_text}

输出格式：
{{
  "terms": [
    {{
      "term": "绢本设色",
      "category": "材质",
      "detail": "文本明确指出作品材质与设色方式",
      "sentence_ids": [0],
      "evidence_sentences": ["原句"]
    }}
  ]
}}
"""


def _build_slot_match_prompt(
    context_name: str,
    terms: Sequence[SemanticTerm],
    slots_spec: Sequence[SlotDefinition],
) -> str:
    return f"""请判断下列术语命中了哪些 slots。

context_name: {context_name}
slots:
{json.dumps([item.model_dump() for item in slots_spec], ensure_ascii=False, indent=2)}

terms:
{json.dumps([item.model_dump() for item in terms], ensure_ascii=False, indent=2)}

输出格式：
{{
  "matches": [
    {{
      "slot_name": "技法",
      "matched_terms": ["披麻皴", "浅绛设色"],
      "matched_categories": ["技法名称", "设色"],
      "sentence_ids": [1, 2],
      "reason": "这些术语直接属于技法与设色维度"
    }}
  ]
}}
"""


def _build_fidelity_prompt(
    context_name: str,
    terms: Sequence[SemanticTerm],
    image_context_v: str,
) -> str:
    return f"""请根据参考视觉线索基准判断术语是否被支持。

context_name: {context_name}
image_context_v:
{image_context_v}

terms:
{json.dumps([item.model_dump() for item in terms], ensure_ascii=False, indent=2)}

输出格式：
{{
  "fidelity": [
    {{
      "term": "绢本设色",
      "category": "材质",
      "supported_by_ground_truth": true,
      "reason": "视觉线索中明确提到绢本与设色"
    }}
  ]
}}
"""


def _format_duplicate_clusters_for_prompt(clusters: Sequence[DuplicateCluster]) -> list[dict[str, Any]]:
    return [
        {
            "cluster_id": cluster.cluster_id,
            "sentence_ids": cluster.sentence_ids,
            "sentences": cluster.sentences,
            "avg_similarity": round(cluster.avg_similarity, 4),
        }
        for cluster in clusters
    ]


def _build_final_judgment_prompt(
    image_context_v: str,
    baseline: ContextMetrics,
    enhanced: ContextMetrics,
    slots_number: int,
) -> str:
    return f"""
参考视觉线索基准(Ground Truth): {image_context_v}

[文本A指标]
- 信息密度: {baseline.information_density:.3f}
- 语义重复率: {baseline.duplicate_rate:.2%}
- 专业Slot覆盖率: {baseline.slot_coverage:.2%}
- 术语准确度(Fidelity): {baseline.accuracy:.2%}

[文本B指标]
- 信息密度: {enhanced.information_density:.3f}
- 语义重复率: {enhanced.duplicate_rate:.2%}
- 专业Slot覆盖率: {enhanced.slot_coverage:.2%}
- 术语准确度(Fidelity): {enhanced.accuracy:.2%}

补充证据：
- slots_number: {slots_number}
- 文本A重复语义簇: {json.dumps(_format_duplicate_clusters_for_prompt(baseline.duplicate_clusters), ensure_ascii=False)}
- 文本B重复语义簇: {json.dumps(_format_duplicate_clusters_for_prompt(enhanced.duplicate_clusters), ensure_ascii=False)}
- 文本A术语与Fidelity: {json.dumps({"terms": [item.model_dump() for item in baseline.terms], "fidelity": [item.model_dump() for item in baseline.fidelity_records]}, ensure_ascii=False)}
- 文本B术语与Fidelity: {json.dumps({"terms": [item.model_dump() for item in enhanced.terms], "fidelity": [item.model_dump() for item in enhanced.fidelity_records]}, ensure_ascii=False)}

请执行以下任务：
1. 判定哪篇赏析更符合专业水准(Win/Loss/Tie)；
2. 为差的一方生成 Textual Loss：具体指出哪些语义簇导致了冗余，哪些视觉线索被忽略。

请输出 JSON：
{{
  "winner": "context_baseline",
  "textual_loss_for": "context_enhanced",
  "reasoning": "高层依据",
  "textual_loss": "详细 loss"
}}
"""


def _extract_content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
            elif hasattr(item, "text"):
                parts.append(str(item.text))
        return "".join(parts)
    return str(content or "")


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    return max(min(dot / (left_norm * right_norm), 1.0), -1.0)


def _average_pair_similarity(indices: Sequence[int], embeddings: Sequence[Sequence[float]]) -> float:
    if len(indices) <= 1:
        return 1.0
    scores: list[float] = []
    for offset, left in enumerate(indices):
        for right in indices[offset + 1 :]:
            scores.append(_cosine_similarity(embeddings[left], embeddings[right]))
    return sum(scores) / len(scores) if scores else 1.0


def _cluster_sentences(
    indexed_sentences: Sequence[dict],
    embeddings: Sequence[Sequence[float]],
    duplicate_threshold: float,
) -> tuple[list[DuplicateCluster], int, int, int]:
    union_find = _UnionFind(len(indexed_sentences))
    for left in range(len(indexed_sentences)):
        for right in range(left + 1, len(indexed_sentences)):
            if _cosine_similarity(embeddings[left], embeddings[right]) >= duplicate_threshold:
                union_find.union(left, right)

    grouped: dict[int, list[int]] = {}
    for item in range(len(indexed_sentences)):
        grouped.setdefault(union_find.find(item), []).append(item)

    unique_semantic_num = len(grouped)
    duplicate_clusters: list[DuplicateCluster] = []
    for cluster_id, members in enumerate(sorted(grouped.values(), key=lambda value: value[0])):
        if len(members) <= 1:
            continue
        duplicate_clusters.append(
            DuplicateCluster(
                cluster_id=cluster_id,
                sentence_ids=[indexed_sentences[index]["sentence_id"] for index in members],
                sentences=[indexed_sentences[index]["text"] for index in members],
                avg_similarity=round(_average_pair_similarity(members, embeddings), 6),
            )
        )
    duplicate_sentence_num = sum(len(cluster.sentence_ids) for cluster in duplicate_clusters)
    return duplicate_clusters, len(duplicate_clusters), unique_semantic_num, duplicate_sentence_num


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(row, ensure_ascii=False) for row in rows]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return str(path)


class GuohuaEvalV2Analyzer:
    def __init__(
        self,
        embedding_model: str = "baai/bge-m3",
        judge_model: str = "minimax/minimax-m2.5",
        base_url: str = "https://api.zjuqx.cn/v1",
        duplicate_threshold: float = 0.83,
        client: OpenAI | None = None,
    ) -> None:
        self.embedding_model = embedding_model
        self.judge_model = judge_model
        self.base_url = base_url
        self.duplicate_threshold = duplicate_threshold
        self.client = client
        self._usage_counter = _UsageCounter()

    def evaluate(
        self,
        context_baseline: str,
        context_enhanced: str,
        slots_input: str,
        output_dir: str | os.PathLike[str] = "artifacts",
        image_context_v: str = "",
    ) -> EvalV2Result:
        baseline_indexed = build_indexed_text(context_baseline)
        enhanced_indexed = build_indexed_text(context_enhanced)
        if not baseline_indexed or not enhanced_indexed:
            raise ValueError("context_baseline 和 context_enhanced 都必须至少包含一句有效文本")

        slots_spec = self._parse_slots_spec(slots_input)
        if not slots_spec:
            raise ValueError("slots_input 解析后为空，无法进行 slot 评测")

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        with ThreadPoolExecutor(max_workers=2) as executor:
            baseline_future = executor.submit(
                self._analyze_context,
                "context_baseline",
                baseline_indexed,
                slots_spec,
                image_context_v,
                output_path,
            )
            enhanced_future = executor.submit(
                self._analyze_context,
                "context_enhanced",
                enhanced_indexed,
                slots_spec,
                image_context_v,
                output_path,
            )
            baseline_metrics = baseline_future.result()
            enhanced_metrics = enhanced_future.result()

        final_judgment = self._judge_final(image_context_v, baseline_metrics, enhanced_metrics, len(slots_spec))
        result_json_path = str(output_path / "eval_v2_result.json")
        result = EvalV2Result(
            base_url=self.base_url,
            embedding_model=self.embedding_model,
            judge_model=self.judge_model,
            duplicate_threshold=self.duplicate_threshold,
            slots_input=slots_input,
            slots_number=len(slots_spec),
            slots_spec=slots_spec,
            image_context_v=image_context_v,
            context_baseline_metrics=baseline_metrics,
            context_enhanced_metrics=enhanced_metrics,
            final_judgment=final_judgment,
            output_dir=str(output_path),
            result_json_path=result_json_path,
            llm_tokens=self._usage_counter.total,
        )
        Path(result_json_path).write_text(result.model_dump_json(indent=2), encoding="utf-8")
        return result

    def _client(self) -> OpenAI:
        if self.client is not None:
            return self.client
        if OpenAI is None:
            raise ImportError("未安装 openai 包；请先安装依赖，或在测试中传入 client。")
        return OpenAI(base_url=self.base_url)

    def _parse_slots_spec(self, slots_input: str) -> list[SlotDefinition]:
        response = self._client().chat.completions.create(
            model=self.judge_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SLOT_PARSE_SYSTEM_PROMPT},
                {"role": "user", "content": _build_slot_parse_prompt(slots_input)},
            ],
        )
        self._usage_counter.add_from_response(response)
        content = _extract_content_text(response.choices[0].message.content)
        parsed = SlotsSpec.model_validate(json.loads(content or "{}"))
        return [slot for slot in parsed.slots if slot.slot_name.strip()]

    def _embed_sentences(self, sentences: Sequence[str]) -> list[list[float]]:
        try:
            response = self._client().embeddings.create(
                model=self.embedding_model,
                input=list(sentences),
                encoding_format="float",
            )
        except APIStatusError as exc:
            status_code = getattr(exc, "status_code", None)
            response_obj = getattr(exc, "response", None)
            payload = getattr(response_obj, "json", None)
            detail = ""
            if callable(payload):
                try:
                    detail = json.dumps(payload(), ensure_ascii=False)
                except Exception:
                    detail = ""
            if status_code == 503:
                raise RuntimeError(
                    f"embedding 模型 `{self.embedding_model}` 当前在 `{self.base_url}` 不可用。"
                    "这通常不是输入文本问题，而是服务端没有可用渠道或模型未开通。"
                    f"{' 服务端返回: ' + detail if detail else ''}"
                ) from exc
            raise
        self._usage_counter.add_from_response(response)
        return [list(item.embedding) for item in response.data]

    def _extract_terms(self, context_name: str, indexed_sentences: Sequence[dict]) -> list[SemanticTerm]:
        response = self._client().chat.completions.create(
            model=self.judge_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": TERM_EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": _build_term_extraction_prompt(context_name, indexed_sentences)},
            ],
        )
        self._usage_counter.add_from_response(response)
        content = _extract_content_text(response.choices[0].message.content)
        parsed = TermExtractionOutput.model_validate(json.loads(content or "{}"))
        return _filter_terms(parsed.terms)

    def _match_slots(
        self,
        context_name: str,
        terms: Sequence[SemanticTerm],
        slots_spec: Sequence[SlotDefinition],
    ) -> list[SlotMatchRecord]:
        if not terms:
            return []
        response = self._client().chat.completions.create(
            model=self.judge_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SLOT_MATCH_SYSTEM_PROMPT},
                {"role": "user", "content": _build_slot_match_prompt(context_name, terms, slots_spec)},
            ],
        )
        self._usage_counter.add_from_response(response)
        content = _extract_content_text(response.choices[0].message.content)
        parsed = SlotMatchOutput.model_validate(json.loads(content or "{}"))
        allowed_slots = {item.slot_name for item in slots_spec}
        return [item for item in parsed.matches if item.slot_name in allowed_slots]

    def _evaluate_fidelity(
        self,
        context_name: str,
        terms: Sequence[SemanticTerm],
        image_context_v: str,
    ) -> list[FidelityRecord]:
        if not terms:
            return []
        if not image_context_v.strip():
            return [
                FidelityRecord(
                    term=item.term,
                    category=item.category,
                    supported_by_ground_truth=False,
                    reason="未提供 image_context_v，无法判定视觉线索支持关系。",
                )
                for item in terms
            ]
        response = self._client().chat.completions.create(
            model=self.judge_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FIDELITY_SYSTEM_PROMPT},
                {"role": "user", "content": _build_fidelity_prompt(context_name, terms, image_context_v)},
            ],
        )
        self._usage_counter.add_from_response(response)
        content = _extract_content_text(response.choices[0].message.content)
        parsed = FidelityOutput.model_validate(json.loads(content or "{}"))
        fidelity_map = {(item.term, item.category): item for item in parsed.fidelity}
        return [
            fidelity_map.get(
                (term.term, term.category),
                FidelityRecord(
                    term=term.term,
                    category=term.category,
                    supported_by_ground_truth=False,
                    reason="模型未返回该术语的 fidelity 结果。",
                ),
            )
            for term in terms
        ]

    def _analyze_context(
        self,
        context_name: str,
        indexed_sentences: Sequence[dict],
        slots_spec: Sequence[SlotDefinition],
        image_context_v: str,
        output_dir: Path,
    ) -> ContextMetrics:
        with ThreadPoolExecutor(max_workers=2) as executor:
            embedding_future = executor.submit(
                self._embed_sentences,
                [item["text"] for item in indexed_sentences],
            )
            terms_future = executor.submit(self._extract_terms, context_name, indexed_sentences)
            embeddings = embedding_future.result()
            terms = terms_future.result()

        duplicate_clusters, similar_semantic_num, unique_semantic_num, duplicate_sentence_num = _cluster_sentences(
            indexed_sentences=indexed_sentences,
            embeddings=embeddings,
            duplicate_threshold=self.duplicate_threshold,
        )

        with ThreadPoolExecutor(max_workers=2) as executor:
            slot_matches_future = executor.submit(self._match_slots, context_name, terms, slots_spec)
            fidelity_future = executor.submit(self._evaluate_fidelity, context_name, terms, image_context_v)
            slot_matches = slot_matches_future.result()
            fidelity_records = fidelity_future.result()

        token_count = estimate_token_count(" ".join(item["text"] for item in indexed_sentences))
        supported_count = sum(1 for item in fidelity_records if item.supported_by_ground_truth)
        accuracy = supported_count / len(fidelity_records) if fidelity_records else 0.0
        slots_match = len({item.slot_name for item in slot_matches})
        information_density = unique_semantic_num / token_count if token_count else 0.0
        duplicate_rate = duplicate_sentence_num / len(indexed_sentences) if indexed_sentences else 0.0
        slot_coverage = slots_match / len(slots_spec) if slots_spec else 0.0

        prefix = "baseline" if context_name == "context_baseline" else "enhanced"
        duplicate_clusters_jsonl = _write_jsonl(
            output_dir / f"{prefix}_duplicate_clusters.jsonl",
            [item.model_dump() for item in duplicate_clusters],
        )
        terms_jsonl = _write_jsonl(
            output_dir / f"{prefix}_terms.jsonl",
            [item.model_dump() for item in terms],
        )
        slot_matches_jsonl = _write_jsonl(
            output_dir / f"{prefix}_slot_matches.jsonl",
            [item.model_dump() for item in slot_matches],
        )
        fidelity_jsonl = _write_jsonl(
            output_dir / f"{prefix}_fidelity.jsonl",
            [item.model_dump() for item in fidelity_records],
        )

        return ContextMetrics(
            context_name=context_name,
            sentence_count=len(indexed_sentences),
            token_count=token_count,
            similar_semantic_num=similar_semantic_num,
            duplicate_sentence_num=duplicate_sentence_num,
            unique_semantic_num=unique_semantic_num,
            term_num=len(terms),
            slots_match=slots_match,
            accuracy=round(accuracy, 6),
            duplicate_rate=round(duplicate_rate, 6),
            information_density=round(information_density, 6),
            slot_coverage=round(slot_coverage, 6),
            duplicate_clusters=duplicate_clusters,
            terms=terms,
            slot_matches=slot_matches,
            fidelity_records=fidelity_records,
            duplicate_clusters_jsonl=duplicate_clusters_jsonl,
            terms_jsonl=terms_jsonl,
            slot_matches_jsonl=slot_matches_jsonl,
            fidelity_jsonl=fidelity_jsonl,
        )

    def _judge_final(
        self,
        image_context_v: str,
        baseline_metrics: ContextMetrics,
        enhanced_metrics: ContextMetrics,
        slots_number: int,
    ) -> FinalJudgment:
        response = self._client().chat.completions.create(
            model=self.judge_model,
            temperature=0.1,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": FINAL_JUDGMENT_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_final_judgment_prompt(
                        image_context_v=image_context_v,
                        baseline=baseline_metrics,
                        enhanced=enhanced_metrics,
                        slots_number=slots_number,
                    ),
                },
            ],
        )
        self._usage_counter.add_from_response(response)
        content = _extract_content_text(response.choices[0].message.content)
        parsed = FinalJudgmentOutput.model_validate(json.loads(content or "{}"))
        return FinalJudgment(**parsed.model_dump())
