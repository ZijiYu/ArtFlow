from __future__ import annotations

import asyncio
import base64
import io
import json
import math
import mimetypes
import re
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Callable, Iterable

from PIL import Image

from .clients import (
    HttpRagClient,
    LLMClient,
    LlmChatLogger,
    OpenAIEmbeddingSimilarityBackend,
    OpenAIJsonClient,
    RagClient,
    RecordingLLMClient,
    SimilarityBackend,
)
from .config import PipelineConfig
from .models import (
    GroundedTerm,
    GroundingQueryRecord,
    OntologyLink,
    PipelineResult,
    RagDocument,
    SlotMetadata,
    SlotRecord,
    TermCandidate,
)

PAINTING_PROFILE_PROMPT = """你是中国画作品类型分析器。
请先基于图像、基础文字和给定的作品主干信息，判断这幅画大致是什么类型、主题和相关的国画知识背景，输出 JSON：
{
  "name": "作品名，没有则为空",
  "author": "作者，没有则为空",
  "dynasty": "朝代，没有则为空",
  "painting_type": "画作类型或体裁",
  "subject": "如果知道名字输出名字，不知道则输出：主题或主要对象",
  "scene_summary": "对画面内容的简要概括",
  "related_background": ["与这幅画、作者或时代直接相关的信息"],
  "guohua_knowledge": ["与该画相关的国画知识点1", "知识点2"],
  "reasoning": "为什么这样判断"
}
要求：
1. 优先使用给定主干信息中的作品名、作者、朝代作为分析主体。
2. 允许基于国画常识做审慎推断，但要保持克制。
3. 输出应服务于后续锚点抽取和 RAG 对齐。
4. `related_background` 优先补充与作品身份和历史脉络直接相关的内容，可从这些维度中择要覆盖：作品名称、尺寸规格、收藏地点、材质形制、画家信息、创作年代、艺术传承、历史语境、艺术地位。
5. `guohua_knowledge` 优先补充与图像分析和鉴赏直接相关的内容，可从这些维度中择要覆盖：构图布局、用笔特点、色彩氛围、题材内容、形神表现、艺术风格、意境营造、象征寓意、题跋印章。
6. 如果某些维度缺少可靠证据，可以跳过，不要为了覆盖维度而编造。
7. `related_background` 和 `guohua_knowledge` 各最多 5 条。"""

VISUAL_CUE_PROMPT = """你是中国画视觉锚点抽取器。
请只根据图像识别稳定、可核验的视觉线索，输出 JSON：
{
  "visual_cues": ["皴法线索", "构图线索", "题跋位置", "材质或设色线索"]
}
要求：
1. 只输出图中可见的信息。
2. 每条线索必须短、稳、可用于后续 RAG 对齐。
3. 最多输出 8 条。"""

TEXT_SIGNAL_PROMPT = """你是中国画文本信号分析器。
请把输入文字拆成可支持后续术语抽取的文本证据，输出 JSON：
{
  "text_signals": ["信号1", "信号2"],
  "salient_entities": ["实体1", "实体2"]
}
要求：
1. 不要引入外部知识。
2. 只保留与国画领域判断有关的内容。"""

ANCHORING_PROMPT = """你是一个高度泛化的中国画领域锚定器。
结合图像和文字，输出 JSON：
{
  "candidates": [
    {
      "term": "术语",
      "description": "一句话说明该术语为何重要",
      "category_guess": "推测的大类",
      "visual_evidence": ["图像证据"],
      "text_evidence": ["文本证据"]
    }
  ]
}
要求：
1. 只输出末端、具体、可被检索和核验的术语。
2. 避免空泛大类，例如“笔墨”“构图”“意境”。
3. 优先保留能够触发 RAG 查询的稳定名词。
4. 输出前参考下面 few-shot 风格：
示例输入：图像显示山石用短促斧劈皴，右上有题跋；文本提到“绢本设色”。
示例输出：{"candidates":[{"term":"斧劈皴","description":"山石纹理以短促折线组织，疑似斧劈皴。","category_guess":"皴法","visual_evidence":["山石边缘呈折线皴擦"],"text_evidence":[]},{"term":"绢本","description":"文本明确指出作品材质为绢本。","category_guess":"材质","visual_evidence":[],"text_evidence":["绢本设色"]}]}"""

TEXT_ANCHORING_PROMPT = """你是中国画文本信号与领域锚定综合分析器。
请先拆解输入文字中的国画相关信号，再结合画作类型分析和视觉线索，直接输出可用于后续 grounding 的候选术语，输出 JSON：
{
  "text_signals": ["信号1", "信号2"],
  "salient_entities": ["实体1", "实体2"],
  "candidates": [
    {
      "term": "术语",
      "description": "一句话说明该术语为何重要",
      "category_guess": "推测的大类",
      "visual_evidence": ["图像证据"],
      "text_evidence": ["文本证据"]
    }
  ]
}
要求：
1. `text_signals` 和 `salient_entities` 不要引入外部知识，只保留输入文字里与国画领域判断有关的内容。
2. `candidates` 可以结合输入文字、画作类型分析和视觉线索做审慎归纳，但只输出末端、具体、可被检索和核验的术语。
3. 避免空泛大类，例如“笔墨”“构图”“意境”。
4. 优先保留能够触发 grounding 查询的稳定名词，并尽量保留与该术语直接对应的原始文本证据。
5. 输出前参考下面 few-shot 风格：
示例输入：图像显示山石用短促斧劈皴，右上有题跋；文本提到“绢本设色”。
示例输出：{"text_signals":["文本提到绢本设色"],"salient_entities":["绢本"],"candidates":[{"term":"斧劈皴","description":"山石纹理以短促折线组织，疑似斧劈皴。","category_guess":"皴法","visual_evidence":["山石边缘呈折线皴擦"],"text_evidence":[]},{"term":"绢本","description":"文本明确指出作品材质为绢本。","category_guess":"材质","visual_evidence":[],"text_evidence":["绢本设色"]}]}"""

FIXED_SLOT_SELECTION_PROMPT = """你是中国画固定分析槽位规划器。
给定固定槽位定义、候选术语和当前画作证据，请只为该槽位选择 0 或 1 个最合适的 term，并输出 JSON：
{
  "applicable": true,
  "slot_name": "固定槽位名",
  "slot_term": "本轮唯一 term；如果没有则为空",
  "description": "围绕该 term、且只针对当前作品的说明",
  "specific_questions": ["问题1", "问题2", "问题3"],
  "metadata": {
    "confidence": 0.0,
    "source_id": "来源索引"
  },
  "pending_terms": ["后续仍可继续推进的候选 term"],
  "reasoning": "为什么选这个 term，或者为什么该槽位当前应关闭"
}
规则：
1. 只能从输入候选 term 中选择，不得凭空创造新 term。
2. 每个槽位一次只能有一个 term。
3. 如果该槽位在当前作品中没有足够证据支持，请返回 `applicable=false`，并把 `slot_term` 置空。
4. progressive 槽位优先选择更上层、适合作为递进起点的 term；已出现过或明显重复的 term 不能再选。
5. enumerative 槽位优先选择当前最关键且尚未使用的信息字段，不要求严格层级关系。
6. 问题必须紧扣当前作品，不要泛化到其他作品或例子。
7. 如果候选里有技法、设色、墨法等术语，但图像或证据并未真正支持它们在本作中被使用，应明确判定为不可用并跳过。
8. 输入里还会给出 `candidate_term_groups`，表示已经按同类术语分组（如皴法、描法、墨法、设色等）；请先在组内比较，再跨组选择最合适的主 term。
9. 如果选择了某一组中的主 term，请尽量把同组里仍值得后续核验的 term 放入 `pending_terms`，避免同类术语在第一轮被直接丢弃。
10. 如果整个槽位都没有可用 term，应返回关闭建议。"""

ONTOLOGY_PROMPT = """你是中国画动态本体推理器。
给定已生成的 Slots，请判断是否存在稳定的父子关系，输出 JSON：
{
  "relations": [
    {
      "child": "马牙皴",
      "parent": "皴法",
      "relation": "is-a",
      "rationale": "马牙皴属于皴法下的具体表现形式"
    }
  ]
}
要求：
1. 只输出高置信度关系。
2. 不要输出重复或循环链接。"""

MIAOFA_TECHNIQUE_NAMES = [
    "减笔描",
    "战笔水纹描",
    "折芦描",
    "撅头钉描",
    "曹衣描",
    "枣核描",
    "柳叶描",
    "柴笔描",
    "橄榄描",
    "混描",
    "琴弦描",
    "竹叶描",
    "蚂蝗描",
    "蚯蚓描",
    "行云流水描",
    "钉头鼠尾描",
    "铁线描",
    "高古游丝描",
]

CUNFA_TECHNIQUE_NAMES = [
    "披麻皴",
    "斧劈皴",
    "小斧劈皴",
    "雨点皴",
    "卷云皴",
    "解索皴",
    "牛毛皴",
    "折带皴",
    "米点皴",
    "荷叶皴",
    "乱柴皴",
    "马牙皴",
    "弹涡皴",
    "鬼面皴",
    "拖泥带水皴",
]

TECHNIQUE_JUDGMENT_PROMPT = (
    """你是中国画图像中的皴法与描法判别器。
请优先只基于图像本身判断当前画面是否存在山体，以及是否能从固定名单中识别出皴法或描法，输出 JSON：
{
  "mountain_present": true,
  "mountain_evidence": ["山体证据1"],
  "cunfa_candidates": [
    {
      "term": "xx皴",
      "reason": "为什么像这种皴法",
      "visual_evidence": ["图像证据"],
      "confidence": 0.0
    }
  ],
  "miaofa_candidates": [
    {
      "term": "xx描",
      "reason": "为什么像这种描法",
      "visual_evidence": ["图像证据"],
      "confidence": 0.0
    }
  ],
  "reasoning": "总体判断说明"
}
规则：
1. 只能从给定名单中选择 term，不得自造名称。
2. 皴法判断必须以“画面中确有山体/山石结构”为前提；如果 `mountain_present=false`，则 `cunfa_candidates` 必须为空。
3. 优先输出看得见、能描述清楚笔触形态的结果；不确定就少报，不要猜。
4. 皴法和描法各最多输出 3 个候选，按置信度从高到低排序。
5. 文字输入与视觉锚点只能辅助你聚焦，但不能替代图像判断。
6. 如果画面只有人物、衣纹、水纹等，没有山体，就不要输出任何皴法。
固定描法名单：
"""
    + json.dumps(MIAOFA_TECHNIQUE_NAMES, ensure_ascii=False)
    + """
固定皴法名单：
"""
    + json.dumps(CUNFA_TECHNIQUE_NAMES, ensure_ascii=False)
)

VISUAL_ANALYSIS_PROMPT = (
    """你是中国画视觉与技法综合分析器。
请优先只基于图像本身，同时输出稳定、可核验的视觉线索，并判断当前画面是否存在山体，以及是否能从固定名单中识别出皴法或描法，输出 JSON：
{
  "visual_cues": ["皴法线索", "构图线索", "题跋位置", "材质或设色线索"],
  "mountain_present": true,
  "mountain_evidence": ["山体证据1"],
  "cunfa_candidates": [
    {
      "term": "xx皴",
      "reason": "为什么像这种皴法",
      "visual_evidence": ["图像证据"],
      "confidence": 0.0
    }
  ],
  "miaofa_candidates": [
    {
      "term": "xx描",
      "reason": "为什么像这种描法",
      "visual_evidence": ["图像证据"],
      "confidence": 0.0
    }
  ],
  "reasoning": "总体判断说明"
}
规则：
1. `visual_cues` 只输出图中可见的信息，每条必须短、稳、可用于后续 grounding 对齐，最多输出 8 条。
2. 皴法和描法只能从给定名单中选择 term，不得自造名称。
3. 皴法判断必须以“画面中确有山体/山石结构”为前提；如果 `mountain_present=false`，则 `cunfa_candidates` 必须为空。
4. 优先输出看得见、能描述清楚笔触形态的结果；不确定就少报，不要猜。
5. 皴法和描法各最多输出 3 个候选，按置信度从高到低排序。
6. 输入文字只能辅助你聚焦，但不能替代图像判断。
7. 如果画面只有人物、衣纹、水纹等，没有山体，就不要输出任何皴法。
固定描法名单：
"""
    + json.dumps(MIAOFA_TECHNIQUE_NAMES, ensure_ascii=False)
    + """
固定皴法名单：
"""
    + json.dumps(CUNFA_TECHNIQUE_NAMES, ensure_ascii=False)
)

_FIXED_SLOT_SPECS = [
    {
        "slot_name": "画作/背景",
        "slot_mode": "progressive",
        "goal": "围绕作品名、卷次、题材主干与更具体的身份信息做层层递进。",
    },
    {
        "slot_name": "作者/时代/流派",
        "slot_mode": "enumerative",
        "goal": "围绕作者、时代、地域、流派及其分支做逐步遍历。",
    },
    {
        "slot_name": "技法/设色/墨法",
        "slot_mode": "progressive",
        "goal": "围绕设色、皴法、描法、墨法、线质等技法信息做递进挖掘。",
    },
    {
        "slot_name": "题跋/诗文/审美语言",
        "slot_mode": "enumerative",
        "goal": "围绕题跋、诗文、题识、落款等可见文字内容做识读与分析；若无证据则关闭。",
    },
    {
        "slot_name": "题跋/印章/用笔",
        "slot_mode": "enumerative",
        "goal": "围绕题跋、印章、落款与用笔线质等可见或可证实线索做并列梳理。",
    },
    {
        "slot_name": "构图/空间/布局",
        "slot_mode": "progressive",
        "goal": "围绕构图组织、空间层次、视觉动线与布局经营做递进分析。",
    },
    {
        "slot_name": "尺寸规格/材质形制/收藏地",
        "slot_mode": "enumerative",
        "goal": "围绕尺寸规格、材质、装裱形制与收藏信息等作品身份线索做并列核对。",
    },
    {
        "slot_name": "意境/题材/象征",
        "slot_mode": "progressive",
        "goal": "围绕题材对象、意境营造与象征寓意做层层递进分析。",
    },
]

_FIXED_SLOT_NAME_ALIASES = {
    "画作/背景": "画作背景",
    "作者/时代/流派": "作者时代流派",
    "墨法/设色/技法": "墨法设色技法",
    "技法/设色/墨法": "墨法设色技法",
    "题跋/诗文/审美语言": "题跋诗文审美语言",
}

_INSCRIPTION_BRUSHWORK_KEYWORDS = (
    "题跋",
    "诗跋",
    "诗文",
    "跋",
    "题识",
    "款识",
    "落款",
    "印章",
    "钤印",
    "用笔",
    "笔法",
    "笔触",
    "线条",
    "中锋",
    "侧锋",
)

_TEXT_INSCRIPTION_KEYWORDS = (
    "题跋",
    "诗文",
    "诗跋",
    "诗",
    "跋",
    "题识",
    "款识",
    "落款",
    "题款",
    "赞",
    "铭",
    "书写文字",
    "文字",
)

_COMPOSITION_LAYOUT_KEYWORDS = (
    "构图",
    "布局",
    "空间",
    "章法",
    "经营",
    "留白",
    "疏密",
    "层次",
    "远近",
    "前景",
    "中景",
    "后景",
    "高远",
    "深远",
    "平远",
    "重心",
    "动线",
    "取景",
    "穿插",
    "呼应",
)

_MATERIAL_FORMAT_COLLECTION_KEYWORDS = (
    "尺寸",
    "规格",
    "绢本",
    "纸本",
    "设色",
    "水墨",
    "手卷",
    "册页",
    "立轴",
    "横披",
    "镜心",
    "扇面",
    "屏风",
    "形制",
    "装裱",
    "馆藏",
    "收藏",
    "博物馆",
    "故宫",
)

_MOOD_SUBJECT_SYMBOLISM_KEYWORDS = (
    "意境",
    "题材",
    "象征",
    "寓意",
    "清逸",
    "静谧",
    "肃穆",
    "庄重",
    "古朴",
    "清旷",
    "空灵",
    "高逸",
    "萧疏",
    "富贵",
    "吉祥",
    "隐逸",
    "山水",
    "花鸟",
    "人物",
    "罗汉",
    "洛神",
)

_TECHNIQUE_GROUP_LABELS = (
    "皴法",
    "描法",
    "墨法",
    "设色",
    "用笔线质",
    "材质",
)

METADATA_SPINE_PROMPT = """你是中国画作品主干信息提取器。
请根据一条最相关的 metadata 文档，提取这幅画后续分析必须围绕的主干信息，输出 JSON：
{
  "name": "作品名，没有则为空",
  "author": "作者，没有则为空",
  "dynasty": "朝代，没有则为空",
  "related_background": ["与作品、作者或时代直接相关的背景信息"],
  "reasoning": "为什么这条 metadata 可以作为主干"
}
要求：
1. 主干信息必须尽量直接取自 metadata 文本，不要编造。
2. `related_background` 只保留与这幅画、作者、朝代直接相关的内容，最多 5 条。
3. 如果字段缺失则输出空字符串或空数组。"""

METADATA_SPINE_MATCH_PROMPT = """你是中国画作品主干核对器。
请结合图像、基础文字和当前候选主干信息，判断这个名字是否真的是目标图片本身，输出 JSON：
{
  "is_target_image": true,
  "suggested_name": "如果当前名字不对，给出更可能的作品名；不确定则为空",
  "suggested_author": "如果能判断则给出作者，否则为空",
  "suggested_dynasty": "如果能判断则给出朝代，否则为空",
  "reasoning": "为什么这样判断"
}
要求：
1. `is_target_image=true` 只在你认为候选主干与图像基本一致时使用。
2. 如果候选主干与图像不一致，请尽量给出更可能的作品名；作者和朝代可空。
3. 不要因为候选文本里出现某个名字，就机械沿用它；应以图像内容是否匹配为准。
4. 如果无法可靠判断，可以输出 `is_target_image=false`，但尽量给出一个可用于后续检索的候选名称。"""

RAG_SPINE_RELEVANCE_PROMPT = """你是中国画 RAG 关联性裁判。
请判断候选文档是否与给定的作品主干信息直接相关，输出 JSON：
{
  "related_indices": [0, 2],
  "reasoning": "简要说明保留依据"
}
要求：
1. 只保留与同一作品、同一作者、同一时代背景或能直接支撑该作品分析的文档。
2. 如果文档只是泛泛讲其他作品、其他题材、其他作者，即使主题相似也不要保留。
3. 如果提到的内容能直接解释这幅画的名字、作者、朝代、画中对象、画法或作者相关背景，可以保留。
4. `related_indices` 只填写需要保留的候选文档下标。"""


class ContextLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("# Dynamic Ontology Context\n\n", encoding="utf-8")

    def append(self, title: str, items: Iterable[str]) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n## {title} [{timestamp}]\n"]
        for item in items:
            lines.append(f"- {item}\n")
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)


class RagSearchLogger:
    def __init__(self, path: Path, *, enabled: bool = True) -> None:
        self._path = path
        self._enabled = enabled
        if self._enabled:
            self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._enabled and not self._path.exists():
            self._path.write_text("# RAG Search Record\n\n", encoding="utf-8")

    def append(
        self,
        *,
        image_path: Path,
        grounded_terms: list[GroundedTerm],
    ) -> None:
        if not self._enabled:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n## Search Batch [{timestamp}]\n", f"- image: `{image_path}`\n"]
        if not grounded_terms:
            lines.append("- no grounded terms\n")
        for item in grounded_terms:
            lines.append(f"- candidate_term: `{item.candidate.term}`\n")
            queries = item.search_queries or [item.candidate.term]
            lines.append(f"  - search_queries: `{ '`, `'.join(queries) }`\n")
            lines.append(f"  - image_attached: `true`\n")
            lines.append(f"  - sources: `{','.join(doc.source_id for doc in item.documents) or 'none'}`\n")
            lines.append(
                f"  - alignment_scores: `{','.join(str(score) for score in item.alignment_scores) or 'none'}`\n"
            )
            if item.query_records:
                for record in item.query_records:
                    lines.append(
                        "  - "
                        f"query=`{record.query_text}` "
                        f"duration_ms=`{record.duration_ms}` "
                        f"initial_top_k_count=`{record.initial_top_k_count}` "
                        f"matched_count=`{record.matched_count}`\n"
                    )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)


class _UnionFind:
    def __init__(self, size: int) -> None:
        self._parent = list(range(size))

    def find(self, index: int) -> int:
        if self._parent[index] != index:
            self._parent[index] = self.find(self._parent[index])
        return self._parent[index]

    def union(self, left: int, right: int) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self._parent[right_root] = left_root

    def groups(self) -> dict[int, list[int]]:
        result: dict[int, list[int]] = {}
        for index in range(len(self._parent)):
            root = self.find(index)
            result.setdefault(root, []).append(index)
        return result


class PerceptionPipeline:
    def __init__(
        self,
        config: PipelineConfig | None = None,
        *,
        llm_client: LLMClient | None = None,
        rag_client: RagClient | None = None,
        rag_client_factory: Callable[[], RagClient] | None = None,
        similarity_backend: SimilarityBackend | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or PipelineConfig.from_env()
        base_llm = llm_client or OpenAIJsonClient(self._config, client=client)
        self._llm_chat_logger = LlmChatLogger(
            self._config.llm_chat_record_path,
            enabled=bool(self._config.write_llm_chat_record),
        )
        self._llm = RecordingLLMClient(base_llm, self._llm_chat_logger)
        if rag_client_factory is not None:
            self._rag_factory = rag_client_factory
        elif rag_client is not None:
            self._rag_factory = lambda: rag_client.clone() if hasattr(rag_client, "clone") else rag_client
        else:
            self._rag_factory = lambda: HttpRagClient(self._config.rag_endpoint)
        self._similarity = similarity_backend or OpenAIEmbeddingSimilarityBackend(self._config, client=client)
        self._context = ContextLogger(self._config.context_path)
        self._rag_search_record = RagSearchLogger(
            self._config.rag_search_record_path,
            enabled=bool(self._config.write_rag_search_record),
        )
        self._spine_relevance_cache: dict[tuple[str, str], bool] = {}
        self._grounding_rag_cache: dict[tuple[str, str, str, int], list[RagDocument]] = {}

    def _new_rag_client(self) -> RagClient:
        return self._rag_factory()

    async def run(
        self,
        *,
        image_file: str | Path,
        input_text: str,
        output_path: str | Path | None = None,
        reset_llm_chat_record: bool = True,
    ) -> PipelineResult:
        if reset_llm_chat_record:
            self._llm.reset_log()
        self._spine_relevance_cache.clear()
        self._grounding_rag_cache.clear()
        stage_timings: dict[str, float] = {}
        run_started_at = perf_counter()
        image_path = Path(image_file)
        stage_started_at = perf_counter()
        image_base64, image_bytes, image_mime_type, image_size = _prepare_image_payload(
            image_path,
            max_pixels=self._config.max_image_pixels,
        )
        stage_timings["prepare_image"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("prepare_image", stage_timings["prepare_image"])

        self._append_debug_context(
            "Run Started",
            [
                f"image=`{image_path}`",
                f"image_size={image_size[0]}x{image_size[1]}",
                f"text={input_text[:160]}",
            ],
        )

        stage_started_at = perf_counter()
        try:
            metadata_spine = await asyncio.to_thread(
                self._build_metadata_spine,
                image_bytes=image_bytes,
                image_filename=image_path.name,
                image_mime_type=image_mime_type,
                input_text=input_text,
            )
        except Exception as exc:  # noqa: BLE001
            metadata_spine = {}
            self._append_debug_context(
                "Metadata Spine",
                [f"metadata spine unavailable: {type(exc).__name__}: {exc}"],
            )
            print(
                f"[perception_stage] stage=metadata_spine_fallback reason={type(exc).__name__}",
                flush=True,
            )
        stage_timings["metadata_spine"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("metadata_spine", stage_timings["metadata_spine"])
        if metadata_spine:
            self._append_debug_context(
                "Metadata Spine",
                [json.dumps(metadata_spine, ensure_ascii=False)],
            )

        stage_started_at = perf_counter()
        painting_profile_task = asyncio.create_task(
            self._run_timed_thread_stage(
                "painting_profile",
                self._analyze_painting_profile,
                image_base64=image_base64,
                image_mime_type=image_mime_type,
                input_text=input_text,
                metadata_spine=metadata_spine,
            )
        )
        visual_analysis_task = asyncio.create_task(
            self._run_timed_thread_stage(
                "visual_analysis",
                self._analyze_visual_bundle,
                image_base64=image_base64,
                image_mime_type=image_mime_type,
                input_text=input_text,
            )
        )
        painting_profile_stage, visual_analysis_stage = await asyncio.gather(
            painting_profile_task,
            visual_analysis_task,
        )
        _, painting_profile, painting_profile_elapsed = painting_profile_stage
        _, visual_bundle, visual_analysis_elapsed = visual_analysis_stage
        stage_timings["painting_profile"] = painting_profile_elapsed
        stage_timings["visual_analysis"] = visual_analysis_elapsed
        stage_timings["profile_visual_parallel"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("painting_profile", stage_timings["painting_profile"])
        self._print_stage_timing("visual_analysis", stage_timings["visual_analysis"])
        self._print_stage_timing("profile_visual_parallel", stage_timings["profile_visual_parallel"])
        self._append_runtime_context(
            "Domain Profile",
            [json.dumps(painting_profile, ensure_ascii=False)],
        )
        visual_cues = list(visual_bundle.get("visual_cues", []))
        technique_judgment = {
            "mountain_present": bool(visual_bundle.get("mountain_present", False)),
            "mountain_evidence": list(visual_bundle.get("mountain_evidence", [])),
            "cunfa_candidates": list(visual_bundle.get("cunfa_candidates", [])),
            "miaofa_candidates": list(visual_bundle.get("miaofa_candidates", [])),
            "reasoning": str(visual_bundle.get("reasoning", "")).strip(),
        }
        self._append_debug_context(
            "Visual Analysis",
            [
                f"visual_cues={json.dumps(visual_cues, ensure_ascii=False)}",
                f"technique_judgment={json.dumps(technique_judgment, ensure_ascii=False)}",
            ],
        )

        stage_started_at = perf_counter()
        text_signals, candidates = await asyncio.to_thread(
            self._analyze_text_anchor_bundle,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
            painting_profile=painting_profile,
            visual_cues=visual_cues,
        )
        candidates = await asyncio.to_thread(self._merge_technique_candidates, candidates, technique_judgment)
        stage_timings["text_anchoring"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("text_anchoring", stage_timings["text_anchoring"])
        self._append_debug_context(
            "Text Anchoring",
            [
                f"text_signals={json.dumps(text_signals, ensure_ascii=False)}",
                *([json.dumps(asdict(candidate), ensure_ascii=False) for candidate in candidates] or ["no candidates"]),
            ],
        )

        stage_started_at = perf_counter()
        grounded_terms = await self._ground_candidates(
            candidates,
            image_bytes=image_bytes,
            image_filename=image_path.name,
            image_mime_type=image_mime_type,
            visual_cues=visual_cues,
            metadata_spine=metadata_spine,
        )
        stage_timings["rag_grounding"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("rag_grounding", stage_timings["rag_grounding"])
        self._append_debug_context(
            "RAG Grounding",
            [
                json.dumps(
                    {
                        "term": item.candidate.term,
                        "sources": [doc.source_id for doc in item.documents],
                        "alignment_scores": item.alignment_scores,
                    },
                    ensure_ascii=False,
                )
                for item in grounded_terms
            ]
            or ["no grounded terms"],
        )
        self._rag_search_record.append(image_path=image_path, grounded_terms=grounded_terms)

        stage_started_at = perf_counter()
        grounded_terms = await asyncio.to_thread(self._extract_text_only_grounded_terms, grounded_terms)
        stage_timings["post_rag_text_extraction"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("post_rag_text_extraction", stage_timings["post_rag_text_extraction"])
        self._append_runtime_context(
            "Post-RAG Text Extraction",
            [
                json.dumps(
                    {
                        "term": item.candidate.term,
                        "description": item.candidate.description,
                        "text_evidence": item.candidate.text_evidence,
                    },
                    ensure_ascii=False,
                )
                for item in grounded_terms
            ]
            or ["no text-only grounded terms"],
        )

        stage_started_at = perf_counter()
        slots = await self._generate_slots_parallel(
            grounded_terms,
            painting_profile,
            visual_cues,
            technique_judgment,
        )
        stage_timings["generate_slots"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("generate_slots", stage_timings["generate_slots"])
        self._append_debug_context(
            "Fixed Slot Planning",
            [json.dumps(slot.to_dict(), ensure_ascii=False) for slot in slots] or ["no fixed slots"],
        )
        stage_started_at = perf_counter()
        deduped_slots, merge_notes = await asyncio.to_thread(self._deduplicate_slots, slots)
        stage_timings["semantic_dedup"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("semantic_dedup", stage_timings["semantic_dedup"])
        self._append_debug_context("Semantic Dedup", merge_notes or ["no merges"])

        ontology_links: list[OntologyLink] = []
        if bool(self._config.enable_ontology_inference):
            stage_started_at = perf_counter()
            ontology_links = await asyncio.to_thread(self._infer_ontology, deduped_slots)
            stage_timings["infer_ontology"] = round(perf_counter() - stage_started_at, 4)
            self._print_stage_timing("infer_ontology", stage_timings["infer_ontology"])
            self._append_runtime_context(
                "Ontology Updates",
                [link.to_markdown() for link in ontology_links] or ["no ontology relations inferred"],
            )
        else:
            stage_timings["infer_ontology"] = 0.0
            self._print_stage_timing("infer_ontology", stage_timings["infer_ontology"])
            self._append_debug_context(
                "Ontology Inference",
                ["disabled"],
            )

        result = PipelineResult(
            slots=deduped_slots,
            grounded_terms=grounded_terms,
            ontology_links=ontology_links,
            output_path=Path(output_path) if output_path else self._config.output_path,
            context_path=self._config.context_path,
            rag_search_record_path=self._config.rag_search_record_path,
            llm_chat_record_path=self._config.llm_chat_record_path,
        )
        result.write_jsonl()
        self._append_debug_context("Run Finished", [f"output=`{result.output_path}`", f"slot_count={len(result.slots)}"])
        total_elapsed = round(perf_counter() - run_started_at, 4)
        slowest_stage, slowest_elapsed = self._slowest_stage(stage_timings)
        self._append_debug_context(
            "Timing Summary",
            [
                json.dumps(stage_timings, ensure_ascii=False),
                f"slowest_stage={slowest_stage}",
                f"slowest_elapsed_s={slowest_elapsed:.4f}",
                f"total_elapsed_s={total_elapsed:.4f}",
            ],
        )
        print(
            f"[perception_bootstrap] total_elapsed_s={total_elapsed:.2f} "
            f"slowest_stage={slowest_stage} slowest_elapsed_s={slowest_elapsed:.2f}",
            flush=True,
        )
        return result

    @staticmethod
    def _print_stage_timing(stage: str, elapsed_s: float) -> None:
        print(f"[perception_stage] stage={stage} elapsed_s={elapsed_s:.2f}", flush=True)

    @staticmethod
    async def _run_timed_thread_stage(
        stage_name: str,
        func: Callable[..., Any],
        /,
        **kwargs: Any,
    ) -> tuple[str, Any, float]:
        started_at = perf_counter()
        result = await asyncio.to_thread(func, **kwargs)
        return stage_name, result, round(perf_counter() - started_at, 4)

    @staticmethod
    def _slowest_stage(stage_timings: dict[str, float]) -> tuple[str, float]:
        if not stage_timings:
            return "none", 0.0
        return max(stage_timings.items(), key=lambda item: item[1])

    def _append_runtime_context(self, title: str, items: Iterable[str]) -> None:
        self._context.append(title, items)

    def _append_debug_context(self, title: str, items: Iterable[str]) -> None:
        if not bool(self._config.emit_debug_context_sections):
            return
        self._context.append(title, items)

    def _build_metadata_spine(
        self,
        *,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
        input_text: str,
    ) -> dict[str, object]:
        image_base64 = base64.b64encode(image_bytes).decode("ascii")
        documents = self._new_rag_client().search(
            query_text=None,
            query_image_bytes=image_bytes,
            query_image_filename=image_filename,
            query_image_mime_type=image_mime_type,
            top_k=max(5, int(self._config.rag_top_k)),
        )
        top_documents = documents[:5]
        selected = self._select_metadata_spine_document(top_documents)
        if selected is None:
            return {}
        candidate_spine = self._extract_metadata_spine_from_document(selected, input_text=input_text)
        if not candidate_spine:
            return {}
        verification = self._verify_metadata_spine_candidate(
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
            metadata_spine=candidate_spine,
        )
        if verification.get("is_target_image", True):
            return self._finalize_metadata_spine(
                metadata_spine=candidate_spine,
                selected=selected,
                top_documents=top_documents,
                strategy="image_rag_top_candidate",
                verification=verification,
            )

        suggested_name = str(verification.get("suggested_name", "")).strip() or str(candidate_spine.get("name", "")).strip()
        if suggested_name:
            metadata_doc = self._search_metadata_document_by_name(suggested_name, metadata_only=True)
            if metadata_doc is not None:
                refined_spine = self._extract_metadata_spine_from_document(metadata_doc, input_text=input_text)
                if refined_spine:
                    return self._finalize_metadata_spine(
                        metadata_spine=refined_spine,
                        selected=metadata_doc,
                        top_documents=top_documents,
                        strategy="metadata_text_search",
                        verification=verification,
                    )
            text_doc = self._search_metadata_document_by_name(suggested_name, metadata_only=False)
            if text_doc is not None:
                refined_spine = self._extract_metadata_spine_from_document(text_doc, input_text=input_text)
                if refined_spine:
                    return self._finalize_metadata_spine(
                        metadata_spine=refined_spine,
                        selected=text_doc,
                        top_documents=top_documents,
                        strategy="text_search_fallback",
                        verification=verification,
                    )

        return self._finalize_metadata_spine(
            metadata_spine=candidate_spine,
            selected=selected,
            top_documents=top_documents,
            strategy="image_rag_fallback_after_failed_recheck",
            verification=verification,
        )

    def _extract_metadata_spine_from_document(
        self,
        document: RagDocument,
        *,
        input_text: str,
    ) -> dict[str, object]:
        payload = self._llm.complete_json(
            system_prompt=METADATA_SPINE_PROMPT,
            user_text=(
                f"基础文字：{input_text}\n"
                f"候选 metadata：{json.dumps(self._serialize_rag_document(document), ensure_ascii=False)}"
            ),
        )
        related_background = [
            str(item).strip()
            for item in payload.get("related_background", [])
            if str(item).strip()
        ]
        return {
            "name": str(payload.get("name", "")).strip(),
            "author": str(payload.get("author", "")).strip(),
            "dynasty": str(payload.get("dynasty", "")).strip(),
            "related_background": _unique_list(related_background)[:5],
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    def _verify_metadata_spine_candidate(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
        metadata_spine: dict[str, object],
    ) -> dict[str, object]:
        payload = self._llm.complete_json(
            system_prompt=METADATA_SPINE_MATCH_PROMPT,
            user_text=(
                f"基础文字：{input_text}\n"
                f"当前候选主干：{json.dumps(metadata_spine, ensure_ascii=False)}"
            ),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        is_target_raw = payload.get("is_target_image", False)
        if isinstance(is_target_raw, str):
            is_target = is_target_raw.strip().lower() in {"true", "1", "yes", "y"}
        else:
            is_target = bool(is_target_raw)
        return {
            "is_target_image": is_target,
            "suggested_name": str(payload.get("suggested_name", "")).strip(),
            "suggested_author": str(payload.get("suggested_author", "")).strip(),
            "suggested_dynasty": str(payload.get("suggested_dynasty", "")).strip(),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    def _search_metadata_document_by_name(self, name: str, *, metadata_only: bool) -> RagDocument | None:
        normalized_name = _normalize_question_text(name)
        if not normalized_name:
            return None
        documents = self._new_rag_client().search(
            query_text=name,
            query_image_bytes=None,
            query_image_filename=None,
            query_image_mime_type=None,
            top_k=max(8, int(self._config.rag_top_k)),
        )
        candidates = documents
        if metadata_only:
            candidates = [
                document
                for document in documents
                if str(document.metadata.get("book_name", "")).strip().lower() == "metadata"
            ]
        if not candidates:
            return None
        for document in candidates:
            if self._document_matches_name(document, normalized_name):
                return document
        return None

    @staticmethod
    def _document_matches_name(document: RagDocument, normalized_name: str) -> bool:
        if not normalized_name:
            return False
        text = _normalize_question_text(
            " ".join(
                [
                    document.content,
                    json.dumps(document.metadata, ensure_ascii=False),
                ]
            )
        )
        return bool(text and normalized_name in text)

    def _finalize_metadata_spine(
        self,
        *,
        metadata_spine: dict[str, object],
        selected: RagDocument,
        top_documents: list[RagDocument],
        strategy: str,
        verification: dict[str, object],
    ) -> dict[str, object]:
        return {
            "name": str(metadata_spine.get("name", "")).strip(),
            "author": str(metadata_spine.get("author", "")).strip(),
            "dynasty": str(metadata_spine.get("dynasty", "")).strip(),
            "related_background": [
                str(item).strip()
                for item in metadata_spine.get("related_background", [])
                if str(item).strip()
            ][:5],
            "reasoning": str(metadata_spine.get("reasoning", "")).strip(),
            "source_id": selected.source_id,
            "book_name": str(selected.metadata.get("book_name", "")).strip(),
            "document_excerpt": selected.content[:600].strip(),
            "top_candidates": [self._serialize_rag_document(item) for item in top_documents],
            "selection_strategy": strategy,
            "verification": verification,
        }

    @staticmethod
    def _select_metadata_spine_document(documents: list[RagDocument]) -> RagDocument | None:
        if not documents:
            return None
        for document in documents:
            book_name = str(document.metadata.get("book_name", "")).strip().lower()
            if book_name == "metadata":
                return document
        return documents[0]

    @staticmethod
    def _serialize_rag_document(document: RagDocument) -> dict[str, object]:
        return {
            "source_id": document.source_id,
            "content": document.content,
            "score": document.score,
            "metadata": document.metadata,
        }

    def _filter_docs_by_metadata_spine(
        self,
        documents: list[RagDocument],
        similarity_scores: list[float],
        *,
        metadata_spine: dict[str, object],
        candidate: TermCandidate,
    ) -> tuple[list[RagDocument], list[float]]:
        if not documents or not metadata_spine:
            return documents, similarity_scores
        spine_name = str(metadata_spine.get("name", "")).strip()
        spine_author = str(metadata_spine.get("author", "")).strip()
        spine_dynasty = str(metadata_spine.get("dynasty", "")).strip()
        if not any([spine_name, spine_author, spine_dynasty]):
            return documents, similarity_scores

        payload = self._llm.complete_json(
            system_prompt=RAG_SPINE_RELEVANCE_PROMPT,
            user_text=json.dumps(
                {
                    "metadata_spine": {
                        "name": spine_name,
                        "author": spine_author,
                        "dynasty": spine_dynasty,
                        "related_background": metadata_spine.get("related_background", []),
                    },
                    "candidate": {
                        "term": candidate.term,
                        "description": candidate.description,
                        "category_guess": candidate.category_guess,
                    },
                    "documents": [
                        {
                            "index": index,
                            "source_id": doc.source_id,
                            "book_name": str(doc.metadata.get("book_name", "")).strip(),
                            "content": doc.content[:1200],
                        }
                        for index, doc in enumerate(documents)
                    ],
                },
                ensure_ascii=False,
            ),
        )
        related_indices = {
            int(item)
            for item in payload.get("related_indices", [])
            if isinstance(item, int) or str(item).isdigit()
        }
        kept_docs: list[RagDocument] = []
        kept_scores: list[float] = []
        for index, (doc, score) in enumerate(zip(documents, similarity_scores)):
            is_related = index in related_indices or self._doc_matches_spine_heuristically(doc, metadata_spine)
            cache_key = (self._spine_cache_key(metadata_spine), doc.source_id)
            self._spine_relevance_cache[cache_key] = is_related
            if is_related:
                kept_docs.append(doc)
                kept_scores.append(score)

        if kept_docs:
            return kept_docs, kept_scores

        fallback_docs: list[RagDocument] = []
        fallback_scores: list[float] = []
        for doc, score in zip(documents, similarity_scores):
            if self._doc_matches_spine_heuristically(doc, metadata_spine):
                fallback_docs.append(doc)
                fallback_scores.append(score)
        return (fallback_docs, fallback_scores) if fallback_docs else (documents[:1], similarity_scores[:1])

    @staticmethod
    def _spine_cache_key(metadata_spine: dict[str, object]) -> str:
        return "|".join(
            [
                str(metadata_spine.get("name", "")).strip(),
                str(metadata_spine.get("author", "")).strip(),
                str(metadata_spine.get("dynasty", "")).strip(),
            ]
        )

    @staticmethod
    def _doc_matches_spine_heuristically(document: RagDocument, metadata_spine: dict[str, object]) -> bool:
        text = _normalize_question_text(
            " ".join(
                [
                    document.content,
                    json.dumps(document.metadata, ensure_ascii=False),
                ]
            )
        )
        if not text:
            return False
        fields = [
            str(metadata_spine.get("name", "")).strip(),
            str(metadata_spine.get("author", "")).strip(),
            str(metadata_spine.get("dynasty", "")).strip(),
        ]
        for field in fields:
            normalized = _normalize_question_text(field)
            if normalized and normalized in text:
                return True
        return False

    @staticmethod
    def _normalize_technique_payload(
        payload: dict[str, object],
    ) -> dict[str, object]:
        mountain_present = bool(payload.get("mountain_present", False))
        mountain_evidence = [
            str(item).strip()
            for item in payload.get("mountain_evidence", [])
            if str(item).strip()
        ]

        def _normalize_items(items: object, *, allowed: list[str], category: str) -> list[dict[str, object]]:
            results: list[dict[str, object]] = []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                term = str(item.get("term", "")).strip()
                if term not in allowed:
                    continue
                if category == "皴法" and not mountain_present:
                    continue
                results.append(
                    {
                        "term": term,
                        "category_guess": category,
                        "description": str(item.get("reason", "")).strip() or f"图像判别更接近{term}。",
                        "visual_evidence": [
                            str(evidence).strip()
                            for evidence in item.get("visual_evidence", [])
                            if str(evidence).strip()
                        ],
                        "confidence": float(item.get("confidence", 0.0) or 0.0),
                    }
                )
            results.sort(key=lambda item: float(item.get("confidence", 0.0) or 0.0), reverse=True)
            return results[:3]

        return {
            "mountain_present": mountain_present,
            "mountain_evidence": mountain_evidence,
            "cunfa_candidates": _normalize_items(payload.get("cunfa_candidates", []), allowed=CUNFA_TECHNIQUE_NAMES, category="皴法"),
            "miaofa_candidates": _normalize_items(payload.get("miaofa_candidates", []), allowed=MIAOFA_TECHNIQUE_NAMES, category="描法"),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    @staticmethod
    def _parse_term_candidates(payload: dict[str, object]) -> list[TermCandidate]:
        candidates: list[TermCandidate] = []
        for item in payload.get("candidates", []):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            description = str(item.get("description", "")).strip()
            category_guess = str(item.get("category_guess", "")).strip()
            if not term or not description or not category_guess:
                continue
            candidates.append(
                TermCandidate(
                    term=term,
                    description=description,
                    category_guess=category_guess,
                    visual_evidence=[str(v).strip() for v in item.get("visual_evidence", []) if str(v).strip()],
                    text_evidence=[str(v).strip() for v in item.get("text_evidence", []) if str(v).strip()],
                )
            )
        return candidates

    def _analyze_visual_bundle(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
    ) -> dict[str, object]:
        payload = self._llm.complete_json(
            system_prompt=VISUAL_ANALYSIS_PROMPT,
            user_text=json.dumps(
                {
                    "input_text": input_text,
                },
                ensure_ascii=False,
            ),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        normalized = self._normalize_technique_payload(payload)
        normalized["visual_cues"] = [
            str(item).strip()
            for item in payload.get("visual_cues", [])
            if str(item).strip()
        ]
        return normalized

    def _extract_visual_cues(self, *, image_base64: str, image_mime_type: str, input_text: str) -> list[str]:
        payload = self._analyze_visual_bundle(
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
        )
        return list(payload.get("visual_cues", []))

    def _judge_image_techniques(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
        visual_cues: list[str],
    ) -> dict[str, object]:
        del visual_cues
        payload = self._analyze_visual_bundle(
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
        )
        return {
            "mountain_present": bool(payload.get("mountain_present", False)),
            "mountain_evidence": list(payload.get("mountain_evidence", [])),
            "cunfa_candidates": list(payload.get("cunfa_candidates", [])),
            "miaofa_candidates": list(payload.get("miaofa_candidates", [])),
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    @staticmethod
    def _merge_candidate_records(
        current: TermCandidate,
        incoming: TermCandidate,
    ) -> TermCandidate:
        return TermCandidate(
            term=current.term,
            description=current.description if len(current.description) >= len(incoming.description) else incoming.description,
            category_guess=current.category_guess if len(current.category_guess) >= len(incoming.category_guess) else incoming.category_guess,
            visual_evidence=_unique_list([*current.visual_evidence, *incoming.visual_evidence]),
            text_evidence=_unique_list([*current.text_evidence, *incoming.text_evidence]),
        )

    def _merge_technique_candidates(
        self,
        candidates: list[TermCandidate],
        technique_judgment: dict[str, object],
    ) -> list[TermCandidate]:
        merged: dict[str, TermCandidate] = {}
        ordered: list[TermCandidate] = []
        for item in technique_judgment.get("cunfa_candidates", []) + technique_judgment.get("miaofa_candidates", []):
            if not isinstance(item, dict):
                continue
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            ordered.append(
                TermCandidate(
                    term=term,
                    description=str(item.get("description", "")).strip() or f"图像判别更接近{term}。",
                    category_guess=str(item.get("category_guess", "")).strip() or "技法",
                    visual_evidence=[str(value).strip() for value in item.get("visual_evidence", []) if str(value).strip()],
                    text_evidence=[],
                )
            )
        ordered.extend(candidates)
        for candidate in ordered:
            key = _normalize_question_text(candidate.term)
            if not key:
                continue
            if key in merged:
                merged[key] = self._merge_candidate_records(merged[key], candidate)
            else:
                merged[key] = candidate
        if not bool(technique_judgment.get("mountain_present", False)):
            merged = {
                key: value
                for key, value in merged.items()
                if not self._is_cunfa_term(value.term)
            }
        return list(merged.values())

    def _analyze_painting_profile(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
        metadata_spine: dict[str, object],
    ) -> dict[str, object]:
        payload = self._llm.complete_json(
            system_prompt=PAINTING_PROFILE_PROMPT,
            user_text=(
                "请先判断这幅图像是什么图像，并结合基础文字与主干信息补充分析。"
                f"\n基础文字：{input_text}"
                f"\n主干信息：{json.dumps(metadata_spine, ensure_ascii=False)}"
            ),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        return {
            "name": str(payload.get("name", "")).strip() or str(metadata_spine.get("name", "")).strip(),
            "author": str(payload.get("author", "")).strip() or str(metadata_spine.get("author", "")).strip(),
            "dynasty": str(payload.get("dynasty", "")).strip() or str(metadata_spine.get("dynasty", "")).strip(),
            "painting_type": str(payload.get("painting_type", "")).strip(),
            "subject": str(payload.get("subject", "")).strip(),
            "scene_summary": str(payload.get("scene_summary", "")).strip(),
            "related_background": _unique_list(
                [
                    *[str(item).strip() for item in metadata_spine.get("related_background", []) if str(item).strip()],
                    *[str(item).strip() for item in payload.get("related_background", []) if str(item).strip()],
                ]
            )[:5],
            "guohua_knowledge": [
                str(item).strip() for item in payload.get("guohua_knowledge", []) if str(item).strip()
            ],
            "reasoning": str(payload.get("reasoning", "")).strip(),
            "spine_source_id": str(metadata_spine.get("source_id", "")).strip(),
            "spine_book_name": str(metadata_spine.get("book_name", "")).strip(),
            "spine_document_excerpt": str(metadata_spine.get("document_excerpt", "")).strip(),
        }

    def _analyze_text_anchor_bundle(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> tuple[dict[str, list[str]], list[TermCandidate]]:
        payload = self._llm.complete_json(
            system_prompt=TEXT_ANCHORING_PROMPT,
            user_text=json.dumps(
                {
                    "input_text": input_text,
                    "painting_profile": painting_profile,
                    "visual_cues": visual_cues,
                },
                ensure_ascii=False,
            ),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        text_signals = {
            "text_signals": [str(item).strip() for item in payload.get("text_signals", []) if str(item).strip()],
            "salient_entities": [str(item).strip() for item in payload.get("salient_entities", []) if str(item).strip()],
        }
        return text_signals, self._parse_term_candidates(payload)

    async def _ground_candidates(
        self,
        candidates: list[TermCandidate],
        *,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
        visual_cues: list[str],
        metadata_spine: dict[str, object],
    ) -> list[GroundedTerm]:
        unique_queries = _unique_list(
            query
            for candidate in candidates
            for query in self._build_primary_grounding_queries(metadata_spine=metadata_spine, candidate=candidate)
            if str(query).strip()
        )
        prefetched_documents: dict[str, list[RagDocument]] = {}
        prefetched_query_durations: dict[str, float] = {}
        worker_limit = max(1, int(self._config.grounding_query_workers or 1))
        semaphore = asyncio.Semaphore(worker_limit)

        async def _prefetch(query_text: str) -> tuple[str, list[RagDocument], float]:
            async with semaphore:
                documents, duration_ms = await asyncio.to_thread(
                    self._search_grounding_documents_with_stats,
                    query_text=query_text,
                    image_bytes=image_bytes,
                    image_filename=image_filename,
                    image_mime_type=image_mime_type,
                )
                return query_text, documents, duration_ms

        prefetch_results = await asyncio.gather(*[_prefetch(query_text) for query_text in unique_queries])
        for query_text, documents, duration_ms in prefetch_results:
            prefetched_documents[query_text] = documents
            prefetched_query_durations[query_text] = duration_ms
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._ground_single_candidate,
                    candidate,
                    image_bytes,
                    image_filename,
                    image_mime_type,
                    visual_cues,
                    metadata_spine,
                    prefetched_documents,
                    prefetched_query_durations,
                )
            )
            for candidate in candidates
        ]
        grounded = await asyncio.gather(*tasks)
        return [item for item in grounded if item.documents]

    def _build_primary_grounding_queries(
        self,
        *,
        metadata_spine: dict[str, object],
        candidate: TermCandidate,
    ) -> list[str]:
        name = str(metadata_spine.get("name", "")).strip()
        author = str(metadata_spine.get("author", "")).strip()
        dynasty = str(metadata_spine.get("dynasty", "")).strip()
        queries: list[str] = []
        if name:
            if author:
                queries.append(f"{name} {author}")
            if dynasty:
                queries.append(f"{name} {dynasty}")
            queries.append(name)
        else:
            if author and dynasty:
                queries.append(f"{author} {dynasty}")
            if author:
                queries.append(author)
            if dynasty:
                queries.append(dynasty)
        deduped = _unique_list(query.strip() for query in queries if str(query).strip())
        if deduped:
            return deduped[:3]
        return [candidate.term] if candidate.term.strip() else []

    def _search_grounding_documents(
        self,
        *,
        query_text: str,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
    ) -> list[RagDocument]:
        documents, _ = self._search_grounding_documents_with_stats(
            query_text=query_text,
            image_bytes=image_bytes,
            image_filename=image_filename,
            image_mime_type=image_mime_type,
        )
        return documents

    def _search_grounding_documents_with_stats(
        self,
        *,
        query_text: str,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
    ) -> tuple[list[RagDocument], float]:
        cache_key = (query_text, image_filename, image_mime_type, hash(image_bytes))
        cached = self._grounding_rag_cache.get(cache_key)
        if cached is not None:
            return cached, 0.0
        started_at = perf_counter()
        documents = self._new_rag_client().search(
            query_text=query_text,
            query_image_bytes=image_bytes,
            query_image_filename=image_filename,
            query_image_mime_type=image_mime_type,
            top_k=self._config.rag_top_k,
        )
        self._grounding_rag_cache[cache_key] = documents
        duration_ms = round((perf_counter() - started_at) * 1000.0, 2)
        return documents, duration_ms

    def _ground_single_candidate(
        self,
        candidate: TermCandidate,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
        visual_cues: list[str],
        metadata_spine: dict[str, object],
        prefetched_documents: dict[str, list[RagDocument]] | None = None,
        prefetched_query_durations: dict[str, float] | None = None,
    ) -> GroundedTerm:
        search_queries = self._build_primary_grounding_queries(metadata_spine=metadata_spine, candidate=candidate)
        documents: list[RagDocument] = []
        query_records: list[GroundingQueryRecord] = []
        seen_docs: set[tuple[str, str]] = set()
        for query_text in search_queries:
            query_documents = prefetched_documents.get(query_text) if prefetched_documents is not None else None
            query_duration_ms = 0.0
            if query_documents is None:
                query_documents, query_duration_ms = self._search_grounding_documents_with_stats(
                    query_text=query_text,
                    image_bytes=image_bytes,
                    image_filename=image_filename,
                    image_mime_type=image_mime_type,
                )
            elif prefetched_query_durations is not None:
                query_duration_ms = float(prefetched_query_durations.get(query_text, 0.0) or 0.0)
            filtered_query_documents, initial_top_k_count, matched_count = self._filter_documents_for_candidate_term(
                candidate=candidate,
                documents=query_documents,
            )
            query_records.append(
                GroundingQueryRecord(
                    query_text=query_text,
                    duration_ms=query_duration_ms,
                    initial_top_k_count=initial_top_k_count,
                    matched_count=matched_count,
                )
            )
            for document in filtered_query_documents:
                doc_key = (document.source_id, document.content)
                if doc_key in seen_docs:
                    continue
                seen_docs.add(doc_key)
                documents.append(document)

        anchor_text = " ".join(
            [
                candidate.term,
                candidate.description,
                candidate.category_guess,
                *visual_cues,
                *candidate.visual_evidence,
                *candidate.text_evidence,
            ]
        )
        scored_documents = [
            (document, round(self._similarity.similarity(anchor_text, document.content), 4))
            for document in documents
        ]
        filtered_docs = [
            document
            for document, score in scored_documents
            if score >= self._config.rag_similarity_threshold
        ]
        similarity_scores = [
            score
            for _, score in scored_documents
            if score >= self._config.rag_similarity_threshold
        ]
        if not filtered_docs and scored_documents:
            best_document, best_score = max(scored_documents, key=lambda item: item[1])
            filtered_docs = [best_document]
            similarity_scores = [best_score]
        filtered_docs, similarity_scores = self._filter_docs_by_metadata_spine(
            filtered_docs,
            similarity_scores,
            metadata_spine=metadata_spine,
            candidate=candidate,
        )
        return GroundedTerm(
            candidate=candidate,
            documents=filtered_docs,
            alignment_scores=similarity_scores,
            search_queries=search_queries,
            query_records=query_records,
        )

    @staticmethod
    def _filter_documents_for_candidate_term(
        *,
        candidate: TermCandidate,
        documents: list[RagDocument],
    ) -> tuple[list[RagDocument], int, int]:
        top_documents = list(documents[:5])
        if not top_documents:
            return [], 0, 0
        normalized_term = _normalize_question_text(candidate.term)
        if not normalized_term:
            return top_documents, len(top_documents), len(top_documents)
        matched_documents = [
            document
            for document in top_documents
            if normalized_term in _normalize_question_text(document.content)
        ]
        return matched_documents, len(top_documents), len(matched_documents)

    def _generate_slots(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
        technique_judgment: dict[str, object],
    ) -> list[SlotRecord]:
        slots: list[SlotRecord] = []
        for spec in _FIXED_SLOT_SPECS:
            candidates = self._build_fixed_slot_candidates(
                spec=spec,
                grounded_terms=grounded_terms,
                painting_profile=painting_profile,
                visual_cues=visual_cues,
                technique_judgment=technique_judgment,
            )
            slot = self._plan_fixed_slot(
                spec=spec,
                candidates=candidates,
                painting_profile=painting_profile,
                visual_cues=visual_cues,
            )
            slots.append(slot)
        return slots

    async def _generate_slots_parallel(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
        technique_judgment: dict[str, object],
    ) -> list[SlotRecord]:
        worker_limit = max(1, int(self._config.slot_planner_workers or 1))
        semaphore = asyncio.Semaphore(worker_limit)

        async def _plan(index: int, spec: dict[str, str]) -> tuple[int, SlotRecord]:
            async with semaphore:
                candidates = await asyncio.to_thread(
                    self._build_fixed_slot_candidates,
                    spec=spec,
                    grounded_terms=grounded_terms,
                    painting_profile=painting_profile,
                    visual_cues=visual_cues,
                    technique_judgment=technique_judgment,
                )
                slot = await asyncio.to_thread(
                    self._plan_fixed_slot,
                    spec=spec,
                    candidates=candidates,
                    painting_profile=painting_profile,
                    visual_cues=visual_cues,
                )
                return index, slot

        planned = await asyncio.gather(
            *[_plan(index, spec) for index, spec in enumerate(_FIXED_SLOT_SPECS)]
        )
        planned.sort(key=lambda item: item[0])
        return [slot for _, slot in planned]

    def _plan_fixed_slot(
        self,
        *,
        spec: dict[str, str],
        candidates: list[dict[str, object]],
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> SlotRecord:
        slot_profile = self._slot_scoped_painting_profile(spec=spec, painting_profile=painting_profile)
        canonical_spec = {
            **spec,
            "slot_name": self._canonical_fixed_slot_name(str(spec.get("slot_name", "")).strip()),
        }
        candidate_term_groups = self._build_candidate_term_groups(
            slot_name=canonical_spec["slot_name"],
            candidates=candidates,
        )
        payload = self._llm.complete_json(
            system_prompt=FIXED_SLOT_SELECTION_PROMPT,
            user_text=json.dumps(
                {
                    "slot_definition": canonical_spec,
                    "painting_profile": slot_profile,
                    "visual_cues": visual_cues,
                    "used_terms": [],
                    "candidate_terms": candidates,
                    "candidate_term_groups": candidate_term_groups,
                },
                ensure_ascii=False,
            ),
        )
        applicable = bool(payload.get("applicable", False))
        slot_name = self._canonical_fixed_slot_name(str(payload.get("slot_name", "")).strip() or canonical_spec["slot_name"])
        slot_term = str(payload.get("slot_term", "")).strip() if applicable else ""
        description = str(payload.get("description", "")).strip()
        questions = [str(question).strip() for question in payload.get("specific_questions", []) if str(question).strip()]
        questions = self._filter_questions_by_theme(
            questions,
            painting_profile=painting_profile,
            slot_name=slot_name,
            slot_term=slot_term,
            description=description,
        )
        metadata_payload = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        if applicable and len(questions) < 2:
            applicable = False
            slot_term = ""
            description = ""
            questions = []

        lifecycle = "ACTIVE" if applicable else "CLOSED"
        pending_terms = [
            str(term).strip()
            for term in payload.get("pending_terms", [])
            if str(term).strip() and str(term).strip() != slot_term
        ]
        slot_terms = self._build_slot_terms(
            slot_name=slot_name,
            slot_term=slot_term,
            candidate_term_groups=candidate_term_groups,
        )
        pending_terms = self._prune_pending_terms(
            pending_terms=pending_terms,
            slot_terms=slot_terms,
        )
        reasoning = str(payload.get("reasoning", "")).strip()
        selected_term_group = self._find_candidate_group_name(
            term=slot_term,
            candidate_term_groups=candidate_term_groups,
        )
        questions = self._ensure_slot_terms_in_questions(
            slot_name=slot_name,
            questions=questions,
            slot_terms=slot_terms,
        )
        extra = {
            "slot_mode": spec["slot_mode"],
            "goal": spec["goal"],
            "lifecycle": lifecycle,
            "used_terms": [slot_term] if slot_term else [],
            "pending_terms": _unique_list(pending_terms),
            "slot_terms": slot_terms,
            "candidate_terms": [str(item.get("term", "")).strip() for item in candidates if str(item.get("term", "")).strip()],
            "candidate_term_groups": candidate_term_groups,
            "selected_term_group": selected_term_group,
            "availability_reason": reasoning,
        }
        return SlotRecord(
            slot_name=slot_name,
            slot_term=slot_term,
            description=description if applicable else (reasoning or f"{slot_name} 当前没有可直接支撑的术语。"),
            specific_questions=questions[:3],
            metadata=SlotMetadata(
                confidence=float(metadata_payload.get("confidence", 0.0) or 0.0),
                source_id=str(metadata_payload.get("source_id", "")).strip(),
                extra=extra,
            ),
        )

    def _slot_scoped_painting_profile(
        self,
        *,
        spec: dict[str, str],
        painting_profile: dict[str, object],
    ) -> dict[str, object]:
        slot_name = self._canonical_fixed_slot_name(str(spec.get("slot_name", "")).strip())
        shared = {
            "painting_type": str(painting_profile.get("painting_type", "")).strip(),
            "subject": str(painting_profile.get("subject", "")).strip(),
            "scene_summary": str(painting_profile.get("scene_summary", "")).strip(),
        }
        if slot_name == "画作背景":
            shared.update(
                {
                    "name": str(painting_profile.get("name", "")).strip(),
                    "related_background": painting_profile.get("related_background", []),
                }
            )
        elif slot_name == "作者时代流派":
            shared.update(
                {
                    "author": str(painting_profile.get("author", "")).strip(),
                    "dynasty": str(painting_profile.get("dynasty", "")).strip(),
                    "related_background": painting_profile.get("related_background", []),
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                }
            )
        elif slot_name == "墨法设色技法":
            shared.update(
                {
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                }
            )
        elif slot_name == "题跋诗文审美语言":
            shared.update(
                {
                    "related_background": [],
                    "guohua_knowledge": [],
                }
            )
        elif slot_name == "题跋/印章/用笔":
            shared.update(
                {
                    "related_background": painting_profile.get("related_background", []),
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                }
            )
        elif slot_name == "构图/空间/布局":
            shared.update(
                {
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                    "related_background": [],
                }
            )
        elif slot_name == "尺寸规格/材质形制/收藏地":
            shared.update(
                {
                    "name": str(painting_profile.get("name", "")).strip(),
                    "related_background": painting_profile.get("related_background", []),
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                }
            )
        elif slot_name == "意境/题材/象征":
            shared.update(
                {
                    "related_background": painting_profile.get("related_background", []),
                    "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                }
            )
        return shared

    def _build_fixed_slot_candidates(
        self,
        *,
        spec: dict[str, str],
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
        technique_judgment: dict[str, object],
    ) -> list[dict[str, object]]:
        slot_name = self._canonical_fixed_slot_name(str(spec.get("slot_name", "")).strip())
        if slot_name == "画作背景":
            return self._build_title_slot_candidates(painting_profile)
        if slot_name == "作者时代流派":
            return self._build_author_slot_candidates(grounded_terms, painting_profile)
        if slot_name == "墨法设色技法":
            return self._build_technique_slot_candidates(grounded_terms, visual_cues, technique_judgment)
        if slot_name == "题跋诗文审美语言":
            return self._build_inscription_slot_candidates(grounded_terms, painting_profile, visual_cues)
        if slot_name == "题跋/印章/用笔":
            return self._build_inscription_brushwork_slot_candidates(grounded_terms, painting_profile, visual_cues)
        if slot_name == "构图/空间/布局":
            return self._build_composition_slot_candidates(grounded_terms, painting_profile, visual_cues)
        if slot_name == "尺寸规格/材质形制/收藏地":
            return self._build_material_slot_candidates(grounded_terms, painting_profile, visual_cues)
        if slot_name == "意境/题材/象征":
            return self._build_mood_slot_candidates(grounded_terms, painting_profile)
        return []

    @staticmethod
    def _canonical_fixed_slot_name(slot_name: str) -> str:
        clean = str(slot_name).strip()
        return _FIXED_SLOT_NAME_ALIASES.get(clean, clean)

    def _build_title_slot_candidates(self, painting_profile: dict[str, object]) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        name = str(painting_profile.get("name", "")).strip()
        subject = str(painting_profile.get("subject", "")).strip()
        source_id = str(painting_profile.get("spine_source_id", "")).strip()
        for priority, term in enumerate(self._derive_title_terms(name, subject), start=1):
            candidates.append(
                {
                    "term": term,
                    "source_id": source_id,
                    "priority": max(1, 10 - priority),
                    "source_type": "painting_profile",
                    "evidence": [name, subject],
                }
            )
        return candidates

    def _build_author_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        ordered = [
            ("author", str(painting_profile.get("author", "")).strip(), 10),
            ("dynasty", str(painting_profile.get("dynasty", "")).strip(), 9),
            ("painting_type", str(painting_profile.get("painting_type", "")).strip(), 8),
        ]
        for source_type, term, priority in ordered:
            if term:
                candidates.append(
                    {
                        "term": term,
                        "source_id": str(painting_profile.get("spine_source_id", "")).strip(),
                        "priority": priority,
                        "source_type": source_type,
                        "evidence": [term],
                    }
                )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            if any(keyword in token for keyword in ("院体", "流派", "道释", "人物画", "山水画")) or "风格" in category:
                candidates.append(
                    {
                        "term": token,
                        "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                        "priority": 6,
                        "source_type": "grounded_term",
                        "evidence": [term.candidate.description, *term.candidate.text_evidence[:1]],
                    }
                )
        return self._dedupe_slot_candidates(candidates)

    def _build_technique_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        visual_cues: list[str],
        technique_judgment: dict[str, object],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        mountain_present = bool(technique_judgment.get("mountain_present", False))
        for item in technique_judgment.get("cunfa_candidates", []) + technique_judgment.get("miaofa_candidates", []):
            if not isinstance(item, dict):
                continue
            token = str(item.get("term", "")).strip()
            if not token:
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": "",
                    "priority": 12 if self._is_cunfa_term(token) else 11,
                    "source_type": "technique_judgment",
                    "evidence": [
                        *[str(value).strip() for value in item.get("visual_evidence", []) if str(value).strip()],
                        str(item.get("description", "")).strip(),
                    ],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            if self._is_cunfa_term(token) and not mountain_present:
                continue
            if self._is_technique_term(token, category):
                candidates.append(
                    {
                        "term": token,
                        "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                        "priority": self._technique_priority(token),
                        "source_type": "grounded_term",
                        "evidence": [term.candidate.description, *term.candidate.text_evidence[:1], *term.candidate.visual_evidence[:1]],
                    }
                )
        for cue in visual_cues:
            cue_term = self._shorten_visual_cue_term(cue)
            if self._is_cunfa_term(cue_term) and not mountain_present:
                continue
            if self._is_technique_term(cue, cue):
                candidates.append(
                    {
                        "term": cue_term,
                        "source_id": "",
                        "priority": 4,
                        "source_type": "visual_cue",
                        "evidence": [cue],
                    }
                )
        return self._dedupe_slot_candidates(candidates)

    def _build_inscription_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for cue in visual_cues:
            if not self._contains_any_keyword(cue, _TEXT_INSCRIPTION_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(cue, _TEXT_INSCRIPTION_KEYWORDS),
                    "source_id": "",
                    "priority": 10,
                    "source_type": "visual_cue",
                    "evidence": [cue],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            description = str(term.candidate.description).strip()
            if not self._contains_any_keyword(" ".join([token, category, description]), _TEXT_INSCRIPTION_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                    "priority": 8,
                    "source_type": "grounded_term",
                    "evidence": [description, *term.candidate.text_evidence[:1]],
                }
            )
        return self._dedupe_slot_candidates(candidates)

    def _build_inscription_brushwork_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for cue in visual_cues:
            if not self._contains_any_keyword(cue, _INSCRIPTION_BRUSHWORK_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(cue, _INSCRIPTION_BRUSHWORK_KEYWORDS),
                    "source_id": "",
                    "priority": 10,
                    "source_type": "visual_cue",
                    "evidence": [cue],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            description = str(term.candidate.description).strip()
            if not self._contains_any_keyword(" ".join([token, category, description]), _INSCRIPTION_BRUSHWORK_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                    "priority": 8,
                    "source_type": "grounded_term",
                    "evidence": [description, *term.candidate.text_evidence[:1], *term.candidate.visual_evidence[:1]],
                }
            )
        for knowledge in painting_profile.get("guohua_knowledge", []):
            text = str(knowledge).strip()
            if not self._contains_any_keyword(text, ("用笔", "笔法", "笔触", "线条")):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(text, ("用笔", "笔法", "笔触", "线条")),
                    "source_id": str(painting_profile.get("spine_source_id", "")).strip(),
                    "priority": 5,
                    "source_type": "painting_profile",
                    "evidence": [text],
                }
            )
        return self._dedupe_slot_candidates(candidates)

    def _build_composition_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        subject = str(painting_profile.get("subject", "")).strip()
        scene_summary = str(painting_profile.get("scene_summary", "")).strip()
        if subject:
            candidates.append(
                {
                    "term": subject,
                    "source_id": str(painting_profile.get("spine_source_id", "")).strip(),
                    "priority": 7,
                    "source_type": "painting_profile",
                    "evidence": [subject, scene_summary],
                }
            )
        for cue in visual_cues:
            if not self._contains_any_keyword(cue, _COMPOSITION_LAYOUT_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(cue, _COMPOSITION_LAYOUT_KEYWORDS),
                    "source_id": "",
                    "priority": 10,
                    "source_type": "visual_cue",
                    "evidence": [cue],
                }
            )
        for knowledge in painting_profile.get("guohua_knowledge", []):
            text = str(knowledge).strip()
            if not self._contains_any_keyword(text, _COMPOSITION_LAYOUT_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(text, _COMPOSITION_LAYOUT_KEYWORDS),
                    "source_id": str(painting_profile.get("spine_source_id", "")).strip(),
                    "priority": 6,
                    "source_type": "painting_profile",
                    "evidence": [text],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            description = str(term.candidate.description).strip()
            if not self._contains_any_keyword(" ".join([token, category, description]), _COMPOSITION_LAYOUT_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                    "priority": 8,
                    "source_type": "grounded_term",
                    "evidence": [description, *term.candidate.text_evidence[:1], *term.candidate.visual_evidence[:1]],
                }
            )
        return self._dedupe_slot_candidates(candidates)

    def _build_material_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
        visual_cues: list[str],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for cue in visual_cues:
            if not self._contains_any_keyword(cue, _MATERIAL_FORMAT_COLLECTION_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(cue, _MATERIAL_FORMAT_COLLECTION_KEYWORDS),
                    "source_id": "",
                    "priority": 9,
                    "source_type": "visual_cue",
                    "evidence": [cue],
                }
            )
        for text in [
            *[str(item).strip() for item in painting_profile.get("related_background", []) if str(item).strip()],
            *[str(item).strip() for item in painting_profile.get("guohua_knowledge", []) if str(item).strip()],
        ]:
            if not self._contains_any_keyword(text, _MATERIAL_FORMAT_COLLECTION_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": self._extract_keyword_term(text, _MATERIAL_FORMAT_COLLECTION_KEYWORDS),
                    "source_id": str(painting_profile.get("spine_source_id", "")).strip(),
                    "priority": 5,
                    "source_type": "painting_profile",
                    "evidence": [text],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            description = str(term.candidate.description).strip()
            if not self._contains_any_keyword(" ".join([token, category, description]), _MATERIAL_FORMAT_COLLECTION_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                    "priority": 10,
                    "source_type": "grounded_term",
                    "evidence": [description, *term.candidate.text_evidence[:1], *term.candidate.visual_evidence[:1]],
                }
            )
        return self._dedupe_slot_candidates(candidates)

    def _build_mood_slot_candidates(
        self,
        grounded_terms: list[GroundedTerm],
        painting_profile: dict[str, object],
    ) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        subject = str(painting_profile.get("subject", "")).strip()
        painting_type = str(painting_profile.get("painting_type", "")).strip()
        source_id = str(painting_profile.get("spine_source_id", "")).strip()
        if subject:
            candidates.append(
                {
                    "term": subject,
                    "source_id": source_id,
                    "priority": 10,
                    "source_type": "painting_profile",
                    "evidence": [subject, str(painting_profile.get("scene_summary", "")).strip()],
                }
            )
        if painting_type:
            candidates.append(
                {
                    "term": painting_type,
                    "source_id": source_id,
                    "priority": 8,
                    "source_type": "painting_profile",
                    "evidence": [painting_type],
                }
            )
        aesthetic_text = " ".join(
            [
                str(painting_profile.get("scene_summary", "")).strip(),
                " ".join(str(item).strip() for item in painting_profile.get("related_background", []) if str(item).strip()),
                " ".join(str(item).strip() for item in painting_profile.get("guohua_knowledge", []) if str(item).strip()),
            ]
        )
        for keyword in _MOOD_SUBJECT_SYMBOLISM_KEYWORDS:
            if keyword not in aesthetic_text:
                continue
            candidates.append(
                {
                    "term": keyword,
                    "source_id": source_id,
                    "priority": 6,
                    "source_type": "painting_profile",
                    "evidence": [aesthetic_text],
                }
            )
        for term in grounded_terms:
            token = str(term.candidate.term).strip()
            if not token:
                continue
            category = str(term.candidate.category_guess).strip()
            description = str(term.candidate.description).strip()
            if not self._contains_any_keyword(" ".join([token, category, description]), _MOOD_SUBJECT_SYMBOLISM_KEYWORDS):
                continue
            candidates.append(
                {
                    "term": token,
                    "source_id": next((doc.source_id for doc in term.documents if doc.source_id), ""),
                    "priority": 7,
                    "source_type": "grounded_term",
                    "evidence": [description, *term.candidate.text_evidence[:1]],
                }
            )
        return self._dedupe_slot_candidates(candidates)

    @staticmethod
    def _derive_title_terms(name: str, subject: str) -> list[str]:
        candidates: list[str] = []
        if name:
            broader = re.sub(r"第[一二三四五六七八九十百零〇0-9]+尊者.*$", "", name).strip()
            if broader and broader != name:
                candidates.append(broader)
            numbered = re.search(r"第[一二三四五六七八九十百零〇0-9]+尊者", name)
            if numbered:
                candidates.append(numbered.group(0))
            candidates.append(name)
        if subject and subject not in candidates:
            candidates.append(subject)
        return _unique_list(candidates)

    @staticmethod
    def _dedupe_slot_candidates(candidates: list[dict[str, object]]) -> list[dict[str, object]]:
        deduped: list[dict[str, object]] = []
        seen: set[str] = set()
        ordered = sorted(candidates, key=lambda item: (-int(item.get("priority", 0) or 0), str(item.get("term", ""))))
        for item in ordered:
            term = str(item.get("term", "")).strip()
            if not term:
                continue
            key = _normalize_question_text(term)
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(item)
        return deduped

    def _build_candidate_term_groups(
        self,
        *,
        slot_name: str,
        candidates: list[dict[str, object]],
    ) -> list[dict[str, object]]:
        grouped: dict[str, list[str]] = {}
        ordered_groups: list[str] = []
        for candidate in candidates:
            term = str(candidate.get("term", "")).strip()
            if not term:
                continue
            group_name = self._infer_candidate_group(slot_name=slot_name, candidate=candidate)
            if group_name not in grouped:
                grouped[group_name] = []
                ordered_groups.append(group_name)
            if term not in grouped[group_name]:
                grouped[group_name].append(term)
        return [
            {
                "group_name": group_name,
                "terms": grouped[group_name],
            }
            for group_name in ordered_groups
        ]

    def _build_slot_terms(
        self,
        *,
        slot_name: str,
        slot_term: str,
        candidate_term_groups: list[dict[str, object]],
    ) -> list[str]:
        if not slot_term:
            return []
        selected_group = self._find_candidate_group_name(
            term=slot_term,
            candidate_term_groups=candidate_term_groups,
        )
        if not selected_group or not self._should_preserve_group_siblings(slot_name=slot_name, group_name=selected_group):
            return [slot_term]
        sibling_terms: list[str] = []
        for item in candidate_term_groups:
            if str(item.get("group_name", "")).strip() != selected_group:
                continue
            sibling_terms = [
                str(term).strip()
                for term in item.get("terms", [])
                if str(term).strip() and str(term).strip() != slot_term
            ]
            break
        return _unique_list([slot_term, *sibling_terms])

    @staticmethod
    def _prune_pending_terms(*, pending_terms: list[str], slot_terms: list[str]) -> list[str]:
        normalized_slot_terms = {_normalize_question_text(term) for term in slot_terms if str(term).strip()}
        return [
            str(term).strip()
            for term in pending_terms
            if str(term).strip() and _normalize_question_text(str(term)) not in normalized_slot_terms
        ]

    @staticmethod
    def _find_candidate_group_name(
        *,
        term: str,
        candidate_term_groups: list[dict[str, object]],
    ) -> str:
        normalized_term = _normalize_question_text(term)
        if not normalized_term:
            return ""
        for item in candidate_term_groups:
            group_name = str(item.get("group_name", "")).strip()
            terms = [
                str(group_term).strip()
                for group_term in item.get("terms", [])
                if str(group_term).strip()
            ]
            if normalized_term in {_normalize_question_text(group_term) for group_term in terms}:
                return group_name
        return ""

    def _infer_candidate_group(self, *, slot_name: str, candidate: dict[str, object]) -> str:
        term = str(candidate.get("term", "")).strip()
        source_type = str(candidate.get("source_type", "")).strip()
        evidence = " ".join(str(item).strip() for item in candidate.get("evidence", []) if str(item).strip())
        text = " ".join([slot_name, term, source_type, evidence])

        if slot_name == "墨法设色技法":
            return self._infer_technique_candidate_group(text)
        if slot_name == "作者时代流派":
            if any(keyword in text for keyword in ("作者", "画家", "书家", "佚名")):
                return "作者"
            if any(keyword in text for keyword in ("朝代", "时代", "宋", "元", "明", "清", "五代")):
                return "时代"
            if any(keyword in text for keyword in ("流派", "院体", "画派", "山水画", "花鸟画", "人物画", "草虫画")):
                return "流派画种"
            return "背景身份"
        if slot_name == "题跋诗文审美语言":
            if any(keyword in text for keyword in _TEXT_INSCRIPTION_KEYWORDS):
                return "文字内容"
            return "文字线索"
        if slot_name == "题跋/印章/用笔":
            if any(keyword in text for keyword in ("印章", "钤印", "鉴藏印")):
                return "印章"
            if any(keyword in text for keyword in ("用笔", "笔法", "笔触", "线条", "中锋", "侧锋")):
                return "用笔线质"
            if any(keyword in text for keyword in _TEXT_INSCRIPTION_KEYWORDS):
                return "文字内容"
            return "题跋印章用笔"
        if slot_name == "构图/空间/布局":
            if any(keyword in text for keyword in ("高远", "深远", "平远", "前景", "中景", "后景", "空间")):
                return "空间层次"
            if any(keyword in text for keyword in ("构图", "布局", "章法", "经营", "动线", "重心", "留白")):
                return "构图布局"
            return "布局线索"
        if slot_name == "尺寸规格/材质形制/收藏地":
            if any(keyword in text for keyword in ("尺寸", "规格", "厘米", "尺幅")):
                return "尺寸规格"
            if any(keyword in text for keyword in ("绢本", "纸本", "设色", "水墨", "材质")):
                return "材质设色"
            if any(keyword in text for keyword in ("手卷", "册页", "立轴", "镜心", "装裱", "形制")):
                return "形制装裱"
            if any(keyword in text for keyword in ("故宫", "博物馆", "馆藏", "收藏")):
                return "收藏地"
            return "作品身份线索"
        if slot_name == "意境/题材/象征":
            if any(keyword in text for keyword in ("意境", "寓意", "象征", "吉祥", "富贵", "隐逸")):
                return "意境象征"
            if any(keyword in text for keyword in ("山水", "花鸟", "人物", "罗汉", "行旅", "题材")):
                return "题材对象"
            return "题材意境"
        if slot_name == "画作背景":
            if any(keyword in text for keyword in ("图", "卷", "册", "作品名", "画名")):
                return "作品名称"
            if any(keyword in text for keyword in ("题材", "主干", "对象", "主题")):
                return "题材主干"
        return "其他"

    def _infer_technique_candidate_group(self, text: str) -> str:
        if any(keyword in text for keyword in ("皴",)):
            return "皴法"
        if any(keyword in text for keyword in ("描",)):
            return "描法"
        if any(keyword in text for keyword in ("墨法", "墨色", "水墨", "积墨", "泼墨", "破墨", "宿墨", "焦墨", "浓墨", "淡墨")):
            return "墨法"
        if any(keyword in text for keyword in ("设色", "赋彩", "青绿", "浅绛", "赭色", "石青", "石绿", "敷色")):
            return "设色"
        if any(keyword in text for keyword in ("用笔", "笔法", "笔触", "线条", "中锋", "侧锋")):
            return "用笔线质"
        if any(keyword in text for keyword in ("绢本", "纸本", "材质")):
            return "材质"
        return "其他"

    @staticmethod
    def _should_preserve_group_siblings(*, slot_name: str, group_name: str) -> bool:
        if slot_name == "墨法设色技法":
            return group_name in _TECHNIQUE_GROUP_LABELS
        if slot_name == "题跋/印章/用笔":
            return group_name in {"印章", "用笔线质", "文字内容"}
        return False

    def _ensure_slot_terms_in_questions(
        self,
        *,
        slot_name: str,
        questions: list[str],
        slot_terms: list[str],
    ) -> list[str]:
        normalized_questions = [_normalize_question_text(question) for question in questions]
        if len(slot_terms) <= 1:
            return questions
        combined_phrase = "、".join(slot_terms)
        normalized_terms = [_normalize_question_text(term) for term in slot_terms if term.strip()]
        if any(all(term in question for term in normalized_terms) for question in normalized_questions):
            return questions
        combined_question = self._build_combined_slot_terms_question(
            slot_name=slot_name,
            slot_terms=slot_terms,
        )
        if not combined_question:
            return questions
        return _unique_list([combined_question, *questions])[:3]

    @staticmethod
    def _build_combined_slot_terms_question(*, slot_name: str, slot_terms: list[str]) -> str:
        combined_phrase = "、".join(slot_terms)
        if slot_name == "墨法设色技法":
            return f"{combined_phrase} 在当前作品中分别对应哪些画面证据？它们之间是并存、互补还是需要区分？"
        if slot_name == "题跋/印章/用笔":
            return f"{combined_phrase} 在当前作品中各自对应哪些可见线索？它们之间如何互相印证？"
        if slot_name == "作者时代流派":
            return f"{combined_phrase} 这组术语如何共同支撑当前作品的作者、时代或流派判断？"
        return f"{combined_phrase} 在当前作品中分别承担什么作用？它们之间如何互相印证？"

    @staticmethod
    def _contains_any_keyword(text: str, keywords: tuple[str, ...]) -> bool:
        return any(keyword in str(text) for keyword in keywords)

    @staticmethod
    def _extract_keyword_term(text: str, keywords: tuple[str, ...]) -> str:
        clean = str(text).strip()
        for keyword in keywords:
            if keyword in clean:
                return keyword
        return clean[:18].strip()

    @staticmethod
    def _is_technique_term(term: str, category: str) -> bool:
        text = f"{term} {category}"
        return any(keyword in text for keyword in ("皴", "描", "设色", "墨", "笔法", "线描", "绢本", "赋彩", "中间色"))

    @staticmethod
    def _is_cunfa_term(term: str) -> bool:
        return str(term).strip() in CUNFA_TECHNIQUE_NAMES

    @staticmethod
    def _is_miaofa_term(term: str) -> bool:
        return str(term).strip() in MIAOFA_TECHNIQUE_NAMES

    @staticmethod
    def _technique_priority(term: str) -> int:
        if any(keyword in term for keyword in ("皴", "描")):
            return 10
        if any(keyword in term for keyword in ("设色", "赋彩", "中间色")):
            return 9
        if any(keyword in term for keyword in ("绢本", "墨")):
            return 8
        return 6

    @staticmethod
    def _shorten_visual_cue_term(cue: str) -> str:
        if "：" in cue:
            cue = cue.split("：", 1)[-1].strip()
        return cue[:18].strip()

    def _filter_questions_by_theme(
        self,
        questions: list[str],
        *,
        painting_profile: dict[str, object],
        slot_name: str,
        slot_term: str,
        description: str,
    ) -> list[str]:
        theme_text = " ".join(
            [
                str(painting_profile.get("name", "")).strip(),
                str(painting_profile.get("author", "")).strip(),
                str(painting_profile.get("dynasty", "")).strip(),
                str(painting_profile.get("painting_type", "")).strip(),
                str(painting_profile.get("subject", "")).strip(),
                str(painting_profile.get("scene_summary", "")).strip(),
                " ".join(str(item).strip() for item in painting_profile.get("related_background", []) if str(item).strip()),
                " ".join(str(item).strip() for item in painting_profile.get("guohua_knowledge", []) if str(item).strip()),
                slot_name,
                slot_term,
                description,
            ]
        )
        return [question for question in questions if not _question_conflicts_with_theme(question, theme_text)]

    def _extract_text_only_grounded_terms(self, grounded_terms: list[GroundedTerm]) -> list[GroundedTerm]:
        text_only_terms: list[GroundedTerm] = []
        for item in grounded_terms:
            text_evidence = _unique_list(
                [
                    *item.candidate.text_evidence,
                    *[document.content.strip() for document in item.documents if document.content.strip()],
                ]
            )
            description = _build_text_only_description(text_evidence, fallback=item.candidate.description)
            text_only_terms.append(
                GroundedTerm(
                    candidate=TermCandidate(
                        term=item.candidate.term,
                        description=description,
                        category_guess=item.candidate.category_guess,
                        visual_evidence=[],
                        text_evidence=text_evidence,
                    ),
                    documents=item.documents,
                    alignment_scores=item.alignment_scores,
                    search_queries=item.search_queries,
                    query_records=item.query_records,
                )
            )
        return text_only_terms

    def _deduplicate_slots(self, slots: list[SlotRecord]) -> tuple[list[SlotRecord], list[str]]:
        if len(slots) < 2:
            return slots, []
        union_find = _UnionFind(len(slots))
        notes: list[str] = []
        for left in range(len(slots)):
            for right in range(left + 1, len(slots)):
                left_text = _slot_signature(slots[left])
                right_text = _slot_signature(slots[right])
                similarity = self._similarity.similarity(left_text, right_text)
                if similarity > self._config.dedup_similarity_threshold:
                    union_find.union(left, right)
                    notes.append(
                        f"merged `{slots[left].slot_term}` with `{slots[right].slot_term}` because similarity={similarity:.4f}"
                    )
        merged_slots = [self._merge_cluster([slots[index] for index in indices]) for indices in union_find.groups().values()]
        return merged_slots, notes

    def _merge_cluster(self, slots: list[SlotRecord]) -> SlotRecord:
        ordered_slots = sorted(slots, key=lambda slot: slot.metadata.confidence, reverse=True)
        representative = ordered_slots[0]
        questions = _unique_list(question for slot in ordered_slots for question in slot.specific_questions)[:3]
        terms = _unique_list(slot.slot_term for slot in ordered_slots)
        sources = ",".join(_unique_list(slot.metadata.source_id for slot in ordered_slots if slot.metadata.source_id))
        merged_extra = dict(representative.metadata.extra)
        merged_extra["candidate_terms"] = _unique_list(
            [
                *[str(item).strip() for item in merged_extra.get("candidate_terms", []) if str(item).strip()],
                *terms,
            ]
        )
        if len(terms) > 1:
            description = f"{representative.description} 合并近义术语：{'、'.join(terms[1:])}。"
        else:
            description = representative.description
        slot_names = Counter(slot.slot_name for slot in ordered_slots if slot.slot_name)
        slot_name = slot_names.most_common(1)[0][0] if slot_names else representative.slot_name
        return SlotRecord(
            slot_name=slot_name,
            slot_term=representative.slot_term,
            description=description,
            specific_questions=questions,
            metadata=SlotMetadata(
                confidence=max(slot.metadata.confidence for slot in ordered_slots),
                source_id=sources or representative.metadata.source_id,
                extra=merged_extra,
            ),
        )

    def _infer_ontology(self, slots: list[SlotRecord]) -> list[OntologyLink]:
        if not slots:
            return []
        payload = self._llm.complete_json(
            system_prompt=ONTOLOGY_PROMPT,
            user_text=json.dumps({"slots": [slot.to_dict() for slot in slots]}, ensure_ascii=False),
        )
        links: list[OntologyLink] = []
        seen: set[tuple[str, str, str]] = set()
        for item in payload.get("relations", []):
            child = str(item.get("child", "")).strip()
            parent = str(item.get("parent", "")).strip()
            relation = str(item.get("relation", "")).strip() or "is-a"
            rationale = str(item.get("rationale", "")).strip()
            key = (child, parent, relation)
            if not child or not parent or child == parent or key in seen:
                continue
            seen.add(key)
            links.append(OntologyLink(child=child, parent=parent, relation=relation, rationale=rationale))
        return links


def _prepare_image_payload(path: Path, *, max_pixels: int) -> tuple[str, bytes, str, tuple[int, int]]:
    with Image.open(path) as image:
        image.load()
        processed = _resize_image_if_needed(image, max_pixels=max_pixels)
        image_format = processed.format or image.format or _infer_pil_format(path)
        mime_type = Image.MIME.get(image_format, mimetypes.guess_type(path.name)[0] or "image/png")
        buffer = io.BytesIO()
        save_image = processed
        if image_format.upper() == "JPEG" and processed.mode not in {"RGB", "L"}:
            save_image = processed.convert("RGB")
        save_image.save(buffer, format=image_format)
        image_bytes = buffer.getvalue()
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return encoded, image_bytes, mime_type, processed.size


def _resize_image_if_needed(image: Image.Image, *, max_pixels: int) -> Image.Image:
    width, height = image.size
    total_pixels = width * height
    if total_pixels <= max_pixels:
        return image.copy()
    scale = math.sqrt(max_pixels / total_pixels)
    resized_width = max(1, int(width * scale))
    resized_height = max(1, int(height * scale))
    resized = image.resize((resized_width, resized_height), Image.Resampling.LANCZOS)
    return resized


def _infer_pil_format(path: Path) -> str:
    mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
    if mime_type == "image/jpeg":
        return "JPEG"
    if mime_type == "image/webp":
        return "WEBP"
    return "PNG"


def _slot_signature(slot: SlotRecord) -> str:
    return " ".join([slot.slot_name, slot.slot_term, slot.description, *slot.specific_questions])


def _unique_list(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique: list[str] = []
    for item in items:
        normalized = item.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        unique.append(normalized)
    return unique


def _build_text_only_description(text_evidence: list[str], *, fallback: str) -> str:
    if not text_evidence:
        return fallback
    merged = " ".join(text_evidence[:2]).strip()
    return merged or fallback


def _question_conflicts_with_theme(question: str, theme_text: str) -> bool:
    normalized_theme = _normalize_question_text(theme_text)
    if not normalized_theme:
        return False

    suspicious_terms: list[str] = []
    suspicious_terms.extend(re.findall(r"《([^》]{2,20})》", question))
    suspicious_terms.extend(re.findall(r"[（(](?:如|例如|比如)?([^）)]{2,20})[）)]", question))
    suspicious_terms.extend(re.findall(r"(?:^|[，。；：（(])(?:例如|比如|如)(?!何)([^，。；！？、]{2,20})", question))
    if "花卉" in question:
        suspicious_terms.append("花卉")

    for term in _unique_list(suspicious_terms):
        normalized_term = _normalize_question_text(term)
        if not normalized_term:
            continue
        if normalized_term in normalized_theme:
            continue
        if len(normalized_term) >= 2:
            return True
    return False


def _normalize_question_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or "")).strip().lower()
