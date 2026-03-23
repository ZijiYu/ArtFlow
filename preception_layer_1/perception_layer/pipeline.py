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
from typing import Any, Iterable

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
from .models import GroundedTerm, OntologyLink, PipelineResult, RagDocument, SlotMetadata, SlotRecord, TermCandidate

PAINTING_PROFILE_PROMPT = """你是中国画作品类型分析器。
请先基于图像和基础文字，判断这幅画大致是什么类型、主题和相关的国画知识背景，输出 JSON：
{
  "painting_type": "画作类型或体裁",
  "subject": "如果知道名字输出名字，不知道则输出：主题或主要对象",
  "scene_summary": "对画面内容的简要概括",
  "guohua_knowledge": ["与该画相关的国画知识点1", "知识点2"],
  "reasoning": "为什么这样判断"
}
要求：
1. 允许基于国画常识做审慎推断，但要保持克制。
2. 输出应服务于后续锚点抽取和 RAG 对齐。
3. `guohua_knowledge` 最多 5 条。"""

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

SLOT_GENERATION_PROMPT = """你是中国画领域感知模块的 Slot 生成器。
请根据术语候选和 RAG 证据输出 JSON：
{
  "slots": [
    {
      "slot_name": "具体领域名",
      "slot_term": "具体术语",
      "description": "整合 RAG 证据后的自然语言描述",
      "specific_questions": ["问题1", "问题2", "问题3"],
      "metadata": {
        "confidence": 0.0,
        "source_id": "来源索引"
      }
    }
  ]
}
要求：
1. 每个 Slot 必须有领域名称、针对性描述、2-3 个深度鉴赏问题。
2. 描述必须明确引用证据中的专业内容，但不要逐字复制大段原文。
3. `confidence` 取 0 到 1 之间的小数。
4. `source_id` 来自对应 RAG 证据。
5. `specific_questions` 必须紧扣当前画作主题、对象和场景，不要把 RAG 文本里举的其他作品、花卉、人物或题材直接搬成当前画作的问题。
6. 如果 RAG 证据里出现了“如某花卉”“如某他作”之类的例子，只能提炼其中的通用技法知识，不能把这些例子本身写进当前画作的问题。"""

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
        with self._path.open("a", encoding="utf-8") as handle:
            handle.writelines(lines)


class RagSearchLogger:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if not self._path.exists():
            self._path.write_text("# RAG Search Record\n\n", encoding="utf-8")

    def append(
        self,
        *,
        image_path: Path,
        grounded_terms: list[GroundedTerm],
    ) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [f"\n## Search Batch [{timestamp}]\n", f"- image: `{image_path}`\n"]
        if not grounded_terms:
            lines.append("- no grounded terms\n")
        for item in grounded_terms:
            lines.append(f"- query_text: `{item.candidate.term}`\n")
            lines.append(f"  - image_attached: `true`\n")
            lines.append(f"  - sources: `{','.join(doc.source_id for doc in item.documents) or 'none'}`\n")
            lines.append(
                f"  - alignment_scores: `{','.join(str(score) for score in item.alignment_scores) or 'none'}`\n"
            )
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
        similarity_backend: SimilarityBackend | None = None,
        client: Any | None = None,
    ) -> None:
        self._config = config or PipelineConfig.from_env()
        base_llm = llm_client or OpenAIJsonClient(self._config, client=client)
        self._llm_chat_logger = LlmChatLogger(self._config.llm_chat_record_path)
        self._llm = RecordingLLMClient(base_llm, self._llm_chat_logger)
        self._rag = rag_client or HttpRagClient(self._config.rag_endpoint)
        self._similarity = similarity_backend or OpenAIEmbeddingSimilarityBackend(self._config, client=client)
        self._context = ContextLogger(self._config.context_path)
        self._rag_search_record = RagSearchLogger(self._config.rag_search_record_path)

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

        self._context.append(
            "Run Started",
            [
                f"image=`{image_path}`",
                f"image_size={image_size[0]}x{image_size[1]}",
                f"text={input_text[:160]}",
            ],
        )

        stage_started_at = perf_counter()
        painting_profile = await asyncio.to_thread(
            self._analyze_painting_profile,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
        )
        stage_timings["painting_profile"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("painting_profile", stage_timings["painting_profile"])
        self._context.append(
            "Painting Profile",
            [json.dumps(painting_profile, ensure_ascii=False)],
        )

        stage_started_at = perf_counter()
        visual_task = asyncio.create_task(
            asyncio.to_thread(
                self._extract_visual_cues,
                image_base64=image_base64,
                image_mime_type=image_mime_type,
                input_text=input_text,
            )
        )
        text_task = asyncio.create_task(asyncio.to_thread(self._analyze_text_signals, input_text))
        visual_cues, text_signals = await asyncio.gather(visual_task, text_task)
        stage_timings["parallel_extraction"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("parallel_extraction", stage_timings["parallel_extraction"])
        self._context.append(
            "Step 1 Parallel Extraction",
            [f"visual_cues={json.dumps(visual_cues, ensure_ascii=False)}", f"text_signals={json.dumps(text_signals, ensure_ascii=False)}"],
        )

        stage_started_at = perf_counter()
        candidates = await asyncio.to_thread(
            self._anchor_candidates,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
            input_text=input_text,
            painting_profile=painting_profile,
            visual_cues=visual_cues,
            text_signals=text_signals,
        )
        stage_timings["anchor_candidates"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("anchor_candidates", stage_timings["anchor_candidates"])
        self._context.append(
            "Anchored Candidates",
            [json.dumps(asdict(candidate), ensure_ascii=False) for candidate in candidates] or ["no candidates"],
        )

        stage_started_at = perf_counter()
        grounded_terms = await self._ground_candidates(
            candidates,
            image_bytes=image_bytes,
            image_filename=image_path.name,
            image_mime_type=image_mime_type,
            visual_cues=visual_cues,
        )
        stage_timings["rag_grounding"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("rag_grounding", stage_timings["rag_grounding"])
        self._context.append(
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
        self._context.append(
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
        slots = await asyncio.to_thread(self._generate_slots, grounded_terms, painting_profile)
        stage_timings["generate_slots"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("generate_slots", stage_timings["generate_slots"])
        stage_started_at = perf_counter()
        deduped_slots, merge_notes = await asyncio.to_thread(self._deduplicate_slots, slots)
        stage_timings["semantic_dedup"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("semantic_dedup", stage_timings["semantic_dedup"])
        self._context.append("Semantic Dedup", merge_notes or ["no merges"])

        stage_started_at = perf_counter()
        ontology_links = await asyncio.to_thread(self._infer_ontology, deduped_slots)
        stage_timings["infer_ontology"] = round(perf_counter() - stage_started_at, 4)
        self._print_stage_timing("infer_ontology", stage_timings["infer_ontology"])
        self._context.append(
            "Ontology Updates",
            [link.to_markdown() for link in ontology_links] or ["no ontology relations inferred"],
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
        self._context.append("Run Finished", [f"output=`{result.output_path}`", f"slot_count={len(result.slots)}"])
        total_elapsed = round(perf_counter() - run_started_at, 4)
        slowest_stage, slowest_elapsed = self._slowest_stage(stage_timings)
        self._context.append(
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
    def _slowest_stage(stage_timings: dict[str, float]) -> tuple[str, float]:
        if not stage_timings:
            return "none", 0.0
        return max(stage_timings.items(), key=lambda item: item[1])

    def _extract_visual_cues(self, *, image_base64: str, image_mime_type: str, input_text: str) -> list[str]:
        payload = self._llm.complete_json(
            system_prompt=VISUAL_CUE_PROMPT,
            user_text=f"请提取视觉线索。补充文字仅用于限制主题，不可作为视觉事实：{input_text}",
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        return [str(item).strip() for item in payload.get("visual_cues", []) if str(item).strip()]

    def _analyze_painting_profile(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
    ) -> dict[str, object]:
        payload = self._llm.complete_json(
            system_prompt=PAINTING_PROFILE_PROMPT,
            user_text=f"请先判断这幅图像是什么图像，并结合基础文字补充分析：{input_text}，如果无法识别该图像出处，则不回答。",
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        return {
            "painting_type": str(payload.get("painting_type", "")).strip(),
            "subject": str(payload.get("subject", "")).strip(),
            "scene_summary": str(payload.get("scene_summary", "")).strip(),
            "guohua_knowledge": [
                str(item).strip() for item in payload.get("guohua_knowledge", []) if str(item).strip()
            ],
            "reasoning": str(payload.get("reasoning", "")).strip(),
        }

    def _analyze_text_signals(self, input_text: str) -> dict[str, list[str]]:
        payload = self._llm.complete_json(
            system_prompt=TEXT_SIGNAL_PROMPT,
            user_text=f"请分析以下文字：\n{input_text}",
        )
        return {
            "text_signals": [str(item).strip() for item in payload.get("text_signals", []) if str(item).strip()],
            "salient_entities": [str(item).strip() for item in payload.get("salient_entities", []) if str(item).strip()],
        }

    def _anchor_candidates(
        self,
        *,
        image_base64: str,
        image_mime_type: str,
        input_text: str,
        painting_profile: dict[str, object],
        visual_cues: list[str],
        text_signals: dict[str, list[str]],
    ) -> list[TermCandidate]:
        payload = self._llm.complete_json(
            system_prompt=ANCHORING_PROMPT,
            user_text=(
                "请抽取候选术语。\n"
                f"基础文字：{input_text}\n"
                f"画作类型分析：{json.dumps(painting_profile, ensure_ascii=False)}\n"
                f"视觉线索：{json.dumps(visual_cues, ensure_ascii=False)}\n"
                f"文本分析：{json.dumps(text_signals, ensure_ascii=False)}"
            ),
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )
        candidates: list[TermCandidate] = []
        for item in payload.get("candidates", []):
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

    async def _ground_candidates(
        self,
        candidates: list[TermCandidate],
        *,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
        visual_cues: list[str],
    ) -> list[GroundedTerm]:
        tasks = [
            asyncio.create_task(
                asyncio.to_thread(
                    self._ground_single_candidate,
                    candidate,
                    image_bytes,
                    image_filename,
                    image_mime_type,
                    visual_cues,
                )
            )
            for candidate in candidates
        ]
        grounded = await asyncio.gather(*tasks)
        return [item for item in grounded if item.documents]

    def _ground_single_candidate(
        self,
        candidate: TermCandidate,
        image_bytes: bytes,
        image_filename: str,
        image_mime_type: str,
        visual_cues: list[str],
    ) -> GroundedTerm:
        documents = self._rag.search(
            query_text=candidate.term,
            query_image_bytes=image_bytes,
            query_image_filename=image_filename,
            query_image_mime_type=image_mime_type,
            top_k=self._config.rag_top_k,
        )
        anchor_text = " ".join([candidate.term, candidate.description, candidate.category_guess, *visual_cues, *candidate.visual_evidence])
        filtered_docs: list[RagDocument] = []
        similarity_scores: list[float] = []
        for document in documents:
            score = self._similarity.similarity(anchor_text, document.content)
            if score >= self._config.rag_similarity_threshold:
                filtered_docs.append(document)
                similarity_scores.append(round(score, 4))
        if not filtered_docs and documents:
            filtered_docs.append(documents[0])
            similarity_scores.append(round(self._similarity.similarity(anchor_text, documents[0].content), 4))
        return GroundedTerm(candidate=candidate, documents=filtered_docs, alignment_scores=similarity_scores)

    def _generate_slots(self, grounded_terms: list[GroundedTerm], painting_profile: dict[str, object]) -> list[SlotRecord]:
        if not grounded_terms:
            return []
        payload = self._llm.complete_json(
            system_prompt=SLOT_GENERATION_PROMPT,
            user_text=json.dumps(
                {
                    "painting_profile": {
                        "painting_type": str(painting_profile.get("painting_type", "")).strip(),
                        "subject": str(painting_profile.get("subject", "")).strip(),
                        "scene_summary": str(painting_profile.get("scene_summary", "")).strip(),
                        "guohua_knowledge": painting_profile.get("guohua_knowledge", []),
                    },
                    "grounded_terms": [
                        {
                            "term": item.candidate.term,
                            "description": item.candidate.description,
                            "category_guess": item.candidate.category_guess,
                            "evidence": [
                                {
                                    "source_id": doc.source_id,
                                    "content": doc.content,
                                    "score": doc.score,
                                    "alignment_score": score,
                                }
                                for doc, score in zip(item.documents, item.alignment_scores)
                            ],
                        }
                        for item in grounded_terms
                    ]
                },
                ensure_ascii=False,
            ),
        )
        slots: list[SlotRecord] = []
        for item in payload.get("slots", []):
            questions = [str(question).strip() for question in item.get("specific_questions", []) if str(question).strip()]
            questions = self._filter_questions_by_theme(
                questions,
                painting_profile=painting_profile,
                slot_name=str(item.get("slot_name", "")).strip(),
                slot_term=str(item.get("slot_term", "")).strip(),
                description=str(item.get("description", "")).strip(),
            )
            if len(questions) < 2:
                continue
            metadata = item.get("metadata", {})
            slots.append(
                SlotRecord(
                    slot_name=str(item.get("slot_name", "")).strip(),
                    slot_term=str(item.get("slot_term", "")).strip(),
                    description=str(item.get("description", "")).strip(),
                    specific_questions=questions[:3],
                    metadata=SlotMetadata(
                        confidence=float(metadata.get("confidence", 0.0)),
                        source_id=str(metadata.get("source_id", "")).strip(),
                    ),
                )
            )
        return [slot for slot in slots if slot.slot_name and slot.slot_term and slot.description]

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
                str(painting_profile.get("painting_type", "")).strip(),
                str(painting_profile.get("subject", "")).strip(),
                str(painting_profile.get("scene_summary", "")).strip(),
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
