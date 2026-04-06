from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from io import BytesIO
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError

from PIL import Image

from perception_layer.cli import (
    _TeeStream,
    build_config_from_args,
    build_parser,
    resolve_terminal_log_path,
    tee_terminal_output,
)
from perception_layer.clients import HttpRagClient, LlmChatLogger, OpenAIEmbeddingSimilarityBackend
from perception_layer.config import PipelineConfig
from perception_layer.downstream import DownstreamPromptRunner
from perception_layer.models import RagDocument, TermCandidate
from perception_layer.pipeline import PerceptionPipeline, _prepare_image_payload


class FakeLLM:
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict:
        assert image_mime_type in {"image/png", "image/jpeg"}
        if "视觉与技法综合分析器" in system_prompt:
            assert image_base64
            return {
                "visual_cues": ["山石边缘呈折线皴擦", "右上角可见题跋", "画面有绢本设色质感"],
                "mountain_present": True,
                "mountain_evidence": ["画面中央为巨大的山石主体"],
                "cunfa_candidates": [
                    {
                        "term": "斧劈皴",
                        "reason": "山石边缘呈折线皴擦，更接近斧劈皴。",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "confidence": 0.92,
                    }
                ],
                "miaofa_candidates": [],
                "reasoning": "画面存在明确山体，因此可先判断皴法。",
            }
        if "文本信号与领域锚定综合分析器" in system_prompt:
            payload = json.loads(user_text)
            assert payload["painting_profile"]["painting_type"] == "山水画"
            assert "山石边缘呈折线皴擦" in payload["visual_cues"]
            return {
                "text_signals": ["文本提到绢本设色", "文字疑似提及斧劈皴"],
                "salient_entities": ["绢本", "斧劈皴"],
                "candidates": [
                    {
                        "term": "斧劈皴",
                        "description": "山石纹理表现与斧劈皴相符。",
                        "category_guess": "皴法",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "text_evidence": ["文字疑似提及斧劈皴"],
                    },
                    {
                        "term": "绢本",
                        "description": "文本明确指出材质为绢本。",
                        "category_guess": "材质",
                        "visual_evidence": ["画面有绢本设色质感"],
                        "text_evidence": ["文本提到绢本设色"],
                    },
                    {
                        "term": "斧劈皴法",
                        "description": "另一条相近描述，后续应与斧劈皴合并。",
                        "category_guess": "皴法",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "text_evidence": ["文字疑似提及斧劈皴"],
                    },
                ],
            }
        if "固定分析槽位规划器" in system_prompt:
            payload = json.loads(user_text)
            slot_def = payload["slot_definition"]
            slot_name = slot_def["slot_name"]
            painting_profile = payload.get("painting_profile", {})
            painting_name = str(painting_profile.get("name", "")).strip() or "溪岸山石图"
            author = str(painting_profile.get("author", "")).strip() or "佚名"
            dynasty = str(painting_profile.get("dynasty", "")).strip() or "宋"
            painting_type = str(painting_profile.get("painting_type", "")).strip() or "山水画"
            if slot_name == "画作背景":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": painting_name,
                    "description": "作品名可作为后续分析图像主干与观看入口的背景起点。",
                    "specific_questions": [
                        f"《{painting_name}》这一画名如何概括画面主体与空间重心？",
                        "画名中的核心对象与图中主要物象、构图关系如何互相印证？",
                    ],
                    "metadata": {"confidence": 0.93, "source_id": "doc-metadata"},
                    "pending_terms": ["山石与溪岸"],
                    "reasoning": "作品名称在 metadata 中明确出现，且与图像主体一致。",
                }
            if slot_name == "作者时代流派":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": author,
                    "description": "作者、时代与流派信息共同构成作品背景判断的核心框架。",
                    "specific_questions": [
                        f"{author}这一作者信息如何影响我们对作品风格来源的判断？",
                        f"{dynasty}与{painting_type}相关的时代或流派线索能为这幅画提供哪些背景约束？",
                    ],
                    "metadata": {"confidence": 0.88, "source_id": "doc-metadata"},
                    "pending_terms": [dynasty, painting_type],
                    "reasoning": "作者字段是当前最关键的背景信息入口，可与时代和流派并行展开。",
                }
            if slot_name == "墨法设色技法":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "斧劈皴",
                    "description": "RAG 证据表明斧劈皴常用于表现山石折转与硬朗纹理，适合作为当前技法槽位的起点。",
                    "specific_questions": [
                        "画家如何利用斧劈皴强化山石的结构感？",
                        "斧劈皴与整幅构图的气势有什么关系？",
                    ],
                    "metadata": {"confidence": 0.92, "source_id": "doc-technique"},
                    "pending_terms": ["绢本"],
                    "reasoning": "斧劈皴在候选术语和证据中都最稳定，且与图像特征直接对应。",
                }
            if slot_name == "题跋诗文审美语言":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "题跋",
                    "description": "图中右上角题跋可作为文字识读与作品阅读方式的重要入口。",
                    "specific_questions": [
                        "右上角题跋的文字内容如何帮助识读这幅画？",
                        "题跋中是否包含题名、落款或其他可辨识文字信息？",
                    ],
                    "metadata": {"confidence": 0.86, "source_id": ""},
                    "pending_terms": ["落款"],
                    "reasoning": "图像中确有题跋可见，适合先从文字内容入口展开。",
                }
            if slot_name == "题跋/印章/用笔":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "题跋",
                    "description": "题跋与用笔信息可作为识读作品和观察线质的重要入口。",
                    "specific_questions": [
                        "题跋与相关笔触线索如何帮助确认这幅画的阅读顺序？",
                        "题跋和用笔特征之间是否形成了相互印证？",
                    ],
                    "metadata": {"confidence": 0.84, "source_id": "doc-technique"},
                    "pending_terms": ["用笔"],
                    "reasoning": "视觉线索中已有题跋可见，同时知识背景提示可关注用笔。",
                }
            if slot_name == "构图/空间/布局":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "构图",
                    "description": "构图可作为分析山石与溪岸空间组织方式的起点。",
                    "specific_questions": [
                        "构图如何组织山石与溪岸的空间层次？",
                        "构图如何引导观者的视线在画面中移动？",
                    ],
                    "metadata": {"confidence": 0.8, "source_id": "doc-metadata"},
                    "pending_terms": ["山石与溪岸"],
                    "reasoning": "画作知识背景明确提示可从构图与空间经营展开。",
                }
            if slot_name == "尺寸规格/材质形制/收藏地":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "绢本",
                    "description": "绢本材质直接影响设色层次与观看质感，可作为作品形制判断入口。",
                    "specific_questions": [
                        "绢本材质如何影响这幅画的设色层次与质感？",
                        "绢本信息对作品的保存与观看方式意味着什么？",
                    ],
                    "metadata": {"confidence": 0.9, "source_id": "doc-material"},
                    "pending_terms": ["设色"],
                    "reasoning": "文本与 RAG 证据都明确提到了绢本设色。",
                }
            if slot_name == "意境/题材/象征":
                return {
                    "applicable": True,
                    "slot_name": slot_name,
                    "slot_term": "山水画",
                    "description": "山水画题材为后续讨论意境营造和象征表达提供了稳定起点。",
                    "specific_questions": [
                        "山水画题材如何塑造这幅画的整体意境？",
                        "山水画题材与山石、溪岸等物象之间形成了怎样的象征关系？",
                    ],
                    "metadata": {"confidence": 0.81, "source_id": "doc-metadata"},
                    "pending_terms": ["山石与溪岸"],
                    "reasoning": "题材信息稳定，适合作为意境与象征分析的上层入口。",
                }
            raise AssertionError(f"Unexpected fixed slot: {slot_name}")
        if "作品主干信息提取器" in system_prompt:
            return {
                "name": "溪岸山石图",
                "author": "佚名",
                "dynasty": "宋",
                "related_background": ["与宋代山水画中的山石表现有关"],
                "reasoning": "metadata 明确给出了作品与时代信息。",
            }
        if "作品主干核对器" in system_prompt:
            assert image_base64
            return {
                "is_target_image": True,
                "suggested_name": "溪岸山石图",
                "suggested_author": "佚名",
                "suggested_dynasty": "宋",
                "reasoning": "候选主干与图像内容一致。",
            }
        if "中国画作品类型分析器" in system_prompt:
            assert image_base64
            return {
                "name": "溪岸山石图",
                "author": "佚名",
                "dynasty": "宋",
                "painting_type": "山水画",
                "subject": "山石与溪岸",
                "scene_summary": "画面以山石结构为主，兼见题跋和绢本设色质感。",
                "related_background": ["宋代山水画重视山石结构经营"],
                "guohua_knowledge": ["山水画常关注皴法、构图和材质", "题跋与绢本信息有助于术语抽取"],
                "reasoning": "图像中山石结构明显，文字提到绢本设色。",
            }
        if "RAG 证据与作品主干关联审核器" in system_prompt or "RAG 关联性裁判" in system_prompt:
            return {"related_indices": [0]}
        if "视觉锚点抽取器" in system_prompt:
            assert image_base64
            return {"visual_cues": ["山石边缘呈折线皴擦", "右上角可见题跋", "画面有绢本设色质感"]}
        if "皴法与描法判别器" in system_prompt:
            assert image_base64
            return {
                "mountain_present": True,
                "mountain_evidence": ["画面中央为巨大的山石主体"],
                "cunfa_candidates": [
                    {
                        "term": "斧劈皴",
                        "reason": "山石边缘呈折线皴擦，更接近斧劈皴。",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "confidence": 0.92,
                    }
                ],
                "miaofa_candidates": [],
                "reasoning": "画面存在明确山体，因此可先判断皴法。",
            }
        if "文本信号分析器" in system_prompt:
            return {"text_signals": ["文本提到绢本设色", "文字疑似提及斧劈皴"], "salient_entities": ["绢本", "斧劈皴"]}
        if "领域锚定器" in system_prompt:
            return {
                "candidates": [
                    {
                        "term": "斧劈皴",
                        "description": "山石纹理表现与斧劈皴相符。",
                        "category_guess": "皴法",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "text_evidence": ["文字疑似提及斧劈皴"],
                    },
                    {
                        "term": "绢本",
                        "description": "文本明确指出材质为绢本。",
                        "category_guess": "材质",
                        "visual_evidence": ["画面有绢本设色质感"],
                        "text_evidence": ["文本提到绢本设色"],
                    },
                    {
                        "term": "斧劈皴法",
                        "description": "另一条相近描述，后续应与斧劈皴合并。",
                        "category_guess": "皴法",
                        "visual_evidence": ["山石边缘呈折线皴擦"],
                        "text_evidence": ["文字疑似提及斧劈皴"],
                    },
                ]
            }
        if "动态本体推理器" in system_prompt:
            return {
                "relations": [
                    {
                        "child": "斧劈皴",
                        "parent": "墨法设色技法",
                        "relation": "is-a",
                        "rationale": "斧劈皴属于皴法中的具体类型。",
                    },
                    {
                        "child": "溪岸山石图",
                        "parent": "画作背景",
                        "relation": "is-a",
                        "rationale": "作品名属于画作背景槽位中的主干术语。",
                    },
                ]
            }
        if "下游测试助手" in system_prompt:
            return {"status": "ok", "task": user_text}
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class FakeRag:
    def __init__(self) -> None:
        self.collection_names: list[str | None] = []

    def clone(self) -> "FakeRag":
        cloned = FakeRag()
        cloned.collection_names = self.collection_names
        return cloned

    def search(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        query_image_filename: str | None,
        query_image_mime_type: str | None,
        top_k: int,
        collection_name: str | None = None,
    ) -> list[RagDocument]:
        self.collection_names.append(collection_name)
        assert query_image_bytes
        assert query_image_filename == "sample.png"
        assert query_image_mime_type == "image/png"
        assert top_k == 5
        if query_text is None:
            return [
                RagDocument(
                    source_id="doc-metadata",
                    content="作品名《溪岸山石图》，作者佚名，宋代作品。",
                    metadata={"book_name": "metadata"},
                )
            ]
        if query_text and "溪岸山石图" in query_text:
            return [
                RagDocument(
                    source_id="doc-technique",
                    content="《溪岸山石图》山石折转处可见斧劈皴，线面关系硬朗。",
                ),
                RagDocument(
                    source_id="doc-material",
                    content="《溪岸山石图》为绢本设色，材质细密，影响设色层次和观感。",
                ),
                RagDocument(source_id="doc-low", content="这是无关的馆藏说明。"),
            ]
        return []


class FakeSimilarity:
    def similarity(self, left: str, right: str) -> float:
        if {"斧劈皴", "斧劈皴法"} <= {left, right}:
            return 0.91
        if "无关的馆藏说明" in {left, right}:
            return 0.1
        if any(keyword in left for keyword in ("斧劈皴", "斧劈皴法")) and any(
            keyword in right for keyword in ("斧劈皴", "折转", "折线皴擦")
        ):
            return 0.78
        if "绢本" in left and any(keyword in right for keyword in ("材质细密", "绢本设色", "设色层次")):
            return 0.76
        if "国画技法" in left and "国画技法" in right:
            return 0.91
        return 0.4


class CountingRagFactory:
    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> FakeRag:
        self.calls += 1
        return FakeRag()


class MetadataMismatchLLM(FakeLLM):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict:
        if "作品主干信息提取器" in system_prompt and "错误图名" in user_text:
            return {
                "name": "错误图名",
                "author": "佚名",
                "dynasty": "宋",
                "related_background": [],
                "reasoning": "首条候选写了错误图名。",
            }
        if "作品主干信息提取器" in system_prompt and "洛神赋图" in user_text:
            return {
                "name": "洛神赋图",
                "author": "顾恺之",
                "dynasty": "东晋",
                "related_background": ["与顾恺之人物画传统直接相关"],
                "reasoning": "metadata 文本明确给出了作品名称、作者和朝代。",
            }
        if "作品主干核对器" in system_prompt:
            return {
                "is_target_image": False,
                "suggested_name": "洛神赋图",
                "suggested_author": "顾恺之",
                "suggested_dynasty": "东晋",
                "reasoning": "当前候选名称与图像不符，图像更接近《洛神赋图》。",
            }
        if "中国画作品类型分析器" in system_prompt:
            return {
                "name": "洛神赋图",
                "author": "顾恺之",
                "dynasty": "东晋",
                "painting_type": "人物画",
                "subject": "洛神与侍从",
                "scene_summary": "人物列队，兼见山石竹木。",
                "related_background": ["顾恺之人物画重传神写照"],
                "guohua_knowledge": ["人物画常结合设色与线描判断身份"],
                "reasoning": "图像与《洛神赋图》主干信息更匹配。",
            }
        if "文本信号与领域锚定综合分析器" in system_prompt:
            return {
                "text_signals": ["请分析图像主体"],
                "salient_entities": ["图像主体"],
                "candidates": [
                    {
                        "term": "传神写照",
                        "description": "人物神态刻画与传神写照传统相关。",
                        "category_guess": "人物画技法",
                        "visual_evidence": ["人物面部神态细腻"],
                        "text_evidence": [],
                    },
                    {
                        "term": "洛神赋图",
                        "description": "图像主体与《洛神赋图》题材线索高度相关。",
                        "category_guess": "题材",
                        "visual_evidence": ["人物列队，兼见山石竹木"],
                        "text_evidence": [],
                    },
                ],
            }
        if "领域锚定器" in system_prompt:
            return {
                "candidates": [
                    {
                        "term": "传神写照",
                        "description": "人物神态刻画与传神写照传统相关。",
                        "category_guess": "人物画技法",
                        "visual_evidence": ["人物面部神态细腻"],
                        "text_evidence": [],
                    },
                    {
                        "term": "洛神赋图",
                        "description": "图像主体与《洛神赋图》题材线索高度相关。",
                        "category_guess": "题材",
                        "visual_evidence": ["人物列队，兼见山石竹木"],
                        "text_evidence": [],
                    },
                ]
            }
        return super().complete_json(
            system_prompt=system_prompt,
            user_text=user_text,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )


class MetadataFallbackRag(FakeRag):
    def __init__(self) -> None:
        super().__init__()
        self.text_queries: list[str] = []

    def clone(self) -> "MetadataFallbackRag":
        cloned = MetadataFallbackRag()
        cloned.text_queries = self.text_queries
        cloned.collection_names = self.collection_names
        return cloned

    def search(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        query_image_filename: str | None,
        query_image_mime_type: str | None,
        top_k: int,
        collection_name: str | None = None,
    ) -> list[RagDocument]:
        self.collection_names.append(collection_name)
        if query_text is None:
            return [
                RagDocument(
                    source_id="doc-wrong",
                    content="作品名《错误图名》，作者佚名，宋代作品。",
                    metadata={"book_name": "metadata"},
                )
            ]
        self.text_queries.append(query_text)
        if query_text == "洛神赋图":
            return [
                RagDocument(
                    source_id="doc-luoshen-metadata",
                    content="东晋·顾恺之《洛神赋图》，描绘洛神与侍从，兼见山石竹木。",
                    metadata={"book_name": "metadata"},
                ),
                RagDocument(
                    source_id="doc-luoshen-generic",
                    content="《洛神赋图》相关介绍。",
                    metadata={"book_name": "图录"},
                ),
            ]
        return super().search(
            query_text=query_text,
            query_image_bytes=query_image_bytes,
            query_image_filename=query_image_filename,
            query_image_mime_type=query_image_mime_type,
            top_k=top_k,
            collection_name=collection_name,
        )


class NoMountainTechniqueLLM(FakeLLM):
    def complete_json(
        self,
        *,
        system_prompt: str,
        user_text: str,
        image_base64: str | None = None,
        image_mime_type: str = "image/png",
    ) -> dict:
        if "视觉与技法综合分析器" in system_prompt:
            return {
                "visual_cues": ["衣纹细劲匀整"],
                "mountain_present": False,
                "mountain_evidence": [],
                "cunfa_candidates": [
                    {
                        "term": "雨点皴",
                        "reason": "误报",
                        "visual_evidence": ["人物衣纹短点"],
                        "confidence": 0.7,
                    }
                ],
                "miaofa_candidates": [
                    {
                        "term": "铁线描",
                        "reason": "人物衣纹线条均匀挺劲。",
                        "visual_evidence": ["衣纹细劲匀整"],
                        "confidence": 0.8,
                    }
                ],
                "reasoning": "画面没有山体。",
            }
        if "皴法与描法判别器" in system_prompt:
            return {
                "mountain_present": False,
                "mountain_evidence": [],
                "cunfa_candidates": [
                    {
                        "term": "雨点皴",
                        "reason": "误报",
                        "visual_evidence": ["人物衣纹短点"],
                        "confidence": 0.7,
                    }
                ],
                "miaofa_candidates": [
                    {
                        "term": "铁线描",
                        "reason": "人物衣纹线条均匀挺劲。",
                        "visual_evidence": ["衣纹细劲匀整"],
                        "confidence": 0.8,
                    }
                ],
                "reasoning": "画面没有山体。",
            }
        return super().complete_json(
            system_prompt=system_prompt,
            user_text=user_text,
            image_base64=image_base64,
            image_mime_type=image_mime_type,
        )


def _build_config(root: Path) -> PipelineConfig:
    return PipelineConfig(
        api_key="test-key",
        context_path=root / "context.md",
        rag_search_record_path=root / "rag_search_record.md",
        llm_chat_record_path=root / "llm_chat_record.jsonl",
        output_path=root / "artifacts" / "slots.jsonl",
    )


def test_pipeline_generates_grounded_slots_and_context() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")

        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        result = asyncio.run(pipeline.run(image_file=image_path, input_text="画面疑似斧劈皴，且文本提到绢本设色。"))

        assert len(result.slots) == 8
        assert {slot.slot_name for slot in result.slots} == {
            "画作背景",
            "作者时代流派",
            "墨法设色技法",
            "题跋诗文审美语言",
            "题跋/印章/用笔",
            "构图/空间/布局",
            "尺寸规格/材质形制/收藏地",
            "意境/题材/象征",
        }
        assert {slot.slot_term for slot in result.slots if slot.slot_term} == {
            "溪岸山石图",
            "佚名",
            "斧劈皴",
            "题跋",
            "构图",
            "绢本",
            "山水画",
        }
        assert all(not item.candidate.visual_evidence for item in result.grounded_terms)
        assert any("斧劈皴" in evidence for evidence in result.grounded_terms[0].candidate.text_evidence)
        technique_slot = next(slot for slot in result.slots if slot.slot_term == "斧劈皴")
        assert all("紫蝶牡丹" not in question for question in technique_slot.specific_questions)
        assert technique_slot.metadata.source_id == "doc-technique"
        assert technique_slot.metadata.extra["slot_mode"] == "progressive"
        assert technique_slot.metadata.extra["used_terms"] == ["斧劈皴"]
        assert result.output_path.exists()
        assert result.rag_search_record_path.exists()
        assert result.llm_chat_record_path.exists()
        lines = result.output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 8
        context_text = result.context_path.read_text(encoding="utf-8")
        rag_search_text = result.rag_search_record_path.read_text(encoding="utf-8")
        llm_chat_text = result.llm_chat_record_path.read_text(encoding="utf-8")
        assert "Ontology Updates" not in context_text
        assert "Post-RAG Text Extraction" in context_text
        assert "斧劈皴" in context_text
        assert "Painting Profile" not in context_text
        assert "Timing Summary" not in context_text
        assert "candidate_term: `斧劈皴`" in rag_search_text
        assert "search_queries: `溪岸山石图 佚名`, `溪岸山石图 宋`, `溪岸山石图`" in rag_search_text
        assert "matched_count=`1`" in rag_search_text
        assert "duration_ms=`" in rag_search_text
        assert "query_text: `斧劈皴`" not in rag_search_text
        assert "doc-technique" in rag_search_text
        assert "doc-low" not in rag_search_text
        assert all(item.search_queries[0].startswith("溪岸山石图") for item in result.grounded_terms if item.search_queries)
        technique_grounded = next(item for item in result.grounded_terms if item.candidate.term == "斧劈皴")
        assert [doc.source_id for doc in technique_grounded.documents] == ["doc-technique"]
        assert len(technique_grounded.query_records) == 3
        assert all(record.initial_top_k_count == 3 for record in technique_grounded.query_records)
        assert all(record.matched_count == 1 for record in technique_grounded.query_records)
        assert '"system_prompt"' in llm_chat_text
        assert "中国画作品类型分析器" in llm_chat_text
        assert "视觉与技法综合分析器" in llm_chat_text
        assert "文本信号与领域锚定综合分析器" in llm_chat_text
        assert "视觉锚点抽取器" not in llm_chat_text
        assert "皴法与描法判别器" not in llm_chat_text
        assert "文本信号分析器" not in llm_chat_text
        assert "领域锚定器" not in llm_chat_text
        assert "动态本体推理器" not in llm_chat_text


def test_pipeline_can_optionally_enable_ontology_inference() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")
        config = _build_config(root)
        config.enable_ontology_inference = True

        pipeline = PerceptionPipeline(
            config,
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        result = asyncio.run(pipeline.run(image_file=image_path, input_text="画面疑似斧劈皴，且文本提到绢本设色。"))

        context_text = result.context_path.read_text(encoding="utf-8")
        llm_chat_text = result.llm_chat_record_path.read_text(encoding="utf-8")

        assert "Ontology Updates" in context_text
        assert result.ontology_links
        assert "动态本体推理器" in llm_chat_text


def test_ground_single_candidate_filters_top5_docs_by_term_and_records_duration() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )
        candidate = TermCandidate(
            term="斧劈皴",
            description="山石纹理表现与斧劈皴相符。",
            category_guess="皴法",
            visual_evidence=["山石边缘呈折线皴擦"],
            text_evidence=["文字疑似提及斧劈皴"],
        )
        documents = [
            RagDocument(source_id="doc-hit", content="《溪岸山石图》山石折转处可见斧劈皴。"),
            RagDocument(source_id="doc-miss", content="《溪岸山石图》为绢本设色。"),
            RagDocument(source_id="doc-miss-2", content="《溪岸山石图》馆藏信息。"),
        ]

        with patch.object(
            pipeline,
            "_search_grounding_documents_with_stats",
            return_value=(documents, 12.34),
        ):
            grounded = pipeline._ground_single_candidate(
                candidate,
                b"fake-image",
                "sample.png",
                "image/png",
                ["山石边缘呈折线皴擦"],
                {},
            )

        assert [doc.source_id for doc in grounded.documents] == ["doc-hit"]
        assert grounded.query_records[0].query_text == "斧劈皴"
        assert grounded.query_records[0].duration_ms == 12.34
        assert grounded.query_records[0].initial_top_k_count == 3
        assert grounded.query_records[0].matched_count == 1


def test_inscription_slot_candidates_ignore_aesthetic_keywords_without_textual_evidence() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        candidates = pipeline._build_inscription_slot_candidates(
            grounded_terms=[],
            painting_profile={
                "scene_summary": "画面气息清逸，整体肃穆庄重。",
                "related_background": ["常见于典雅题材。"],
            },
            visual_cues=["右上角可见题跋", "画面有绢本设色质感"],
        )

        assert [item["term"] for item in candidates] == ["题跋"]


def test_fixed_slot_selection_uses_slot_scoped_painting_profile() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        profile = {
            "name": "十六罗汉图第十一尊者",
            "author": "金大受",
            "dynasty": "南宋",
            "painting_type": "道释人物画",
            "subject": "罗汉与侍者",
            "scene_summary": "主尊坐于山石座具上，侍者在侧。",
            "related_background": ["与南宋罗汉题材传播有关"],
            "guohua_knowledge": ["设色与描法可作为技法分析入口"],
        }

        technique_profile = pipeline._slot_scoped_painting_profile(
            spec={"slot_name": "墨法设色技法", "slot_mode": "progressive", "goal": "测试"},
            painting_profile=profile,
        )
        background_profile = pipeline._slot_scoped_painting_profile(
            spec={"slot_name": "画作背景", "slot_mode": "progressive", "goal": "测试"},
            painting_profile=profile,
        )
        material_profile = pipeline._slot_scoped_painting_profile(
            spec={"slot_name": "尺寸规格/材质形制/收藏地", "slot_mode": "enumerative", "goal": "测试"},
            painting_profile=profile,
        )

        assert "name" not in technique_profile
        assert "author" not in technique_profile
        assert "dynasty" not in technique_profile
        assert technique_profile["painting_type"] == "道释人物画"
        assert technique_profile["subject"] == "罗汉与侍者"
        assert background_profile["name"] == "十六罗汉图第十一尊者"
        assert "author" not in background_profile
        assert material_profile["name"] == "十六罗汉图第十一尊者"
        assert material_profile["related_background"] == ["与南宋罗汉题材传播有关"]


def test_fixed_slot_selection_groups_similar_terms_and_preserves_group_siblings() -> None:
    class GroupedTechniqueLLM:
        def complete_json(
            self,
            *,
            system_prompt: str,
            user_text: str,
            image_base64: str | None = None,
            image_mime_type: str = "image/png",
        ) -> dict:
            payload = json.loads(user_text)
            groups = payload.get("candidate_term_groups", [])
            assert groups
            assert groups[0]["group_name"] == "皴法"
            assert groups[0]["terms"][:2] == ["披麻皴", "雨点皴"]
            return {
                "applicable": True,
                "slot_name": "墨法设色技法",
                "slot_term": "披麻皴",
                "description": "以披麻皴作为当前技法主 term。",
                "specific_questions": [
                    "披麻皴如何表现山体结构？",
                    "披麻皴与墨色渲染怎样共同塑造空间层次？",
                ],
                "metadata": {"confidence": 0.91, "source_id": "doc-technique"},
                "pending_terms": ["墨色渲染"],
                "reasoning": "披麻皴是当前最稳的技法主 term。",
            }

    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=GroupedTechniqueLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        slot = pipeline._plan_fixed_slot(
            spec={"slot_name": "墨法设色技法", "slot_mode": "progressive", "goal": "测试"},
            candidates=[
                {"term": "披麻皴", "priority": 12, "source_type": "technique_judgment", "evidence": ["山体线条纵向柔韧"]},
                {"term": "雨点皴", "priority": 11, "source_type": "grounded_term", "evidence": ["文献称范宽擅雨点皴"]},
                {"term": "墨色渲染", "priority": 9, "source_type": "visual_cue", "evidence": ["远景山顶墨色渲染"]},
            ],
            painting_profile={
                "name": "溪山行旅图",
                "author": "范宽",
                "dynasty": "宋代",
                "painting_type": "山水画",
                "subject": "山水与行旅",
                "scene_summary": "巨山占据主体，远景墨色渲染明显。",
            },
            visual_cues=["巨山主体明显", "远景山顶墨色渲染"],
        )

        assert slot.slot_term == "披麻皴"
        assert slot.metadata.extra["selected_term_group"] == "皴法"
        assert slot.metadata.extra["candidate_term_groups"][0]["group_name"] == "皴法"
        assert slot.metadata.extra["slot_terms"] == ["披麻皴", "雨点皴"]
        assert "雨点皴" not in slot.metadata.extra["pending_terms"]
        assert "墨色渲染" in slot.metadata.extra["pending_terms"]
        assert any("披麻皴、雨点皴" in question for question in slot.specific_questions)


def test_technique_judgment_blocks_cunfa_without_mountain() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=NoMountainTechniqueLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        judgment = pipeline._judge_image_techniques(
            image_base64="ZmFrZQ==",
            image_mime_type="image/png",
            input_text="人物画测试",
            visual_cues=["衣纹细劲匀整"],
        )
        merged = pipeline._merge_technique_candidates(
            [
                TermCandidate(
                    term="雨点皴",
                    description="误报皴法",
                    category_guess="皴法",
                    visual_evidence=["人物衣纹短点"],
                    text_evidence=[],
                ),
                TermCandidate(
                    term="减笔描",
                    description="备用描法",
                    category_guess="描法",
                    visual_evidence=[],
                    text_evidence=[],
                ),
            ],
            judgment,
        )

        assert judgment["mountain_present"] is False
        assert all(item.term != "雨点皴" for item in merged)
        assert any(item.term == "铁线描" for item in merged)


def test_llm_chat_record_is_refreshed_on_each_run() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")

        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )

        asyncio.run(pipeline.run(image_file=image_path, input_text="第一次运行"))
        first_run_records = json.loads((root / "llm_chat_record.jsonl").read_text(encoding="utf-8"))

        asyncio.run(pipeline.run(image_file=image_path, input_text="第二次运行"))
        second_run_records = json.loads((root / "llm_chat_record.jsonl").read_text(encoding="utf-8"))

    assert first_run_records
    assert second_run_records
    assert len(second_run_records) == len(first_run_records)
    assert "第二次运行" in second_run_records[0]["user_text"]
    assert "第一次运行" not in json.dumps(second_run_records, ensure_ascii=False)


def test_llm_chat_logger_recreates_parent_directory_before_append() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        log_path = root / "nested" / "llm_chat_record.jsonl"
        logger = LlmChatLogger(log_path)
        logger.reset()

        existing_dir = log_path.parent
        if log_path.exists():
            log_path.unlink()
        existing_dir.rmdir()

        logger.append(
            system_prompt="测试 prompt",
            user_text="测试 user_text",
            image_base64=None,
            image_mime_type="image/png",
            response_payload={"ok": True},
        )

        payload = json.loads(log_path.read_text(encoding="utf-8"))
        assert payload[0]["response"]["ok"] is True


def test_pipeline_creates_fresh_rag_client_for_each_search() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")
        rag_factory = CountingRagFactory()

        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=FakeLLM(),
            rag_client_factory=rag_factory,
            similarity_backend=FakeSimilarity(),
        )

        asyncio.run(pipeline.run(image_file=image_path, input_text="画面疑似斧劈皴，且文本提到绢本设色。"))

    assert rag_factory.calls >= 4


def test_pipeline_rechecks_metadata_spine_with_text_search_when_name_mismatches() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")
        rag = MetadataFallbackRag()

        pipeline = PerceptionPipeline(
            _build_config(root),
            llm_client=MetadataMismatchLLM(),
            rag_client=rag,
            similarity_backend=FakeSimilarity(),
        )

        result = asyncio.run(pipeline.run(image_file=image_path, input_text="请分析图像主体。"))
        context_text = result.context_path.read_text(encoding="utf-8")
        assert "洛神赋图" in context_text
        assert "洛神赋图" in rag.text_queries
        records = json.loads(result.llm_chat_record_path.read_text(encoding="utf-8"))
        assert any("作品主干核对器" in record["system_prompt"] for record in records)


def test_pipeline_uses_info_collection_name_only_for_metadata_spine_searches() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")
        rag = MetadataFallbackRag()
        config = _build_config(root)
        config.rag_info_collection_name = "image_blip"

        pipeline = PerceptionPipeline(
            config,
            llm_client=MetadataMismatchLLM(),
            rag_client=rag,
            similarity_backend=FakeSimilarity(),
        )

        asyncio.run(pipeline.run(image_file=image_path, input_text="请分析图像主体。"))

    assert rag.collection_names[:2] == ["image_blip", "image_blip"]
    assert all(name is None for name in rag.collection_names[2:])


def test_downstream_prompt_runner_preserves_existing_llm_chat_record() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        image_path = root / "sample.png"
        Image.new("RGB", (32, 32), color="white").save(image_path, format="PNG")
        config = _build_config(root)

        pipeline = PerceptionPipeline(
            config,
            llm_client=FakeLLM(),
            rag_client=FakeRag(),
            similarity_backend=FakeSimilarity(),
        )
        asyncio.run(pipeline.run(image_file=image_path, input_text="主流程运行"))
        first_run_records = json.loads((root / "llm_chat_record.jsonl").read_text(encoding="utf-8"))

        downstream = DownstreamPromptRunner(config, llm_client=FakeLLM())
        response = downstream.run_json(
            task_name="补充问题",
            system_prompt="你是下游测试助手",
            user_text="请补充两个更细的问题",
        )
        final_records = json.loads((root / "llm_chat_record.jsonl").read_text(encoding="utf-8"))

    assert response["status"] == "ok"
    assert len(first_run_records) >= 1
    assert "主流程运行" in json.dumps(final_records, ensure_ascii=False)
    assert "请补充两个更细的问题" in json.dumps(final_records, ensure_ascii=False)


def test_config_from_env_reads_defaults_and_overrides() -> None:
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "env-key",
            "PERCEPTION_RAG_TOP_K": "7",
            "PERCEPTION_RAG_COLLECTION_NAME": "general_collection",
            "PERCEPTION_RAG_INFO_COLLECTION_NAME": "image_blip",
            "PERCEPTION_DUPLICATE_THRESHOLD": "0.9",
            "PERCEPTION_MAX_IMAGE_PIXELS": "1003520",
        },
        clear=False,
    ):
        config = PipelineConfig.from_env()

    assert config.api_key == "env-key"
    assert config.base_url == "https://api.zjuqx.cn/v1"
    assert config.rag_top_k == 7
    assert config.rag_collection_name == "general_collection"
    assert config.rag_info_collection_name == "image_blip"
    assert config.duplicate_threshold == 0.9
    assert config.max_image_pixels == 1003520
    assert config.rag_search_record_path == Path("/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/rag_search_record.md")
    assert config.llm_chat_record_path == Path("/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/llm_chat_record.jsonl")
    assert config.grounding_query_workers == 4
    assert config.slot_planner_workers == 4
    assert config.enable_ontology_inference is False
    assert config.emit_debug_context_sections is False
    assert config.write_rag_search_record is True
    assert config.write_llm_chat_record is True


def test_config_from_env_reads_debug_log_flags() -> None:
    with patch.dict(
        "os.environ",
        {
            "OPENAI_API_KEY": "env-key",
            "PERCEPTION_EMIT_DEBUG_CONTEXT_SECTIONS": "true",
            "PERCEPTION_WRITE_RAG_SEARCH_RECORD": "false",
            "PERCEPTION_WRITE_LLM_CHAT_RECORD": "0",
            "PERCEPTION_GROUNDING_QUERY_WORKERS": "3",
            "PERCEPTION_SLOT_PLANNER_WORKERS": "2",
            "PERCEPTION_ENABLE_ONTOLOGY_INFERENCE": "1",
        },
        clear=False,
    ):
        config = PipelineConfig.from_env()

    assert config.emit_debug_context_sections is True
    assert config.write_rag_search_record is False
    assert config.write_llm_chat_record is False
    assert config.grounding_query_workers == 3
    assert config.slot_planner_workers == 2
    assert config.enable_ontology_inference is True


def test_cli_accepts_model_overrides() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "/tmp/demo.png",
            "--text",
            "demo",
            "--base-url",
            "https://api.zjuqx.cn/v1",
            "--embedding-model",
            "baai/bge-m3",
            "--judge-model",
            "gemini-3pro",
            "--max-image-pixels",
            "1003520",
            "--api-key",
            "cli-key",
        ]
    )

    assert args.base_url == "https://api.zjuqx.cn/v1"
    assert args.embedding_model == "baai/bge-m3"
    assert args.judge_model == "gemini-3pro"
    assert args.max_image_pixels == 1003520
    assert args.api_key == "cli-key"


def test_cli_builds_config_from_args_without_model_env_override() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "/tmp/demo.png",
            "--text",
            "demo",
            "--api-key",
            "cli-key",
            "--base-url",
            "https://api.zjuqx.cn/v1",
            "--embedding-model",
            "baai/bge-m3",
            "--judge-model",
            "gemini-3pro",
        ]
    )

    with patch.dict(
        "os.environ",
        {
            "PERCEPTION_EMBEDDING_MODEL": "qwen/qwen3-embedding-8b",
            "PERCEPTION_JUDGE_MODEL": "bad-env-model",
        },
        clear=False,
    ):
        config = build_config_from_args(args)

    assert config.api_key == "cli-key"
    assert config.embedding_model == "baai/bge-m3"
    assert config.judge_model == "gemini-3pro"


def test_cli_resolves_default_terminal_log_path_from_output_directory() -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--image",
            "/tmp/demo.png",
            "--text",
            "demo",
            "--output",
            "/tmp/result/slots.jsonl",
        ]
    )
    config = build_config_from_args(args)

    log_path = resolve_terminal_log_path(args, config)

    assert log_path == Path("/tmp/result/terminal_output.log")


def test_tee_terminal_output_saves_stdout_and_stderr() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        log_path = Path(temp_dir) / "terminal.log"
        with tee_terminal_output(log_path):
            print("stdout line")
            print("stderr line", file=sys.stderr)

        content = log_path.read_text(encoding="utf-8")

    assert "stdout line" in content
    assert "stderr line" in content


def test_rag_http_error_surfaces_server_response() -> None:
    client = HttpRagClient("http://example.com/api/search")
    http_error = HTTPError(
        url="http://example.com/api/search",
        code=400,
        msg="Bad Request",
        hdrs=None,
        fp=BytesIO(b'{"detail":"query_image format invalid"}'),
    )

    with patch("urllib.request.urlopen", side_effect=http_error):
        try:
            client.search(
                query_text="斧劈皴",
                query_image_bytes=b"abc123",
                query_image_filename="demo.png",
                query_image_mime_type="image/png",
                top_k=5,
            )
        except RuntimeError as exc:
            message = str(exc)
        else:
            raise AssertionError("Expected RuntimeError for HTTP 400")

    assert "query_image format invalid" in message
    assert '"query_image_present": true' in message


class _FakeHttpResponse:
    def __init__(self, body: bytes) -> None:
        self._body = body

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeHttpResponse":
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None


def test_rag_uses_multipart_form_data_with_image_upload() -> None:
    client = HttpRagClient("http://example.com/api/search")
    captured: dict[str, object] = {}

    def _fake_urlopen(req: object, timeout: int = 30) -> _FakeHttpResponse:
        assert timeout == 30
        captured["content_type"] = req.headers.get("Content-type")
        captured["body"] = req.data
        return _FakeHttpResponse('{"results":[{"source_id":"doc-1","content":"绢本材质细密。"}]}'.encode("utf-8"))

    with patch("urllib.request.urlopen", side_effect=_fake_urlopen):
        results = client.search(
            query_text="绢本",
            query_image_bytes=b"fake-image-bytes",
            query_image_filename="demo.png",
            query_image_mime_type="image/png",
            top_k=5,
            collection_name="image_blip",
        )

    assert len(results) == 1
    assert results[0].source_id == "doc-1"
    content_type = str(captured["content_type"])
    body = bytes(captured["body"])
    assert "multipart/form-data; boundary=" in content_type
    assert b'name="query_text"' in body
    assert b'name="top_k"' in body
    assert b'name="collection_name"' in body
    assert b"image_blip" in body
    assert b'name="query_image"; filename="demo.png"' in body
    assert b"Content-Type: image/png" in body
    assert b"fake-image-bytes" in body


def test_prepare_image_payload_resizes_large_images_to_max_pixels() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        image_path = Path(temp_dir) / "large.png"
        Image.new("RGB", (2048, 1024), color="white").save(image_path, format="PNG")

        encoded, image_bytes, mime_type, size = _prepare_image_payload(image_path, max_pixels=1003520)

    assert mime_type == "image/png"
    assert size[0] * size[1] <= 1003520
    assert encoded
    assert image_bytes


class _FailingEmbeddings:
    def create(self, **_: object) -> object:
        raise RuntimeError("503 distributor unavailable")


class _FailingEmbeddingClient:
    def __init__(self) -> None:
        self.embeddings = _FailingEmbeddings()


def test_embedding_backend_falls_back_to_lexical_similarity_on_api_error() -> None:
    backend = OpenAIEmbeddingSimilarityBackend(
        PipelineConfig(api_key="test-key"),
        client=_FailingEmbeddingClient(),
    )

    similar_score = backend.similarity("斧劈皴用于山石皴擦", "山石皴擦可见斧劈皴")
    different_score = backend.similarity("斧劈皴用于山石皴擦", "馆藏机构为故宫博物院")

    assert similar_score > different_score
    assert 0.0 <= different_score <= 1.0
