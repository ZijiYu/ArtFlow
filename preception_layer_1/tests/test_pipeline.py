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
from perception_layer.clients import HttpRagClient, OpenAIEmbeddingSimilarityBackend
from perception_layer.config import PipelineConfig
from perception_layer.downstream import DownstreamPromptRunner
from perception_layer.models import RagDocument
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
        if "中国画作品类型分析器" in system_prompt:
            assert image_base64
            return {
                "painting_type": "山水画",
                "subject": "山石与溪岸",
                "scene_summary": "画面以山石结构为主，兼见题跋和绢本设色质感。",
                "guohua_knowledge": ["山水画常关注皴法、构图和材质", "题跋与绢本信息有助于术语抽取"],
                "reasoning": "图像中山石结构明显，文字提到绢本设色。",
            }
        if "视觉锚点抽取器" in system_prompt:
            assert image_base64
            return {"visual_cues": ["山石边缘呈折线皴擦", "右上角可见题跋", "画面有绢本设色质感"]}
        if "文本信号分析器" in system_prompt:
            return {"text_signals": ["文本提到绢本设色", "文字疑似提及斧劈皴"], "salient_entities": ["绢本", "斧劈皴"]}
        if "领域锚定器" in system_prompt:
            assert "画作类型分析" in user_text
            assert "山水画" in user_text
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
        if "Slot 生成器" in system_prompt:
            return {
                "slots": [
                    {
                        "slot_name": "国画技法",
                        "slot_term": "斧劈皴",
                        "description": "RAG 证据表明斧劈皴常用于表现山石折转与硬朗纹理。",
                        "specific_questions": [
                            "画家如何利用斧劈皴强化山石的结构感？",
                            "斧劈皴与整幅构图的气势有什么关系？",
                            "在花卉（如紫蝶牡丹）的写生中，铁线描与兰叶描结合使用时如何分配主次？",
                        ],
                        "metadata": {"confidence": 0.92, "source_id": "doc-technique"},
                    },
                    {
                        "slot_name": "国画技法",
                        "slot_term": "斧劈皴法",
                        "description": "RAG 证据指出该皴法以折线组织石纹，语义上与斧劈皴高度接近。",
                        "specific_questions": [
                            "皴擦节奏如何影响山石的质感？",
                            "这种皴法是否与题跋中的审美判断形成呼应？",
                        ],
                        "metadata": {"confidence": 0.88, "source_id": "doc-technique-2"},
                    },
                    {
                        "slot_name": "材质形制",
                        "slot_term": "绢本",
                        "description": "RAG 证据显示绢本材质会影响设色层次和笔触呈现。",
                        "specific_questions": [
                            "绢本如何改变设色后的光泽和细节表现？",
                            "材质选择对观看者理解作品精致度有何影响？",
                        ],
                        "metadata": {"confidence": 0.85, "source_id": "doc-material"},
                    },
                ]
            }
        if "动态本体推理器" in system_prompt:
            return {
                "relations": [
                    {
                        "child": "斧劈皴",
                        "parent": "皴法",
                        "relation": "is-a",
                        "rationale": "斧劈皴属于皴法中的具体类型。",
                    },
                    {
                        "child": "绢本",
                        "parent": "材质",
                        "relation": "is-a",
                        "rationale": "绢本对应作品承载材料。",
                    },
                ]
            }
        if "下游测试助手" in system_prompt:
            return {"status": "ok", "task": user_text}
        raise AssertionError(f"Unexpected prompt: {system_prompt}")


class FakeRag:
    def search(
        self,
        *,
        query_text: str | None,
        query_image_bytes: bytes | None,
        query_image_filename: str | None,
        query_image_mime_type: str | None,
        top_k: int,
    ) -> list[RagDocument]:
        assert query_image_bytes
        assert query_image_filename == "sample.png"
        assert query_image_mime_type == "image/png"
        assert top_k == 5
        if query_text == "斧劈皴":
            return [
                RagDocument(source_id="doc-technique", content="斧劈皴多用于表现山石折转，线面关系硬朗。"),
                RagDocument(source_id="doc-low", content="这是无关的馆藏说明。"),
            ]
        if query_text == "绢本":
            return [RagDocument(source_id="doc-material", content="绢本材质细密，常影响设色层次和观感。")]
        if query_text == "斧劈皴法":
            return [RagDocument(source_id="doc-technique-2", content="斧劈皴法以折线皴擦构成山石肌理。")]
        return []


class FakeSimilarity:
    def similarity(self, left: str, right: str) -> float:
        if {"斧劈皴", "斧劈皴法"} <= {left, right}:
            return 0.91
        if "无关的馆藏说明" in {left, right}:
            return 0.1
        if "斧劈皴" in left and "折转" in right:
            return 0.78
        if "斧劈皴法" in left and "折线皴擦" in right:
            return 0.82
        if "绢本" in left and "材质细密" in right:
            return 0.76
        if "国画技法" in left and "国画技法" in right:
            return 0.91
        return 0.4


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

        assert len(result.slots) == 2
        assert {slot.slot_term for slot in result.slots} == {"斧劈皴", "绢本"}
        assert all(not item.candidate.visual_evidence for item in result.grounded_terms)
        assert any("斧劈皴多用于表现山石折转" in evidence for evidence in result.grounded_terms[0].candidate.text_evidence)
        merged_slot = next(slot for slot in result.slots if slot.slot_term == "斧劈皴")
        assert all("紫蝶牡丹" not in question for question in merged_slot.specific_questions)
        assert "合并近义术语" in merged_slot.description
        assert merged_slot.metadata.source_id == "doc-technique,doc-technique-2"
        assert result.output_path.exists()
        assert result.rag_search_record_path.exists()
        assert result.llm_chat_record_path.exists()
        lines = result.output_path.read_text(encoding="utf-8").strip().splitlines()
        assert len(lines) == 2
        context_text = result.context_path.read_text(encoding="utf-8")
        rag_search_text = result.rag_search_record_path.read_text(encoding="utf-8")
        llm_chat_text = result.llm_chat_record_path.read_text(encoding="utf-8")
        assert "Ontology Updates" in context_text
        assert "Post-RAG Text Extraction" in context_text
        assert "斧劈皴" in context_text
        assert "query_text: `斧劈皴`" in rag_search_text
        assert "sources: `doc-technique`" in rag_search_text or "sources: `doc-technique,doc-low`" in rag_search_text
        assert '"system_prompt"' in llm_chat_text
        assert "中国画作品类型分析器" in llm_chat_text
        assert "视觉锚点抽取器" in llm_chat_text


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
            "PERCEPTION_DUPLICATE_THRESHOLD": "0.9",
            "PERCEPTION_MAX_IMAGE_PIXELS": "1003520",
        },
        clear=False,
    ):
        config = PipelineConfig.from_env()

    assert config.api_key == "env-key"
    assert config.base_url == "https://api.zjuqx.cn/v1"
    assert config.rag_top_k == 7
    assert config.duplicate_threshold == 0.9
    assert config.max_image_pixels == 1003520
    assert config.rag_search_record_path == Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/rag_search_record.md")
    assert config.llm_chat_record_path == Path("/Users/ken/MM/Pipeline/preception_layer/artifacts/llm_chat_record.jsonl")


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
        )

    assert len(results) == 1
    assert results[0].source_id == "doc-1"
    content_type = str(captured["content_type"])
    body = bytes(captured["body"])
    assert "multipart/form-data; boundary=" in content_type
    assert b'name="query_text"' in body
    assert b'name="top_k"' in body
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
