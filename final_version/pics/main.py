from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.cot_layer import DynamicAgentPipeline, PipelineConfig, load_context_meta, merge_meta
from src.cot_layer.config_loader import DEFAULT_CONFIG_PATH, DEFAULT_SLOTS_FILE, get_config_value, load_yaml_config
from src.cot_layer.new_api_client import NewAPIClient


def _parse_meta(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--meta 必须是合法 JSON: {exc}") from exc


def _config_bool(value: object, default: bool = True) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off"}:
        return False
    return default


def build_parser(file_config: dict) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dynamic agent pipeline for Chinese painting appreciation")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="共享 YAML 配置路径")
    parser.add_argument("--image", default=str(get_config_value(file_config, "image", "path", default="")).strip(), help="图像路径")
    parser.add_argument(
        "--meta",
        default=json.dumps(get_config_value(file_config, "image", "meta", default={}), ensure_ascii=False),
        help='meta JSON，例如 {"system_metadata":{"dynasty":"北宋"}}',
    )
    parser.add_argument(
        "--slots-file",
        default=str(get_config_value(file_config, "slots_v2", "slots_file", default=DEFAULT_SLOTS_FILE)).strip(),
        help="槽位 JSONL 文件路径",
    )
    parser.add_argument(
        "--meta-context-file",
        default=str(get_config_value(file_config, "image", "meta_context_file", default="")).strip(),
        help="可选 Markdown 上下文文件路径，会抽取 Domain Profile、Post-RAG Text Extraction、Ontology Updates 作为 meta",
    )
    parser.add_argument(
        "--domain-model",
        default=str(
            get_config_value(
                file_config,
                "models",
                "domain",
                default=get_config_value(file_config, "models", "default", default=""),
            )
        ).strip(),
    )
    parser.add_argument(
        "--validation-model",
        default=str(
            get_config_value(
                file_config,
                "models",
                "validation",
                default=get_config_value(file_config, "models", "default", default=""),
            )
        ).strip(),
    )
    parser.add_argument(
        "--final-prompt-model",
        default=str(
            get_config_value(
                file_config,
                "models",
                "final_prompt",
                default=get_config_value(file_config, "models", "default", default=""),
            )
        ).strip(),
    )
    parser.add_argument(
        "--final-answer-model",
        default=str(
            get_config_value(
                file_config,
                "models",
                "final_answer",
                default=get_config_value(file_config, "models", "final_prompt", default=get_config_value(file_config, "models", "default", default="")),
            )
        ).strip(),
    )
    parser.add_argument("--api-key", default=str(get_config_value(file_config, "api", "key", default="")).strip())
    parser.add_argument("--api-key-file", default=str(get_config_value(file_config, "api", "key_file", default="")).strip())
    parser.add_argument("--api-key-line", type=int, default=int(get_config_value(file_config, "api", "key_line", default=1) or 1))
    parser.add_argument("--api-base-url", default=str(get_config_value(file_config, "api", "base_url", default="")).strip())
    parser.add_argument("--api-model", default=str(get_config_value(file_config, "api", "model", default="")).strip())
    parser.add_argument("--api-timeout", type=int, default=int(get_config_value(file_config, "api", "timeout", default=60) or 60))
    parser.add_argument("--max-pixel", type=int, default=int(get_config_value(file_config, "slots_v2", "max_pixel", default=1_003_520) or 1_003_520))
    parser.add_argument(
        "--rag-query-max-blocks",
        type=int,
        default=int(get_config_value(file_config, "slots_v2", "rag_query_max_blocks", default=2) or 2),
    )
    parser.add_argument(
        "--enable-rag-verification",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "slots_v2", "enable_rag_verification", default=True), True),
        help="是否启用 retrieval_gain 与 downstream RAG 核实流程",
    )
    parser.add_argument(
        "--retrieval-gain",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "slots_v2", "retrieval_gain", default=False), False),
        help="是否在 domain_cot 阶段启用 retrieval_gain 输出",
    )
    parser.add_argument(
        "--enable-web-search",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "web_search", "enabled", default=False), False),
        help="是否启用 Serper 联网搜索与网页抓取",
    )
    parser.add_argument("--web-search-url", default=str(get_config_value(file_config, "web_search", "url", default="")).strip())
    parser.add_argument("--web-search-api-key", default=str(get_config_value(file_config, "web_search", "api_key", default="")).strip())
    parser.add_argument(
        "--web-search-api-key-file",
        default=str(get_config_value(file_config, "web_search", "api_key_file", default="")).strip(),
    )
    parser.add_argument(
        "--web-search-api-key-line",
        type=int,
        default=int(get_config_value(file_config, "web_search", "api_key_line", default=1) or 1),
    )
    parser.add_argument(
        "--web-search-timeout",
        type=int,
        default=int(get_config_value(file_config, "web_search", "timeout", default=20) or 20),
    )
    parser.add_argument(
        "--web-search-top-k",
        type=int,
        default=int(get_config_value(file_config, "web_search", "search_top_k", default=5) or 5),
    )
    parser.add_argument(
        "--web-search-fetch-top-n",
        type=int,
        default=int(get_config_value(file_config, "web_search", "fetch_top_n", default=2) or 2),
    )
    parser.add_argument(
        "--web-search-use-llm-rerank",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "web_search", "use_llm_rerank", default=True), True),
    )
    parser.add_argument(
        "--web-search-skip-llm-if-confident",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "web_search", "skip_llm_if_confident", default=True), True),
    )
    parser.add_argument(
        "--web-search-fallback-on-empty-rag",
        action=argparse.BooleanOptionalAction,
        default=_config_bool(get_config_value(file_config, "web_search", "fallback_on_empty_rag", default=True), True),
    )
    parser.add_argument("--max-spawn-rounds", type=int, default=int(get_config_value(file_config, "slots_v2", "max_spawn_rounds", default=1) or 1))
    parser.add_argument("--max-dialogue-rounds", type=int, default=int(get_config_value(file_config, "slots_v2", "max_dialogue_rounds", default=4) or 4))
    parser.add_argument("--max-threads-per-round", type=int, default=int(get_config_value(file_config, "slots_v2", "max_threads_per_round", default=4) or 4))
    parser.add_argument("--thread-attempt-limit", type=int, default=int(get_config_value(file_config, "slots_v2", "thread_attempt_limit", default=2) or 2))
    parser.add_argument("--convergence-patience", type=int, default=int(get_config_value(file_config, "slots_v2", "convergence_patience", default=2) or 2))
    parser.add_argument("--concurrent-workers", type=int, default=int(get_config_value(file_config, "slots_v2", "concurrent_workers", default=4) or 4))
    parser.add_argument(
        "--output-dir",
        default=str(
            get_config_value(
                file_config,
                "slots_v2",
                "output_dir",
                default=get_config_value(file_config, "runtime", "output_dir", default="artifacts"),
            )
        ).strip(),
    )
    parser.add_argument("--no-resize", action="store_true", help="禁用图像像素约束预处理")
    return parser


def main() -> None:
    started_at = perf_counter()
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    config_args, _ = config_parser.parse_known_args()
    file_config = load_yaml_config(config_args.config)
    parser = build_parser(file_config)
    args = parser.parse_args()

    if not str(args.image or "").strip():
        raise SystemExit("缺少图像输入，请通过 --image 指定国画图片。")

    meta = _parse_meta(args.meta)
    if str(args.meta_context_file or "").strip():
        context_meta = load_context_meta(args.meta_context_file)
        meta = merge_meta(context_meta, meta)
    final_question = str(
        get_config_value(
            file_config,
            "image",
            "final_question",
            default=get_config_value(file_config, "image", "initial_question", default=""),
        )
    ).strip()
    if final_question:
        meta.setdefault("final_user_question", final_question)
    if final_question or str(get_config_value(file_config, "image", "initial_question", default="")).strip():
        meta.setdefault(
            "input_text",
            final_question or str(get_config_value(file_config, "image", "initial_question", default="")).strip(),
        )

    pipeline = DynamicAgentPipeline(
        config=PipelineConfig(
            slots_file=args.slots_file,
            max_pixel=max(1, int(args.max_pixel)),
            concurrent_workers=max(1, int(args.concurrent_workers)),
            rag_query_max_blocks=max(1, int(args.rag_query_max_blocks)),
            enable_rag_verification=bool(args.enable_rag_verification),
            retrieval_gain=bool(args.retrieval_gain),
            enable_web_search=bool(args.enable_web_search),
            web_search_url=args.web_search_url or None,
            web_search_api_key=args.web_search_api_key or None,
            web_search_api_key_file=args.web_search_api_key_file or None,
            web_search_api_key_line=max(1, int(args.web_search_api_key_line)),
            web_search_timeout=max(1, int(args.web_search_timeout)),
            web_search_top_k=max(1, int(args.web_search_top_k)),
            web_search_fetch_top_n=max(1, int(args.web_search_fetch_top_n)),
            web_search_use_llm_rerank=bool(args.web_search_use_llm_rerank),
            web_search_skip_llm_if_confident=bool(args.web_search_skip_llm_if_confident),
            web_search_fallback_on_empty_rag=bool(args.web_search_fallback_on_empty_rag),
            max_spawn_rounds=max(0, int(args.max_spawn_rounds)),
            max_dialogue_rounds=max(1, int(args.max_dialogue_rounds)),
            max_threads_per_round=max(1, int(args.max_threads_per_round)),
            thread_attempt_limit=max(1, int(args.thread_attempt_limit)),
            convergence_patience=max(1, int(args.convergence_patience)),
            resize_image=not args.no_resize,
            domain_model=args.domain_model or None,
            validation_model=args.validation_model or None,
            final_prompt_model=args.final_prompt_model or None,
            final_answer_model=args.final_answer_model or None,
            output_dir=args.output_dir,
        ),
        api_client=NewAPIClient(
            api_key=args.api_key or None,
            api_key_file=args.api_key_file or None,
            api_key_line=max(1, int(args.api_key_line)),
            base_url=args.api_base_url or None,
            model=args.api_model or None,
            timeout=max(1, int(args.api_timeout)),
            config_path=args.config,
        ),
    )

    result = pipeline.run(image_path=args.image, meta=meta)
    result = pipeline.finalize_result(result, meta=meta)
    outputs = pipeline.save_result(result, output_dir=args.output_dir)

    print("=== slots_v2 finished ===")
    print(f"Routing: {result.routing.action}")
    print(f"Converged: {result.routing.converged}")
    print(f"Convergence Reason: {result.routing.convergence_reason}")
    print(f"Issues: {len(result.cross_validation.issues)}")
    print(f"Spawned Tasks: {len(result.routing.spawned_tasks)}")
    print(f"Prepared Image: {result.prepared_image.path}")
    print(f"Total Elapsed: {perf_counter() - started_at:.2f}s")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
