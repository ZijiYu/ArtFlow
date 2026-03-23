from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from src.slots_v2 import DynamicAgentPipeline, PipelineConfig, load_context_meta, merge_meta
from src.slots_v2.config_loader import DEFAULT_CONFIG_PATH, DEFAULT_SLOTS_FILE, get_config_value, load_yaml_config
from src.slots_v2.new_api_client import NewAPIClient


def _parse_meta(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--meta 必须是合法 JSON: {exc}") from exc


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
        "--summary-model",
        default=str(
            get_config_value(
                file_config,
                "models",
                "summary",
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
                default=get_config_value(file_config, "models", "summary", default=""),
            )
        ).strip(),
    )
    parser.add_argument("--api-base-url", default=str(get_config_value(file_config, "api", "base_url", default="")).strip())
    parser.add_argument("--api-model", default=str(get_config_value(file_config, "api", "model", default="")).strip())
    parser.add_argument("--api-timeout", type=int, default=int(get_config_value(file_config, "api", "timeout", default=60) or 60))
    parser.add_argument("--max-pixel", type=int, default=int(get_config_value(file_config, "slots_v2", "max_pixel", default=1_003_520) or 1_003_520))
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

    pipeline = DynamicAgentPipeline(
        config=PipelineConfig(
            slots_file=args.slots_file,
            max_pixel=max(1, int(args.max_pixel)),
            concurrent_workers=max(1, int(args.concurrent_workers)),
            max_spawn_rounds=max(0, int(args.max_spawn_rounds)),
            max_dialogue_rounds=max(1, int(args.max_dialogue_rounds)),
            max_threads_per_round=max(1, int(args.max_threads_per_round)),
            thread_attempt_limit=max(1, int(args.thread_attempt_limit)),
            convergence_patience=max(1, int(args.convergence_patience)),
            resize_image=not args.no_resize,
            domain_model=args.domain_model or None,
            validation_model=args.validation_model or None,
            summary_model=args.summary_model or None,
            final_prompt_model=args.final_prompt_model or args.summary_model or None,
            output_dir=args.output_dir,
        ),
        api_client=NewAPIClient(
            base_url=args.api_base_url or None,
            model=args.api_model or None,
            timeout=max(1, int(args.api_timeout)),
            config_path=args.config,
        ),
    )

    result = pipeline.run(image_path=args.image, meta=meta)
    outputs = pipeline.save_result(result, output_dir=args.output_dir)

    print("=== slots_v2 finished ===")
    print(f"Routing: {result.routing.action}")
    print(f"Converged: {result.routing.converged}")
    print(f"Convergence Reason: {result.routing.convergence_reason}")
    print(f"Issues: {len(result.cross_validation.issues)}")
    print(f"Spawned Tasks: {len(result.routing.spawned_tasks)}")
    print(f"Prepared Image: {result.prepared_image.path}")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
