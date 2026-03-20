from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Keep local execution simple without requiring package install.
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.tcp_pipeline import PipelineConfig, TcpPromptPipeline
from src.tcp_pipeline.new_api_client import NewAPIClient


def _parse_meta(raw: str | None) -> dict:
    if not raw:
        return {}
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--meta 必须是合法JSON: {exc}") from exc


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TCP 国画鉴赏 prompt v1 pipeline")
    parser.add_argument(
        "--mode",
        default="slot",
        choices=["slot", "solitary", "communal", "slot_pipe"],
        help="运行模式：slot / solitary / communal / slot_pipe",
    )
    parser.add_argument("--image", required=True, help="图像路径或图像标识")
    parser.add_argument(
        "--meta",
        default="{}",
        help='可选meta JSON，例如: {"caption":"雨后山水", "tags":["山水","水墨"]}',
    )
    parser.add_argument(
        "--slots",
        default="",
        help="slot列表，逗号分隔；例如 brush-and-ink,literary,history。留空或all则使用默认全部slot",
    )
    parser.add_argument("--agent-temperature", type=float, default=0.7, help="agent生成温度")
    parser.add_argument("--vlm-temperature", type=float, default=0.2, help="VLM鉴赏温度")
    parser.add_argument("--solitary-model", default="", help="solitary模式模型名称，三轮复用同一模型")
    parser.add_argument("--solitary-rounds", type=int, default=3, help="solitary模式反思轮数，最少1轮，默认3")
    parser.add_argument("--guest-num", type=int, default=3, help="communal模式客人数量，默认3")
    parser.add_argument("--guest-model", default="", help="communal模式客人模型名称，留空则用NEW_API_MODEL")
    parser.add_argument("--judge-model", default="", help="judge模型名称，供slot_pipe与其他需要裁决的流程使用")
    parser.add_argument(
        "--slot-pipe-agents-per-slot",
        type=int,
        default=2,
        help="slot_pipe模式每个slot并行agent数量，默认2",
    )
    parser.add_argument(
        "--slot-pipe-max-retries",
        type=int,
        default=3,
        help="slot_pipe模式每个slot最多返工轮数，默认3",
    )
    parser.add_argument(
        "--slot-pipe-version",
        type=int,
        default=4,
        help="slot_pipe版本，默认4；可设为3以回退旧流程",
    )
    parser.add_argument(
        "--slot-pipe-slots-file",
        default="artifacts/slots.jsonl",
        help="slot_pipe v4的slots输入文件(JSONL)，默认artifacts/slots.jsonl",
    )
    parser.add_argument(
        "--slot-pipe-content-agents",
        type=int,
        default=2,
        help="slot_pipe v4内容层每个slot调用次数，默认2",
    )
    parser.add_argument(
        "--slot-pipe-expression-guests",
        type=int,
        default=3,
        help="slot_pipe v4表达层每个问题的群赏人数，默认3",
    )
    parser.add_argument("--checker-model", default="", help="slot_pipe v4要素确认层checker模型")
    parser.add_argument("--reviewer-model", default="", help="slot_pipe v4要素确认层reviewer模型")
    parser.add_argument("--content-model", default="", help="slot_pipe v4内容要点层模型")
    parser.add_argument("--expression-guest-model", default="", help="slot_pipe v4群赏客人模型")
    parser.add_argument("--expression-summary-model", default="", help="slot_pipe v4群赏池化摘要模型")
    parser.add_argument("--agent-model", default="", help="slot小模型名称，留空则用NEW_API_MODEL")
    parser.add_argument(
        "--embedding-model",
        default="",
        help="slot_pipe降重embedding模型名称（通过chat/completions调用），留空则回退默认模型",
    )
    parser.add_argument("--baseline-model", default="", help="baseline鉴赏模型名称，留空则用NEW_API_MODEL")
    parser.add_argument("--enhanced-model", default="", help="enhanced鉴赏模型名称，留空则用NEW_API_MODEL")
    parser.add_argument(
        "--final-appreciation-model",
        default="",
        help="slot_pipe最终鉴赏模型名称（使用最终补充prompt+图像），留空则回退enhanced/baseline模型",
    )
    parser.add_argument("--api-timeout", type=int, default=60, help="单次API请求超时秒数，默认60")
    parser.add_argument("--output-dir", default="outputs", help="结果输出目录")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    meta = _parse_meta(args.meta)
    slots = [x.strip() for x in args.slots.split(",") if x.strip()]
    api_client = NewAPIClient(timeout=max(1, args.api_timeout))

    pipeline = TcpPromptPipeline(
        config=PipelineConfig(
            mode=args.mode,
            slots=slots,
            agent_temperature=args.agent_temperature,
            vlm_temperature=args.vlm_temperature,
            solitary_model=args.solitary_model or None,
            solitary_rounds=max(1, args.solitary_rounds),
            guest_num=max(1, args.guest_num),
            guest_model=args.guest_model or None,
            judge_model=args.judge_model or None,
            slot_pipe_agents_per_slot=max(1, args.slot_pipe_agents_per_slot),
            slot_pipe_max_retries=max(0, args.slot_pipe_max_retries),
            slot_pipe_version=max(3, args.slot_pipe_version),
            slot_pipe_slots_file=args.slot_pipe_slots_file or None,
            slot_pipe_content_agents=max(1, args.slot_pipe_content_agents),
            slot_pipe_expression_guests=max(1, args.slot_pipe_expression_guests),
            agent_model=args.agent_model or None,
            embedding_model=args.embedding_model or None,
            baseline_model=args.baseline_model or None,
            enhanced_model=args.enhanced_model or None,
            final_appreciation_model=args.final_appreciation_model or None,
            checker_model=args.checker_model or None,
            reviewer_model=args.reviewer_model or None,
            content_model=args.content_model or None,
            expression_guest_model=args.expression_guest_model or None,
            expression_summary_model=args.expression_summary_model or None,
        ),
        api_client=api_client,
    )
    result = pipeline.run(image_path=args.image, meta=meta)
    output_files = pipeline.save_result(result, output_dir=args.output_dir)

    print("=== Pipeline Finished ===")
    print(f"Selected Slots: {result.selected_slots}")
    print(f"Token Usage: {result.token_usage}")
    failed_calls = [x for x in result.api_logs if not x.get("ok")]
    print(f"API Calls: {len(result.api_logs)}, Failed: {len(failed_calls)}")
    if result.mode == "slot_pipe":
        embedding_logs = [x for x in result.api_logs if x.get("stage") == "slot_pipe_embedding"]
        if embedding_logs:
            total = len(embedding_logs)
            api_ok = sum(1 for x in embedding_logs if bool(x.get("ok")))
            parse_ok = sum(1 for x in embedding_logs if bool(x.get("embedding_parse_ok")))
            fallback = sum(1 for x in embedding_logs if bool(x.get("fallback_used")))
            print(
                "Embedding Summary: "
                f"total={total}, api_ok={api_ok}, parse_ok={parse_ok}, fallback={fallback}"
            )
    for item in failed_calls[:5]:
        stage = item.get("stage", "unknown")
        slot = item.get("slot")
        location = f"{stage}:{slot}" if slot else str(stage)
        print(f"  - {location} -> {item.get('error')}")
    print("Output Files:")
    for name, path in output_files.items():
        print(f"- {name}: {path}")
    print("\n--- Enhanced Prompt Preview (first 600 chars) ---")
    print(result.enhanced_prompt[:600])


if __name__ == "__main__":
    main()
