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
        choices=["slot", "solitary", "communal"],
        help="运行模式：slot / solitary / communal",
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
    parser.add_argument("--agent-model", default="", help="slot小模型名称，留空则用NEW_API_MODEL")
    parser.add_argument("--baseline-model", default="", help="baseline鉴赏模型名称，留空则用NEW_API_MODEL")
    parser.add_argument("--enhanced-model", default="", help="enhanced鉴赏模型名称，留空则用NEW_API_MODEL")
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
            agent_model=args.agent_model or None,
            baseline_model=args.baseline_model or None,
            enhanced_model=args.enhanced_model or None,
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
