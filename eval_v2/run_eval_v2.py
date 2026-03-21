#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
from pathlib import Path

from guohua_eval import GuohuaEvalV2Analyzer


def _read_text_arg(
    text: str | None,
    file_path: str | None,
    label: str,
    fallback_paths: list[Path] | None = None,
) -> str:
    if text:
        return text
    if file_path:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"{label} 文件不存在: {path}")
        return path.read_text(encoding="utf-8")
    for path in fallback_paths or []:
        if path.exists():
            return path.read_text(encoding="utf-8")
    looked_paths = "、".join(str(path) for path in (fallback_paths or []))
    if looked_paths:
        raise ValueError(f"{label} 必须通过文本参数、文件参数提供，或确保默认文件存在: {looked_paths}")
    raise ValueError(f"{label} 必须通过文本参数或文件参数提供")


def main() -> None:
    root = Path(__file__).resolve().parent
    default_inputs = {
        "context_baseline": [
            root / "inputs" / "context_baseline.txt",
            root / "tests" / "inputs" / "context_baseline.txt",
        ],
        "context_enhanced": [
            root / "inputs" / "context_enhanced.txt",
            root / "tests" / "inputs" / "context_enhanced.txt",
        ],
        "slots": [
            root / "inputs" / "slots.txt",
            root / "tests" / "inputs" / "slots.txt",
        ],
        "image_context_v": [
            root / "inputs" / "image_context_v.txt",
            root / "tests" / "inputs" / "image_context_v.txt",
        ],
    }

    parser = argparse.ArgumentParser(description="评测两个国画赏析长文本的专业性差异。")
    parser.add_argument("--context-baseline", help="基准文本内容")
    parser.add_argument("--context-baseline-file", help="基准文本文件路径")
    parser.add_argument("--context-enhanced", help="增强文本内容")
    parser.add_argument("--context-enhanced-file", help="增强文本文件路径")
    parser.add_argument("--slots-text", help="自然语言 slots 描述")
    parser.add_argument("--slots-file", help="自然语言 slots 描述文件")
    parser.add_argument("--image-context-v", default="", help="视觉线索基准文本")
    parser.add_argument("--image-context-v-file", help="视觉线索基准文件")
    parser.add_argument("--base-url", default="https://api.zjuqx.cn/v1", help="OpenAI 兼容接口地址")
    parser.add_argument("--embedding-model", default="baai/bge-m3", help="embedding 模型名")
    parser.add_argument("--judge-model", default="openai/gpt-4.1", help="判定模型名")
    parser.add_argument("--duplicate-threshold", type=float, default=0.83, help="重复语义簇阈值")
    parser.add_argument("--output-dir", default="artifacts", help="输出目录")
    args = parser.parse_args()

    context_baseline = _read_text_arg(
        args.context_baseline,
        args.context_baseline_file,
        "context_baseline",
        default_inputs["context_baseline"],
    )
    context_enhanced = _read_text_arg(
        args.context_enhanced,
        args.context_enhanced_file,
        "context_enhanced",
        default_inputs["context_enhanced"],
    )
    slots_text = _read_text_arg(args.slots_text, args.slots_file, "slots", default_inputs["slots"])
    image_context_v = _read_text_arg(
        args.image_context_v,
        args.image_context_v_file,
        "image_context_v",
        default_inputs["image_context_v"],
    )

    analyzer = GuohuaEvalV2Analyzer(
        embedding_model=args.embedding_model,
        judge_model=args.judge_model,
        base_url=args.base_url,
        duplicate_threshold=args.duplicate_threshold,
    )
    result = analyzer.evaluate(
        context_baseline=context_baseline,
        context_enhanced=context_enhanced,
        slots_input=slots_text,
        image_context_v=image_context_v,
        output_dir=args.output_dir,
    )
    print(
        json.dumps(
            {
                "result_json_path": result.result_json_path,
                "baseline_duplicate_clusters_jsonl": result.context_baseline_metrics.duplicate_clusters_jsonl,
                "baseline_terms_jsonl": result.context_baseline_metrics.terms_jsonl,
                "baseline_slot_matches_jsonl": result.context_baseline_metrics.slot_matches_jsonl,
                "baseline_fidelity_jsonl": result.context_baseline_metrics.fidelity_jsonl,
                "enhanced_duplicate_clusters_jsonl": result.context_enhanced_metrics.duplicate_clusters_jsonl,
                "enhanced_terms_jsonl": result.context_enhanced_metrics.terms_jsonl,
                "enhanced_slot_matches_jsonl": result.context_enhanced_metrics.slot_matches_jsonl,
                "enhanced_fidelity_jsonl": result.context_enhanced_metrics.fidelity_jsonl,
                "winner": result.final_judgment.winner,
                "textual_loss_for": result.final_judgment.textual_loss_for,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
