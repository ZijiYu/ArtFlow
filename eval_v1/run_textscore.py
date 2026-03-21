from __future__ import annotations

import argparse
import json
from pathlib import Path

from textscore import TextScoreAnalyzer


DEFAULT_CONTEXT_1_PATH = Path(__file__).resolve().parent / "textscore" / "baseline.txt"
DEFAULT_CONTEXT_2_PATH = Path(__file__).resolve().parent / "textscore" / "enhanced.txt"


def _read_default_context(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"默认 context 文件不存在: {path}")
    return path.read_text(encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="TextScore: compare two long contexts with slot-aware scoring.")
    parser.add_argument("--slots", nargs="+", required=True, help="重要领域词列表")
    parser.add_argument("--model", default="openai/gpt-4o-mini", help="模型名")
    parser.add_argument("--base-url", default="https://api.zjuqx.cn/v1", help="OpenAI 兼容接口地址")
    parser.add_argument("--output", default="artifacts/textscore_result.json", help="JSON 输出路径")
    parser.add_argument(
        "--visualization",
        default="artifacts/textscore_comparison.html",
        help="HTML 可视化输出路径",
    )
    args = parser.parse_args()

    analyzer = TextScoreAnalyzer(model=args.model, base_url=args.base_url)
    result, html_path = analyzer.score(
        context_1=_read_default_context(DEFAULT_CONTEXT_1_PATH),
        context_2=_read_default_context(DEFAULT_CONTEXT_2_PATH),
        slots=args.slots,
        slots_number=len(args.slots),
        visualization_path=args.visualization,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(result.model_dump_json(indent=2), encoding="utf-8")

    print(json.dumps({"result_path": str(output_path), "visualization_path": html_path}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
