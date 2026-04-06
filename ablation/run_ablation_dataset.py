from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


AB_ROOT = Path("/Users/ken/MM/Pipeline/ablation")
FINAL_VERSION_ROOT = Path("/Users/ken/MM/Pipeline/final_version")
DEFAULT_CONFIG = FINAL_VERSION_ROOT / "config.yaml"
DEFAULT_DATASET = AB_ROOT / "test_image.jsonl"
DEFAULT_ENTRYPOINT = FINAL_VERSION_ROOT / "pics" / "closed_loop.py"
DEFAULT_OUTPUT_ROOT = AB_ROOT


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def format_slug(value: str) -> str:
    safe = "".join(ch if ch.isalnum() else "_" for ch in str(value or "").strip())
    while "__" in safe:
        safe = safe.replace("__", "_")
    return safe.strip("_").lower() or "item"


def _resolve_runner_cmd(raw: str) -> list[str]:
    text = str(raw or "").strip()
    if not text:
        return [sys.executable]
    return shlex.split(text)


def ensure_variant_configs(config_path: Path, output_root: Path, runner_cmd: list[str]) -> Path:
    command = [
        *runner_cmd,
        str(AB_ROOT / "ablation_layers.py"),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "生成 ablation 配置失败:\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    return output_root / "ablation_manifest.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ablation variants over a JSONL image list.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--text", default="请对这幅国画做严谨分析。")
    parser.add_argument("--limit", type=int, default=0, help="只跑前 N 张，0 表示全部")
    parser.add_argument("--variant", action="append", default=[], help="只跑指定 ablation variant，可重复传入")
    parser.add_argument(
        "--runner-cmd",
        default="",
        help='用于执行子进程的命令前缀，例如 `uv run --with pillow --with openai python`。',
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config_path = Path(args.config).expanduser()
    dataset_path = Path(args.dataset_jsonl).expanduser()
    output_root = Path(args.output_root).expanduser()
    runner_cmd = _resolve_runner_cmd(args.runner_cmd)
    manifest_path = ensure_variant_configs(config_path, output_root, runner_cmd)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    variants = manifest.get("variants", [])
    if args.variant:
        requested = {item.strip() for item in args.variant if item.strip()}
        variants = [item for item in variants if item.get("name") in requested]
    if not variants:
        raise SystemExit("没有可运行的 ablation variant。")

    rows = read_jsonl(dataset_path)
    if args.limit > 0:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("数据集为空，无法启动 ablation。")

    run_index: list[dict[str, Any]] = []
    for variant in variants:
        variant_name = str(variant.get("name", "")).strip() or "unknown_variant"
        variant_config = Path(str(variant.get("config_path", "")).strip()).expanduser()
        variant_dir = output_root / variant_name
        batch_dir = variant_dir / "batch_runs"
        batch_dir.mkdir(parents=True, exist_ok=True)
        for row in rows:
            image_path = Path(str(row.get("image_path", "")).strip()).expanduser()
            if not image_path.exists():
                run_index.append(
                    {
                        "variant": variant_name,
                        "title": row.get("title", ""),
                        "image_path": str(image_path),
                        "status": "missing_image",
                    }
                )
                continue
            title = str(row.get("title", "")).strip() or image_path.stem
            sample_slug = format_slug(f"{row.get('index', '')}_{title}")
            sample_output = batch_dir / sample_slug
            sample_output.mkdir(parents=True, exist_ok=True)
            command = [
                *runner_cmd,
                str(DEFAULT_ENTRYPOINT),
                "--config",
                str(variant_config),
                "--image",
                str(image_path),
                "--text",
                str(args.text),
                "--output-dir",
                str(sample_output),
            ]
            print(
                f"[ablation] variant={variant_name} title={title} image={image_path.name}",
                flush=True,
            )
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            stdout_path = sample_output / "stdout.log"
            stderr_path = sample_output / "stderr.log"
            stdout_path.write_text(completed.stdout, encoding="utf-8")
            stderr_path.write_text(completed.stderr, encoding="utf-8")
            run_dirs = sorted(path for path in sample_output.iterdir() if path.is_dir())
            latest_run_dir = run_dirs[-1] if run_dirs else None
            run_index.append(
                {
                    "variant": variant_name,
                    "title": title,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "sample_output_dir": str(sample_output),
                    "latest_run_dir": str(latest_run_dir) if latest_run_dir else "",
                    "returncode": completed.returncode,
                    "status": "ok" if completed.returncode == 0 else "failed",
                    "stdout_log": str(stdout_path),
                    "stderr_log": str(stderr_path),
                }
            )

    index_path = output_root / "ablation_run_index.jsonl"
    with index_path.open("w", encoding="utf-8") as handle:
        for item in run_index:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[ablation] run_index={index_path}", flush=True)


if __name__ == "__main__":
    main()
