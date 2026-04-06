#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any


AB_ROOT = Path("/Users/ken/MM/Pipeline/ablation")
DEFAULT_RUN_INDEX = AB_ROOT / "ablation_run_index.jsonl"
DEFAULT_OUTPUT_ROOT = AB_ROOT / "acc_export"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def sample_id_from_row(row: dict[str, Any]) -> str:
    image_name = str(row.get("image_name", "")).strip()
    if image_name:
        return Path(image_name).stem
    image_path = str(row.get("image_path", "")).strip()
    if image_path:
        return Path(image_path).stem
    return ""


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_if_exists(src: Path, dst: Path) -> str:
    if not src.is_file():
        return ""
    ensure_dir(dst.parent)
    shutil.copy2(src, dst)
    return str(dst)


def export_variant_record(
    *,
    row: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    variant = str(row.get("variant", "")).strip() or "unknown_variant"
    title = str(row.get("title", "")).strip()
    status = str(row.get("status", "")).strip()
    sample_id = sample_id_from_row(row)
    latest_run_dir = Path(str(row.get("latest_run_dir", "")).strip()).expanduser()

    variant_root = ensure_dir(output_root / variant)
    prompts_dir = ensure_dir(variant_root / "final_prompts")
    reports_dir = ensure_dir(variant_root / "reports")
    runtime_dir = ensure_dir(variant_root / "runtime_state")
    raw_dir = ensure_dir(variant_root / "raw")

    final_prompt_src = latest_run_dir / "final_appreciation_prompt.md"
    closed_loop_src = latest_run_dir / "closed_loop_report.json"
    slots_final_src = latest_run_dir / "runtime_state" / "slots_final.jsonl"
    stdout_src = Path(str(row.get("stdout_log", "")).strip()).expanduser()
    stderr_src = Path(str(row.get("stderr_log", "")).strip()).expanduser()

    final_prompt_dst = prompts_dir / f"{sample_id}.md"
    report_dst = reports_dir / f"{sample_id}.json"
    slots_final_dst = runtime_dir / f"{sample_id}.jsonl"
    stdout_dst = raw_dir / f"{sample_id}.stdout.log"
    stderr_dst = raw_dir / f"{sample_id}.stderr.log"

    final_prompt_path = copy_if_exists(final_prompt_src, final_prompt_dst)
    report_path = copy_if_exists(closed_loop_src, report_dst)
    slots_final_path = copy_if_exists(slots_final_src, slots_final_dst)
    stdout_path = copy_if_exists(stdout_src, stdout_dst)
    stderr_path = copy_if_exists(stderr_src, stderr_dst)

    final_appreciation = ""
    if final_prompt_src.is_file():
        final_appreciation = final_prompt_src.read_text(encoding="utf-8").strip()

    return {
        "variant": variant,
        "sample_id": sample_id,
        "title": title,
        "image_name": str(row.get("image_name", "")).strip(),
        "image_path": str(row.get("image_path", "")).strip(),
        "status": status,
        "returncode": int(row.get("returncode", 0) or 0),
        "sample_output_dir": str(row.get("sample_output_dir", "")).strip(),
        "latest_run_dir": str(latest_run_dir) if str(latest_run_dir) else "",
        "final_prompt_path": final_prompt_path,
        "closed_loop_report_path": report_path,
        "slots_final_path": slots_final_path,
        "stdout_log_path": stdout_path,
        "stderr_log_path": stderr_path,
        "acc_input_path": final_prompt_path,
        "final_appreciation": final_appreciation,
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]], header: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in header})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Copy and summarize ablation outputs into a unified ACC-friendly directory."
    )
    parser.add_argument("--run-index", default=str(DEFAULT_RUN_INDEX))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_index_path = Path(args.run_index).expanduser()
    output_root = ensure_dir(Path(args.output_root).expanduser())

    rows = read_jsonl(run_index_path)
    exported_rows = [export_variant_record(row=row, output_root=output_root) for row in rows]

    index_jsonl = output_root / "ablation_acc_index.jsonl"
    index_csv = output_root / "ablation_acc_index.csv"
    summary_json = output_root / "summary.json"
    variant_manifest = output_root / "variant_manifest.json"

    write_jsonl(index_jsonl, exported_rows)
    write_csv(
        index_csv,
        exported_rows,
        header=[
            "variant",
            "sample_id",
            "title",
            "image_name",
            "status",
            "returncode",
            "acc_input_path",
            "closed_loop_report_path",
            "slots_final_path",
            "latest_run_dir",
        ],
    )

    per_variant: dict[str, list[dict[str, Any]]] = {}
    for row in exported_rows:
        per_variant.setdefault(row["variant"], []).append(row)

    manifest_rows: list[dict[str, Any]] = []
    status_counter = Counter(row["status"] for row in exported_rows)
    for variant, variant_rows in sorted(per_variant.items()):
        variant_index = output_root / variant / "index.jsonl"
        variant_csv = output_root / variant / "index.csv"
        write_jsonl(variant_index, variant_rows)
        write_csv(
            variant_csv,
            variant_rows,
            header=[
                "sample_id",
                "title",
                "image_name",
                "status",
                "acc_input_path",
                "closed_loop_report_path",
                "slots_final_path",
                "latest_run_dir",
            ],
        )
        manifest_rows.append(
            {
                "variant": variant,
                "count": len(variant_rows),
                "ok": sum(1 for row in variant_rows if row["status"] == "ok"),
                "acc_input_dir": str(output_root / variant / "final_prompts"),
                "index_jsonl": str(variant_index),
                "index_csv": str(variant_csv),
            }
        )

    variant_manifest.write_text(json.dumps(manifest_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {
        "run_index": str(run_index_path),
        "output_root": str(output_root),
        "total_rows": len(exported_rows),
        "status_counts": dict(status_counter),
        "index_jsonl": str(index_jsonl),
        "index_csv": str(index_csv),
        "variant_manifest": str(variant_manifest),
    }
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"ACC export root: {output_root}")
    print(f"ACC index jsonl: {index_jsonl}")
    print(f"ACC index csv: {index_csv}")
    print(f"Variant manifest: {variant_manifest}")


if __name__ == "__main__":
    main()
