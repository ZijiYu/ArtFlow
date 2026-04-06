#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

PIPELINE_ROOT = Path("/Users/ken/MM/Pipeline")
CLUSTER_MATCH_ROOT = PIPELINE_ROOT / "cluster_match"
sys.path.insert(0, str(CLUSTER_MATCH_ROOT))

from cluster_match.runner import main as runner_main
from run_multisource_extract_eval import compute_pair_metrics


AB_ROOT = PIPELINE_ROOT / "ablation"
DEFAULT_ACC_INDEX = AB_ROOT / "acc_export" / "ablation_acc_index.jsonl"
DEFAULT_GT_DIR = CLUSTER_MATCH_ROOT / "extracted_result_v2" / "gt"
DEFAULT_OUTPUT_ROOT = CLUSTER_MATCH_ROOT / "extracted_result_v2" / "ablation"
DEFAULT_CONFIG = CLUSTER_MATCH_ROOT / "test_config.yaml"
DEFAULT_QUESTION = "请对这幅国画做严谨分析。"


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[ablation_cluster_match][{now}] {message}", flush=True)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


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
        description="Run cluster_match extraction and ACC metrics for ablation final prompts."
    )
    parser.add_argument("--acc-index", default=str(DEFAULT_ACC_INDEX))
    parser.add_argument("--gt-dir", default=str(DEFAULT_GT_DIR))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--question", default=DEFAULT_QUESTION)
    parser.add_argument("--schema-profile", default="simple_v1")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--model", default="")
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--overwrite", action="store_true")
    return parser


def build_variant_dataset(
    *,
    rows: list[dict[str, Any]],
    question: str,
    gt_dir: Path,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    dataset_rows: list[dict[str, Any]] = []
    missing_gt: list[str] = []
    missing_prompt: list[str] = []

    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        prompt_path = Path(str(row.get("acc_input_path", "")).strip()).expanduser()
        gt_path = gt_dir / f"{sample_id}.json"
        if not sample_id:
            continue
        if not prompt_path.is_file():
            missing_prompt.append(sample_id)
            continue
        if not gt_path.is_file():
            missing_gt.append(sample_id)
            continue
        dataset_rows.append({"sample_id": sample_id, "question": question})

    sample_ids = [row["sample_id"] for row in dataset_rows]
    return dataset_rows, sample_ids, missing_gt + missing_prompt


def run_variant_extraction(
    *,
    variant: str,
    rows: list[dict[str, Any]],
    gt_dir: Path,
    output_root: Path,
    config_path: Path,
    question: str,
    schema_profile: str,
    temperature: float,
    max_retries: int,
    timeout: int,
    model: str,
    overwrite: bool,
) -> dict[str, Any]:
    variant_root = ensure_dir(output_root / variant)
    dataset_dir = ensure_dir(output_root / "_datasets")
    dataset_rows, sample_ids, missing_inputs = build_variant_dataset(rows=rows, question=question, gt_dir=gt_dir)
    dataset_path = dataset_dir / f"{variant}.jsonl"
    write_jsonl(dataset_path, dataset_rows)

    prompt_dir = Path(str(rows[0].get("acc_input_path", "")).strip()).expanduser().parent if rows else Path()
    extracted_json_dir = ensure_dir(variant_root / "enhanced")

    existing = sum(1 for sample_id in sample_ids if (extracted_json_dir / f"{sample_id}.json").is_file())
    missing_extract = [sample_id for sample_id in sample_ids if not (extracted_json_dir / f"{sample_id}.json").is_file()]

    log(
        f"variant={variant} samples={len(sample_ids)} existing={existing} "
        f"to_extract={len(missing_extract)} missing_inputs={len(missing_inputs)}"
    )

    if missing_extract or overwrite:
        args = [
            "--config",
            str(config_path),
            "--dataset-jsonl",
            str(dataset_path),
            "--enhanced-dir",
            str(prompt_dir),
            "--output-dir",
            str(variant_root),
            "--schema-profile",
            schema_profile,
            "--sources",
            "enhanced",
            "--temperature",
            str(temperature),
            "--max-retries",
            str(max_retries),
            "--timeout",
            str(timeout),
        ]
        if model.strip():
            args.extend(["--model", model.strip()])
        if overwrite:
            args.append("--overwrite")
        for sample_id in sample_ids:
            args.extend(["--sample-id", sample_id])
        code = runner_main(args)
        if code != 0:
            raise RuntimeError(f"cluster_match extraction failed for variant={variant} code={code}")
    else:
        log(f"variant={variant} reuse existing extracted jsons")

    metrics = compute_pair_metrics(
        baseline_json_dir=gt_dir,
        target_json_dir=extracted_json_dir,
        sample_ids=sample_ids,
    )
    metrics["variant"] = variant
    metrics["n_samples"] = float(len(sample_ids))
    metrics["missing_inputs"] = float(len(missing_inputs))

    index_rows: list[dict[str, Any]] = []
    for row in rows:
        sample_id = str(row.get("sample_id", "")).strip()
        if sample_id not in sample_ids:
            continue
        index_rows.append(
            {
                "variant": variant,
                "sample_id": sample_id,
                "title": str(row.get("title", "")).strip(),
                "acc_input_path": str(row.get("acc_input_path", "")).strip(),
                "gt_path": str(gt_dir / f"{sample_id}.json"),
                "extracted_json_path": str(extracted_json_dir / f"{sample_id}.json"),
            }
        )

    write_json(variant_root / "metrics.json", metrics)
    write_jsonl(variant_root / "acc_index.jsonl", index_rows)
    write_csv(
        variant_root / "acc_index.csv",
        index_rows,
        header=["variant", "sample_id", "title", "acc_input_path", "gt_path", "extracted_json_path"],
    )
    return {
        "variant": variant,
        "metrics": metrics,
        "index_rows": index_rows,
        "variant_root": str(variant_root),
        "prompt_dir": str(prompt_dir),
        "dataset_path": str(dataset_path),
    }


def main() -> None:
    args = build_parser().parse_args()
    acc_index_path = Path(args.acc_index).expanduser()
    gt_dir = Path(args.gt_dir).expanduser()
    output_root = ensure_dir(Path(args.output_root).expanduser())
    config_path = Path(args.config).expanduser()

    rows = read_jsonl(acc_index_path)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        variant = str(row.get("variant", "")).strip()
        if not variant:
            continue
        grouped[variant].append(row)

    if args.variant:
        selected = {item.strip() for item in args.variant if item.strip()}
        grouped = {variant: variant_rows for variant, variant_rows in grouped.items() if variant in selected}

    if not grouped:
        raise SystemExit("No ablation variants selected for ACC evaluation.")

    summary_rows: list[dict[str, Any]] = []
    overall_index_rows: list[dict[str, Any]] = []
    for variant in sorted(grouped):
        result = run_variant_extraction(
            variant=variant,
            rows=grouped[variant],
            gt_dir=gt_dir,
            output_root=output_root,
            config_path=config_path,
            question=args.question,
            schema_profile=args.schema_profile,
            temperature=args.temperature,
            max_retries=args.max_retries,
            timeout=args.timeout,
            model=args.model,
            overwrite=args.overwrite,
        )
        summary_rows.append(result["metrics"])
        overall_index_rows.extend(result["index_rows"])
        log(
            f"variant={variant} ACC={result['metrics']['ACC']:.4f} "
            f"Precision={result['metrics']['Precision']:.4f} F1={result['metrics']['F1']:.4f}"
        )

    write_jsonl(output_root / "acc_eval_index.jsonl", overall_index_rows)
    write_csv(
        output_root / "acc_eval_index.csv",
        overall_index_rows,
        header=["variant", "sample_id", "title", "acc_input_path", "gt_path", "extracted_json_path"],
    )
    write_csv(
        output_root / "overall_metrics.csv",
        summary_rows,
        header=[
            "variant",
            "n_samples",
            "missing_inputs",
            "ACC",
            "Precision",
            "Recall",
            "F1",
            "TP_w",
            "Matched_TP_w",
            "N_extra_gen",
            "N_extra_gen_success",
            "Extra_Rate",
            "N_Gen",
            "N_GT",
            "N_matched",
        ],
    )
    summary = {
        "acc_index": str(acc_index_path),
        "gt_dir": str(gt_dir),
        "output_root": str(output_root),
        "variants": sorted(grouped),
        "metrics_table": str(output_root / "overall_metrics.csv"),
        "index_jsonl": str(output_root / "acc_eval_index.jsonl"),
        "index_csv": str(output_root / "acc_eval_index.csv"),
    }
    write_json(output_root / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
