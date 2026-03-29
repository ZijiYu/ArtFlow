#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from cluster_match.config import DEFAULT_FINAL_CONFIG_PATH, RuntimeConfig
from cluster_match.evaluation import SemanticJudge, compute_weighted_metrics, extract_leaf_items, match_leaf_items
from cluster_match.client import ChatCompletionsClient


DEFAULT_RESULTS_ROOT = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts")
DEFAULT_OUTPUT_ROOT = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/evaluation")
DEFAULT_DATASET_PATH = Path("/Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl")
DEFAULT_CACHE_PATH = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/eval_cache/llm_judge_cache.json")
SOURCE_DIRS = {
    "baseline": "pilot30_baseline",
    "enhanced": "enhanced",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate cluster_match results with weighted ACC / Precision / F1.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--config", default=str(DEFAULT_FINAL_CONFIG_PATH))
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--judge-mode", choices=("llm", "exact"), default="llm")
    parser.add_argument("--cache-path", default=str(DEFAULT_CACHE_PATH))
    parser.add_argument("--sample-id", default="")
    return parser


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[cluster_match_eval][{now}] {message}", flush=True)


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_eta(processed: int, total: int, started_at: float) -> tuple[float, float, float]:
    if processed <= 0:
        return 0.0, 0.0, 0.0
    elapsed = max(0.0, perf_counter() - started_at)
    average = elapsed / processed
    remaining = max(0, total - processed)
    eta = average * remaining
    return elapsed, average, eta


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Songti SC", "STSong", "Noto Serif CJK SC", "SimSun", "DejaVu Serif"],
            "axes.unicode_minus": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
        }
    )


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_dataset(path: Path) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            sample_id = str(payload.get("sample_id", "")).strip()
            if sample_id:
                rows[sample_id] = payload
    return rows


def load_jsons(directory: Path) -> dict[str, dict]:
    payloads: dict[str, dict] = {}
    if not directory.exists():
        return payloads
    for path in sorted(directory.glob("*.json")):
        if not path.is_file():
            continue
        payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def create_figures(output_dir: Path, sample_df: pd.DataFrame, summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return

    figure_dir = ensure_dir(output_dir / "figures")

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    sns.barplot(data=summary_df, x="metric", y="value", hue="scope", palette="Set2", ax=ax)
    ax.set_title("Weighted Evaluation Metrics")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    save_figure(fig, figure_dir / "overall_metrics")

    if sample_df.empty:
        return

    sample_plot_df = sample_df.melt(
        id_vars=["sample_id"],
        value_vars=["ACC", "Precision", "F1"],
        var_name="metric",
        value_name="value",
    )
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    sns.boxplot(data=sample_plot_df, x="metric", y="value", color="#D9E6F2", width=0.55, ax=ax)
    sns.stripplot(data=sample_plot_df, x="metric", y="value", color="#406E8E", alpha=0.75, size=4, ax=ax)
    ax.set_title("Per-Sample Metric Distribution")
    ax.set_xlabel("")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    save_figure(fig, figure_dir / "sample_metric_distribution")


def write_report(
    output_dir: Path,
    *,
    results_root: Path,
    judge_mode: str,
    judge_fallback_count: int,
    paired_ids: list[str],
    missing_enhanced: list[str],
    overall_metrics: dict[str, float],
    macro_metrics: dict[str, float],
) -> None:
    report = f"""# cluster_match 评测报告

- 结果目录：`{results_root}`
- 评测口径：`baseline = GT`，`enhanced = Gen`
- 额外规则：`enhanced` 中未被 `baseline` 覆盖的额外细节，按成功项计分
- 匹配方式：`{judge_mode}`
- LLM 降级次数：`{judge_fallback_count}`
- 配对样本数：`{len(paired_ids)}`
- 缺失 enhanced 样本数：`{len(missing_enhanced)}`

## Overall Micro Metrics

- `ACC`: {overall_metrics["ACC"]:.4f}
- `Precision`: {overall_metrics["Precision"]:.4f}
- `F1`: {overall_metrics["F1"]:.4f}
- `TP_w`: {overall_metrics["TP_w"]:.2f}
- `Matched_TP_w`: {overall_metrics["Matched_TP_w"]:.2f}
- `N_extra_gen_success`: {int(overall_metrics["N_extra_gen_success"])}
- `N_GT`: {int(overall_metrics["N_GT"])}
- `N_Gen`: {int(overall_metrics["N_Gen"])}
- `N_strong`: {int(overall_metrics["N_strong"])}
- `N_weak`: {int(overall_metrics["N_weak"])}

## Sample Macro Means

- `ACC`: {macro_metrics["ACC"]:.4f}
- `Precision`: {macro_metrics["Precision"]:.4f}
- `F1`: {macro_metrics["F1"]:.4f}

## Missing Enhanced Samples

{chr(10).join(f"- `{sample_id}`" for sample_id in missing_enhanced) if missing_enhanced else "- 无"}
"""
    (output_dir / "evaluation_report.md").write_text(report, encoding="utf-8")


def main() -> int:
    args = build_parser().parse_args()
    results_root = Path(args.results_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    dataset_path = Path(args.dataset_jsonl).expanduser()
    run_dir = ensure_dir(output_root / datetime.now().strftime("%Y%m%d_%H%M%S"))
    table_dir = ensure_dir(run_dir / "tables")
    started_at = perf_counter()

    dataset_map = read_dataset(dataset_path)
    baseline_payloads = load_jsons(results_root / SOURCE_DIRS["baseline"])
    enhanced_payloads = load_jsons(results_root / SOURCE_DIRS["enhanced"])

    if args.sample_id:
        requested = args.sample_id.strip()
        baseline_payloads = {key: value for key, value in baseline_payloads.items() if key == requested}
        enhanced_payloads = {key: value for key, value in enhanced_payloads.items() if key == requested}

    paired_ids = sorted(set(baseline_payloads) & set(enhanced_payloads))
    missing_enhanced = sorted(set(baseline_payloads) - set(enhanced_payloads))

    client = None
    model = args.model.strip()
    if args.judge_mode == "llm":
        runtime = RuntimeConfig.from_sources(
            config_path=args.config,
            model=model or None,
            timeout=args.timeout,
        )
        client = ChatCompletionsClient(
            api_key=runtime.api_key,
            base_url=runtime.base_url,
            timeout=runtime.timeout,
            max_retries=args.max_retries,
        )
        model = runtime.model

    judge = SemanticJudge(
        mode=args.judge_mode,
        client=client,
        model=model,
        cache_path=args.cache_path,
    )

    log(
        f"start results_root={results_root} judge_mode={args.judge_mode} "
        f"baseline={len(baseline_payloads)} enhanced={len(enhanced_payloads)} paired={len(paired_ids)}"
    )
    log(
        f"params model={model or '<none>'} timeout={args.timeout}s max_retries={args.max_retries} "
        f"cache_path={Path(args.cache_path).expanduser()} output_dir={run_dir}"
    )
    log(
        f"paths dataset={dataset_path} baseline_dir={results_root / SOURCE_DIRS['baseline']} "
        f"enhanced_dir={results_root / SOURCE_DIRS['enhanced']}"
    )
    if missing_enhanced:
        log(f"missing_enhanced count={len(missing_enhanced)} samples={','.join(missing_enhanced)}")

    sample_rows: list[dict] = []
    category_rows: list[dict] = []
    match_rows: list[dict] = []

    for index, sample_id in enumerate(paired_ids, start=1):
        processed_before = index - 1
        elapsed_before, average_before, eta_before = estimate_eta(processed_before, len(paired_ids), started_at)
        question = str(dataset_map.get(sample_id, {}).get("question", "")).strip()
        sample_started_at = perf_counter()
        log(
            f"[{index}/{len(paired_ids)}] start sample_id={sample_id} "
            f"elapsed={format_duration(elapsed_before)} avg={format_duration(average_before)} "
            f"eta={format_duration(eta_before)}"
        )
        log(f"[{index}/{len(paired_ids)}] detail question={question}")
        gt_map = extract_leaf_items(baseline_payloads[sample_id])
        gen_map = extract_leaf_items(enhanced_payloads[sample_id])
        sample_counts = {
            "N_GT": 0,
            "N_Gen": 0,
            "N_strong": 0,
            "N_weak": 0,
            "N_extra_gen_success": 0,
        }

        for category_key in sorted(set(gt_map) | set(gen_map)):
            group_name, leaf_name = category_key
            category_path = f"{group_name}/{leaf_name}" if group_name else leaf_name
            gt_items = gt_map.get(category_key, [])
            gen_items = gen_map.get(category_key, [])
            pair_details, metrics = match_leaf_items(
                category_path=category_path,
                gt_items=gt_items,
                gen_items=gen_items,
                judge=judge,
            )

            sample_counts["N_GT"] += int(metrics["N_GT"])
            sample_counts["N_Gen"] += int(metrics["N_Gen"])
            sample_counts["N_strong"] += int(metrics["N_strong"])
            sample_counts["N_weak"] += int(metrics["N_weak"])
            sample_counts["N_extra_gen_success"] += int(metrics["N_extra_gen_success"])

            category_rows.append(
                {
                    "sample_id": sample_id,
                    "category_group": group_name,
                    "category_name": leaf_name,
                    "category_path": category_path,
                    **metrics,
                }
            )

            for detail in pair_details:
                match_rows.append(
                    {
                        "sample_id": sample_id,
                        "category_group": group_name,
                        "category_name": leaf_name,
                        **detail,
                    }
                )

        sample_metrics = compute_weighted_metrics(
            sample_counts["N_GT"],
            sample_counts["N_Gen"],
            sample_counts["N_strong"],
            sample_counts["N_weak"],
            sample_counts["N_extra_gen_success"],
        )
        sample_rows.append(
            {
                "sample_id": sample_id,
                "question": question,
                **sample_metrics,
            }
        )
        elapsed_after, average_after, eta_after = estimate_eta(index, len(paired_ids), started_at)
        sample_seconds = perf_counter() - sample_started_at
        log(
            f"[{index}/{len(paired_ids)}] done sample_id={sample_id} "
            f"job_time={format_duration(sample_seconds)} elapsed={format_duration(elapsed_after)} "
            f"avg={format_duration(average_after)} eta={format_duration(eta_after)} "
            f"N_GT={int(sample_metrics['N_GT'])} N_Gen={int(sample_metrics['N_Gen'])} "
            f"N_strong={int(sample_metrics['N_strong'])} N_weak={int(sample_metrics['N_weak'])} "
            f"N_extra_gen_success={int(sample_metrics['N_extra_gen_success'])} "
            f"ACC={sample_metrics['ACC']:.4f} Precision={sample_metrics['Precision']:.4f} F1={sample_metrics['F1']:.4f}"
        )

    judge.save_cache()

    sample_df = pd.DataFrame(sample_rows)
    category_df = pd.DataFrame(category_rows)
    match_df = pd.DataFrame(match_rows)

    if not sample_df.empty:
        overall_metrics = compute_weighted_metrics(
            int(sample_df["N_GT"].sum()),
            int(sample_df["N_Gen"].sum()),
            int(sample_df["N_strong"].sum()),
            int(sample_df["N_weak"].sum()),
            int(sample_df["N_extra_gen_success"].sum()),
        )
        macro_metrics = {
            "ACC": float(sample_df["ACC"].mean()),
            "Precision": float(sample_df["Precision"].mean()),
            "F1": float(sample_df["F1"].mean()),
        }
    else:
        overall_metrics = compute_weighted_metrics(0, 0, 0, 0, 0)
        macro_metrics = {"ACC": 0.0, "Precision": 0.0, "F1": 0.0}

    if not category_df.empty:
        category_summary_df = (
            category_df.groupby(["category_group", "category_name", "category_path"], dropna=False, as_index=False)[
                ["N_GT", "N_Gen", "N_strong", "N_weak", "N_extra_gen_success"]
            ]
            .sum()
        )
        category_metric_rows = []
        for row in category_summary_df.to_dict(orient="records"):
            metrics = compute_weighted_metrics(
                int(row["N_GT"]),
                int(row["N_Gen"]),
                int(row["N_strong"]),
                int(row["N_weak"]),
                int(row["N_extra_gen_success"]),
            )
            category_metric_rows.append({**row, **metrics})
        category_summary_df = pd.DataFrame(category_metric_rows)
    else:
        category_summary_df = pd.DataFrame()

    summary_df = pd.DataFrame(
        [
            {"scope": "Micro overall", "metric": "ACC", "value": overall_metrics["ACC"]},
            {"scope": "Micro overall", "metric": "Precision", "value": overall_metrics["Precision"]},
            {"scope": "Micro overall", "metric": "F1", "value": overall_metrics["F1"]},
            {"scope": "Macro sample mean", "metric": "ACC", "value": macro_metrics["ACC"]},
            {"scope": "Macro sample mean", "metric": "Precision", "value": macro_metrics["Precision"]},
            {"scope": "Macro sample mean", "metric": "F1", "value": macro_metrics["F1"]},
        ]
    )

    sample_df.to_csv(table_dir / "sample_scores.csv", index=False, encoding="utf-8")
    category_df.to_csv(table_dir / "category_scores_by_sample.csv", index=False, encoding="utf-8")
    category_summary_df.to_csv(table_dir / "category_scores_summary.csv", index=False, encoding="utf-8")
    match_df.to_csv(table_dir / "match_details.csv", index=False, encoding="utf-8")
    summary_df.to_csv(table_dir / "overall_metrics.csv", index=False, encoding="utf-8")

    summary_payload = {
        "results_root": str(results_root),
        "output_dir": str(run_dir),
        "judge_mode": args.judge_mode,
        "model": model,
        "judge_fallback_count": judge.fallback_count,
        "judge_fallback_errors": judge.fallback_errors,
        "paired_sample_count": len(paired_ids),
        "missing_enhanced_count": len(missing_enhanced),
        "missing_enhanced_samples": missing_enhanced,
        "overall_metrics": overall_metrics,
        "macro_metrics": macro_metrics,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    configure_plot_style()
    create_figures(run_dir, sample_df, summary_df)
    write_report(
        run_dir,
        results_root=results_root,
        judge_mode=args.judge_mode,
        judge_fallback_count=judge.fallback_count,
        paired_ids=paired_ids,
        missing_enhanced=missing_enhanced,
        overall_metrics=overall_metrics,
        macro_metrics=macro_metrics,
    )

    log(
        f"done output_dir={run_dir} ACC={overall_metrics['ACC']:.4f} "
        f"Precision={overall_metrics['Precision']:.4f} F1={overall_metrics['F1']:.4f} "
        f"judge_fallbacks={judge.fallback_count}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
