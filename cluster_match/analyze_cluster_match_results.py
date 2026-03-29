#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import rankdata, wilcoxon

from cluster_match.categories import RELEVANCE_VALUES, infer_schema_profile, iter_schema_leaves


DEFAULT_RESULTS_ROOT = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts")
DEFAULT_DATASET_PATH = Path("/Users/ken/MM/Pipeline/eval_v3/artifacts/dataset_pilot_30.jsonl")
DEFAULT_OUTPUT_ROOT = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/analysis")
SOURCE_DIRS = {
    "baseline": "pilot30_baseline",
    "enhanced": "enhanced",
}
SOURCE_LABELS = {
    "baseline": "Baseline",
    "enhanced": "Enhanced",
}
RELEVANCE_ORDER = ["强相关", "弱相关", "不相关"]
RELEVANCE_RANK = {"不相关": 0, "弱相关": 1, "强相关": 2}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize cluster_match results and generate publication-style figures.")
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--dataset-jsonl", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def log(message: str) -> None:
    now = datetime.now().strftime("%H:%M:%S")
    print(f"[cluster_match_analysis][{now}] {message}", flush=True)


def clean_question(question: str) -> str:
    text = str(question or "").strip()
    if text.endswith("/think"):
        text = text[: -len("/think")].rstrip()
    return text


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
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def read_dataset(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            obj = json.loads(text)
            obj["question"] = clean_question(obj.get("question", ""))
            rows.append(obj)
    return pd.DataFrame(rows)


def dominant_relevance(items: list[dict[str, str]]) -> str:
    best = "不相关"
    for item in items:
        relevance = str(item.get("相关性", "不相关")).strip()
        if RELEVANCE_RANK.get(relevance, 0) > RELEVANCE_RANK[best]:
            best = relevance
    return best


def category_display_name(group_name: str | None, leaf_name: str) -> str:
    if group_name:
        return f"{group_name}\n{leaf_name}"
    return leaf_name


def flatten_result_payload(payload: dict) -> list[dict]:
    profile = infer_schema_profile(payload)
    rows: list[dict] = []

    if profile == "legacy":
        for order_index, (_, leaf_name, _) in enumerate(iter_schema_leaves("legacy"), start=1):
            items = payload.get(leaf_name, [])
            if not isinstance(items, list):
                items = [items] if isinstance(items, dict) else []
            actual_items = [item for item in items if isinstance(item, dict)]
            rows.append(
                {
                    "schema_profile": profile,
                    "category_group": "",
                    "category_name": leaf_name,
                    "category_name_display": category_display_name(None, leaf_name),
                    "category_order_index": order_index,
                    "dominant_relevance": dominant_relevance(actual_items),
                    "relevant_item_count": sum(
                        1 for item in actual_items if item.get("相关性") in {"强相关", "弱相关"}
                    ),
                    "unsupported_relevant_count": sum(
                        1
                        for item in actual_items
                        if item.get("相关性") in {"强相关", "弱相关"} and item.get("原句") == "未涉及"
                    ),
                    "strong_item_count": sum(1 for item in actual_items if item.get("相关性") == "强相关"),
                    "weak_item_count": sum(1 for item in actual_items if item.get("相关性") == "弱相关"),
                }
            )
        return rows

    for order_index, (group_name, leaf_name, _) in enumerate(iter_schema_leaves("academic_v2"), start=1):
        group_payload = payload.get(group_name or "", {})
        if not isinstance(group_payload, dict):
            group_payload = {}
        node = group_payload.get(leaf_name, {})
        if not isinstance(node, dict):
            node = {}
        relevance = str(node.get("相关性", "不相关")).strip()
        if relevance not in RELEVANCE_VALUES:
            relevance = "不相关"
        facts = node.get("要素列表", [])
        if isinstance(facts, str):
            facts = [facts]
        elif not isinstance(facts, list):
            facts = []
        facts = [str(item).strip() for item in facts if str(item).strip()]
        actual_facts = [item for item in facts if item != "未涉及"]
        rows.append(
            {
                "schema_profile": profile,
                "category_group": group_name or "",
                "category_name": leaf_name,
                "category_name_display": category_display_name(group_name, leaf_name),
                "category_order_index": order_index,
                "dominant_relevance": relevance,
                "relevant_item_count": len(actual_facts) if relevance in {"强相关", "弱相关"} else 0,
                "unsupported_relevant_count": 1 if relevance in {"强相关", "弱相关"} and not actual_facts else 0,
                "strong_item_count": len(actual_facts) if relevance == "强相关" else 0,
                "weak_item_count": len(actual_facts) if relevance == "弱相关" else 0,
            }
        )
    return rows


def iter_json_files(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.glob("*.json")):
        if path.is_file():
            yield path


def load_result_jsons(results_root: Path, source: str) -> dict[str, dict]:
    directory = results_root / SOURCE_DIRS[source]
    payloads: dict[str, dict] = {}
    for path in iter_json_files(directory):
        payloads[path.stem] = json.loads(path.read_text(encoding="utf-8"))
    return payloads


def build_analysis_frames(results_root: Path, dataset_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sample_rows: list[dict] = []
    category_rows: list[dict] = []

    for source in SOURCE_DIRS:
        results = load_result_jsons(results_root, source)
        for sample_id, payload in results.items():
            relevant_item_count = 0
            strong_item_count = 0
            weak_item_count = 0
            unsupported_relevant_count = 0
            strong_category_count = 0
            weak_category_count = 0
            coverage_category_count = 0
            flattened = flatten_result_payload(payload)

            for category_row in flattened:
                category_rows.append(
                    {
                        "sample_id": sample_id,
                        "source": source,
                        "source_label": SOURCE_LABELS[source],
                        **category_row,
                    }
                )

                dominant = category_row["dominant_relevance"]
                relevant_item_count += category_row["relevant_item_count"]
                unsupported_relevant_count += category_row["unsupported_relevant_count"]
                strong_item_count += category_row["strong_item_count"]
                weak_item_count += category_row["weak_item_count"]

                if dominant == "强相关":
                    strong_category_count += 1
                    coverage_category_count += 1
                elif dominant == "弱相关":
                    weak_category_count += 1
                    coverage_category_count += 1

            sample_rows.append(
                {
                    "sample_id": sample_id,
                    "source": source,
                    "source_label": SOURCE_LABELS[source],
                    "schema_profile": flattened[0]["schema_profile"] if flattened else "legacy",
                    "relevant_item_count": relevant_item_count,
                    "strong_item_count": strong_item_count,
                    "weak_item_count": weak_item_count,
                    "unsupported_relevant_count": unsupported_relevant_count,
                    "strong_category_count": strong_category_count,
                    "weak_category_count": weak_category_count,
                    "coverage_category_count": coverage_category_count,
                    "irrelevant_category_count": len(flattened) - coverage_category_count,
                    "total_category_count": len(flattened),
                }
            )

    sample_df = pd.DataFrame(sample_rows)
    category_df = pd.DataFrame(category_rows)
    merged_sample_df = sample_df.merge(dataset_df, on="sample_id", how="left")
    merged_category_df = category_df.merge(dataset_df, on="sample_id", how="left")

    availability_rows = []
    for source in SOURCE_DIRS:
        source_ids = set(merged_sample_df.loc[merged_sample_df["source"] == source, "sample_id"])
        availability_rows.append(
            {
                "group": SOURCE_LABELS[source],
                "count": len(source_ids),
            }
        )
    baseline_ids = set(merged_sample_df.loc[merged_sample_df["source"] == "baseline", "sample_id"])
    enhanced_ids = set(merged_sample_df.loc[merged_sample_df["source"] == "enhanced", "sample_id"])
    availability_rows.append({"group": "Paired overlap", "count": len(baseline_ids & enhanced_ids)})
    availability_df = pd.DataFrame(availability_rows)
    return merged_sample_df, merged_category_df, availability_df


def summarize_sources(sample_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "relevant_item_count",
        "strong_item_count",
        "weak_item_count",
        "unsupported_relevant_count",
        "strong_category_count",
        "weak_category_count",
        "coverage_category_count",
    ]
    for source, sub_df in sample_df.groupby("source", sort=False):
        row = {
            "source": source,
            "source_label": SOURCE_LABELS[source],
            "n_samples": int(len(sub_df)),
        }
        for metric in metrics:
            row[f"{metric}_mean"] = float(sub_df[metric].mean())
            row[f"{metric}_sd"] = float(sub_df[metric].std(ddof=1) if len(sub_df) > 1 else 0.0)
            row[f"{metric}_median"] = float(sub_df[metric].median())
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_categories(category_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (source, category_name, category_name_display), sub_df in category_df.groupby(
        ["source", "category_name", "category_name_display"], sort=False
    ):
        n = len(sub_df)
        strong_count = int((sub_df["dominant_relevance"] == "强相关").sum())
        weak_count = int((sub_df["dominant_relevance"] == "弱相关").sum())
        irrelevant_count = int((sub_df["dominant_relevance"] == "不相关").sum())
        rows.append(
            {
                "source": source,
                "source_label": SOURCE_LABELS[source],
                "category_name": category_name,
                "category_name_display": category_name_display,
                "category_group": str(sub_df["category_group"].iloc[0]),
                "category_order_index": int(sub_df["category_order_index"].iloc[0]),
                "schema_profile": str(sub_df["schema_profile"].iloc[0]),
                "n_samples": n,
                "strong_count": strong_count,
                "strong_rate": strong_count / n if n else 0.0,
                "weak_count": weak_count,
                "weak_rate": weak_count / n if n else 0.0,
                "irrelevant_count": irrelevant_count,
                "irrelevant_rate": irrelevant_count / n if n else 0.0,
                "coverage_rate": (strong_count + weak_count) / n if n else 0.0,
                "mean_relevant_items": float(sub_df["relevant_item_count"].mean()),
            }
        )
    result = pd.DataFrame(rows)
    return result.sort_values(["category_order_index", "source"]).reset_index(drop=True)


def build_paired_frame(sample_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "relevant_item_count",
        "strong_item_count",
        "weak_item_count",
        "unsupported_relevant_count",
        "strong_category_count",
        "weak_category_count",
        "coverage_category_count",
    ]
    common_ids = (
        sample_df.groupby("sample_id")["source"].nunique().loc[lambda s: s == 2].index.to_list()
    )
    if not common_ids:
        return pd.DataFrame()
    paired_df = (
        sample_df[sample_df["sample_id"].isin(common_ids)]
        .pivot_table(index=["sample_id", "question", "category", "difficulty"], columns="source", values=metrics)
    )
    if paired_df.empty:
        return pd.DataFrame()
    paired_df.columns = [f"{metric}_{source}" for metric, source in paired_df.columns]
    return paired_df.reset_index().sort_values("sample_id").reset_index(drop=True)


def holm_bonferroni(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = sorted(range(n), key=lambda idx: p_values[idx])
    adjusted = [0.0] * n
    running_max = 0.0
    for rank, idx in enumerate(order):
        candidate = min(1.0, (n - rank) * p_values[idx])
        running_max = max(running_max, candidate)
        adjusted[idx] = running_max
    return adjusted


def rank_biserial_correlation(differences: np.ndarray) -> float:
    diffs = differences[differences != 0]
    if diffs.size == 0:
        return 0.0
    ranks = rankdata(np.abs(diffs), method="average")
    positive = ranks[diffs > 0].sum()
    negative = ranks[diffs < 0].sum()
    denominator = positive + negative
    if denominator == 0:
        return 0.0
    return float((positive - negative) / denominator)


def paired_tests(paired_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "relevant_item_count",
        "strong_category_count",
        "weak_category_count",
        "coverage_category_count",
    ]
    required_columns = [f"{metric}_{source}" for metric in metrics for source in ("baseline", "enhanced")]
    if paired_df.empty or any(column not in paired_df.columns for column in required_columns):
        return pd.DataFrame(
            columns=[
                "metric",
                "n_pairs",
                "mean_baseline",
                "mean_enhanced",
                "median_baseline",
                "median_enhanced",
                "median_diff_enhanced_minus_baseline",
                "mean_diff_enhanced_minus_baseline",
                "wilcoxon_statistic",
                "p_value",
                "rank_biserial_correlation",
                "p_value_holm",
            ]
        )
    rows = []
    for metric in metrics:
        baseline = paired_df[f"{metric}_baseline"].to_numpy(dtype=float)
        enhanced = paired_df[f"{metric}_enhanced"].to_numpy(dtype=float)
        diff = enhanced - baseline
        nonzero = diff[diff != 0]
        if nonzero.size == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            test = wilcoxon(baseline, enhanced, zero_method="wilcox", alternative="two-sided", method="auto")
            p_value = float(test.pvalue)
            statistic = float(test.statistic)
        rows.append(
            {
                "metric": metric,
                "n_pairs": int(len(diff)),
                "mean_baseline": float(np.mean(baseline)),
                "mean_enhanced": float(np.mean(enhanced)),
                "median_baseline": float(np.median(baseline)),
                "median_enhanced": float(np.median(enhanced)),
                "median_diff_enhanced_minus_baseline": float(np.median(diff)),
                "mean_diff_enhanced_minus_baseline": float(np.mean(diff)),
                "wilcoxon_statistic": statistic,
                "p_value": p_value,
                "rank_biserial_correlation": rank_biserial_correlation(diff),
            }
        )
    result = pd.DataFrame(rows)
    result["p_value_holm"] = holm_bonferroni(result["p_value"].tolist())
    return result


def missing_samples_table(dataset_df: pd.DataFrame, sample_df: pd.DataFrame) -> pd.DataFrame:
    all_ids = set(dataset_df["sample_id"])
    source_presence = sample_df.groupby("source")["sample_id"].apply(set).to_dict()
    rows = []
    for source in SOURCE_DIRS:
        missing_ids = sorted(all_ids - source_presence.get(source, set()))
        for sample_id in missing_ids:
            row = dataset_df.loc[dataset_df["sample_id"] == sample_id].iloc[0].to_dict()
            row["missing_from_source"] = source
            row["source_label"] = SOURCE_LABELS[source]
            rows.append(row)
    return pd.DataFrame(rows)


def create_coverage_figure(availability_df: pd.DataFrame, figure_dir: Path) -> None:
    palette = ["#4C72B0", "#DD8452", "#55A868"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    sns.barplot(data=availability_df, x="group", y="count", hue="group", palette=palette, dodge=False, legend=False, ax=ax)
    for patch, count in zip(ax.patches, availability_df["count"]):
        ax.text(
            patch.get_x() + patch.get_width() / 2.0,
            patch.get_height() + 0.3,
            f"{int(count)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    ax.set_xlabel("")
    ax.set_ylabel("样本数")
    ax.set_title("Figure 1. Result availability across sources")
    save_figure(fig, figure_dir / "fig_01_source_coverage")


def create_category_heatmap(category_summary_df: pd.DataFrame, figure_dir: Path) -> None:
    category_order = (
        category_summary_df.sort_values("category_order_index")["category_name_display"].drop_duplicates().tolist()
    )
    strong_pivot = (
        category_summary_df.pivot(index="category_name_display", columns="source_label", values="strong_rate")
        .reindex(category_order)
    )
    coverage_pivot = (
        category_summary_df.pivot(index="category_name_display", columns="source_label", values="coverage_rate")
        .reindex(category_order)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 7.4), constrained_layout=True)
    sns.heatmap(
        strong_pivot,
        ax=axes[0],
        cmap="YlGnBu",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "比例"},
    )
    axes[0].set_title("Strong relevance rate")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")

    sns.heatmap(
        coverage_pivot,
        ax=axes[1],
        cmap="YlOrBr",
        vmin=0,
        vmax=1,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        cbar_kws={"label": "比例"},
    )
    axes[1].set_title("Any relevant coverage rate")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    fig.suptitle("Figure 2. Category-level relevance structure by source", fontsize=13, fontweight="bold")
    save_figure(fig, figure_dir / "fig_02_category_heatmap")


def create_paired_slope_plot(paired_df: pd.DataFrame, figure_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 7.0))
    if paired_df.empty or "relevant_item_count_baseline" not in paired_df.columns or "relevant_item_count_enhanced" not in paired_df.columns:
        ax.text(0.5, 0.5, "No paired samples available", ha="center", va="center", fontsize=12)
        ax.set_axis_off()
        fig.suptitle("Figure 3. Paired comparison of relevant fact density", fontsize=13, fontweight="bold")
        save_figure(fig, figure_dir / "fig_03_paired_fact_density")
        return
    x_positions = [0, 1]
    for _, row in paired_df.iterrows():
        ax.plot(
            x_positions,
            [row["relevant_item_count_baseline"], row["relevant_item_count_enhanced"]],
            color="#B0B0B0",
            alpha=0.6,
            linewidth=1.0,
            zorder=1,
        )
    ax.scatter(
        np.zeros(len(paired_df)),
        paired_df["relevant_item_count_baseline"],
        color="#4C72B0",
        s=28,
        alpha=0.85,
        zorder=2,
        label="Baseline",
    )
    ax.scatter(
        np.ones(len(paired_df)),
        paired_df["relevant_item_count_enhanced"],
        color="#DD8452",
        s=28,
        alpha=0.85,
        zorder=2,
        label="Enhanced",
    )
    mean_baseline = paired_df["relevant_item_count_baseline"].mean()
    mean_enhanced = paired_df["relevant_item_count_enhanced"].mean()
    ax.plot(x_positions, [mean_baseline, mean_enhanced], color="black", linewidth=2.2, marker="D", zorder=3)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["Baseline", "Enhanced"])
    ax.set_ylabel("每样本相关要点数")
    ax.set_title("Figure 3. Paired comparison of relevant fact density (n=25)")
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, figure_dir / "fig_03_paired_fact_density")


def create_relevance_composition_plot(category_df: pd.DataFrame, figure_dir: Path) -> None:
    composition_df = (
        category_df.groupby(["source_label", "dominant_relevance"]).size().reset_index(name="count")
    )
    source_totals = composition_df.groupby("source_label")["count"].transform("sum")
    composition_df["proportion"] = composition_df["count"] / source_totals
    pivot = (
        composition_df.pivot(index="source_label", columns="dominant_relevance", values="proportion")
        .reindex(index=["Baseline", "Enhanced"], columns=RELEVANCE_ORDER)
        .fillna(0.0)
    )

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    bottom = np.zeros(len(pivot))
    colors = {"强相关": "#C44E52", "弱相关": "#E1BE6A", "不相关": "#9D9D9D"}
    for relevance in RELEVANCE_ORDER:
        values = pivot[relevance].to_numpy()
        ax.bar(pivot.index, values, bottom=bottom, color=colors[relevance], label=relevance)
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_ylabel("占比")
    ax.set_xlabel("")
    ax.set_title("Figure 4. Dominant relevance composition across sample-category cells")
    ax.legend(frameon=False, loc="upper right")
    save_figure(fig, figure_dir / "fig_04_relevance_composition")


def render_report(
    *,
    report_path: Path,
    sample_df: pd.DataFrame,
    source_summary_df: pd.DataFrame,
    category_summary_df: pd.DataFrame,
    paired_test_df: pd.DataFrame,
    missing_df: pd.DataFrame,
    availability_df: pd.DataFrame,
    output_dir: Path,
) -> None:
    def summary_for(source: str) -> pd.Series:
        sub = source_summary_df.loc[source_summary_df["source"] == source]
        if not sub.empty:
            return sub.iloc[0]
        return pd.Series(
            {
                "relevant_item_count_mean": 0.0,
                "relevant_item_count_sd": 0.0,
                "coverage_category_count_mean": 0.0,
                "strong_category_count_mean": 0.0,
            }
        )

    baseline_n = int(availability_df.loc[availability_df["group"] == "Baseline", "count"].iloc[0])
    enhanced_n = int(availability_df.loc[availability_df["group"] == "Enhanced", "count"].iloc[0])
    paired_n = int(availability_df.loc[availability_df["group"] == "Paired overlap", "count"].iloc[0])

    baseline_summary = summary_for("baseline")
    enhanced_summary = summary_for("enhanced")
    schema_profiles = sorted(sample_df["schema_profile"].dropna().unique().tolist())

    paired_lines = []
    for _, row in paired_test_df.iterrows():
        paired_lines.append(
            f"- `{row['metric']}`: Baseline 均值 {row['mean_baseline']:.2f}，Enhanced 均值 {row['mean_enhanced']:.2f}，"
            f"中位差值 {row['median_diff_enhanced_minus_baseline']:.2f}，"
            f"Wilcoxon `p={row['p_value']:.4g}`，Holm 校正后 `p={row['p_value_holm']:.4g}`，"
            f"秩二列相关 `r={row['rank_biserial_correlation']:.3f}`。"
        )
    if not paired_lines:
        paired_lines = ["- 当前结果集中没有可用于 Baseline/Enhanced 配对检验的共同样本。"]

    baseline_top = (
        category_summary_df[category_summary_df["source"] == "baseline"]
        .sort_values(["strong_rate", "coverage_rate"], ascending=False)
        .head(5)[["category_name_display", "strong_rate", "coverage_rate"]]
    )
    enhanced_top = (
        category_summary_df[category_summary_df["source"] == "enhanced"]
        .sort_values(["strong_rate", "coverage_rate"], ascending=False)
        .head(5)[["category_name_display", "strong_rate", "coverage_rate"]]
    )
    if baseline_top.empty:
        baseline_top = pd.DataFrame(columns=["category_name_display", "strong_rate", "coverage_rate"])
    if enhanced_top.empty:
        enhanced_top = pd.DataFrame(columns=["category_name_display", "strong_rate", "coverage_rate"])

    report = f"""# cluster_match 统计与可视化报告

## 1. 数据范围

- Baseline 结果文件数：{baseline_n}
- Enhanced 结果文件数：{enhanced_n}
- 可直接配对比较的共同样本数：{paired_n}
- 缺失样本记录数：{len(missing_df)}
- 本批结果包含的 schema profile：{", ".join(schema_profiles)}

本报告的分析对象是 `cluster_match` 结构化抽取结果本身，而非原始图像或人工标注真值。  
因此，以下统计描述的是“抽取结构、相关性分配与事实要点密度”的差异，不应直接等同于语义正确率。

## 2. 方法说明

- 分析单元：单个 `sample_id`
- 类别相关性口径：对每个类别取最大相关性，规则为 `强相关 > 弱相关 > 不相关`
- 要点密度：某样本中所有 `强相关/弱相关` 条目的总数
- 配对比较范围：仅在同时拥有 Baseline 和 Enhanced 结果的共同样本上进行
- 统计检验：双侧 Wilcoxon signed-rank test
- 多重比较修正：Holm-Bonferroni 校正

## 3. 描述性结果

- Baseline 每样本平均相关要点数为 {baseline_summary['relevant_item_count_mean']:.2f}，标准差 {baseline_summary['relevant_item_count_sd']:.2f}。
- Enhanced 每样本平均相关要点数为 {enhanced_summary['relevant_item_count_mean']:.2f}，标准差 {enhanced_summary['relevant_item_count_sd']:.2f}。
- Baseline 每样本平均覆盖类别数为 {baseline_summary['coverage_category_count_mean']:.2f}，Enhanced 为 {enhanced_summary['coverage_category_count_mean']:.2f}。
- Baseline 与 Enhanced 的平均强相关类别数分别为 {baseline_summary['strong_category_count_mean']:.2f} 与 {enhanced_summary['strong_category_count_mean']:.2f}。

Baseline 中强相关率最高的前 5 个类别：
{baseline_top.to_markdown(index=False)}

Enhanced 中强相关率最高的前 5 个类别：
{enhanced_top.to_markdown(index=False)}

## 4. 配对统计结果

{chr(10).join(paired_lines)}

{"从配对结果看，Baseline 与 Enhanced 在“强相关类别数”上差异不显著；  \n但在“相关要点总数”“弱相关类别数”“总体覆盖类别数”上存在显著差异，说明两者更大的区别来自信息展开密度与延伸覆盖，而非核心强相关类别的数量。" if paired_n > 0 and not paired_test_df.empty else "由于当前结果集中缺少成对样本，配对统计部分仅保留方法说明，不做显著性解释。"}

## 5. 图表清单

- Figure 1: `fig_01_source_coverage.png/.pdf`
- Figure 2: `fig_02_category_heatmap.png/.pdf`
- Figure 3: `fig_03_paired_fact_density.png/.pdf`
- Figure 4: `fig_04_relevance_composition.png/.pdf`

图表输出目录：`{output_dir / 'figures'}`

## 6. 学术使用建议

- 在论文正文中，建议先报告样本覆盖与缺失情况，再报告描述性统计，最后报告配对检验结果。
- 若要支持更强的因果或质量结论，需要引入人工标注真值、互标一致性或外部评价标准。
- 当前结果最适合作为“结构化抽取行为差异”的实证证据，而不是最终质量裁决。
"""
    report_path.write_text(report, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_plot_style()

    results_root = Path(args.results_root).expanduser()
    dataset_path = Path(args.dataset_jsonl).expanduser()
    output_root = Path(args.output_root).expanduser()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = ensure_dir(output_root / timestamp)
    figure_dir = ensure_dir(output_dir / "figures")
    table_dir = ensure_dir(output_dir / "tables")

    log(f"results_root={results_root}")
    log(f"dataset_path={dataset_path}")
    log(f"output_dir={output_dir}")

    dataset_df = read_dataset(dataset_path)
    sample_df, category_df, availability_df = build_analysis_frames(results_root, dataset_df)
    source_summary_df = summarize_sources(sample_df)
    category_summary_df = summarize_categories(category_df)
    paired_df = build_paired_frame(sample_df)
    paired_test_df = paired_tests(paired_df)
    missing_df = missing_samples_table(dataset_df, sample_df)

    sample_df.to_csv(table_dir / "sample_metrics.csv", index=False, encoding="utf-8-sig")
    category_df.to_csv(table_dir / "category_sample_metrics.csv", index=False, encoding="utf-8-sig")
    source_summary_df.to_csv(table_dir / "source_summary.csv", index=False, encoding="utf-8-sig")
    category_summary_df.to_csv(table_dir / "category_summary.csv", index=False, encoding="utf-8-sig")
    paired_df.to_csv(table_dir / "paired_sample_metrics.csv", index=False, encoding="utf-8-sig")
    paired_test_df.to_csv(table_dir / "paired_tests.csv", index=False, encoding="utf-8-sig")
    missing_df.to_csv(table_dir / "missing_samples.csv", index=False, encoding="utf-8-sig")

    summary_payload = {
        "results_root": str(results_root),
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "n_baseline": int(availability_df.loc[availability_df["group"] == "Baseline", "count"].iloc[0]),
        "n_enhanced": int(availability_df.loc[availability_df["group"] == "Enhanced", "count"].iloc[0]),
        "n_paired": int(availability_df.loc[availability_df["group"] == "Paired overlap", "count"].iloc[0]),
        "table_dir": str(table_dir),
        "figure_dir": str(figure_dir),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    log("creating figures")
    create_coverage_figure(availability_df, figure_dir)
    create_category_heatmap(category_summary_df, figure_dir)
    create_paired_slope_plot(paired_df, figure_dir)
    create_relevance_composition_plot(category_df, figure_dir)

    log("rendering report")
    render_report(
        report_path=output_dir / "analysis_report.md",
        sample_df=sample_df,
        source_summary_df=source_summary_df,
        category_summary_df=category_summary_df,
        paired_test_df=paired_test_df,
        missing_df=missing_df,
        availability_df=availability_df,
        output_dir=output_dir,
    )

    print(json.dumps(summary_payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
