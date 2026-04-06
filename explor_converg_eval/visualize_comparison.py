"""
对比结果可视化与报告导出脚本
读取 comparison CSV 文件并生成图表、汇总表和术语报告
"""
import argparse
import json
import ast
import os
from pathlib import Path

MPLCONFIGDIR = Path(__file__).parent / ".mplconfig"
MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPLCONFIGDIR))

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# 数据源显示名称映射
SOURCE_DISPLAY_NAMES = {
    "final_prompts": "Pipeline最终版本",
    "ground_truth": "Ground Truth",
    "zhihua_0": "Zhihua初版"
}


def normalize_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    将 comparison CSV 中的字符串字典列还原为 Python dict。
    """
    normalized_df = df.copy()

    for column in ["dimension_slots", "weight_distribution"]:
        if column in normalized_df.columns:
            normalized_df[column] = normalized_df[column].apply(
                lambda value: ast.literal_eval(value) if isinstance(value, str) and value else value
            )

    return normalized_df


def build_dimension_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    由 dimension_slots 列构建“数据源 x 维度”的平均命中矩阵。
    """
    rows = []
    for source in df["model_name"].unique():
        source_df = df[df["model_name"] == source]
        dimension_mean = {}
        for dim_slots in source_df["dimension_slots"]:
            for dim_name, count in dim_slots.items():
                dimension_mean.setdefault(dim_name, []).append(count)

        rows.append({
            "model_name": source,
            **{dim_name: sum(values) / len(values) for dim_name, values in dimension_mean.items()}
        })

    matrix_df = pd.DataFrame(rows).fillna(0)
    matrix_df["model_name"] = matrix_df["model_name"].map(lambda x: SOURCE_DISPLAY_NAMES.get(x, x))
    matrix_df = matrix_df.set_index("model_name")
    return matrix_df


def visualize_scatter(df: pd.DataFrame, output_dir: Path):
    """
    图1: Diversity vs Depth 散点图（按模型分子图）
    """
    sources = list(df["model_name"].dropna().unique())
    if not sources:
        return

    colors = {
        "ground_truth": "#d62728",   # red
        "final_prompts": "#1f77b4",  # blue
        "pipeline": "#1f77b4",       # blue (alias)
        "zhihua_0": "#ff7f0e",       # orange
        "zhihua": "#ff7f0e",         # orange (alias)
        "gpt_5.4": "#9467bd",        # purple
        "gemini_3.1": "#2ca02c",     # green
        "qwen3.5-397b": "#17becf",   # cyan
    }
    markers = {
        "ground_truth": "*",
        "final_prompts": "o",
        "pipeline": "o",
        "zhihua_0": "s",
        "zhihua": "s",
        "gpt_5.4": "^",
        "gemini_3.1": "D",
        "qwen3.5-397b": "P",
    }

    entropy_min, entropy_max = df["entropy"].min(), df["entropy"].max()
    depth_min, depth_max = df["weighted_density"].min(), df["weighted_density"].max()
    x_pad = (entropy_max - entropy_min) * 0.08 if entropy_max > entropy_min else 0.1
    y_pad = (depth_max - depth_min) * 0.08 if depth_max > depth_min else 0.005
    xlim = (entropy_min - x_pad, entropy_max + x_pad)
    ylim = (depth_min - y_pad, depth_max + y_pad)

    # GT 参考均值（若存在）
    gt_df = df[df['model_name'] == 'ground_truth']
    gt_entropy_mean = None
    gt_wdensity_mean = None
    if len(gt_df) > 0:
        gt_entropy_mean = gt_df['entropy'].mean()
        gt_wdensity_mean = gt_df['weighted_density'].mean()

    source_count = len(sources)
    n_cols = 3 if source_count > 4 else 2
    n_rows = (source_count + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.6 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    for idx, source in enumerate(sources):
        ax = axes_flat[idx]
        source_df = df[df['model_name'] == source]
        display_name = SOURCE_DISPLAY_NAMES.get(source, source)

        ax.scatter(
            source_df['entropy'],
            source_df['weighted_density'],
            s=130 if source == 'ground_truth' else 75,
            alpha=0.78,
            color=colors.get(source, 'gray'),
            marker=markers.get(source, 'o'),
            edgecolors='black',
            linewidths=0.45
        )

        if gt_entropy_mean is not None and gt_wdensity_mean is not None:
            ax.axvline(gt_entropy_mean, color='black', linestyle='--', alpha=0.35, linewidth=1.0)
            ax.axhline(gt_wdensity_mean, color='black', linestyle='--', alpha=0.35, linewidth=1.0)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_title(f"{display_name} (n={len(source_df)})", fontsize=11, fontweight='bold')
        ax.set_xlabel('Entropy', fontsize=10)
        ax.set_ylabel('W_Density', fontsize=10)
        ax.grid(True, alpha=0.25, linestyle='--')

    for idx in range(source_count, len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle('多数据源 Diversity vs Depth（分面子图）', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    output_path = output_dir / 'scatter_diversity_depth.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已生成散点图: {output_path}")
    plt.close()


def visualize_bars(df: pd.DataFrame, output_dir: Path):
    """
    图2: 平均指标对比柱状图
    """
    summary = df.groupby('model_name').agg({
        'entropy': 'mean',
        'weighted_density': 'mean',
        'avg_weight': 'mean',
        'density': 'mean'
    }).round(4)
    
    # 重命名索引为中文
    summary.index = [SOURCE_DISPLAY_NAMES.get(idx, idx) for idx in summary.index]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 图2.1: Entropy
    summary['entropy'].plot(kind='bar', ax=axes[0, 0], color='steelblue', alpha=0.8)
    axes[0, 0].set_title('Diversity (Entropy) - 知识广度', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('熵值', fontsize=10)
    axes[0, 0].set_ylim([0, 4])
    axes[0, 0].axhline(y=3.2, color='red', linestyle='--', linewidth=1, alpha=0.5, label='优秀线 (3.2)')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    axes[0, 0].legend(fontsize=9)
    
    # 图2.2: Weighted_Density
    summary['weighted_density'].plot(kind='bar', ax=axes[0, 1], color='coral', alpha=0.8)
    axes[0, 1].set_title('Depth (Weighted_Density) - 知识深度', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('加权密度', fontsize=10)
    axes[0, 1].axhline(y=0.06, color='red', linestyle='--', linewidth=1, alpha=0.5, label='优秀线 (0.06)')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    axes[0, 1].legend(fontsize=9)
    
    # 图2.3: avg_weight
    summary['avg_weight'].plot(kind='bar', ax=axes[1, 0], color='mediumseagreen', alpha=0.8)
    axes[1, 0].set_title('平均权重 - 专业水平', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('权重值', fontsize=10)
    axes[1, 0].set_ylim([0, 3.5])
    axes[1, 0].axhline(y=2.0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='目标线 (2.0)')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    axes[1, 0].legend(fontsize=9)
    
    # 图2.4: Density (传统)
    summary['density'].plot(kind='bar', ax=axes[1, 1], color='mediumpurple', alpha=0.8)
    axes[1, 1].set_title('Density (传统密度) - 参考', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('密度值', fontsize=10)
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    for ax in axes.flat:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
        # 在柱子上方标注数值
        for container in ax.containers:
            ax.bar_label(container, fmt='%.3f', fontsize=9)
    
    plt.suptitle('三数据源核心指标对比', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = output_dir / 'bar_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已生成柱状图: {output_path}")
    plt.close()


def visualize_weight_distribution(df: pd.DataFrame, output_dir: Path):
    """
    图3: 权重分布堆叠柱状图
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 收集每个数据源的权重分布
    weight_data = {}
    
    for source in df['model_name'].unique():
        source_df = df[df['model_name'] == source]
        
        # 聚合权重分布
        level_1_total = 0
        level_2_total = 0
        level_3_total = 0
        
        for dist_str in source_df['weight_distribution']:
            if isinstance(dist_str, str):
                dist = eval(dist_str)
            else:
                dist = dist_str
            
            level_1_total += dist.get(1, 0)
            level_2_total += dist.get(2, 0)
            level_3_total += dist.get(3, 0)
        
        total = level_1_total + level_2_total + level_3_total
        
        weight_data[source] = {
            'Level 1 (宏观)': level_1_total / total * 100 if total > 0 else 0,
            'Level 2 (中观)': level_2_total / total * 100 if total > 0 else 0,
            'Level 3 (微观)': level_3_total / total * 100 if total > 0 else 0
        }
    
    # 转为 DataFrame
    weight_df = pd.DataFrame(weight_data).T
    weight_df.index = [SOURCE_DISPLAY_NAMES.get(idx, idx) for idx in weight_df.index]
    
    # 绘制堆叠柱状图
    weight_df.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=['lightcoral', 'lightskyblue', 'lightgreen'],
        alpha=0.8,
        edgecolor='black',
        linewidth=0.5
    )
    
    ax.set_title('三数据源权重分布对比', fontsize=14, fontweight='bold', pad=20)
    ax.set_ylabel('占比 (%)', fontsize=12)
    ax.set_xlabel('')
    ax.set_ylim([0, 100])
    ax.legend(title='权重级别', loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha='right')
    
    # 添加百分比标注
    for container in ax.containers:
        labels = [f'{v:.1f}%' if v > 5 else '' for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    output_path = output_dir / 'weight_distribution_stacked.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已生成权重分布图: {output_path}")
    plt.close()


def visualize_boxplot(df: pd.DataFrame, output_dir: Path):
    """
    图4: 箱线图 - 展示各指标的分布情况
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 重命名数据源
    df_renamed = df.copy()
    df_renamed['model_name'] = df_renamed['model_name'].map(
        lambda x: SOURCE_DISPLAY_NAMES.get(x, x)
    )
    
    # 箱线图1: Entropy
    sns.boxplot(
        data=df_renamed,
        x='model_name',
        y='entropy',
        hue='model_name',
        ax=axes[0],
        palette=['steelblue', 'crimson', 'darkorange'],
        legend=False
    )
    axes[0].set_title('Diversity (Entropy) 分布', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('')
    axes[0].set_ylabel('熵值', fontsize=10)
    axes[0].axhline(y=3.2, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=15)
    
    # 箱线图2: Weighted_Density
    sns.boxplot(
        data=df_renamed,
        x='model_name',
        y='weighted_density',
        hue='model_name',
        ax=axes[1],
        palette=['steelblue', 'crimson', 'darkorange'],
        legend=False
    )
    axes[1].set_title('Depth (Weighted_Density) 分布', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('')
    axes[1].set_ylabel('加权密度', fontsize=10)
    axes[1].axhline(y=0.06, color='red', linestyle='--', linewidth=1, alpha=0.5)
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=15)
    
    plt.suptitle('指标分布对比（箱线图）', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    output_path = output_dir / 'boxplot_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✅ 已生成箱线图: {output_path}")
    plt.close()


def visualize_dimension_matrix(df: pd.DataFrame, output_dir: Path):
    """
    图5: 维度矩阵热力图（matrix）
    """
    matrix_df = build_dimension_matrix(df)

    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "平均 Slot 数"}
    )

    ax.set_title("三数据源维度矩阵热力图", fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("评价维度", fontsize=11)
    ax.set_ylabel("数据源", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    output_path = output_dir / "dimension_matrix_heatmap.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"✅ 已生成矩阵热力图: {output_path}")
    plt.close()


def export_summary_tables(df: pd.DataFrame, output_dir: Path):
    """
    导出对比汇总表。
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_df = df.groupby("model_name").agg({
        "entropy": ["mean", "std"],
        "weighted_density": ["mean", "std"],
        "density": ["mean", "std"],
        "avg_weight": ["mean", "std"],
        "total_slots": ["mean", "std"],
        "cleaned_text_length": ["mean", "std"],
        "dimension_coverage": ["mean", "std"],
    }).round(4)
    summary_df.columns = ["_".join(col).strip("_") for col in summary_df.columns]
    summary_df = summary_df.reset_index()
    summary_df["model_name"] = summary_df["model_name"].map(lambda x: SOURCE_DISPLAY_NAMES.get(x, x))
    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    matrix_df = build_dimension_matrix(df)
    matrix_path = output_dir / "dimension_matrix.csv"
    matrix_df.to_csv(matrix_path, encoding="utf-8-sig")

    return {
        "summary_metrics": summary_path,
        "dimension_matrix": matrix_path,
    }


def load_extracted_terms(extracted_jsonl_path: Path) -> pd.DataFrame:
    """
    将 structured_data 中的术语拉平成表格。
    """
    rows = []
    with open(extracted_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            structured_data = record.get("structured_data", {})
            for dimension, slots in structured_data.items():
                for slot in slots:
                    rows.append({
                        "image_id": record["image_id"],
                        "source_model": record["source_model"],
                        "extraction_model": record["extraction_model"],
                        "dimension": dimension,
                        "keyword": slot.get("关键词", ""),
                        "relevance": slot.get("相关性", ""),
                        "weight": slot.get("权重", ""),
                        "sentence": slot.get("原句", ""),
                    })

    return pd.DataFrame(rows)


def export_terms_reports(extracted_jsonl_path: Path, output_dir: Path):
    """
    导出术语明细与聚合统计。
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    terms_df = load_extracted_terms(extracted_jsonl_path)

    if terms_df.empty:
        return {}

    terms_df["source_model"] = terms_df["source_model"].map(lambda x: SOURCE_DISPLAY_NAMES.get(x, x))

    flat_path = output_dir / "terms_flattened.csv"
    terms_df.to_csv(flat_path, index=False, encoding="utf-8-sig")

    summary_path = output_dir / "terms_summary_by_source_dimension.csv"
    summary_df = terms_df.groupby(["source_model", "dimension"]).agg(
        term_count=("keyword", "count"),
        unique_term_count=("keyword", "nunique"),
        avg_weight=("weight", "mean"),
    ).reset_index().round(4)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    top_terms_path = output_dir / "top_terms_by_source.csv"
    top_terms_df = terms_df.groupby(["source_model", "keyword"]).agg(
        frequency=("keyword", "count"),
        avg_weight=("weight", "mean"),
        dimensions=("dimension", lambda s: " | ".join(sorted(set(s))))
    ).reset_index()
    top_terms_df = top_terms_df.sort_values(["source_model", "frequency", "avg_weight"], ascending=[True, False, False])
    top_terms_df.to_csv(top_terms_path, index=False, encoding="utf-8-sig")

    return {
        "terms_flattened": flat_path,
        "terms_summary": summary_path,
        "top_terms": top_terms_path,
    }


def write_report_summary(
    df: pd.DataFrame,
    report_dir: Path,
    csv_path: Path,
    extracted_jsonl_path: Path | None,
    figure_paths: dict,
    table_paths: dict,
    term_paths: dict,
):
    """
    生成一份简洁的 Markdown 报告索引。
    """
    summary_df = df.groupby("model_name").agg({
        "entropy": "mean",
        "weighted_density": "mean",
        "density": "mean",
        "avg_weight": "mean",
    }).round(4)
    summary_df.index = [SOURCE_DISPLAY_NAMES.get(idx, idx) for idx in summary_df.index]

    lines = [
        "# Comparison Report",
        "",
        f"- 源 CSV: `{csv_path}`",
        f"- 源 JSONL: `{extracted_jsonl_path}`" if extracted_jsonl_path else "- 源 JSONL: `(none)`",
        f"- 样本总数: `{len(df)}`",
        "",
        "## Summary",
        "",
    ]

    for source_name, row in summary_df.iterrows():
        lines.append(
            f"- `{source_name}`: Entropy={row['entropy']}, "
            f"W_Density={row['weighted_density']}, Density={row['density']}, AvgWeight={row['avg_weight']}"
        )

    lines.extend([
        "",
        "## Figures",
        "",
    ])
    for name, path in figure_paths.items():
        lines.append(f"- `{name}`: `{path}`")

    lines.extend([
        "",
        "## Tables",
        "",
    ])
    for name, path in table_paths.items():
        lines.append(f"- `{name}`: `{path}`")

    if term_paths:
        lines.extend([
            "",
            "## Terms",
            "",
        ])
        for name, path in term_paths.items():
            lines.append(f"- `{name}`: `{path}`")

    report_path = report_dir / "report_summary.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def generate_all_reports(csv_path: str, extracted_jsonl_path: str | None = None) -> Path:
    """
    生成 figures/tables/terms/report_summary.md 的完整报告目录。
    """
    csv_file = Path(csv_path)
    df = pd.read_csv(csv_file, encoding="utf-8-sig")
    df = normalize_comparison_df(df)

    run_name = csv_file.stem
    report_dir = csv_file.parent / "reports" / run_name
    figures_dir = report_dir / "figures"
    tables_dir = report_dir / "tables"
    terms_dir = report_dir / "terms"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    visualize_scatter(df, figures_dir)
    visualize_bars(df, figures_dir)
    visualize_weight_distribution(df, figures_dir)
    visualize_boxplot(df, figures_dir)
    visualize_dimension_matrix(df, figures_dir)

    figure_paths = {
        "scatter_diversity_depth": figures_dir / "scatter_diversity_depth.png",
        "bar_comparison": figures_dir / "bar_comparison.png",
        "weight_distribution_stacked": figures_dir / "weight_distribution_stacked.png",
        "boxplot_comparison": figures_dir / "boxplot_comparison.png",
        "dimension_matrix_heatmap": figures_dir / "dimension_matrix_heatmap.png",
    }

    table_paths = export_summary_tables(df, tables_dir)

    term_paths = {}
    extracted_path = Path(extracted_jsonl_path) if extracted_jsonl_path else None
    if extracted_path and extracted_path.exists():
        term_paths = export_terms_reports(extracted_path, terms_dir)

    report_summary_path = write_report_summary(
        df=df,
        report_dir=report_dir,
        csv_path=csv_file,
        extracted_jsonl_path=extracted_path,
        figure_paths=figure_paths,
        table_paths=table_paths,
        term_paths=term_paths,
    )

    print(f"✅ 已生成 reports 目录: {report_dir}")
    print(f"✅ 报告索引: {report_summary_path}")
    return report_dir


def main(csv_path: str):
    """
    主函数：生成所有可视化图表
    """
    print("=" * 80)
    print("📊 开始生成对比可视化与报告")
    print("=" * 80)

    csv_file = Path(csv_path)
    extracted_candidate = csv_file.with_name(csv_file.name.replace("comparison_", "extracted_").replace(".csv", ".jsonl"))
    extracted_jsonl_path = str(extracted_candidate) if extracted_candidate.exists() else None
    report_dir = generate_all_reports(csv_path, extracted_jsonl_path)

    print("\n" + "=" * 80)
    print(f"✅ 所有报告已生成到: {report_dir}")
    print("=" * 80)
    print("\n生成内容:")
    print("  1. figures/scatter_diversity_depth.png")
    print("  2. figures/bar_comparison.png")
    print("  3. figures/weight_distribution_stacked.png")
    print("  4. figures/boxplot_comparison.png")
    print("  5. figures/dimension_matrix_heatmap.png")
    print("  6. tables/summary_metrics.csv")
    print("  7. tables/dimension_matrix.csv")
    if extracted_jsonl_path:
        print("  8. terms/terms_flattened.csv")
        print("  9. terms/terms_summary_by_source_dimension.csv")
        print(" 10. terms/top_terms_by_source.csv")
    print(" 11. report_summary.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比结果可视化")
    
    parser.add_argument(
        'csv_path',
        type=str,
        help='comparison CSV 文件路径'
    )
    
    args = parser.parse_args()
    
    main(args.csv_path)
