"""
Redraw the five comparison figures in English with a fixed source order.
"""
import argparse
import ast
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


SOURCE_ORDER = [
    "gemini_3.1",
    "qwen3.5-397b",
    "kimi_2.5",
    "gpt_5.4",
    "gpt_4.1",
    "pipeline",
]

DISPLAY_NAMES = {
    "gemini_3.1": "Gemini-3.1-Pro-Preview",
    "qwen3.5-397b": "Qwen-3.5-397B",
    "kimi_2.5": "Kimi-2.5",
    "gpt_5.4": "GPT-5.4",
    "gpt_4.1": "GPT-4.1",
    "pipeline": "ArtFlow",
}

DIMENSION_NAMES = {
    "材质形制": "Material and Format",
    "构图布局": "Composition and Layout",
    "用笔特点": "Brushwork",
    "色彩氛围": "Color and Atmosphere",
    "题材内容": "Subject Matter",
    "形神表现": "Form and Spirit",
    "艺术风格": "Artistic Style",
    "意境营造": "Artistic Conception",
    "象征寓意": "Symbolism",
    "画家信息": "Artist Information",
    "创作年代": "Creation Period",
    "题跋印章": "Inscriptions and Seals",
    "艺术传承": "Artistic Lineage",
    "历史语境": "Historical Context",
    "艺术地位": "Artistic Significance",
}

PALETTE = {
    "gemini_3.1": "#2E86AB",
    "qwen3.5-397b": "#4F6D7A",
    "kimi_2.5": "#F18F01",
    "gpt_5.4": "#8E5572",
    "gpt_4.1": "#C73E1D",
    "pipeline": "#1B998B",
}


def normalize_comparison_df(df: pd.DataFrame) -> pd.DataFrame:
    normalized_df = df.copy()
    for column in ["dimension_slots", "weight_distribution"]:
        if column in normalized_df.columns:
            normalized_df[column] = normalized_df[column].apply(
                lambda value: ast.literal_eval(value) if isinstance(value, str) and value else value
            )
    return normalized_df


def filter_and_order(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df[df["model_name"].isin(SOURCE_ORDER)].copy()
    filtered["model_name"] = pd.Categorical(filtered["model_name"], SOURCE_ORDER, ordered=True)
    return filtered.sort_values(["model_name", "image_id"]).reset_index(drop=True)


def build_dimension_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in SOURCE_ORDER:
        source_df = df[df["model_name"] == source]
        if source_df.empty:
            continue
        dimension_mean = {}
        for dim_slots in source_df["dimension_slots"]:
            for dim_name, count in dim_slots.items():
                dimension_mean.setdefault(dim_name, []).append(count)

        rows.append({
            "model_name": DISPLAY_NAMES[source],
            **{dim_name: sum(values) / len(values) for dim_name, values in dimension_mean.items()},
        })

    matrix_df = pd.DataFrame(rows).fillna(0)
    matrix_df = matrix_df.rename(columns=DIMENSION_NAMES)
    return matrix_df.set_index("model_name")


def plot_scatter(df: pd.DataFrame, output_dir: Path):
    sources = [source for source in SOURCE_ORDER if source in set(df["model_name"].astype(str))]
    n_cols = 3
    n_rows = (len(sources) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4.8 * n_rows), squeeze=False)
    axes_flat = axes.flatten()

    entropy_min, entropy_max = df["entropy"].min(), df["entropy"].max()
    wd_min, wd_max = df["weighted_density"].min(), df["weighted_density"].max()
    x_pad = (entropy_max - entropy_min) * 0.08 if entropy_max > entropy_min else 0.1
    y_pad = (wd_max - wd_min) * 0.08 if wd_max > wd_min else 0.005

    for idx, source in enumerate(sources):
        ax = axes_flat[idx]
        source_df = df[df["model_name"] == source]
        ax.scatter(
            source_df["entropy"],
            source_df["weighted_density"],
            s=70,
            alpha=0.8,
            color=PALETTE[source],
            edgecolors="black",
            linewidths=0.4,
        )
        ax.set_xlim(entropy_min - x_pad, entropy_max + x_pad)
        ax.set_ylim(wd_min - y_pad, wd_max + y_pad)
        ax.set_title(DISPLAY_NAMES[source], fontsize=11, fontweight="bold")
        ax.set_xlabel("Entropy")
        ax.set_ylabel("Weighted Density")
        ax.grid(True, alpha=0.25, linestyle="--")

    for idx in range(len(sources), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle("Entropy vs Weighted Density", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_dir / "scatter_diversity_depth.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_bars(df: pd.DataFrame, output_dir: Path):
    summary = (
        df.groupby("model_name", observed=False)[["entropy", "weighted_density", "avg_weight", "density"]]
        .mean()
        .reindex(SOURCE_ORDER)
    )
    summary.index = [DISPLAY_NAMES[idx] for idx in summary.index]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metric_specs = [
        ("entropy", "Entropy", "#2E86AB"),
        ("weighted_density", "Weighted Density", "#F18F01"),
        ("avg_weight", "Average Weight", "#1B998B"),
        ("density", "Density", "#8E5572"),
    ]

    for ax, (column, title, color) in zip(axes.flat, metric_specs):
        bars = ax.bar(summary.index, summary[column], color=color, alpha=0.85)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", labelrotation=15)
        ax.bar_label(bars, fmt="%.3f", fontsize=9)

    plt.suptitle("Core Metric Comparison", fontsize=16, fontweight="bold", y=0.99)
    plt.tight_layout()
    plt.savefig(output_dir / "bar_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_weight_distribution(df: pd.DataFrame, output_dir: Path):
    weight_rows = []
    for source in SOURCE_ORDER:
        source_df = df[df["model_name"] == source]
        if source_df.empty:
            continue
        level_1_total = 0
        level_2_total = 0
        level_3_total = 0
        for dist in source_df["weight_distribution"]:
            level_1_total += dist.get(1, 0)
            level_2_total += dist.get(2, 0)
            level_3_total += dist.get(3, 0)
        total = level_1_total + level_2_total + level_3_total
        weight_rows.append({
            "source": DISPLAY_NAMES[source],
            "Level 1": level_1_total / total * 100 if total else 0,
            "Level 2": level_2_total / total * 100 if total else 0,
            "Level 3": level_3_total / total * 100 if total else 0,
        })

    weight_df = pd.DataFrame(weight_rows).set_index("source")
    ax = weight_df.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
        color=["#D95D39", "#4F6D7A", "#7FB069"],
        alpha=0.85,
        edgecolor="black",
        linewidth=0.4,
    )
    ax.set_title("Weight Distribution", fontsize=14, fontweight="bold", pad=18)
    ax.set_ylabel("Percentage (%)")
    ax.set_xlabel("")
    ax.set_ylim(0, 100)
    ax.legend(title="Weight Level", loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
    for container in ax.containers:
        labels = [f"{v:.1f}%" if v > 5 else "" for v in container.datavalues]
        ax.bar_label(container, labels=labels, label_type="center", fontsize=9)
    plt.tight_layout()
    plt.savefig(output_dir / "weight_distribution_stacked.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_boxplot(df: pd.DataFrame, output_dir: Path):
    df_plot = df.copy()
    df_plot["display_name"] = df_plot["model_name"].map(DISPLAY_NAMES)
    order = [DISPLAY_NAMES[source] for source in SOURCE_ORDER if source in set(df["model_name"].astype(str))]
    palette = [PALETTE[source] for source in SOURCE_ORDER if source in set(df["model_name"].astype(str))]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    sns.boxplot(
        data=df_plot,
        x="display_name",
        y="entropy",
        hue="display_name",
        order=order,
        hue_order=order,
        palette=dict(zip(order, palette)),
        ax=axes[0],
        legend=False,
    )
    axes[0].set_title("Entropy", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Value")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].tick_params(axis="x", rotation=15)

    sns.boxplot(
        data=df_plot,
        x="display_name",
        y="weighted_density",
        hue="display_name",
        order=order,
        hue_order=order,
        palette=dict(zip(order, palette)),
        ax=axes[1],
        legend=False,
    )
    axes[1].set_title("Weighted Density", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Value")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].tick_params(axis="x", rotation=15)

    plt.suptitle("Distribution Comparison", fontsize=15, fontweight="bold", y=0.98)
    plt.tight_layout()
    plt.savefig(output_dir / "boxplot_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_dimension_heatmap(df: pd.DataFrame, output_dir: Path):
    matrix_df = build_dimension_matrix(df)
    fig, ax = plt.subplots(figsize=(16, 6))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Average Slot Count"},
    )
    ax.set_title("Dimension Matrix Heatmap", fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Source")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(output_dir / "dimension_matrix_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Redraw comparison figures in English.")
    parser.add_argument("csv_path", type=str, help="Path to the comparison CSV.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for the regenerated figures. Defaults to a sibling 'figures_en'.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "figures_en"
    output_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")
    plt.rcParams["font.family"] = "DejaVu Sans"

    df = pd.read_csv(csv_path, encoding="utf-8-sig")
    df = normalize_comparison_df(df)
    df = filter_and_order(df)

    plot_scatter(df, output_dir)
    plot_bars(df, output_dir)
    plot_weight_distribution(df, output_dir)
    plot_boxplot(df, output_dir)
    plot_dimension_heatmap(df, output_dir)

    print(f"Saved figures to: {output_dir}")


if __name__ == "__main__":
    main()
