"""
模块4：可视化与报告生成
生成雷达图、散点图等多种可视化图表
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict
import logging
from matplotlib import rcParams

from config import DIMENSIONS, METRICS_DIR, REPORTS_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 配置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'STHeiti']
plt.rcParams['axes.unicode_minus'] = False


class VisualizationEngine:
    """可视化引擎"""
    
    def __init__(self, metrics_df: pd.DataFrame):
        self.df = metrics_df
        self.dimensions = DIMENSIONS
    
    def plot_radar_chart(self, output_path: Path):
        """
        雷达图：展示各模型在15个维度上的命中频率（包含GT作为基准）
        """
        # 计算各模型在各维度的平均Slot数
        model_groups = self.df.groupby('model_name')
        
        fig, ax = plt.subplots(figsize=(14, 11), subplot_kw=dict(projection='polar'))
        
        # 设置角度
        angles = np.linspace(0, 2 * np.pi, len(self.dimensions), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # GT使用特殊样式（红色粗线）
        gt_drawn = False
        other_models = []
        
        for model_name, group in model_groups:
            values = []
            for dim in self.dimensions:
                col_name = dim
                if col_name in group.columns:
                    avg_slots = group[col_name].mean()
                else:
                    avg_slots = 0
                values.append(avg_slots)
            
            values += values[:1]  # 闭合
            
            if model_name == "ground_truth":
                # GT用红色粗线标注
                ax.plot(angles, values, 'o-', linewidth=3, label='Ground Truth (基准)', 
                       color='red', markersize=8, zorder=10)
                ax.fill(angles, values, alpha=0.1, color='red')
                gt_drawn = True
            else:
                other_models.append((model_name, values))
        
        # 绘制其他模型
        colors = plt.cm.tab10(np.linspace(0, 1, len(other_models)))
        for (model_name, values), color in zip(other_models, colors):
            ax.plot(angles, values, 'o-', linewidth=2, label=model_name, color=color, alpha=0.8)
            ax.fill(angles, values, alpha=0.1, color=color)
        
        # 设置标签
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(self.dimensions, fontsize=10)
        ax.set_ylim(0, None)
        ax.set_title('各模型在15个维度上的平均Slot数量\n（红线为Ground Truth基准）', 
                    fontsize=16, pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=10)
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"雷达图已保存: {output_path}")
    
    def plot_entropy_density_scatter(self, output_path: Path):
        """
        散点图：Entropy vs Density
        展示各模型的广度-深度分布（包含GT基准点）
        """
        fig, ax = plt.subplots(figsize=(14, 9))
        
        # 分离GT和其他模型
        gt_df = self.df[self.df['model_name'] == 'ground_truth']
        other_df = self.df[self.df['model_name'] != 'ground_truth']
        
        # 先绘制GT（红色五角星，较大）
        if not gt_df.empty:
            ax.scatter(
                gt_df['entropy'],
                gt_df['density'],
                s=300,
                alpha=0.8,
                label='Ground Truth (基准)',
                color='red',
                marker='*',
                edgecolors='darkred',
                linewidths=2,
                zorder=10
            )
            
            # 使用GT的均值作为象限划分线
            gt_entropy_mean = gt_df['entropy'].mean()
            gt_density_mean = gt_df['density'].mean()
            
            ax.axvline(gt_entropy_mean, color='red', linestyle='--', alpha=0.4, linewidth=2, 
                      label=f'GT均值: Entropy={gt_entropy_mean:.3f}')
            ax.axhline(gt_density_mean, color='red', linestyle='--', alpha=0.4, linewidth=2,
                      label=f'GT均值: Density={gt_density_mean:.4f}')
        else:
            # 如果没有GT，使用所有数据的中位数
            gt_entropy_mean = self.df['entropy'].median()
            gt_density_mean = self.df['density'].median()
            ax.axvline(gt_entropy_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.axhline(gt_density_mean, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        
        # 绘制其他模型
        if not other_df.empty:
            model_names = other_df['model_name'].unique()
            colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))
            
            for model_name, color in zip(model_names, colors):
                model_df = other_df[other_df['model_name'] == model_name]
                
                ax.scatter(
                    model_df['entropy'],
                    model_df['density'],
                    s=120,
                    alpha=0.6,
                    label=model_name,
                    color=color,
                    edgecolors='white',
                    linewidths=1
                )
        
        # 添加象限标注（基于GT均值）
        x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
        y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
        
        ax.text(ax.get_xlim()[0] + x_range*0.05, ax.get_ylim()[1] - y_range*0.05, 
                '狭窄但精炼', fontsize=11, alpha=0.7, weight='bold')
        ax.text(ax.get_xlim()[1] - x_range*0.15, ax.get_ylim()[1] - y_range*0.05, 
                '🏆 理想区域\n（广度+深度）', fontsize=11, alpha=0.7, weight='bold', 
                ha='right', color='green')
        ax.text(ax.get_xlim()[0] + x_range*0.05, ax.get_ylim()[0] + y_range*0.05, 
                '❌ 又窄又浅', fontsize=11, alpha=0.7, weight='bold', color='darkred')
        ax.text(ax.get_xlim()[1] - x_range*0.15, ax.get_ylim()[0] + y_range*0.05, 
                '⚠️ 又臭又长\n（广而不精）', fontsize=11, alpha=0.7, weight='bold', 
                ha='right', color='orange')
        
        ax.set_xlabel('Entropy（知识点分布熵 - 广度）', fontsize=13, weight='bold')
        ax.set_ylabel('Density（干货密度 - 深度）', fontsize=13, weight='bold')
        ax.set_title('模型赏析质量分布图：Entropy vs Density\n（红色虚线=GT基准）', 
                    fontsize=16, weight='bold', pad=15)
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"散点图已保存: {output_path}")
    
    def plot_dimension_heatmap(self, output_path: Path):
        """
        热力图：各模型在15个维度上的Slot数量
        """
        # 计算各模型在各维度的平均Slot数
        pivot_data = []
        
        for model_name in self.df['model_name'].unique():
            model_df = self.df[self.df['model_name'] == model_name]
            row = {'model': model_name}
            
            for dim in self.dimensions:
                if dim in model_df.columns:
                    row[dim] = model_df[dim].mean()
                else:
                    row[dim] = 0
            
            pivot_data.append(row)
        
        heatmap_df = pd.DataFrame(pivot_data)
        heatmap_df = heatmap_df.set_index('model')
        
        # 绘制热力图
        fig, ax = plt.subplots(figsize=(16, 6))
        sns.heatmap(
            heatmap_df,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            linewidths=0.5,
            ax=ax,
            cbar_kws={'label': '平均Slot数'}
        )
        
        ax.set_title('各模型在15个维度上的平均Slot数量热力图', fontsize=16, pad=15)
        ax.set_xlabel('评价维度', fontsize=12)
        ax.set_ylabel('模型', fontsize=12)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"热力图已保存: {output_path}")
    
    def plot_slot_distribution(self, output_path: Path):
        """
        箱线图：各模型的Slot数量分布
        """
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # 左图：总Slot数分布
        self.df.boxplot(column='total_slots', by='model_name', ax=axes[0])
        axes[0].set_title('各模型的总Slot数量分布', fontsize=14)
        axes[0].set_xlabel('模型', fontsize=12)
        axes[0].set_ylabel('总Slot数量', fontsize=12)
        axes[0].get_figure().suptitle('')  # 移除默认标题
        
        # 右图：维度覆盖分布
        self.df.boxplot(column='dimension_coverage', by='model_name', ax=axes[1])
        axes[1].set_title('各模型的维度覆盖数量分布', fontsize=14)
        axes[1].set_xlabel('模型', fontsize=12)
        axes[1].set_ylabel('覆盖的维度数（最多15）', fontsize=12)
        axes[1].get_figure().suptitle('')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"箱线图已保存: {output_path}")
    
    def plot_gt_comparison_bars(self, output_path: Path):
        """
        柱状图：各模型与GT的差异对比
        """
        # 只取非GT的模型
        model_df = self.df[self.df['model_name'] != 'ground_truth']
        
        if model_df.empty or 'entropy_diff_from_gt' not in model_df.columns:
            logger.warning("没有与GT的对比数据，跳过对比图表")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        
        # 按模型分组，计算均值
        comparison_data = model_df.groupby('model_name').agg({
            'entropy_diff_from_gt': 'mean',
            'density_diff_from_gt': 'mean',
            'dimension_distance_from_gt': 'mean'
        }).reset_index()
        
        # 1. Entropy差异
        axes[0].bar(comparison_data['model_name'], comparison_data['entropy_diff_from_gt'])
        axes[0].set_title('与GT的Entropy差异\n（越低越接近GT）', fontsize=12, weight='bold')
        axes[0].set_ylabel('|Entropy - GT_Entropy|', fontsize=11)
        axes[0].tick_params(axis='x', rotation=45)
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 2. Density差异
        axes[1].bar(comparison_data['model_name'], comparison_data['density_diff_from_gt'], color='orange')
        axes[1].set_title('与GT的Density差异\n（越低越接近GT）', fontsize=12, weight='bold')
        axes[1].set_ylabel('|Density - GT_Density|', fontsize=11)
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 3. 维度分布距离
        axes[2].bar(comparison_data['model_name'], comparison_data['dimension_distance_from_gt'], color='green')
        axes[2].set_title('与GT的维度分布距离\n（余弦距离，越低越相似）', fontsize=12, weight='bold')
        axes[2].set_ylabel('Cosine Distance', fontsize=11)
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"GT对比图已保存: {output_path}")
    
    def generate_summary_table(self, output_path: Path):
        """
        生成汇总表：各模型的综合指标对比
        """
        summary = self.df.groupby('model_name').agg({
            'total_slots': ['mean', 'std'],
            'cleaned_text_length': ['mean', 'std'],
            'dimension_coverage': ['mean', 'std'],
            'entropy': ['mean', 'std'],
            'density': ['mean', 'std'],
            'strong_relevant_ratio': 'mean',
            'weak_relevant_ratio': 'mean',
            'irrelevant_ratio': 'mean',
        }).round(4)
        
        # 保存为CSV
        summary.to_csv(output_path, encoding='utf-8-sig')
        
        # 打印到控制台
        logger.info("\n" + "="*80)
        logger.info("各模型综合指标对比:")
        logger.info("="*80)
        logger.info("\n" + str(summary))
        logger.info("="*80)
        
        logger.info(f"汇总表已保存: {output_path}")
    
    def generate_all_visualizations(self, output_prefix: str = "report"):
        """
        生成所有可视化图表
        """
        from datetime import datetime
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 1. 雷达图
        self.plot_radar_chart(REPORTS_DIR / f"{output_prefix}_radar_{timestamp}.png")
        
        # 2. 散点图
        self.plot_entropy_density_scatter(REPORTS_DIR / f"{output_prefix}_scatter_{timestamp}.png")
        
        # 3. 热力图
        self.plot_dimension_heatmap(REPORTS_DIR / f"{output_prefix}_heatmap_{timestamp}.png")
        
        # 4. 箱线图
        self.plot_slot_distribution(REPORTS_DIR / f"{output_prefix}_boxplot_{timestamp}.png")
        
        # 5. 汇总表
        self.generate_summary_table(REPORTS_DIR / f"{output_prefix}_summary_{timestamp}.csv")
        
        logger.info(f"\n所有可视化图表已生成到: {REPORTS_DIR}")


def main(metrics_file: str):
    """
    模块4主函数
    
    参数:
        metrics_file: 模块3生成的指标结果文件路径
    """
    # 加载指标数据
    metrics_path = Path(metrics_file)
    if not metrics_path.exists():
        logger.error(f"文件不存在: {metrics_path}")
        return
    
    df = pd.read_csv(metrics_path, encoding='utf-8-sig')
    logger.info(f"加载了 {len(df)} 条指标数据")
    
    # 生成可视化
    viz = VisualizationEngine(df)
    viz.generate_all_visualizations()


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        logger.error("请提供指标结果文件路径")
        logger.info("用法: python module4_visualization.py <metrics_file.csv>")
