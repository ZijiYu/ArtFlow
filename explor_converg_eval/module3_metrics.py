"""
模块3：指标计算引擎
计算Entropy、Density等核心指标
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional
import logging

from config import DIMENSIONS, EXTRACTED_DIR, METRICS_DIR
from models import ExtractedData, MetricsResult
from utils import filter_relevant_slots, count_relevance_stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MetricsCalculator:
    """指标计算引擎"""
    
    def __init__(self):
        self.dimensions = DIMENSIONS
    
    def calculate_entropy(self, dimension_slots: Dict[str, int], total_slots: int) -> float:
        """
        计算知识点分布熵（Entropy）
        
        公式: Entropy = -∑(p_i * log2(p_i))
        其中 p_i = 第i个维度的Slot数 / 总Slot数
        
        参数:
            dimension_slots: 各维度的Slot数量字典
            total_slots: 总Slot数量
        
        返回:
            熵值
        """
        if total_slots == 0:
            return 0.0
        
        entropy = 0.0
        for dim in self.dimensions:
            slot_count = dimension_slots.get(dim, 0)
            if slot_count == 0:
                continue  # p_i = 0 时该项为0
            
            p_i = slot_count / total_slots
            entropy -= p_i * np.log2(p_i)
        
        return entropy
    
    def calculate_density(self, total_slots: int, text_length: int) -> float:
        """
        计算干货密度（Density）
        
        公式: Density = N_total_slots / L_text
        
        参数:
            total_slots: 有效Slot总数（去重后）
            text_length: 清洗后的文本字符数
        
        返回:
            密度值
        """
        if text_length == 0:
            return 0.0
        
        return total_slots / text_length
    
    def calculate_weighted_density(self, relevant_slots: List, text_length: int) -> float:
        """
        计算加权密度（Weighted_Density）
        
        公式: Weighted_Density = Σ(Weight_i) / L_text
        
        参数:
            relevant_slots: 有效Slot列表（强相关+弱相关）
            text_length: 清洗后的文本字符数
        
        返回:
            加权密度值
        """
        if text_length == 0:
            return 0.0
        
        total_weight = sum(slot.权重 for slot in relevant_slots)
        return total_weight / text_length
    
    def calculate_weight_statistics(self, relevant_slots: List) -> tuple:
        """
        计算权重统计信息
        
        返回:
            (total_weight, avg_weight, weight_distribution)
        """
        if len(relevant_slots) == 0:
            return 0, 0.0, {1: 0, 2: 0, 3: 0}
        
        total_weight = sum(slot.权重 for slot in relevant_slots)
        avg_weight = total_weight / len(relevant_slots)
        
        weight_distribution = {1: 0, 2: 0, 3: 0}
        for slot in relevant_slots:
            weight_distribution[slot.权重] = weight_distribution.get(slot.权重, 0) + 1
        
        return total_weight, avg_weight, weight_distribution
    
    def calculate_dimension_difference(
        self,
        model_slots: Dict[str, int],
        gt_slots: Dict[str, int]
    ) -> float:
        """
        计算模型与GT在维度分布上的差异（使用余弦距离）
        
        返回值越小表示越接近GT
        """
        # 转为向量
        model_vec = np.array([model_slots.get(dim, 0) for dim in self.dimensions])
        gt_vec = np.array([gt_slots.get(dim, 0) for dim in self.dimensions])
        
        # 避免除零
        model_norm = np.linalg.norm(model_vec)
        gt_norm = np.linalg.norm(gt_vec)
        
        if model_norm == 0 or gt_norm == 0:
            return 1.0  # 最大差异
        
        # 余弦相似度
        cosine_sim = np.dot(model_vec, gt_vec) / (model_norm * gt_norm)
        
        # 转为距离（0=完全相同，1=完全不同）
        return 1.0 - cosine_sim
    
    def calculate_metrics_for_single(self, extracted: ExtractedData) -> MetricsResult:
        """
        为单个抽取结果计算所有指标
        """
        structured_data = extracted.structured_data
        
        # 1. 统计各维度的Slot数量（只统计相关的）
        dimension_slots = {}
        all_slots = []
        
        for dim in self.dimensions:
            slots = getattr(structured_data, dim, [])
            # 过滤掉不相关的
            relevant_slots = filter_relevant_slots(slots)
            dimension_slots[dim] = len(relevant_slots)
            all_slots.extend(relevant_slots)
        
        # 2. 计算总Slot数
        total_slots = len(all_slots)
        
        # 3. 计算维度覆盖率
        dimension_coverage = sum(1 for count in dimension_slots.values() if count > 0)
        
        # 4. 计算Entropy
        entropy = self.calculate_entropy(dimension_slots, total_slots)
        
        # 5. 计算Density
        density = self.calculate_density(total_slots, extracted.cleaned_text_length)
        
        # 6. 计算Weighted_Density（新增）
        weighted_density = self.calculate_weighted_density(all_slots, extracted.cleaned_text_length)
        
        # 7. 计算权重统计（新增）
        total_weight, avg_weight, weight_distribution = self.calculate_weight_statistics(all_slots)
        
        # 8. 统计相关性分布
        strong, weak, irrelevant = count_relevance_stats(
            [slot for dim in self.dimensions for slot in getattr(structured_data, dim, [])]
        )
        total_with_irrelevant = strong + weak + irrelevant
        
        strong_ratio = strong / total_with_irrelevant if total_with_irrelevant > 0 else 0
        weak_ratio = weak / total_with_irrelevant if total_with_irrelevant > 0 else 0
        irrelevant_ratio = irrelevant / total_with_irrelevant if total_with_irrelevant > 0 else 0
        
        return MetricsResult(
            image_id=extracted.image_id,
            model_name=extracted.source_model,
            total_slots=total_slots,
            cleaned_text_length=extracted.cleaned_text_length,
            dimension_coverage=dimension_coverage,
            dimension_slots=dimension_slots,
            entropy=entropy,
            density=density,
            weighted_density=weighted_density,
            total_weight=total_weight,
            avg_weight=avg_weight,
            weight_distribution=weight_distribution,
            strong_relevant_ratio=strong_ratio,
            weak_relevant_ratio=weak_ratio,
            irrelevant_ratio=irrelevant_ratio
        )
    
    def batch_calculate(
        self, 
        extracted_list: List[ExtractedData],
        gt_metrics_dict: Optional[Dict[str, MetricsResult]] = None
    ) -> List[MetricsResult]:
        """
        批量计算指标
        
        参数:
            extracted_list: 待计算的抽取数据列表
            gt_metrics_dict: GT的指标字典 {image_id: MetricsResult}，用于对比
        """
        results = []
        for extracted in extracted_list:
            try:
                metrics = self.calculate_metrics_for_single(extracted)
                
                # 如果有GT数据，计算与GT的差异
                if gt_metrics_dict and extracted.image_id in gt_metrics_dict:
                    gt_metrics = gt_metrics_dict[extracted.image_id]
                    
                    # 计算Entropy差异
                    metrics.entropy_diff_from_gt = abs(metrics.entropy - gt_metrics.entropy)
                    
                    # 计算Density差异
                    metrics.density_diff_from_gt = abs(metrics.density - gt_metrics.density)
                    
                    # 计算Weighted_Density差异（新增）
                    metrics.weighted_density_diff_from_gt = abs(metrics.weighted_density - gt_metrics.weighted_density)
                    
                    # 计算维度分布差异
                    metrics.dimension_distance_from_gt = self.calculate_dimension_difference(
                        metrics.dimension_slots,
                        gt_metrics.dimension_slots
                    )
                
                results.append(metrics)
            except Exception as e:
                logger.error(f"指标计算失败 {extracted.image_id}: {str(e)}")
        
        logger.info(f"完成 {len(results)} 个样本的指标计算")
        return results
    
    def save_results(self, results: List[MetricsResult], output_path: Path):
        """
        保存指标结果到CSV和JSONL
        """
        # 保存为JSONL
        jsonl_path = output_path.with_suffix('.jsonl')
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result.model_dump_json(ensure_ascii=False) + '\n')
        
        # 保存为CSV（方便Excel查看）
        csv_path = output_path.with_suffix('.csv')
        df = pd.DataFrame([r.model_dump() for r in results])
        
        # 展开dimension_slots为单独的列
        dimension_cols = pd.json_normalize(df['dimension_slots'])
        df = pd.concat([df.drop('dimension_slots', axis=1), dimension_cols], axis=1)
        
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        logger.info(f"已保存指标结果:")
        logger.info(f"  - JSONL: {jsonl_path}")
        logger.info(f"  - CSV: {csv_path}")


def load_extracted_data(extracted_path: Path) -> List[ExtractedData]:
    """
    加载抽取的结构化数据
    """
    results = []
    with open(extracted_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                results.append(ExtractedData(**data))
    
    logger.info(f"加载了 {len(results)} 条抽取数据")
    return results


def main(extracted_file: str, gt_file: Optional[str] = None):
    """
    模块3主函数
    
    参数:
        extracted_file: 模块2生成的抽取结果文件路径（全部或只有模型数据）
        gt_file: GT的抽取结果文件路径（可选，如果extracted_file已包含GT则不需要）
    """
    # 加载抽取数据
    extracted_path = Path(extracted_file)
    if not extracted_path.exists():
        logger.error(f"文件不存在: {extracted_path}")
        return
    
    extracted_list = load_extracted_data(extracted_path)
    
    # 分离GT和模型数据
    gt_list = [e for e in extracted_list if e.source_model == "ground_truth"]
    model_list = [e for e in extracted_list if e.source_model != "ground_truth"]
    
    logger.info(f"加载数据: GT={len(gt_list)}, 模型={len(model_list)}")
    
    # 如果指定了单独的GT文件，加载它
    if gt_file and Path(gt_file).exists():
        gt_list = load_extracted_data(Path(gt_file))
        logger.info(f"从单独文件加载GT: {len(gt_list)} 条")
    
    calculator = MetricsCalculator()
    
    # 1. 计算GT的指标（作为基准）
    gt_metrics_list = calculator.batch_calculate(gt_list)
    gt_metrics_dict = {m.image_id: m for m in gt_metrics_list}
    
    logger.info(f"\nGT基准指标:")
    if gt_metrics_list:
        gt_df = pd.DataFrame([m.model_dump() for m in gt_metrics_list])
        logger.info(f"  平均Entropy: {gt_df['entropy'].mean():.4f}")
        logger.info(f"  平均Density: {gt_df['density'].mean():.4f}")
        logger.info(f"  平均Weighted_Density: {gt_df['weighted_density'].mean():.4f}")
        logger.info(f"  平均Slot数: {gt_df['total_slots'].mean():.2f}")
        logger.info(f"  平均权重: {gt_df['avg_weight'].mean():.2f}")
    
    # 2. 计算模型的指标（并与GT对比）
    model_metrics_list = calculator.batch_calculate(model_list, gt_metrics_dict)
    
    # 3. 合并所有指标
    all_metrics = gt_metrics_list + model_metrics_list
    
    # 保存结果
    from datetime import datetime
    output_path = METRICS_DIR / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    calculator.save_results(all_metrics, output_path)
    
    # 打印统计摘要
    if model_metrics_list:
        model_df = pd.DataFrame([m.model_dump() for m in model_metrics_list])
        logger.info("\n" + "="*60)
        logger.info("模型指标统计摘要:")
        logger.info("="*60)
        logger.info(f"\n平均Entropy: {model_df['entropy'].mean():.4f} (±{model_df['entropy'].std():.4f})")
        logger.info(f"平均Density: {model_df['density'].mean():.4f} (±{model_df['density'].std():.4f})")
        logger.info(f"平均Weighted_Density: {model_df['weighted_density'].mean():.4f} (±{model_df['weighted_density'].std():.4f})")
        logger.info(f"平均Slot数: {model_df['total_slots'].mean():.2f} (±{model_df['total_slots'].std():.2f})")
        logger.info(f"平均权重: {model_df['avg_weight'].mean():.2f} (±{model_df['avg_weight'].std():.2f})")
        logger.info(f"平均维度覆盖: {model_df['dimension_coverage'].mean():.2f}/15")
        
        # 如果有GT对比数据
        if 'entropy_diff_from_gt' in model_df.columns:
            logger.info(f"\n与GT的差异:")
            logger.info(f"  Entropy差异: {model_df['entropy_diff_from_gt'].mean():.4f}")
            logger.info(f"  Density差异: {model_df['density_diff_from_gt'].mean():.4f}")
            logger.info(f"  Weighted_Density差异: {model_df['weighted_density_diff_from_gt'].mean():.4f}")
            logger.info(f"  维度分布距离: {model_df['dimension_distance_from_gt'].mean():.4f}")
        
        logger.info("="*60)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        logger.error("请提供抽取结果文件路径")
        logger.info("用法: python module3_metrics.py <extracted_file.jsonl>")
