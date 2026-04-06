"""
主流程编排：串联四个模块
"""
import asyncio
import argparse
import logging
from pathlib import Path
from datetime import datetime

from config import EXTRACTED_DIR, METRICS_DIR, REPORTS_DIR

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def run_pipeline(
    skip_generation: bool = True,
    skip_extraction: bool = False,
    skip_metrics: bool = False,
    skip_visualization: bool = False,
    extracted_file: str = None
):
    """
    运行完整的评估Pipeline
    
    参数:
        skip_generation: 跳过模块1（默认True，因为zhihua的txt已经生成好了）
        skip_extraction: 跳过模块2
        skip_metrics: 跳过模块3
        skip_visualization: 跳过模块4
        extracted_file: 指定已有的抽取结果文件（跳过模块1和2）
    """
    logger.info("="*80)
    logger.info("开始运行多模型赏析评估Pipeline")
    logger.info("="*80)
    
    latest_extracted_file = extracted_file
    latest_metrics_file = None
    
    # ==================== 模块1：API生成 ====================
    if not skip_generation:
        logger.info("\n[模块1] 开始批量调用API生成赏析文本...")
        from module1_api_generator import APIGenerator, load_ground_truth
        from config import GT_JSONL_PATH, MODEL_CONFIGS, GENERATED_DIR
        
        image_list = load_ground_truth(GT_JSONL_PATH)
        generator = APIGenerator()
        results = await generator.batch_generate(image_list, MODEL_CONFIGS)
        
        output_path = GENERATED_DIR / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        generator.save_results(results, output_path)
        
        logger.info(f"[模块1] 完成！生成了 {len(results)} 条赏析文本")
    else:
        logger.info("\n[模块1] 跳过API生成（使用现有zhihua txt文件）")
    
    # ==================== 模块2：结构化抽取 ====================
    if not skip_extraction and not extracted_file:
        logger.info("\n[模块2] 开始结构化信息抽取...")
        from module2_extractor import StructuredExtractor
        from config import EXTRACTED_DIR
        
        extractor = StructuredExtractor()
        
        # 同时处理GT和zhihua
        all_results = []
        
        # 2.1 处理GT
        logger.info("\n处理Ground Truth...")
        gt_results = await extractor.batch_extract_from_gt()
        all_results.extend(gt_results)
        
        # 2.2 处理zhihua
        logger.info("\n处理zhihua模型输出...")
        zhihua_results = await extractor.batch_extract_from_zhihua()
        all_results.extend(zhihua_results)
        
        # 保存合并结果
        output_path = EXTRACTED_DIR / f"extracted_all_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        extractor.save_results(all_results, output_path)
        
        latest_extracted_file = str(output_path)
        logger.info(f"[模块2] 完成！抽取了 {len(all_results)} 条结构化数据 (GT={len(gt_results)}, Zhihua={len(zhihua_results)})")
    else:
        if extracted_file:
            logger.info(f"\n[模块2] 跳过抽取（使用指定文件: {extracted_file}）")
            latest_extracted_file = extracted_file
        else:
            logger.info("\n[模块2] 跳过抽取")
    
    # ==================== 模块3：指标计算 ====================
    if not skip_metrics and latest_extracted_file:
        logger.info("\n[模块3] 开始计算指标...")
        from module3_metrics import MetricsCalculator, load_extracted_data
        from config import METRICS_DIR
        
        extracted_list = load_extracted_data(Path(latest_extracted_file))
        calculator = MetricsCalculator()
        metrics_results = calculator.batch_calculate(extracted_list)
        
        output_path = METRICS_DIR / f"metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        calculator.save_results(metrics_results, output_path)
        
        latest_metrics_file = str(output_path.with_suffix('.csv'))
        logger.info(f"[模块3] 完成！计算了 {len(metrics_results)} 个样本的指标")
    else:
        logger.info("\n[模块3] 跳过指标计算")
    
    # ==================== 模块4：可视化 ====================
    if not skip_visualization and latest_metrics_file:
        logger.info("\n[模块4] 开始生成可视化报告...")
        from module4_visualization import VisualizationEngine
        import pandas as pd
        
        df = pd.read_csv(latest_metrics_file, encoding='utf-8-sig')
        viz = VisualizationEngine(df)
        viz.generate_all_visualizations()
        
        logger.info(f"[模块4] 完成！可视化报告已生成到: {REPORTS_DIR}")
    else:
        logger.info("\n[模块4] 跳过可视化")
    
    logger.info("\n" + "="*80)
    logger.info("Pipeline执行完毕！")
    logger.info("="*80)


def main():
    """
    命令行入口
    """
    parser = argparse.ArgumentParser(description="多模型赏析评估Pipeline")
    
    parser.add_argument(
        '--skip-generation',
        action='store_true',
        default=True,
        help='跳过模块1（API生成）'
    )
    
    parser.add_argument(
        '--skip-extraction',
        action='store_true',
        help='跳过模块2（结构化抽取）'
    )
    
    parser.add_argument(
        '--skip-metrics',
        action='store_true',
        help='跳过模块3（指标计算）'
    )
    
    parser.add_argument(
        '--skip-visualization',
        action='store_true',
        help='跳过模块4（可视化）'
    )
    
    parser.add_argument(
        '--extracted-file',
        type=str,
        help='指定已有的抽取结果文件（跳过模块1和2）'
    )
    
    parser.add_argument(
        '--run-all',
        action='store_true',
        help='运行完整Pipeline（从zhihua txt开始）'
    )
    
    args = parser.parse_args()
    
    # 如果指定了run-all，清除所有skip标志
    if args.run_all:
        args.skip_extraction = False
        args.skip_metrics = False
        args.skip_visualization = False
    
    asyncio.run(run_pipeline(
        skip_generation=args.skip_generation,
        skip_extraction=args.skip_extraction,
        skip_metrics=args.skip_metrics,
        skip_visualization=args.skip_visualization,
        extracted_file=args.extracted_file
    ))


if __name__ == "__main__":
    main()
