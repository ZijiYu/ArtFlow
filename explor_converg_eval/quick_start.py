"""
快速启动脚本：用于测试和快速运行
"""
import asyncio
import logging
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def quick_test():
    """
    快速测试：先处理3个样本验证流程
    """
    from module2_extractor import StructuredExtractor
    from module3_metrics import MetricsCalculator
    from module4_visualization import VisualizationEngine
    import pandas as pd
    import json
    from config import ZHIHUA_TXT_DIR, GT_JSONL_PATH, EXTRACTED_DIR, METRICS_DIR
    from utils import clean_text
    
    logger.info("="*80)
    logger.info("快速测试模式：处理前3个样本")
    logger.info("="*80)
    
    extractor = StructuredExtractor()
    
    # 1. 测试GT抽取（取1个）
    logger.info("\n[测试1] 抽取GT样本...")
    gt_results = []
    with open(GT_JSONL_PATH, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 1:  # 只取1个
                break
            data = json.loads(line)
            image_id = Path(data['image']).stem
            text = data['assistant']
            
            result = await extractor.process_single_text(image_id, text, "ground_truth")
            if result:
                gt_results.append(result)
                logger.info(f"  ✓ GT样本 {image_id}: {result.total_slots_after_dedup} slots")
    
    # 2. 测试zhihua抽取（取2个）
    logger.info("\n[测试2] 抽取zhihua样本...")
    zhihua_results = []
    txt_files = list(ZHIHUA_TXT_DIR.glob("*.txt"))[:2]
    
    for txt_file in txt_files:
        image_id = txt_file.stem
        with open(txt_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        result = await extractor.process_single_text(image_id, text, "zhihua")
        if result:
            zhihua_results.append(result)
            logger.info(f"  ✓ Zhihua样本 {image_id}: {result.total_slots_after_dedup} slots")
    
    # 3. 测试指标计算
    logger.info("\n[测试3] 计算指标...")
    calculator = MetricsCalculator()
    
    gt_metrics = calculator.batch_calculate(gt_results)
    gt_dict = {m.image_id: m for m in gt_metrics}
    
    model_metrics = calculator.batch_calculate(zhihua_results, gt_dict)
    
    all_metrics = gt_metrics + model_metrics
    
    for m in all_metrics:
        logger.info(f"  {m.model_name} | {m.image_id[:20]}... | "
                   f"Entropy={m.entropy:.3f}, Density={m.density:.4f}")
    
    logger.info("\n" + "="*80)
    logger.info("✅ 快速测试完成！流程验证成功")
    logger.info("="*80)
    logger.info("\n提示：运行完整Pipeline请使用:")
    logger.info("  python main.py --run-all")


async def run_full_pipeline():
    """
    运行完整Pipeline
    """
    from main import run_pipeline
    await run_pipeline(
        skip_generation=True,
        skip_extraction=False,
        skip_metrics=False,
        skip_visualization=False
    )


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        # 运行完整流程
        asyncio.run(run_full_pipeline())
    else:
        # 快速测试
        asyncio.run(quick_test())
