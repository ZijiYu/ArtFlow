"""
增量运行器：支持逐步添加模型，结果持续累积到result文件夹
"""
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Set
import pandas as pd

from config import (
    EXTRACTED_DIR, 
    METRICS_DIR, 
    REPORTS_DIR,
    EXISTING_MODELS_CACHE,
    LOGS_DIR
)
from module2_extractor import StructuredExtractor
from module3_metrics import MetricsCalculator, load_extracted_data
from module4_visualization import VisualizationEngine

# 配置日志
log_file = LOGS_DIR / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class IncrementalRunner:
    """增量运行器：管理持续累积的评估结果"""
    
    def __init__(self):
        self.master_extracted_file = EXTRACTED_DIR / "master_extracted.jsonl"
        self.master_metrics_file = METRICS_DIR / "master_metrics.csv"
        self.processed_models_cache = EXISTING_MODELS_CACHE
    
    def load_processed_models(self) -> Set[str]:
        """加载已处理的模型列表"""
        if self.processed_models_cache.exists():
            with open(self.processed_models_cache, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_models', []))
        return set()
    
    def save_processed_models(self, models: Set[str]):
        """保存已处理的模型列表"""
        with open(self.processed_models_cache, 'w') as f:
            json.dump({
                'processed_models': list(models),
                'last_updated': datetime.now().isoformat()
            }, f, indent=2)
    
    def append_extracted_data(self, new_data: List, source_label: str):
        """追加新的抽取数据到master文件"""
        # 读取现有数据
        existing_ids = set()
        if self.master_extracted_file.exists():
            with open(self.master_extracted_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        data = json.loads(line)
                        existing_ids.add(f"{data['source_model']}_{data['image_id']}")
        
        # 追加新数据（避免重复）
        new_count = 0
        with open(self.master_extracted_file, 'a', encoding='utf-8') as f:
            for item in new_data:
                key = f"{item.source_model}_{item.image_id}"
                if key not in existing_ids:
                    f.write(item.model_dump_json(ensure_ascii=False) + '\n')
                    new_count += 1
        
        logger.info(f"✅ 追加了 {new_count} 条{source_label}数据到master文件")
        return new_count
    
    def recalculate_all_metrics(self):
        """重新计算所有指标（含新旧数据）"""
        if not self.master_extracted_file.exists():
            logger.error("master_extracted.jsonl不存在，无法计算指标")
            return None
        
        logger.info("🔄 重新计算所有数据的指标...")
        
        # 加载所有抽取数据
        all_extracted = load_extracted_data(self.master_extracted_file)
        
        # 分离GT和模型数据
        gt_list = [e for e in all_extracted if e.source_model == "ground_truth"]
        model_list = [e for e in all_extracted if e.source_model != "ground_truth"]
        
        logger.info(f"数据统计: GT={len(gt_list)}, 模型={len(model_list)}")
        
        # 计算指标
        calculator = MetricsCalculator()
        
        gt_metrics = calculator.batch_calculate(gt_list)
        gt_dict = {m.image_id: m for m in gt_metrics}
        
        model_metrics = calculator.batch_calculate(model_list, gt_dict)
        
        all_metrics = gt_metrics + model_metrics
        
        # 保存为master_metrics
        calculator.save_results(all_metrics, self.master_metrics_file)
        
        logger.info(f"✅ 已更新master_metrics.csv（共{len(all_metrics)}条）")
        
        return all_metrics
    
    def generate_reports(self):
        """生成最新的可视化报告"""
        if not self.master_metrics_file.exists():
            logger.error("master_metrics.csv不存在，无法生成报告")
            return
        
        logger.info("📊 生成可视化报告...")
        
        df = pd.read_csv(self.master_metrics_file, encoding='utf-8-sig')
        viz = VisualizationEngine(df)
        
        # 使用固定文件名（latest_），方便查看最新结果
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # 生成所有图表
        viz.plot_radar_chart(REPORTS_DIR / f"latest_radar.png")
        viz.plot_entropy_density_scatter(REPORTS_DIR / f"latest_scatter.png")
        viz.plot_gt_comparison_bars(REPORTS_DIR / f"latest_gt_comparison.png")
        viz.plot_dimension_heatmap(REPORTS_DIR / f"latest_heatmap.png")
        viz.plot_slot_distribution(REPORTS_DIR / f"latest_boxplot.png")
        viz.generate_summary_table(REPORTS_DIR / f"latest_summary.csv")
        
        # 同时保存带时间戳的版本（存档）
        viz.plot_radar_chart(REPORTS_DIR / f"radar_{timestamp}.png")
        viz.plot_entropy_density_scatter(REPORTS_DIR / f"scatter_{timestamp}.png")
        viz.plot_gt_comparison_bars(REPORTS_DIR / f"gt_comparison_{timestamp}.png")
        
        logger.info(f"✅ 报告已生成到 {REPORTS_DIR}")
        logger.info(f"   查看最新结果: latest_*.png")
    
    async def add_model(self, model_name: str, process_func):
        """
        增量添加一个模型的评估
        
        参数:
            model_name: 模型名称
            process_func: 异步函数，返回ExtractedData列表
        """
        processed_models = self.load_processed_models()
        
        if model_name in processed_models:
            logger.warning(f"⚠️  模型 {model_name} 已处理过，跳过")
            return
        
        logger.info(f"🚀 开始处理模型: {model_name}")
        
        # 执行处理函数
        results = await process_func()
        
        if not results:
            logger.error(f"❌ {model_name} 处理失败，无结果")
            return
        
        # 追加数据
        self.append_extracted_data(results, model_name)
        
        # 标记为已处理
        processed_models.add(model_name)
        self.save_processed_models(processed_models)
        
        logger.info(f"✅ {model_name} 处理完成")


async def run_initial_baseline():
    """
    第一次运行：处理GT和zhihua作为基准
    """
    logger.info("="*80)
    logger.info("🎯 初始基准运行：GT + zhihua")
    logger.info("="*80)
    
    runner = IncrementalRunner()
    extractor = StructuredExtractor()
    
    # 1. 处理GT
    logger.info("\n" + "="*60)
    logger.info("第1步：处理Ground Truth（参考标准）")
    logger.info("="*60)
    
    await runner.add_model("ground_truth", extractor.batch_extract_from_gt)
    
    # 2. 处理zhihua
    logger.info("\n" + "="*60)
    logger.info("第2步：处理zhihua模型输出")
    logger.info("="*60)
    
    await runner.add_model("zhihua", extractor.batch_extract_from_zhihua)
    
    # 3. 计算指标
    logger.info("\n" + "="*60)
    logger.info("第3步：计算指标")
    logger.info("="*60)
    
    metrics = runner.recalculate_all_metrics()
    
    if metrics:
        # 打印统计摘要
        gt_metrics = [m for m in metrics if m.model_name == "ground_truth"]
        zhihua_metrics = [m for m in metrics if m.model_name == "zhihua"]
        
        logger.info("\n" + "📊 统计摘要".center(60, "="))
        
        if gt_metrics:
            logger.info(f"\nGround Truth (基准):")
            logger.info(f"  样本数: {len(gt_metrics)}")
            logger.info(f"  平均Entropy: {sum(m.entropy for m in gt_metrics)/len(gt_metrics):.4f}")
            logger.info(f"  平均Density: {sum(m.density for m in gt_metrics)/len(gt_metrics):.4f}")
            logger.info(f"  平均Slot数: {sum(m.total_slots for m in gt_metrics)/len(gt_metrics):.2f}")
        
        if zhihua_metrics:
            logger.info(f"\nzhihua模型:")
            logger.info(f"  样本数: {len(zhihua_metrics)}")
            logger.info(f"  平均Entropy: {sum(m.entropy for m in zhihua_metrics)/len(zhihua_metrics):.4f}")
            logger.info(f"  平均Density: {sum(m.density for m in zhihua_metrics)/len(zhihua_metrics):.4f}")
            logger.info(f"  平均Slot数: {sum(m.total_slots for m in zhihua_metrics)/len(zhihua_metrics):.2f}")
            
            if gt_metrics:
                avg_entropy_diff = sum(m.entropy_diff_from_gt or 0 for m in zhihua_metrics) / len(zhihua_metrics)
                avg_density_diff = sum(m.density_diff_from_gt or 0 for m in zhihua_metrics) / len(zhihua_metrics)
                logger.info(f"\n与GT的差异:")
                logger.info(f"  Entropy差异: {avg_entropy_diff:.4f}")
                logger.info(f"  Density差异: {avg_density_diff:.4f}")
        
        logger.info("="*60)
    
    # 4. 生成可视化
    logger.info("\n" + "="*60)
    logger.info("第4步：生成可视化报告")
    logger.info("="*60)
    
    runner.generate_reports()
    
    # 完成
    logger.info("\n" + "="*80)
    logger.info("🎉 基准评估完成！")
    logger.info("="*80)
    logger.info(f"\n📂 结果保存在: {runner.master_metrics_file.parent.parent}")
    logger.info(f"\n📊 核心图表:")
    logger.info(f"  - {REPORTS_DIR}/latest_scatter.png  (⭐最重要)")
    logger.info(f"  - {REPORTS_DIR}/latest_gt_comparison.png")
    logger.info(f"  - {REPORTS_DIR}/latest_radar.png")
    logger.info(f"\n📈 数据文件:")
    logger.info(f"  - {runner.master_extracted_file}")
    logger.info(f"  - {runner.master_metrics_file}")
    logger.info(f"\n📝 日志文件: {log_file}")


async def add_new_model(model_name: str, input_dir: Path):
    """
    增量添加新模型的评估
    
    参数:
        model_name: 模型名称（如"gpt-4o", "claude-3.5"）
        input_dir: 该模型输出的txt文件目录
    """
    logger.info("="*80)
    logger.info(f"🔄 增量添加模型: {model_name}")
    logger.info("="*80)
    
    runner = IncrementalRunner()
    extractor = StructuredExtractor()
    
    # 1. 处理新模型的txt文件
    logger.info(f"\n处理 {model_name} 的输出文件（{input_dir}）...")
    
    txt_files = list(input_dir.glob("*.txt"))
    logger.info(f"找到 {len(txt_files)} 个文件")
    
    if not txt_files:
        logger.error(f"❌ 未找到任何txt文件")
        return
    
    # 抽取结构化信息
    from tqdm.asyncio import tqdm
    
    tasks = []
    for txt_file in txt_files:
        image_id = txt_file.stem
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read()
            task = extractor.process_single_text(image_id, text, source_model=model_name)
            tasks.append(task)
        except Exception as e:
            logger.error(f"读取失败 {txt_file}: {e}")
    
    results = []
    for coro in tqdm.as_completed(tasks, desc=f"抽取{model_name}"):
        result = await coro
        if result:
            results.append(result)
    
    # 2. 追加到master文件
    runner.append_extracted_data(results, model_name)
    
    # 3. 重新计算所有指标
    logger.info("\n🔄 重新计算指标（含新模型）...")
    runner.recalculate_all_metrics()
    
    # 4. 重新生成可视化
    logger.info("\n📊 重新生成可视化...")
    runner.generate_reports()
    
    # 5. 标记为已处理
    processed = runner.load_processed_models()
    processed.add(model_name)
    runner.save_processed_models(processed)
    
    logger.info("\n" + "="*80)
    logger.info(f"✅ {model_name} 已添加到评估系统")
    logger.info("="*80)
    logger.info(f"\n查看更新后的结果:")
    logger.info(f"  open {REPORTS_DIR}/latest_scatter.png")


async def show_current_status():
    """显示当前的评估状态"""
    runner = IncrementalRunner()
    
    print("\n" + "="*80)
    print("📊 当前评估状态")
    print("="*80)
    
    # 已处理的模型
    processed = runner.load_processed_models()
    print(f"\n已处理的模型 ({len(processed)}):")
    for model in sorted(processed):
        print(f"  ✓ {model}")
    
    # 数据统计
    if runner.master_extracted_file.exists():
        with open(runner.master_extracted_file, 'r') as f:
            lines = f.readlines()
        print(f"\n抽取数据总量: {len(lines)} 条")
    
    if runner.master_metrics_file.exists():
        df = pd.read_csv(runner.master_metrics_file)
        print(f"指标数据总量: {len(df)} 条")
        print(f"\n各模型样本数:")
        print(df['model_name'].value_counts().to_string())
    
    # 最新报告
    latest_reports = list(REPORTS_DIR.glob("latest_*.png"))
    if latest_reports:
        print(f"\n最新报告 ({len(latest_reports)} 个):")
        for report in sorted(latest_reports):
            print(f"  📊 {report.name}")
    
    print("="*80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) == 1:
        # 默认：运行初始基准（GT + zhihua）
        asyncio.run(run_initial_baseline())
    
    elif sys.argv[1] == "status":
        # 查看状态
        asyncio.run(show_current_status())
    
    elif sys.argv[1] == "add":
        # 增量添加新模型
        if len(sys.argv) < 4:
            print("用法: python incremental_runner.py add <model_name> <input_dir>")
            print("示例: python incremental_runner.py add gpt-4o /path/to/gpt4o_outputs")
            sys.exit(1)
        
        model_name = sys.argv[2]
        input_dir = Path(sys.argv[3])
        
        asyncio.run(add_new_model(model_name, input_dir))
    
    else:
        print("用法:")
        print("  python incremental_runner.py              # 初始运行（GT + zhihua）")
        print("  python incremental_runner.py status       # 查看当前状态")
        print("  python incremental_runner.py add <name> <dir>  # 添加新模型")
