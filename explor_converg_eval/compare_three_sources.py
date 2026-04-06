"""
对比三个数据源的 Depth 和 Diversity
数据源:
1. final_prompts_aggregated.jsonl (字段: final_appreciation)
2. mixed_data_ground_train_3rd_v3_tmp_cleaned_representative_guohua_200.jsonl (字段: assistant)
3. zhihua_0/*.txt (纯文本文件)
"""
import asyncio
import argparse
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Set, Optional
import pandas as pd
from tqdm.asyncio import tqdm

from module2_extractor import StructuredExtractor
from module3_metrics import MetricsCalculator
from models import ExtractedData
from visualize_comparison import generate_all_reports

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

METRICS_COLUMNS = [
    "image_id",
    "model_name",
    "total_slots",
    "cleaned_text_length",
    "dimension_coverage",
    "dimension_slots",
    "entropy",
    "density",
    "weighted_density",
    "total_weight",
    "avg_weight",
    "weight_distribution",
    "strong_relevant_ratio",
    "weak_relevant_ratio",
    "irrelevant_ratio",
    "entropy_diff_from_gt",
    "density_diff_from_gt",
    "weighted_density_diff_from_gt",
    "dimension_distance_from_gt",
]

# 数据源路径
SOURCE1_PATH = Path("/Users/ken/MM/Pipeline/cluster_match/artifacts/200_image_pipeline/final_prompts_aggregated.jsonl")
SOURCE2_PATH = Path("/Users/ken/MM/Pipeline/test/tcp_info/mixed_data_ground_train_3rd_v3_tmp_cleaned_representative_guohua_200.jsonl")
SOURCE3_DIR = Path("/Users/ken/MM/zhihua_0")

# 数据源配置
SOURCES = {
    "final_prompts": {
        "path": SOURCE1_PATH,
        "type": "jsonl",
        "id_field": "image_stem",
        "text_field": "final_appreciation",
        "display_name": "Pipeline最终版本"
    },
    "ground_truth": {
        "path": SOURCE2_PATH,
        "type": "jsonl",
        "id_field": "image",  # 需要从路径中提取stem
        "text_field": "assistant",
        "display_name": "Ground Truth"
    },
    "zhihua_0": {
        "path": SOURCE3_DIR,
        "type": "txt_dir",
        "display_name": "Zhihua初版"
    }
}

SOURCE_ORDER = ["final_prompts", "ground_truth", "zhihua_0"]


def parse_source_arg(source_arg: str) -> Tuple[str, Dict]:
    if "=" not in source_arg:
        raise ValueError(f"无效 --source: {source_arg}，应为 name=/path[:k=v...]")
    name, payload = source_arg.split("=", 1)
    source_name = name.strip()
    if not source_name:
        raise ValueError(f"无效 source 名称: {source_arg}")

    parts = payload.split(":")
    path = Path(parts[0].strip())
    config: Dict[str, object] = {"path": path, "display_name": source_name}
    for token in parts[1:]:
        if "=" not in token:
            continue
        k, v = token.split("=", 1)
        config[k.strip()] = v.strip()

    if "type" not in config:
        config["type"] = "jsonl" if path.suffix.lower() == ".jsonl" else "txt_dir"
    if config["type"] == "jsonl":
        config.setdefault("id_field", "image_id")
        config.setdefault("text_field", "text")
    return source_name, config


def resolve_sources(custom_sources: Optional[List[str]]) -> Tuple[Dict[str, Dict], List[str]]:
    if not custom_sources:
        return dict(SOURCES), list(SOURCE_ORDER)
    sources: Dict[str, Dict] = {}
    order: List[str] = []
    for source_arg in custom_sources:
        source_name, config = parse_source_arg(source_arg)
        sources[source_name] = config
        order.append(source_name)
    if len(order) < 2:
        raise ValueError("至少需要 2 个数据源，--source 可重复传入")
    return sources, order


def load_data_from_source(source_name: str, sources: Dict[str, Dict], limit: int = None) -> List[Dict[str, str]]:
    """
    从指定数据源加载数据
    
    返回: [{"image_id": "xxx", "text": "xxx", "source": "xxx"}, ...]
    """
    config = sources[source_name]
    results = []
    
    if config["type"] == "jsonl":
        logger.info(f"加载 {config['display_name']} 从 {config['path']}")
        
        with open(config['path'], 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                    
                if line.strip():
                    data = json.loads(line)
                    
                    # 提取 image_id
                    raw_id = data.get(config['id_field'], "")
                    image_id = raw_id
                    if isinstance(raw_id, str):
                        if "/" in raw_id or raw_id.endswith((".jpg", ".jpeg", ".png", ".webp")):
                            image_id = Path(raw_id).stem
                    
                    # 提取文本
                    text = data.get(config['text_field'], "")
                    
                    if image_id and text:
                        results.append({
                            "image_id": image_id,
                            "text": text,
                            "source": source_name
                        })
    
    elif config["type"] == "txt_dir":
        logger.info(f"加载 {config['display_name']} 从 {config['path']}")
        
        txt_files = sorted(set(list(config['path'].glob("*.txt")) + list(config['path'].glob("*.md"))))
        
        for i, txt_file in enumerate(txt_files):
            if limit and i >= limit:
                break
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                image_id = txt_file.stem  # 去掉 .jpg.txt 后缀
                if image_id.endswith('.jpg'):
                    image_id = image_id[:-4]
                
                if text:
                    results.append({
                        "image_id": image_id,
                        "text": text,
                        "source": source_name
                    })
            except Exception as e:
                logger.error(f"读取文件失败 {txt_file}: {str(e)}")
    
    logger.info(f"  加载了 {len(results)} 条数据")
    return results


def index_source_records(data_list: List[Dict[str, str]], source_name: str) -> Dict[str, Dict[str, str]]:
    """
    将单个数据源的数据按 image_id 建立索引。
    如果同一数据源内出现重复 image_id，保留首次出现的记录。
    """
    indexed_records: Dict[str, Dict[str, str]] = {}
    duplicate_count = 0

    for item in data_list:
        image_id = item["image_id"]
        if image_id in indexed_records:
            duplicate_count += 1
            continue
        indexed_records[image_id] = item

    if duplicate_count > 0:
        logger.warning(f"{source_name} 中发现 {duplicate_count} 条重复 image_id，已忽略后续重复项")

    return indexed_records


def build_comparison_dataset(
    mode: str,
    compare_scope: str,
    sources: Dict[str, Dict],
    source_order: List[str],
) -> Tuple[List[Dict[str, str]], Dict]:
    """
    根据对比范围构建待抽取的数据列表。

    compare_scope:
        - intersection: 仅保留三个数据源共同拥有的 image_id
        - union: 每个数据源各自全部参与统计（旧行为）
    """
    indexed_by_source: Dict[str, Dict[str, Dict[str, str]]] = {}
    raw_counts: Dict[str, int] = {}

    for source_name in source_order:
        data = load_data_from_source(source_name, sources=sources, limit=None)
        indexed = index_source_records(data, source_name)
        indexed_by_source[source_name] = indexed
        raw_counts[source_name] = len(indexed)

    if compare_scope == "intersection":
        common_ids: Set[str] = set.intersection(
            *(set(indexed_by_source[source_name].keys()) for source_name in source_order)
        )
        selected_ids = sorted(common_ids)

        if mode == "test":
            selected_ids = selected_ids[:10]

        all_data: List[Dict[str, str]] = []
        for image_id in selected_ids:
            for source_name in source_order:
                all_data.append(indexed_by_source[source_name][image_id])

        selection_summary = {
            "compare_scope": compare_scope,
            "mode": mode,
            "raw_counts": raw_counts,
            "selected_common_image_count": len(selected_ids),
            "selected_record_count": len(all_data),
            "selected_counts_by_source": {
                source_name: len(selected_ids) for source_name in source_order
            },
        }
    else:
        per_source_limit = 10 if mode == "test" else None
        all_data = []
        selected_counts_by_source: Dict[str, int] = {}

        for source_name in source_order:
            records = list(indexed_by_source[source_name].values())
            if per_source_limit is not None:
                records = records[:per_source_limit]
            selected_counts_by_source[source_name] = len(records)
            all_data.extend(records)

        selection_summary = {
            "compare_scope": compare_scope,
            "mode": mode,
            "raw_counts": raw_counts,
            "selected_common_image_count": None,
            "selected_record_count": len(all_data),
            "selected_counts_by_source": selected_counts_by_source,
        }

    return all_data, selection_summary


def print_selection_summary(selection_summary: Dict, sources: Dict[str, Dict], source_order: List[str]):
    """
    打印样本选择摘要，说明本次对比使用了哪些样本。
    """
    print("\n" + "=" * 80)
    print("📦 样本选择摘要")
    print("=" * 80)
    print(f"对比范围: {selection_summary['compare_scope']}")
    print(f"运行模式: {selection_summary['mode']}")

    for source_name in source_order:
        display_name = sources[source_name]["display_name"]
        raw_count = selection_summary["raw_counts"].get(source_name, 0)
        selected_count = selection_summary["selected_counts_by_source"].get(source_name, 0)
        print(f"{display_name:<20} 原始={raw_count:<4} 参与统计={selected_count:<4}")

    if selection_summary["compare_scope"] == "intersection":
        print(f"公共 image_id 数量: {selection_summary['selected_common_image_count']}")
        print("说明: 仅比较所有数据源都存在的样本")
    else:
        print("说明: 每个数据源各自独立统计，不强制要求 image_id 交集")


async def extract_and_calculate(data_list: List[Dict[str, str]]) -> List[ExtractedData]:
    """
    对数据列表进行结构化抽取
    """
    extractor = StructuredExtractor()
    
    tasks = []
    for item in data_list:
        task = extractor.process_single_text(
            image_id=item['image_id'],
            text=item['text'],
            source_model=item['source']
        )
        tasks.append(task)
    
    logger.info(f"开始抽取 {len(tasks)} 条数据...")
    
    results = []
    for coro in tqdm.as_completed(tasks, desc="结构化抽取"):
        result = await coro
        if result:
            results.append(result)
    
    return results


def calculate_metrics(extracted_list: List[ExtractedData], gt_source: Optional[str]) -> pd.DataFrame:
    """
    计算所有指标并返回 DataFrame
    """
    calculator = MetricsCalculator()
    metrics_list = calculator.batch_calculate(extracted_list)

    if not metrics_list:
        return pd.DataFrame(columns=METRICS_COLUMNS)

    if gt_source:
        gt_metrics_dict = {m.image_id: m for m in metrics_list if m.model_name == gt_source}
        if gt_metrics_dict:
            metrics_list = calculator.batch_calculate(extracted_list, gt_metrics_dict=gt_metrics_dict)

    # 转为 DataFrame
    df = pd.DataFrame([m.model_dump() for m in metrics_list])

    return df


def print_comparison_summary(df: pd.DataFrame, sources: Dict[str, Dict], source_order: List[str]):
    """
    打印对比摘要
    """
    print("\n" + "=" * 80)
    print("📊 多数据源对比摘要")
    print("=" * 80)

    if df.empty:
        print("没有成功抽取到任何样本，无法生成对比摘要。")
        return
    
    for source_name in source_order:
        config = sources[source_name]
        source_df = df[df['model_name'] == source_name]
        
        if len(source_df) == 0:
            continue
        
        print(f"\n【{config['display_name']}】（n={len(source_df)}）")
        print(f"  Diversity (Entropy):      {source_df['entropy'].mean():.4f} ± {source_df['entropy'].std():.4f}")
        print(f"  Depth (W_Density):        {source_df['weighted_density'].mean():.4f} ± {source_df['weighted_density'].std():.4f}")
        print(f"  Density (传统):            {source_df['density'].mean():.4f} ± {source_df['density'].std():.4f}")
        print(f"  平均权重:                  {source_df['avg_weight'].mean():.2f} ± {source_df['avg_weight'].std():.2f}")
        print(f"  平均Slot数:               {source_df['total_slots'].mean():.1f} ± {source_df['total_slots'].std():.1f}")
        print(f"  平均文本长度:              {source_df['cleaned_text_length'].mean():.0f} ± {source_df['cleaned_text_length'].std():.0f}")
        print(f"  维度覆盖:                  {source_df['dimension_coverage'].mean():.1f}/15")
    
    # 三者对比
    print("\n" + "-" * 80)
    print("📈 排名对比")
    print("-" * 80)
    
    summary = df.groupby('model_name').agg({
        'entropy': 'mean',
        'weighted_density': 'mean',
        'density': 'mean',
        'avg_weight': 'mean'
    }).round(4)
    
    print("\n按 Diversity (Entropy) 排名:")
    entropy_rank = summary.sort_values('entropy', ascending=False)
    for i, (source, row) in enumerate(entropy_rank.iterrows(), 1):
        display_name = sources.get(source, {}).get('display_name', source)
        print(f"  {i}. {display_name:<20} {row['entropy']:.4f}")
    
    print("\n按 Depth (Weighted_Density) 排名:")
    depth_rank = summary.sort_values('weighted_density', ascending=False)
    for i, (source, row) in enumerate(depth_rank.iterrows(), 1):
        display_name = sources.get(source, {}).get('display_name', source)
        print(f"  {i}. {display_name:<20} {row['weighted_density']:.4f}")
    
    print("\n按平均权重排名:")
    weight_rank = summary.sort_values('avg_weight', ascending=False)
    for i, (source, row) in enumerate(weight_rank.iterrows(), 1):
        display_name = sources.get(source, {}).get('display_name', source)
        print(f"  {i}. {display_name:<20} {row['avg_weight']:.2f}")


async def main(
    mode: str = "test",
    compare_scope: str = "intersection",
    custom_sources: Optional[List[str]] = None,
    gt_source: Optional[str] = "ground_truth",
):
    """
    主函数
    
    参数:
        mode: "test" (10条) 或 "all" (全部)
        compare_scope: "intersection" (默认，仅比较交集) 或 "union"（旧行为）
    """
    logger.info("="*80)
    logger.info("开始多数据源对比分析")
    logger.info("="*80)
    logger.info(f"运行模式: {mode.upper()}")
    logger.info(f"对比范围: {compare_scope.upper()}")

    sources, source_order = resolve_sources(custom_sources)
    if gt_source and gt_source not in sources:
        logger.warning(f"未找到 gt_source={gt_source}，将跳过 *_diff_from_gt 指标")
        gt_source = None

    # 加载并筛选数据源
    all_data, selection_summary = build_comparison_dataset(
        mode, compare_scope, sources=sources, source_order=source_order
    )

    logger.info(f"\n总计加载 {len(all_data)} 条待处理记录")
    
    # 结构化抽取
    extracted_list = await extract_and_calculate(all_data)
    
    logger.info(f"\n成功抽取 {len(extracted_list)} 条数据")
    
    # 计算指标
    logger.info("\n开始计算指标...")
    df = calculate_metrics(extracted_list, gt_source=gt_source)
    
    # 保存结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = Path(__file__).parent / "result" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存 CSV 指标
    csv_path = output_dir / f"comparison_{mode}_{compare_scope}_{timestamp}.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    logger.info(f"\n指标结果已保存到: {csv_path}")
    
    # 保存详细的抽取数据（JSONL）
    jsonl_path = output_dir / f"extracted_{mode}_{compare_scope}_{timestamp}.jsonl"
    with open(jsonl_path, 'w', encoding='utf-8') as f:
        for extracted in extracted_list:
            f.write(extracted.model_dump_json(ensure_ascii=False) + '\n')
    logger.info(f"详细抽取数据已保存到: {jsonl_path}")

    print_selection_summary(selection_summary, sources=sources, source_order=source_order)

    if df.empty:
        logger.warning("没有成功抽取到任何样本，请检查模型可用性、API 配置或上游数据。")
        print_comparison_summary(df, sources=sources, source_order=source_order)
        logger.info("对比分析完成！")
        return
    
    # 打印对比摘要
    print_comparison_summary(df, sources=sources, source_order=source_order)
    
    # 保存详细的权重分布
    print("\n" + "=" * 80)
    print("🔍 权重分布详情")
    print("=" * 80)
    
    for source_name in source_order:
        config = sources[source_name]
        source_df = df[df['model_name'] == source_name]
        
        if len(source_df) == 0:
            continue
        
        # 统计权重分布
        weight_dist = {1: 0, 2: 0, 3: 0}
        for dist_str in source_df['weight_distribution']:
            if isinstance(dist_str, str):
                dist = eval(dist_str)
            else:
                dist = dist_str
            for k, v in dist.items():
                weight_dist[int(k)] += v
        
        total = sum(weight_dist.values())
        
        print(f"\n【{config['display_name']}】")
        print(f"  Level 1 (宏观): {weight_dist[1]:>4} 个 ({weight_dist[1]/total*100:>5.1f}%)")
        print(f"  Level 2 (中观): {weight_dist[2]:>4} 个 ({weight_dist[2]/total*100:>5.1f}%)")
        print(f"  Level 3 (微观): {weight_dist[3]:>4} 个 ({weight_dist[3]/total*100:>5.1f}%)")
        print(f"  总计:           {total:>4} 个")
    
    print("\n" + "=" * 80)
    logger.info("对比分析完成！")
    print("=" * 80)

    report_dir = generate_all_reports(str(csv_path), str(jsonl_path))
    logger.info(f"报告目录已生成: {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比多个数据源的Depth和Diversity")
    
    parser.add_argument(
        '--mode',
        type=str,
        default='test',
        choices=['test', 'all'],
        help='运行模式: test (10条测试) 或 all (全部数据)'
    )

    parser.add_argument(
        '--compare-scope',
        type=str,
        default='intersection',
        choices=['intersection', 'union'],
        help='样本范围: intersection=仅比较多源交集，union=各数据源独立全量统计'
    )

    parser.add_argument(
        '--source',
        action='append',
        default=None,
        help='自定义数据源，可重复。格式: name=/path[:type=jsonl|txt_dir:id_field=...:text_field=...:display_name=...]'
    )

    parser.add_argument(
        '--gt-source',
        type=str,
        default='ground_truth',
        help='用于计算 *_diff_from_gt 的参考源名称；若不存在则自动跳过差分指标'
    )
    
    args = parser.parse_args()
    
    asyncio.run(
        main(
            mode=args.mode,
            compare_scope=args.compare_scope,
            custom_sources=args.source,
            gt_source=args.gt_source,
        )
    )
