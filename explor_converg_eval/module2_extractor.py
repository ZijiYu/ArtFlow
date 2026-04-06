"""
模块2：LLM-as-a-Judge结构化信息抽取
使用最强模型+Strict JSON约束，将文本转化为15维结构
"""
import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging
from tqdm.asyncio import tqdm

import instructor
from openai import AsyncOpenAI

from config import (
    OPENAI_CONFIG,
    EXTRACTION_PROMPT_TEMPLATE,
    ZHIHUA_TXT_DIR,
    EXTRACTED_DIR,
    DIMENSIONS,
    SIMILARITY_THRESHOLD,
    GT_JSONL_PATH
)
from models import StructuredAppreciation, ExtractedData, Slot
from utils import clean_text, semantic_deduplication, filter_relevant_slots

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StructuredExtractor:
    """结构化信息抽取器"""
    
    def __init__(self):
        # 初始化Instructor客户端
        self.client = instructor.from_openai(
            AsyncOpenAI(
                api_key=os.getenv("OPENAI_API_KEY", OPENAI_CONFIG["api_key"]),
                base_url=OPENAI_CONFIG["base_url"]
            )
        )
        self.extraction_model = OPENAI_CONFIG["model"]
    
    async def extract_structure(self, text: str) -> Optional[StructuredAppreciation]:
        """
        使用LLM抽取结构化信息
        """
        prompt = EXTRACTION_PROMPT_TEMPLATE.format(text=text)
        
        try:
            response = await self.client.chat.completions.create(
                model=self.extraction_model,
                response_model=StructuredAppreciation,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,  # 低温度保证稳定性
                max_tokens=4000,
            )
            return response
        except Exception as e:
            logger.error(f"结构化抽取失败: {str(e)}")
            return None
    
    def deduplicate_dimension_slots(self, slots: List[Slot]) -> List[Slot]:
        """
        对同一维度内的Slots进行语义去重
        """
        if len(slots) <= 1:
            return slots
        
        # 提取关键词列表
        keywords = [slot.关键词 for slot in slots]
        
        # 进行语义去重
        keep_indices = semantic_deduplication(keywords, SIMILARITY_THRESHOLD)
        
        # 返回保留的Slots
        return [slots[i] for i in keep_indices]
    
    def deduplicate_all_dimensions(
        self, 
        structured_data: StructuredAppreciation
    ) -> StructuredAppreciation:
        """
        对所有15个维度进行去重
        """
        deduped_data = {}
        
        for dim_name in DIMENSIONS:
            slots = getattr(structured_data, dim_name, [])
            deduped_slots = self.deduplicate_dimension_slots(slots)
            deduped_data[dim_name] = deduped_slots
        
        return StructuredAppreciation(**deduped_data)
    
    async def process_single_text(
        self,
        image_id: str,
        text: str,
        source_model: str = "zhihua"
    ) -> Optional[ExtractedData]:
        """
        处理单个赏析文本
        """
        # 1. 清洗文本
        cleaned_text, cleaned_length = clean_text(text)
        
        if cleaned_length == 0:
            logger.warning(f"文本清洗后为空: {image_id}")
            return None
        
        # 2. 结构化抽取
        structured_data = await self.extract_structure(cleaned_text)
        if not structured_data:
            return None
        
        # 3. 统计去重前的Slot数量
        total_before = sum(
            len(getattr(structured_data, dim, []))
            for dim in DIMENSIONS
        )
        
        # 4. 语义去重
        deduped_data = self.deduplicate_all_dimensions(structured_data)
        
        # 5. 统计去重后的Slot数量
        total_after = sum(
            len(getattr(deduped_data, dim, []))
            for dim in DIMENSIONS
        )
        
        logger.info(f"{image_id}: 去重前={total_before}, 去重后={total_after}")
        
        return ExtractedData(
            image_id=image_id,
            source_model=source_model,
            extraction_model=self.extraction_model,
            structured_data=deduped_data,
            cleaned_text=cleaned_text,
            cleaned_text_length=cleaned_length,
            total_slots_before_dedup=total_before,
            total_slots_after_dedup=total_after,
            timestamp=datetime.now().isoformat()
        )
    
    async def batch_extract_from_zhihua(self) -> List[ExtractedData]:
        """
        批量处理zhihua文件夹下的txt文件
        """
        txt_files = list(ZHIHUA_TXT_DIR.glob("*.txt"))
        logger.info(f"找到 {len(txt_files)} 个zhihua txt文件")
        
        tasks = []
        for txt_file in txt_files:
            image_id = txt_file.stem  # 去掉.txt后缀
            
            try:
                with open(txt_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                task = self.process_single_text(image_id, text, source_model="zhihua")
                tasks.append(task)
            except Exception as e:
                logger.error(f"读取文件失败 {txt_file}: {str(e)}")
        
        # 并发处理
        results = []
        for coro in tqdm.as_completed(tasks, desc="抽取zhihua文本"):
            result = await coro
            if result:
                results.append(result)
        
        return results
    
    async def batch_extract_from_gt(self) -> List[ExtractedData]:
        """
        批量处理Ground Truth文件中的赏析文本
        """
        logger.info(f"开始处理Ground Truth: {GT_JSONL_PATH}")
        
        tasks = []
        with open(GT_JSONL_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    image_path = data.get("image", "")
                    text = data.get("assistant", "")
                    
                    if not text:
                        continue
                    
                    # 从路径中提取image_id
                    image_id = Path(image_path).stem
                    
                    task = self.process_single_text(image_id, text, source_model="ground_truth")
                    tasks.append(task)
        
        logger.info(f"GT中有 {len(tasks)} 个样本")
        
        # 并发处理
        results = []
        for coro in tqdm.as_completed(tasks, desc="抽取GT文本"):
            result = await coro
            if result:
                results.append(result)
        
        return results
    
    def save_results(self, results: List[ExtractedData], output_path: Path):
        """
        保存抽取结果到JSONL文件
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result.model_dump_json(ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {len(results)} 条抽取结果到 {output_path}")


async def main():
    """
    模块2主函数
    """
    extractor = StructuredExtractor()
    
    all_results = []
    
    # 1. 处理Ground Truth
    logger.info("\n" + "="*60)
    logger.info("开始处理Ground Truth...")
    logger.info("="*60)
    gt_results = await extractor.batch_extract_from_gt()
    all_results.extend(gt_results)
    logger.info(f"GT抽取完成：{len(gt_results)} 条")
    
    # 2. 处理zhihua的txt文件
    logger.info("\n" + "="*60)
    logger.info("开始处理zhihua txt文件...")
    logger.info("="*60)
    zhihua_results = await extractor.batch_extract_from_zhihua()
    all_results.extend(zhihua_results)
    logger.info(f"Zhihua抽取完成：{len(zhihua_results)} 条")
    
    # 3. 保存所有结果
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = EXTRACTED_DIR / f"extracted_all_{timestamp}.jsonl"
    extractor.save_results(all_results, output_path)
    
    # 4. 分别保存GT和zhihua的结果（便于单独分析）
    gt_output_path = EXTRACTED_DIR / f"extracted_gt_{timestamp}.jsonl"
    extractor.save_results(gt_results, gt_output_path)
    
    zhihua_output_path = EXTRACTED_DIR / f"extracted_zhihua_{timestamp}.jsonl"
    extractor.save_results(zhihua_results, zhihua_output_path)
    
    logger.info("\n" + "="*60)
    logger.info(f"模块2完成！")
    logger.info(f"  - GT: {len(gt_results)} 条")
    logger.info(f"  - Zhihua: {len(zhihua_results)} 条")
    logger.info(f"  - 总计: {len(all_results)} 条")
    logger.info("="*60)


if __name__ == "__main__":
    asyncio.run(main())
