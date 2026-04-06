"""
模块1：多模型API并发生成器
负责调用不同大模型的API，批量生成初始的赏析文本
"""
import asyncio
import aiohttp
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
from tqdm.asyncio import tqdm
import logging

from config import (
    MODEL_CONFIGS, 
    GENERATION_PROMPT_TEMPLATE,
    GT_JSONL_PATH,
    GENERATED_DIR,
    MAX_CONCURRENT_REQUESTS,
    REQUEST_TIMEOUT,
    MAX_RETRIES
)
from models import GeneratedText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class APIGenerator:
    """多模型API并发生成器"""
    
    def __init__(self):
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
        self.results: List[GeneratedText] = []
    
    async def call_openai_api(
        self, 
        session: aiohttp.ClientSession,
        config: Dict,
        prompt: str,
        image_id: str
    ) -> Optional[GeneratedText]:
        """
        调用OpenAI兼容的API
        """
        headers = {
            "Authorization": f"Bearer {config['api_key']}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": config["model"],
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": config.get("max_tokens", 4000),
            "temperature": config.get("temperature", 0.7),
        }
        
        async with self.semaphore:
            for attempt in range(MAX_RETRIES):
                try:
                    async with session.post(
                        f"{config['base_url']}/chat/completions",
                        headers=headers,
                        json=payload,
                        timeout=aiohttp.ClientTimeout(total=REQUEST_TIMEOUT)
                    ) as response:
                        if response.status == 200:
                            data = await response.json()
                            content = data["choices"][0]["message"]["content"]
                            
                            return GeneratedText(
                                image_id=image_id,
                                model_name=config["name"],
                                prompt=prompt,
                                response=content,
                                timestamp=datetime.now().isoformat(),
                                raw_text_length=len(content)
                            )
                        else:
                            error_text = await response.text()
                            logger.warning(f"API请求失败 (status {response.status}): {error_text}")
                            
                except asyncio.TimeoutError:
                    logger.warning(f"请求超时 ({image_id}, {config['name']}, attempt {attempt + 1})")
                except Exception as e:
                    logger.error(f"API调用异常 ({image_id}, {config['name']}): {str(e)}")
                
                if attempt < MAX_RETRIES - 1:
                    await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return None
    
    async def generate_for_image(
        self,
        session: aiohttp.ClientSession,
        image_info: Dict,
        config: Dict
    ) -> Optional[GeneratedText]:
        """
        为单张图片调用某个模型生成赏析
        """
        image_id = Path(image_info.get("image", "unknown")).stem
        
        # 构造prompt
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            image_info=json.dumps(image_info, ensure_ascii=False, indent=2)
        )
        
        if config["provider"] == "openai":
            return await self.call_openai_api(session, config, prompt, image_id)
        # 可以扩展支持其他provider
        else:
            logger.warning(f"不支持的provider: {config['provider']}")
            return None
    
    async def batch_generate(
        self,
        image_list: List[Dict],
        model_configs: List[Dict]
    ) -> List[GeneratedText]:
        """
        批量生成：对所有图片×所有模型进行并发调用
        """
        async with aiohttp.ClientSession() as session:
            tasks = []
            
            for image_info in image_list:
                for config in model_configs:
                    # 获取API key
                    import os
                    api_key = os.getenv(config.get("api_key_env", ""))
                    if not api_key:
                        logger.warning(f"未找到API key: {config.get('api_key_env')}")
                        continue
                    
                    config["api_key"] = api_key
                    task = self.generate_for_image(session, image_info, config)
                    tasks.append(task)
            
            logger.info(f"开始并发生成，共 {len(tasks)} 个任务")
            
            # 使用tqdm显示进度
            results = []
            for coro in tqdm.as_completed(tasks, desc="生成赏析文本"):
                result = await coro
                if result:
                    results.append(result)
            
            return results
    
    def save_results(self, results: List[GeneratedText], output_path: Path):
        """
        保存生成结果到JSONL文件
        """
        with open(output_path, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(result.model_dump_json(ensure_ascii=False) + '\n')
        
        logger.info(f"已保存 {len(results)} 条生成结果到 {output_path}")


def load_ground_truth(gt_path: Path) -> List[Dict]:
    """
    加载Ground Truth数据，提取图片列表
    """
    images = []
    with open(gt_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                images.append(data)
    
    logger.info(f"加载了 {len(images)} 条Ground Truth数据")
    return images


async def main():
    """
    模块1主函数
    """
    # 加载图片列表
    image_list = load_ground_truth(GT_JSONL_PATH)
    
    # 初始化生成器
    generator = APIGenerator()
    
    # 批量生成
    results = await generator.batch_generate(image_list, MODEL_CONFIGS)
    
    # 保存结果
    output_path = GENERATED_DIR / f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    generator.save_results(results, output_path)
    
    logger.info(f"模块1完成！生成了 {len(results)} 条赏析文本")


if __name__ == "__main__":
    asyncio.run(main())
