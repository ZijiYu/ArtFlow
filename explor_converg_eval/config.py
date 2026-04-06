"""
配置文件：管理API密钥、模型配置、路径等
"""
import os
import yaml
from typing import Dict, List
from pathlib import Path

# ==================== 路径配置 ====================
BASE_DIR = Path(__file__).parent
RESULT_DIR = BASE_DIR / "result"  # 统一使用result文件夹
EXTRACTED_DIR = RESULT_DIR / "extracted"
METRICS_DIR = RESULT_DIR / "metrics"
REPORTS_DIR = RESULT_DIR / "reports"
LOGS_DIR = RESULT_DIR / "logs"

# 输入数据路径
GT_JSONL_PATH = Path("/Users/ken/MM/Pipeline/test/tcp_info/mixed_data_ground_train_3rd_v3_tmp_cleaned_representative_guohua_200.jsonl")
ZHIHUA_TXT_DIR = Path("/Users/ken/MM/zhihua")

# 创建必要的目录
for dir_path in [RESULT_DIR, EXTRACTED_DIR, METRICS_DIR, REPORTS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ==================== API配置 ====================
# 从 final_version/config.yaml 读取配置
def load_api_config_from_yaml():
    """从 yaml 配置文件加载 API 配置"""
    yaml_path = Path("/Users/ken/MM/Pipeline/final_version/config.yaml")
    
    if not yaml_path.exists():
        # 如果 yaml 不存在，使用默认配置
        return {
            "api_key": os.getenv("OPENAI_API_KEY", ""),
            "base_url": "https://api.zjuqx.cn/v1",
            "model": "google/gemini-3.1-pro-preview",
        }
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    api_config = config.get('api', {})
    
    # 读取 API Key
    api_key = ""
    
    # 优先从环境变量读取
    if api_config.get('key_env'):
        api_key = os.getenv(api_config['key_env'], "")
    
    # 如果环境变量为空，从文件读取
    if not api_key and api_config.get('key_file'):
        key_file = Path(api_config['key_file'])
        if key_file.exists():
            with open(key_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                key_line = api_config.get('key_line', 1)
                if len(lines) >= key_line:
                    api_key = lines[key_line - 1].strip()
    
    # 如果都为空，使用配置中的 key
    if not api_key:
        api_key = api_config.get('key', "")
    
    return {
        "api_key": api_key,
        "base_url": api_config.get('base_url', "https://api.zjuqx.cn/v1"),
        "model": "openai/gpt-5.4",  # 用于结构化抽取的裁判模型（强制使用 openai/gpt-5.4）
    }

# OpenAI API配置
OPENAI_CONFIG = load_api_config_from_yaml()

# 增量模式配置
INCREMENTAL_MODE = True  # 启用增量处理
EXISTING_MODELS_CACHE = EXTRACTED_DIR / ".processed_models.json"  # 记录已处理的模型

# 模型列表（用于模块1批量生成）
MODEL_CONFIGS: List[Dict] = [
    {
        "name": "gpt-4o",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.zjuqx.cn/v1",
        "model": "gpt-4o",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    {
        "name": "gpt-4o-mini",
        "provider": "openai",
        "api_key_env": "OPENAI_API_KEY",
        "base_url": "https://api.zjuqx.cn/v1",
        "model": "gpt-4o-mini",
        "max_tokens": 4000,
        "temperature": 0.7,
    },
    # 可以添加更多模型配置
    # {
    #     "name": "claude-3.5-sonnet",
    #     "provider": "anthropic",
    #     "api_key_env": "ANTHROPIC_API_KEY",
    #     "model": "claude-3-5-sonnet-20241022",
    #     "max_tokens": 4000,
    #     "temperature": 0.7,
    # },
]

# ==================== Prompt模板 ====================
GENERATION_PROMPT_TEMPLATE = """请对以下中国古代书画作品进行详细赏析。

要求：
1. 从多个维度进行分析，包括但不限于：材质形制、构图布局、用笔特点、色彩氛围、题材内容、形神表现、艺术风格、意境营造、象征寓意、画家信息、创作年代、题跋印章、艺术传承、历史语境、艺术地位
2. 分析要具体、专业，尽量提及具体的技法名称、艺术流派、历史典故
3. 避免空泛的赞美，聚焦客观描述和专业分析

图片信息：{image_info}

请开始你的赏析：
"""

EXTRACTION_PROMPT_TEMPLATE = """你是一位专业的中国古代书画鉴赏专家。请仔细阅读以下赏析文本，并严格按照JSON Schema提取结构化信息。

**15个评价维度说明**：
1. 材质形制：纸本/绢本、立轴/手卷/册页等
2. 构图布局：平远/高远/深远、留白、对角线等
3. 用笔特点：披麻皴/斧劈皴、游丝描、中锋/侧锋等
4. 色彩氛围：水墨/设色、浓淡干湿、冷暖色调等
5. 题材内容：山水/人物/花鸟、具体描绘对象
6. 形神表现：形似/神似、气韵生动等
7. 艺术风格：工笔/写意、院体/文人画、具体画派
8. 意境营造：诗意、意境层次、情感表达
9. 象征寓意：符号象征、文化隐喻
10. 画家信息：画家姓名、字号、生平
11. 创作年代：具体朝代、时期、年份
12. 题跋印章：题跋内容、书法风格、印章信息
13. 艺术传承：师承关系、流派传承、影响
14. 历史语境：时代背景、创作环境、文化典故
15. 艺术地位：艺术史地位、代表性、影响力

**三级本体映射与权重赋值**（极其重要）：
每个提取的关键词必须根据专业深度赋予权重值（1、2或3）：

- **Level 1（权重=1）- 宏观概念**：
  宽泛的艺术风格、通用美术术语
  示例：水墨画、设色、山水、花鸟、写意、工笔、线条、构图

- **Level 2（权重=2）- 中观分类**：
  具体的技法大类、明确的历史流派
  示例：皴法、描法、青绿山水、浅绛山水、浙派、吴门画派、院体画

- **Level 3（权重=3）- 微观操作**：
  极具专业壁垒的底层操作细节、特定技法名词、特定历史印记
  示例：点染、披麻皴、斧劈皴、雨点皴、双钩填彩、没骨法、积墨法、游丝描、折带皴、解索皴、荷叶皴、牛毛皴、大斧劈、小斧劈

**抽取规则**：
- 每个维度下提取多个关键词Slot
- 每个Slot必须包含：关键词、相关性（强相关/弱相关/不相关）、权重（1/2/3）、原句（从文本中摘录的原句）
- 如果某个维度完全未涉及，可以返回空数组
- 关键词要具体，避免空泛词汇（如"很好"、"美丽"等）
- **权重赋值必须准确**：仔细判断每个关键词的专业深度级别

**赏析文本**：
{text}

请严格按照Pydantic Schema输出结构化JSON。
"""

# ==================== 结构化抽取配置 ====================
# 15个维度的类别名称
DIMENSIONS = [
    "材质形制",
    "构图布局",
    "用笔特点",
    "色彩氛围",
    "题材内容",
    "形神表现",
    "艺术风格",
    "意境营造",
    "象征寓意",
    "画家信息",
    "创作年代",
    "题跋印章",
    "艺术传承",
    "历史语境",
    "艺术地位",
]

# 语义去重阈值
SIMILARITY_THRESHOLD = 0.85

# ==================== 三级本体映射（权重体系）====================
# Level 1: 宏观概念（权重=1）
LEVEL_1_KEYWORDS = [
    "水墨画", "设色", "山水", "花鸟", "写意", "工笔", "线条", "构图",
    "人物", "鸟兽", "色彩", "笔墨", "绘画", "书法", "题材", "艺术"
]

# Level 2: 中观分类（权重=2）
LEVEL_2_KEYWORDS = [
    "皴法", "描法", "青绿山水", "浅绛山水", "浙派", "吴门画派", "院体画",
    "文人画", "水墨山水", "工笔花鸟", "写意花鸟", "界画", "白描", "泼墨",
    "南宋四家", "元四家", "明四家", "清四僧", "扬州八怪"
]

# Level 3: 微观操作（权重=3）
LEVEL_3_KEYWORDS = [
    "点染", "披麻皴", "斧劈皴", "雨点皴", "双钩填彩", "没骨法", "积墨法", 
    "游丝描", "折带皴", "解索皴", "荷叶皴", "牛毛皴", "大斧劈", "小斧劈",
    "铁线描", "兰叶描", "钉头鼠尾描", "破墨法", "泼彩法", "米点皴", "云头皴",
    "撞水撞粉", "骨法用笔", "随类赋彩", "经营位置", "气韵生动"
]

# ==================== 并发控制 ====================
MAX_CONCURRENT_REQUESTS = 10  # 最大并发API请求数
REQUEST_TIMEOUT = 60  # 请求超时时间（秒）
MAX_RETRIES = 3  # 最大重试次数

# ==================== Embedding模型配置 ====================
# 用于语义去重的Embedding模型
EMBEDDING_MODEL = "BAAI/bge-m3"  # 可以改为其他模型
