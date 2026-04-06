"""
Pydantic数据模型定义
"""
from typing import List, Optional, Literal, Dict
from pydantic import BaseModel, Field


class Slot(BaseModel):
    """单个知识点Slot"""
    关键词: str = Field(..., description="提取的关键词，必须具体专业")
    相关性: Literal["强相关", "弱相关", "不相关"] = Field(..., description="该关键词与维度的相关性")
    原句: str = Field(..., description="从原文摘录的支撑句子")
    权重: Literal[1, 2, 3] = Field(..., description="专业深度权重：1=宏观概念，2=中观分类，3=微观操作")


class DimensionSlots(BaseModel):
    """某个维度下的所有Slots"""
    维度名称: str = Field(..., description="维度名称")
    slots: List[Slot] = Field(default_factory=list, description="该维度下的所有关键词")


class StructuredAppreciation(BaseModel):
    """15维结构化赏析"""
    材质形制: List[Slot] = Field(default_factory=list, description="材质与形制相关的关键词")
    构图布局: List[Slot] = Field(default_factory=list, description="构图与布局相关的关键词")
    用笔特点: List[Slot] = Field(default_factory=list, description="用笔与笔法相关的关键词")
    色彩氛围: List[Slot] = Field(default_factory=list, description="色彩与氛围相关的关键词")
    题材内容: List[Slot] = Field(default_factory=list, description="题材与内容相关的关键词")
    形神表现: List[Slot] = Field(default_factory=list, description="形神表现相关的关键词")
    艺术风格: List[Slot] = Field(default_factory=list, description="艺术风格相关的关键词")
    意境营造: List[Slot] = Field(default_factory=list, description="意境营造相关的关键词")
    象征寓意: List[Slot] = Field(default_factory=list, description="象征与寓意相关的关键词")
    画家信息: List[Slot] = Field(default_factory=list, description="画家相关信息")
    创作年代: List[Slot] = Field(default_factory=list, description="创作年代相关信息")
    题跋印章: List[Slot] = Field(default_factory=list, description="题跋与印章相关信息")
    艺术传承: List[Slot] = Field(default_factory=list, description="艺术传承相关信息")
    历史语境: List[Slot] = Field(default_factory=list, description="历史语境相关信息")
    艺术地位: List[Slot] = Field(default_factory=list, description="艺术地位相关信息")


class GeneratedText(BaseModel):
    """模块1生成的原始文本"""
    image_id: str = Field(..., description="图片ID")
    model_name: str = Field(..., description="模型名称")
    prompt: str = Field(..., description="使用的Prompt")
    response: str = Field(..., description="模型生成的赏析文本")
    timestamp: str = Field(..., description="生成时间")
    raw_text_length: int = Field(..., description="原始文本长度")
    

class ExtractedData(BaseModel):
    """模块2抽取的结构化数据"""
    image_id: str = Field(..., description="图片ID")
    source_model: str = Field(..., description="生成赏析文本的模型")
    extraction_model: str = Field(..., description="进行结构化抽取的裁判模型")
    structured_data: StructuredAppreciation = Field(..., description="15维结构化数据")
    cleaned_text: str = Field(..., description="清洗后的文本（去掉开头结尾废话）")
    cleaned_text_length: int = Field(..., description="清洗后的文本长度")
    total_slots_before_dedup: int = Field(..., description="去重前的Slot总数")
    total_slots_after_dedup: int = Field(..., description="去重后的Slot总数")
    timestamp: str = Field(..., description="抽取时间")


class MetricsResult(BaseModel):
    """模块3计算的指标结果"""
    image_id: str = Field(..., description="图片ID")
    model_name: str = Field(..., description="模型名称")
    
    # 基础统计
    total_slots: int = Field(..., description="有效Slot总数（去重后）")
    cleaned_text_length: int = Field(..., description="清洗后文本长度")
    
    # 维度覆盖统计
    dimension_coverage: int = Field(..., description="有效维度数量（Slot>0的维度数）")
    dimension_slots: Dict[str, int] = Field(..., description="各维度的Slot数量")
    
    # 核心指标
    entropy: float = Field(..., description="知识点分布熵（广度指标）")
    density: float = Field(..., description="干货密度（深度指标）")
    weighted_density: float = Field(..., description="加权密度（考虑专业深度）")
    
    # 权重统计
    total_weight: int = Field(..., description="总权重分（所有Slot的权重之和）")
    avg_weight: float = Field(..., description="平均权重（总权重/Slot数）")
    weight_distribution: Dict[int, int] = Field(..., description="权重分布 {1: count, 2: count, 3: count}")
    
    # 相关性统计
    strong_relevant_ratio: float = Field(..., description="强相关Slot占比")
    weak_relevant_ratio: float = Field(..., description="弱相关Slot占比")
    irrelevant_ratio: float = Field(..., description="不相关Slot占比")
    
    # 与GT的对比指标（如果有GT）
    entropy_diff_from_gt: Optional[float] = Field(None, description="与GT的Entropy差异")
    density_diff_from_gt: Optional[float] = Field(None, description="与GT的Density差异")
    weighted_density_diff_from_gt: Optional[float] = Field(None, description="与GT的Weighted_Density差异")
    dimension_distance_from_gt: Optional[float] = Field(None, description="与GT的维度分布距离（余弦距离）")
