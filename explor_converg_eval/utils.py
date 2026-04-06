"""
工具函数：文本清洗、语义去重、embedding计算等
"""
import re
import numpy as np
from typing import List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text: str) -> Tuple[str, int]:
    """
    清洗文本：去掉开头和结尾的废话
    
    返回：(清洗后的文本, 字符数)
    """
    # 去掉常见的开头废话
    opening_patterns = [
        r"^好的[，,。].*?赏析[：:：]\s*",
        r"^以下是.*?赏析[：:：]\s*",
        r"^这是一份?.*?赏析[：:：]\s*",
        r"^关于.*?赏析如下[：:：]\s*",
        r"^让我.*?进行赏析[：:：]\s*",
    ]
    
    for pattern in opening_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE)
    
    # 去掉常见的结尾废话
    ending_patterns = [
        r"总而言之.*$",
        r"综上所述.*$",
        r"如果你也喜欢.*$",
        r"希望.*?帮助.*$",
        r"以上就是.*?赏析.*$",
    ]
    
    for pattern in ending_patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE | re.MULTILINE | re.DOTALL)
    
    # 去掉多余的空白
    text = text.strip()
    
    # 计算有效字符数（不包括空格和换行）
    char_count = len(re.sub(r'\s+', '', text))
    
    return text, char_count


def compute_text_embeddings(texts: List[str]) -> np.ndarray:
    """
    计算文本的TF-IDF向量（简化版，实际应使用BGE-M3）
    
    参数：
        texts: 文本列表
    
    返回：
        embeddings矩阵 (n_texts, n_features)
    """
    if not texts or len(texts) == 0:
        return np.array([])
    
    # 使用TF-IDF作为简化实现
    # 实际应该使用BGE-M3等预训练模型
    vectorizer = TfidfVectorizer(
        max_features=512,
        ngram_range=(1, 2),
        analyzer='char'  # 中文使用字符级
    )
    
    try:
        embeddings = vectorizer.fit_transform(texts).toarray()
        return embeddings
    except:
        # 如果文本太少或为空，返回零向量
        return np.zeros((len(texts), 512))


def semantic_deduplication(keywords: List[str], threshold: float = 0.85) -> List[int]:
    """
    语义去重：找出需要保留的关键词索引
    
    参数：
        keywords: 关键词列表
        threshold: 相似度阈值
    
    返回：
        保留的关键词索引列表
    """
    if len(keywords) <= 1:
        return list(range(len(keywords)))
    
    # 计算embeddings
    embeddings = compute_text_embeddings(keywords)
    
    if embeddings.shape[0] == 0:
        return list(range(len(keywords)))
    
    # 计算余弦相似度矩阵
    similarity_matrix = cosine_similarity(embeddings)
    
    # 贪心去重：按顺序遍历，如果与前面的相似度都<阈值，则保留
    keep_indices = []
    
    for i in range(len(keywords)):
        should_keep = True
        for j in keep_indices:
            if similarity_matrix[i, j] >= threshold:
                should_keep = False
                break
        if should_keep:
            keep_indices.append(i)
    
    return keep_indices


def filter_relevant_slots(slots: List) -> List:
    """
    过滤掉不相关的Slots
    
    参数：
        slots: Slot对象列表
    
    返回：
        过滤后的Slot列表（只保留强相关和弱相关）
    """
    return [slot for slot in slots if slot.相关性 in ["强相关", "弱相关"]]


def count_relevance_stats(slots: List) -> Tuple[int, int, int]:
    """
    统计相关性分布
    
    返回：(强相关数, 弱相关数, 不相关数)
    """
    strong = sum(1 for slot in slots if slot.相关性 == "强相关")
    weak = sum(1 for slot in slots if slot.相关性 == "弱相关")
    irrelevant = sum(1 for slot in slots if slot.相关性 == "不相关")
    
    return strong, weak, irrelevant
