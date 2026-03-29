from __future__ import annotations

import json

from ..common.prompt_utils import meta_payload
from ..cot_layer.models import CrossValidationResult, DomainCoTRecord


def build_round_table_prompt(
    outputs: list[DomainCoTRecord],
    validation: CrossValidationResult,
    meta: dict,
    *,
    enable_web_search: bool = False,
) -> str:
    payload = [
        {
            "slot_name": output.slot_name,
            "slot_term": output.slot_term,
            "visual_anchoring": [item.observation for item in output.visual_anchoring[:4]],
            "domain_decoding": [item.term for item in output.domain_decoding[:4]],
            "cultural_mapping": [item.insight for item in output.cultural_mapping[:3]],
            "question_coverage": [item.question for item in output.question_coverage if item.answered][:4],
            "unresolved_points": output.unresolved_points[:4],
            "statuses": output.statuses[:3],
        }
        for output in outputs
    ]
    issues = [
        {
            "type": issue.issue_type,
            "severity": issue.severity,
            "slots": issue.slot_names,
            "detail": issue.detail,
        }
        for issue in validation.issues[:12]
    ]
    web_instruction = ""
    follow_up_schema = '    {"slot_name": "槽位名", "question": "需要继续追问的问题", "why": "为什么值得追问", "priority": "high|medium", "rag_queries": ["检索词"]}\n'
    if enable_web_search:
        web_instruction = (
            "若判断需要联网搜索，请同时输出 retrieval_mode 和 web_queries。`web_queries` 可以是多词组合，例如作品名+作者名+馆藏、题跋释文+关键词、作者比较等；"
            "技法、构图、设色、图像学描述优先 rag，作者、作品名、馆藏、题跋、印章、年代、版本、流传、比较研究优先 web，两者都重要时用 hybrid。\n"
        )
        follow_up_schema = (
            '    {"slot_name": "槽位名", "question": "需要继续追问的问题", "why": "为什么值得追问", "priority": "high|medium", '
            '"retrieval_mode": "rag|web|hybrid", "rag_queries": ["短词"], "web_queries": ["多词查询"], "retrieval_reason": "为什么这么分"}\n'
        )
    return (
        "你是圆桌细节复核节点。请基于各槽位草稿、memory card 与已有本地结果，专门找出回答里还没注意到的细节，"
        "并生成值得继续核实的问题与需要调用 RAG 的信息点。\n"
        "不要过度关注赏析、修辞或美学感受，重点关注对象身份、图像细节、图像与术语的对应、时代/宗教/人物属性、构图元素、材质技法、图像证据缺口。\n"
        "例如：如果提到了“佛陀”，应继续追问“是哪一尊佛”“依据是什么”“画面中有哪些识别线索”“哪些信息需要检索”。\n"
        "如果现有回答只完成了对象命名、尊者编号、题材归类或术语贴标签，但没有展开可见物象、识别依据、属性组合和相关图像信息，"
        "应视为未充分回答，而不是已完成回答。\n"
        "请优先检查：对象是否只有名称没有形象描述，是否只有身份没有持物/姿态/服饰/座具/环境等支撑细节，"
        "以及这些物象中哪些真正具有身份识别作用。\n"
        "如果 context_meta 里已经提供了 post_rag_text_extraction 或 rag_cache，请先判断这些已有文献证据是否足够，"
        "只有在 cache 明显不足时，才提出新的 rag_queries 或 rag_needs。\n"
        "rag_queries 应优先输出稳定术语或单一实体名，不要输出自然语言短句。\n"
        f"{web_instruction}"
        "follow_up_questions 中的 slot_name 必须严格使用已有输出里出现过的 slot_name，不要自造新的槽位名。\n"
        "请只输出 JSON，不要输出解释性散文。\n"
        "JSON 结构固定为：\n"
        "{\n"
        '  "review_summary": "一句话总结当前回答最大的细节缺口",\n'
        '  "blind_spots": ["被忽略的细节或识别盲点"],\n'
        '  "follow_up_questions": [\n'
        f"{follow_up_schema}"
        "  ],\n"
        '  "rag_needs": [\n'
        '    {"topic": "需要补检索的信息主题", "reason": "为什么要查", "queries": ["检索词"]}\n'
        "  ]\n"
        "}\n\n"
        f"context_meta: {json.dumps(meta_payload(meta), ensure_ascii=False)}\n"
        f"已有输出: {json.dumps(payload, ensure_ascii=False)}\n"
        f"本地校验: {json.dumps(issues, ensure_ascii=False)}\n"
    )
