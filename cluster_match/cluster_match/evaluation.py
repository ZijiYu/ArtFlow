from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

from .categories import infer_schema_profile, iter_schema_leaves
from .client import ChatCompletionsClient
from .json_utils import parse_json_object, split_atomic_facts


GENERIC_SUFFIXES = (
    "法",
    "技法",
    "画法",
    "笔法",
    "皴法",
    "描法",
    "墨法",
    "设色",
    "风格",
    "体系",
    "图式",
    "布局",
    "构图",
    "形式",
)
NORMALIZE_RE = re.compile(r"[\s，,；;。:：/|·•“”\"'《》〈〉【】\[\]（）()\-]+")
GENERIC_TAILS = {"法", "技法", "画法", "笔法", "皴法", "描法", "墨法", "设色"}


@dataclass(slots=True)
class MatchDecision:
    score: float
    method: str
    reason: str
    cacheable: bool = True


def canonicalize_factor(text: str) -> str:
    return NORMALIZE_RE.sub("", str(text or "")).strip()


def strip_generic_suffixes(text: str) -> str:
    cleaned = canonicalize_factor(text)
    for suffix in GENERIC_SUFFIXES:
        if cleaned.endswith(suffix) and len(cleaned) > len(suffix) + 1:
            return cleaned[: -len(suffix)]
    return cleaned


def dedupe_items(items: list[str]) -> list[str]:
    deduped: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = str(item or "").strip()
        if not cleaned:
            continue
        if cleaned in {"未涉及", "不相关"}:
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        deduped.append(cleaned)
    return deduped


def extract_leaf_items(payload: dict[str, Any]) -> dict[tuple[str, str], list[str]]:
    profile = infer_schema_profile(payload)
    leaf_map: dict[tuple[str, str], list[str]] = {}

    if profile == "simple_v1":
        for _, leaf_name, _ in iter_schema_leaves("simple_v1"):
            items: list[str] = []
            raw_items = payload.get(leaf_name, [])
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            elif not isinstance(raw_items, list):
                raw_items = []
            for item in raw_items:
                items.extend(split_atomic_facts(str(item)))
            leaf_map[("", leaf_name)] = dedupe_items(items)
        return leaf_map

    if profile == "legacy":
        for _, leaf_name, _ in iter_schema_leaves("legacy"):
            items: list[str] = []
            raw_items = payload.get(leaf_name, [])
            if isinstance(raw_items, dict):
                raw_items = [raw_items]
            elif not isinstance(raw_items, list):
                raw_items = []
            for entry in raw_items:
                if not isinstance(entry, dict):
                    continue
                relevance = str(entry.get("相关性", "")).strip()
                if relevance not in {"强相关", "弱相关"}:
                    continue
                keyword = str(entry.get("关键词", "")).strip()
                items.extend(split_atomic_facts(keyword))
            leaf_map[("", leaf_name)] = dedupe_items(items)
        return leaf_map

    for group_name, leaf_name, _ in iter_schema_leaves("academic_v2"):
        items: list[str] = []
        group_payload = payload.get(group_name or "", {})
        if not isinstance(group_payload, dict):
            group_payload = {}
        node = group_payload.get(leaf_name, {})
        if not isinstance(node, dict):
            node = {}
        relevance = str(node.get("相关性", "")).strip()
        if relevance in {"强相关", "弱相关"}:
            raw_items = node.get("要素列表", [])
            if isinstance(raw_items, str):
                raw_items = [raw_items]
            elif not isinstance(raw_items, list):
                raw_items = []
            for item in raw_items:
                items.extend(split_atomic_facts(str(item)))
        leaf_map[(group_name or "", leaf_name)] = dedupe_items(items)
    return leaf_map


def compute_weighted_metrics(
    n_gt: int,
    n_gen: int,
    n_strong: int,
    n_weak: int,
    n_extra_gen_success: int = 0,
) -> dict[str, float]:
    matched_tp_w = float(n_strong) + (0.5 * float(n_weak))
    extra_gen_success = float(n_extra_gen_success)
    tp_w = matched_tp_w + extra_gen_success
    acc_base = float(n_gt) + extra_gen_success
    acc = tp_w / acc_base if acc_base > 0 else 0.0
    precision = tp_w / float(n_gen) if n_gen > 0 else 0.0
    f1 = (2.0 * precision * acc / (precision + acc)) if (precision + acc) > 0 else 0.0
    return {
        "N_GT": float(n_gt),
        "N_Gen": float(n_gen),
        "N_strong": float(n_strong),
        "N_weak": float(n_weak),
        "N_extra_gen_success": extra_gen_success,
        "Matched_TP_w": matched_tp_w,
        "TP_w": tp_w,
        "ACC": acc,
        "Precision": precision,
        "F1": f1,
    }


class SemanticJudge:
    def __init__(
        self,
        *,
        mode: str = "llm",
        client: ChatCompletionsClient | None = None,
        model: str = "",
        cache_path: str | Path | None = None,
    ) -> None:
        self.mode = mode
        self.client = client
        self.model = model
        self.cache_path = Path(cache_path).expanduser() if cache_path else None
        self.cache: dict[str, dict[str, Any]] = {}
        self.fallback_count = 0
        self.fallback_errors: list[str] = []
        if self.cache_path and self.cache_path.exists():
            try:
                parsed = json.loads(self.cache_path.read_text(encoding="utf-8"))
                if isinstance(parsed, dict):
                    self.cache = parsed
            except json.JSONDecodeError:
                self.cache = {}

    def _record_fallback(self, error: str) -> None:
        message = str(error or "unknown_error").strip() or "unknown_error"
        self.fallback_count += 1
        if len(self.fallback_errors) < 10:
            self.fallback_errors.append(message)

    def save_cache(self) -> None:
        if not self.cache_path:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.cache_path.write_text(json.dumps(self.cache, ensure_ascii=False, indent=2), encoding="utf-8")

    def _cache_key(self, category_path: str, gt_item: str, gen_item: str) -> str:
        return json.dumps(
            {
                "mode": self.mode,
                "model": self.model,
                "category_path": category_path,
                "gt_item": gt_item,
                "gen_item": gen_item,
            },
            ensure_ascii=False,
            sort_keys=True,
        )

    def _heuristic_match(self, gt_item: str, gen_item: str) -> MatchDecision | None:
        gt_norm = canonicalize_factor(gt_item)
        gen_norm = canonicalize_factor(gen_item)
        if not gt_norm or not gen_norm:
            return MatchDecision(score=0.0, method="empty", reason="空要素")

        if gt_norm == gen_norm:
            return MatchDecision(score=1.0, method="exact", reason="规范化后完全一致")

        gt_trimmed = strip_generic_suffixes(gt_norm)
        gen_trimmed = strip_generic_suffixes(gen_norm)
        if gt_trimmed and gen_trimmed and gt_trimmed == gen_trimmed:
            return MatchDecision(score=1.0, method="generic_suffix", reason="去通用后缀后一致")

        if gt_norm.startswith(gen_norm) and gt_norm[len(gen_norm) :] in GENERIC_TAILS:
            return MatchDecision(score=1.0, method="generic_tail", reason="仅多出通用后缀")
        if gen_norm.startswith(gt_norm) and gen_norm[len(gt_norm) :] in GENERIC_TAILS:
            return MatchDecision(score=1.0, method="generic_tail", reason="仅多出通用后缀")

        shorter = min(len(gt_norm), len(gen_norm))
        if shorter >= 2 and (gt_norm in gen_norm or gen_norm in gt_norm):
            return MatchDecision(score=0.5, method="containment", reason="存在包含关系")

        if self.mode == "exact":
            return MatchDecision(score=0.0, method="exact_only", reason="未达到严格字符匹配")

        return None

    def _llm_judge(self, category_path: str, gt_item: str, gen_item: str) -> MatchDecision:
        if self.client is None or not self.model:
            raise RuntimeError("LLM judge requires a configured client and model.")

        system_prompt = """你是中国画结构化要素匹配裁判。

你只比较 GT 要素 和 Gen 要素 在同一层级中的语义对应程度，禁止补充常识。
你必须严格使用以下分值：
- 1.0：语义完全一致，或属于直接同义/专业改写，可视为完全命中
- 0.5：语义相关，但存在上位/下位替代、描述模糊或方向正确但不完全对应
- 0.0：未提及、明显不对应、事实冲突，或只是空泛近义

只输出 JSON，不要解释文字。"""
        user_prompt = f"""请判断以下两个中国画要素的匹配程度。

类别路径：{category_path}
GT 要素：{gt_item}
Gen 要素：{gen_item}

返回格式：
{{
  "score": 1.0,
  "reason": "不超过20字"
}}

注意：
- 只允许输出 1.0、0.5、0.0 三种分值
- 必须忠于短语本身，不能因为常识接近就判高分
- 如果只是大类接近但不精确，给 0.5
- 如果完全不是同一要素，给 0.0"""

        try:
            result = self.client.chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                model=self.model,
                temperature=0.0,
            )
            if not result.content:
                raise RuntimeError(f"LLM judge failed: {result.error or 'empty_response'}")

            parsed = parse_json_object(result.content)
            raw_score = parsed.get("score", 0.0)
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = 0.0
            if score not in {0.0, 0.5, 1.0}:
                score = 0.0
            reason = str(parsed.get("reason", "")).strip() or "无说明"
            return MatchDecision(score=score, method="llm", reason=reason)
        except Exception as exc:  # noqa: BLE001
            self._record_fallback(str(exc))
            return MatchDecision(
                score=0.0,
                method="llm_error_fallback",
                reason="LLM裁判失败，按未命中处理",
                cacheable=False,
            )

    def score_pair(self, category_path: str, gt_item: str, gen_item: str) -> MatchDecision:
        cache_key = self._cache_key(category_path, gt_item, gen_item)
        cached = self.cache.get(cache_key)
        if isinstance(cached, dict):
            try:
                return MatchDecision(
                    score=float(cached.get("score", 0.0)),
                    method=str(cached.get("method", "cache")),
                    reason=str(cached.get("reason", "")),
                )
            except (TypeError, ValueError):
                pass

        heuristic = self._heuristic_match(gt_item, gen_item)
        decision = heuristic if heuristic is not None else self._llm_judge(category_path, gt_item, gen_item)
        if decision.cacheable:
            self.cache[cache_key] = {
                "score": decision.score,
                "method": decision.method,
                "reason": decision.reason,
            }
        return decision


def match_leaf_items(
    *,
    category_path: str,
    gt_items: list[str],
    gen_items: list[str],
    judge: SemanticJudge,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    gt_items = dedupe_items(gt_items)
    gen_items = dedupe_items(gen_items)
    pair_details: list[dict[str, Any]] = []

    if not gt_items and not gen_items:
        return pair_details, compute_weighted_metrics(0, 0, 0, 0, 0)

    if not gt_items:
        for gen_item in gen_items:
            pair_details.append(
                {
                    "category_path": category_path,
                    "gt_item": "",
                    "gen_item": gen_item,
                    "score": 1.0,
                    "method": "extra_gen_success",
                    "reason": "Gen 额外细节，按成功计",
                }
            )
        return pair_details, compute_weighted_metrics(0, len(gen_items), 0, 0, len(gen_items))

    if not gen_items:
        for gt_item in gt_items:
            pair_details.append(
                {
                    "category_path": category_path,
                    "gt_item": gt_item,
                    "gen_item": "",
                    "score": 0.0,
                    "method": "gt_only",
                    "reason": "Gen 为空",
                }
            )
        return pair_details, compute_weighted_metrics(len(gt_items), 0, 0, 0, 0)

    score_matrix = np.zeros((len(gt_items), len(gen_items)), dtype=float)
    decisions: dict[tuple[int, int], MatchDecision] = {}
    for gt_index, gt_item in enumerate(gt_items):
        for gen_index, gen_item in enumerate(gen_items):
            decision = judge.score_pair(category_path, gt_item, gen_item)
            score_matrix[gt_index, gen_index] = decision.score
            decisions[(gt_index, gen_index)] = decision

    cost_matrix = 1.0 - score_matrix
    row_indices, col_indices = linear_sum_assignment(cost_matrix)
    matched_gt: set[int] = set()
    matched_gen: set[int] = set()
    n_strong = 0
    n_weak = 0
    n_extra_gen_success = 0

    for gt_index, gen_index in zip(row_indices.tolist(), col_indices.tolist()):
        decision = decisions[(gt_index, gen_index)]
        if decision.score <= 0.0:
            continue
        matched_gt.add(gt_index)
        matched_gen.add(gen_index)
        if decision.score == 1.0:
            n_strong += 1
        elif decision.score == 0.5:
            n_weak += 1
        pair_details.append(
            {
                "category_path": category_path,
                "gt_item": gt_items[gt_index],
                "gen_item": gen_items[gen_index],
                "score": decision.score,
                "method": decision.method,
                "reason": decision.reason,
            }
        )

    for gt_index, gt_item in enumerate(gt_items):
        if gt_index in matched_gt:
            continue
        pair_details.append(
            {
                "category_path": category_path,
                "gt_item": gt_item,
                "gen_item": "",
                "score": 0.0,
                "method": "unmatched_gt",
                "reason": "未找到匹配",
            }
        )

    for gen_index, gen_item in enumerate(gen_items):
        if gen_index in matched_gen:
            continue
        n_extra_gen_success += 1
        pair_details.append(
            {
                "category_path": category_path,
                "gt_item": "",
                "gen_item": gen_item,
                "score": 1.0,
                "method": "extra_gen_success",
                "reason": "Gen 额外细节，按成功计",
            }
        )

    metrics = compute_weighted_metrics(len(gt_items), len(gen_items), n_strong, n_weak, n_extra_gen_success)
    return pair_details, metrics
