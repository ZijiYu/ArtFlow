from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from cluster_match.categories import default_schema, infer_schema_profile
from cluster_match.client import ChatResult
from cluster_match.config import RuntimeConfig
from cluster_match.dataset import build_jobs, clean_question
from cluster_match.evaluation import SemanticJudge, compute_weighted_metrics, extract_leaf_items, match_leaf_items
from cluster_match.json_utils import normalize_result, parse_json_object, split_atomic_facts
from cluster_match.runner import _format_duration, _mask_secret


class JsonUtilsTests(unittest.TestCase):
    def test_parse_json_object_accepts_code_fence(self) -> None:
        payload = """```json
{"材质形制":[{"关键词":"","相关性":"不相关","原句":"不相关"}]}
```"""
        parsed = parse_json_object(payload)
        self.assertEqual(parsed["材质形制"][0]["相关性"], "不相关")

    def test_normalize_result_fills_all_categories(self) -> None:
        normalized = normalize_result(
            {
                "画家信息": [
                    {
                        "关键词": "宋徽宗",
                        "相关性": "强相关",
                        "原句": "无法确认是否为宋徽宗亲笔。",
                    }
                ]
            }
        )
        self.assertEqual(normalized["画家信息"][0]["关键词"], "宋徽宗")
        self.assertEqual(normalized["材质形制"][0]["相关性"], "不相关")
        self.assertEqual(normalized["材质形制"][0]["原句"], "不相关")

    def test_normalize_result_filters_subjective_keywords(self) -> None:
        normalized = normalize_result(
            {
                "意境营造": [
                    {
                        "关键词": "绚烂之极归于平淡",
                        "相关性": "弱相关",
                        "原句": "恰如北宋文人“绚烂之极归于平淡”的美学追求。",
                    }
                ]
            }
        )
        self.assertEqual(normalized["意境营造"][0]["相关性"], "不相关")

        normalized = normalize_result(
            {
                "色彩氛围": [
                    {
                        "关键词": "设色浓丽清雅",
                        "相关性": "强相关",
                        "原句": "设色浓丽清雅",
                    }
                ]
            }
        )
        self.assertEqual(normalized["色彩氛围"][0]["相关性"], "不相关")

    def test_academic_schema_defaults_and_normalization(self) -> None:
        schema = default_schema("academic_v2")
        self.assertIn("基本信息", schema)
        self.assertIn("作品名称", schema["基本信息"])
        self.assertEqual(schema["基本信息"]["作品名称"]["相关性"], "不相关")

        normalized = normalize_result(
            {
                "基本信息": {
                    "作品名称": {
                        "相关性": "强相关",
                        "要素列表": ["溪山行旅图", "溪山行旅图", ""],
                    }
                },
                "基础视觉理解层": {
                    "笔墨技法": {
                        "相关性": "弱相关",
                        "要素列表": ["使用了淡墨披麻皴描绘远山"],
                    }
                },
            },
            schema_profile="academic_v2",
        )
        self.assertEqual(normalized["基本信息"]["作品名称"]["要素列表"], ["溪山行旅图"])
        self.assertEqual(normalized["基础视觉理解层"]["笔墨技法"]["要素列表"], ["淡墨", "披麻皴", "远山"])
        self.assertEqual(normalized["语义理解层"]["艺术风格"]["相关性"], "不相关")

    def test_simple_schema_defaults_and_normalization(self) -> None:
        schema = default_schema("simple_v1")
        self.assertIn("画名", schema)
        self.assertEqual(schema["画名"], [])

        normalized = normalize_result(
            {
                "画名": "溪山行旅图",
                "作者": ["范宽", "范宽"],
                "技法": ["使用了淡墨披麻皴描绘远山"],
                "构图": ["全景式布局与留白"],
                "题材": ["花鸟、山水"],
                "形制": ["立轴、手卷"],
                "材质": ["绢本、纸本"],
                "设色方式": ["青绿设色"],
                "题跋": ["题诗一则", "题诗一则"],
            },
            schema_profile="simple_v1",
        )
        self.assertEqual(normalized["画名"], ["溪山行旅图"])
        self.assertEqual(normalized["作者"], ["范宽"])
        self.assertEqual(normalized["技法"], ["披麻皴", "淡墨"])
        self.assertEqual(normalized["构图"], ["留白"])
        self.assertEqual(normalized["题材"], ["花鸟", "山水"])
        self.assertEqual(normalized["形制"], ["手卷"])
        self.assertEqual(normalized["材质"], [])
        self.assertEqual(normalized["设色方式"], ["青绿"])
        self.assertEqual(normalized["题跋"], ["题诗一则"])
        self.assertEqual(normalized["印章"], [])

    def test_infer_schema_profile(self) -> None:
        self.assertEqual(infer_schema_profile({"材质形制": []}), "legacy")
        self.assertEqual(
            infer_schema_profile(
                {
                    "画名": [],
                    "作者": [],
                    "朝代": [],
                    "技法": [],
                    "构图": [],
                    "题材": [],
                    "形制": [],
                    "材质": [],
                    "设色方式": [],
                    "题跋": [],
                    "印章": [],
                }
            ),
            "simple_v1",
        )
        self.assertEqual(infer_schema_profile({"基本信息": {"作品名称": {"相关性": "不相关", "要素列表": []}}}), "academic_v2")

    def test_split_atomic_facts(self) -> None:
        self.assertEqual(split_atomic_facts("使用了淡墨披麻皴描绘远山"), ["淡墨", "披麻皴", "远山"])
        self.assertEqual(split_atomic_facts("青绿设色"), ["青绿", "设色"])

    def test_academic_empty_relevant_becomes_irrelevant(self) -> None:
        normalized = normalize_result(
            {
                "语义理解层": {
                    "艺术风格": {
                        "相关性": "强相关",
                        "要素列表": [],
                    }
                }
            },
            schema_profile="academic_v2",
        )
        self.assertEqual(normalized["语义理解层"]["艺术风格"]["相关性"], "不相关")
        self.assertEqual(normalized["语义理解层"]["艺术风格"]["要素列表"], [])


class DatasetTests(unittest.TestCase):
    def test_clean_question_removes_think_suffix(self) -> None:
        self.assertEqual(clean_question("这幅画的作者是谁？/think"), "这幅画的作者是谁？")

    def test_build_jobs_uses_existing_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "dataset.jsonl"
            baseline_dir = root / "baseline"
            enhanced_dir = root / "enhanced"
            baseline_dir.mkdir()
            enhanced_dir.mkdir()
            (baseline_dir / "s001.md").write_text("## 赏析\n文本A", encoding="utf-8")
            (enhanced_dir / "s001.md").write_text("## 赏析\n文本B", encoding="utf-8")
            dataset_path.write_text(
                json.dumps({"sample_id": "s001", "question": "请赏析这幅画/think"}, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

            jobs = build_jobs(
                dataset_path=dataset_path,
                baseline_dir=baseline_dir,
                enhanced_dir=enhanced_dir,
                sources=["baseline", "enhanced"],
            )

            self.assertEqual(len(jobs), 2)
            self.assertEqual(jobs[0].question, "请赏析这幅画")


class RunnerHelpersTests(unittest.TestCase):
    def test_format_duration(self) -> None:
        self.assertEqual(_format_duration(65), "01:05")
        self.assertEqual(_format_duration(3661), "01:01:01")

    def test_mask_secret(self) -> None:
        self.assertEqual(_mask_secret("abcdefgh1234"), "abcd...1234")

    def test_runtime_config_reads_api_key_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            secret_path = root / "api.txt"
            secret_path.write_text("first-key\nsecond-key\n", encoding="utf-8")
            config_path = root / "config.yaml"
            config_path.write_text(
                "\n".join(
                    [
                        "api:",
                        "  key: \"\"",
                        f"  key_file: \"{secret_path}\"",
                        "  key_line: 2",
                        "  base_url: \"https://example.com/v1\"",
                    ]
                ),
                encoding="utf-8",
            )

            runtime = RuntimeConfig.from_sources(config_path=config_path)

        self.assertEqual("second-key", runtime.api_key)
        self.assertEqual("https://example.com/v1", runtime.base_url)


class EvaluationTests(unittest.TestCase):
    def test_compute_weighted_metrics(self) -> None:
        metrics = compute_weighted_metrics(10, 8, 4, 2)
        self.assertEqual(metrics["TP_w"], 5.0)
        self.assertEqual(metrics["Matched_TP_w"], 5.0)
        self.assertEqual(metrics["N_extra_gen_success"], 0.0)
        self.assertEqual(metrics["ACC"], 0.5)
        self.assertEqual(metrics["Precision"], 0.625)
        self.assertAlmostEqual(metrics["F1"], 0.5555555556)

    def test_compute_weighted_metrics_rewards_extra_gen(self) -> None:
        metrics = compute_weighted_metrics(10, 8, 4, 2, 2)
        self.assertEqual(metrics["Matched_TP_w"], 5.0)
        self.assertEqual(metrics["N_extra_gen_success"], 2.0)
        self.assertEqual(metrics["TP_w"], 7.0)
        self.assertAlmostEqual(metrics["ACC"], 7.0 / 12.0)
        self.assertAlmostEqual(metrics["Precision"], 0.875)
        self.assertAlmostEqual(metrics["F1"], 0.7)

    def test_match_leaf_items_exact(self) -> None:
        judge = SemanticJudge(mode="exact")
        details, metrics = match_leaf_items(
            category_path="基础视觉理解层/笔墨技法",
            gt_items=["披麻皴", "淡墨"],
            gen_items=["披麻皴法", "石青"],
            judge=judge,
        )
        self.assertEqual(metrics["N_GT"], 2.0)
        self.assertEqual(metrics["N_Gen"], 2.0)
        self.assertEqual(metrics["N_strong"], 1.0)
        self.assertEqual(metrics["N_weak"], 0.0)
        self.assertEqual(metrics["N_extra_gen_success"], 1.0)
        self.assertAlmostEqual(metrics["ACC"], 2.0 / 3.0)
        self.assertEqual(metrics["Precision"], 1.0)
        self.assertTrue(any(row["score"] == 1.0 for row in details))
        self.assertTrue(any(row["method"] == "extra_gen_success" for row in details))

    def test_match_leaf_items_gen_only_counts_as_success(self) -> None:
        judge = SemanticJudge(mode="exact")
        details, metrics = match_leaf_items(
            category_path="设色方式",
            gt_items=[],
            gen_items=["青绿"],
            judge=judge,
        )
        self.assertEqual(metrics["N_GT"], 0.0)
        self.assertEqual(metrics["N_Gen"], 1.0)
        self.assertEqual(metrics["N_extra_gen_success"], 1.0)
        self.assertEqual(metrics["TP_w"], 1.0)
        self.assertEqual(metrics["ACC"], 1.0)
        self.assertEqual(metrics["Precision"], 1.0)
        self.assertEqual(metrics["F1"], 1.0)
        self.assertTrue(any(row["method"] == "extra_gen_success" for row in details))

    def test_extract_leaf_items_simple_v1(self) -> None:
        payload = {
            "画名": ["溪山行旅图"],
            "作者": ["范宽"],
            "朝代": ["北宋"],
            "技法": ["披麻皴"],
            "构图": ["全景式布局"],
            "题材": ["山水"],
            "形制": ["立轴"],
            "材质": ["绢本"],
            "设色方式": ["水墨"],
            "题跋": ["题诗"],
            "印章": ["朱文印"],
        }
        leaf_map = extract_leaf_items(payload)
        self.assertEqual(leaf_map[("", "作者")], ["范宽"])
        self.assertEqual(leaf_map[("", "技法")], ["披麻皴"])
        self.assertEqual(leaf_map[("", "印章")], ["朱文印"])

    def test_llm_judge_timeout_falls_back_without_crashing(self) -> None:
        class FailingClient:
            def chat(self, **_: object) -> ChatResult:
                return ChatResult(
                    content=None,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    model="judge-model",
                    endpoint="https://example.com/chat/completions",
                    status_code=None,
                    duration_ms=10.0,
                    error="The read operation timed out",
                )

        judge = SemanticJudge(mode="llm", client=FailingClient(), model="judge-model")
        details, metrics = match_leaf_items(
            category_path="基础视觉理解层/笔墨技法",
            gt_items=["披麻皴"],
            gen_items=["皴擦"],
            judge=judge,
        )
        self.assertEqual(metrics["N_GT"], 1.0)
        self.assertEqual(metrics["N_Gen"], 1.0)
        self.assertEqual(metrics["N_strong"], 0.0)
        self.assertEqual(metrics["N_weak"], 0.0)
        self.assertEqual(judge.fallback_count, 1)
        self.assertTrue(any(row["method"] == "unmatched_gt" for row in details))


if __name__ == "__main__":
    unittest.main()
