from __future__ import annotations

from collections import OrderedDict
from typing import Any


LEGACY_CATEGORY_SPECS: tuple[tuple[str, str], ...] = (
    ("材质形制", "绘画材料和装裱形式，例如绢本、纸本、绫本、立轴、手卷、扇面等。"),
    ("构图布局", "画面空间安排、主体位置、前后景关系、层次结构等。"),
    ("用笔特点", "线条表现、笔触特征、用笔方法，以及白描、皴法、点染、勾勒等技法。"),
    ("色彩氛围", "主要色调、设色方式、色彩搭配、明暗对比等。"),
    ("题材内容", "画面描绘的对象、人物、景物、场景和叙事内容。"),
    ("形神表现", "对象的形态特征、姿态、精神气质或神态刻画。"),
    ("艺术风格", "艺术特色、流派归属、工笔/写意等风格类型。"),
    ("意境营造", "文本明确写出的情感氛围、艺术境界或画面氛围。"),
    ("象征寓意", "文本明确写出的象征意义、文化寓意或隐喻。"),
    ("画家信息", "创作者的姓名、身份、生平、艺术成就等。"),
    ("创作年代", "作品朝代、年代、时期、时代归属。"),
    ("题跋印章", "题字、题跋、印章及可辨识内容。"),
    ("艺术传承", "与前人作品、师承、流派承接、风格传承的关系。"),
    ("历史语境", "作品创作时的社会历史背景、制度背景或时代语境。"),
    ("艺术地位", "文本明确写出的艺术史定位、影响或地位。"),
)

CATEGORY_SPECS = LEGACY_CATEGORY_SPECS

SIMPLE_V1_FIELD_SPECS: tuple[tuple[str, str], ...] = (
    ("画名", "作品名称、题名或通行名称。"),
    ("作者", "创作者姓名、款署作者或明确归属作者。"),
    ("朝代", "朝代、时代、时期归属。"),
    ("技法", "最具代表性的一种笔墨或造型技法，例如披麻皴、白描、工笔。"),
    ("构图", "最具代表性的一种构图或空间组织方式，例如全景式、留白、三段式纵深。"),
    ("题材", "画面主要题材或主体内容，例如山水、花鸟、仕女、历史故事。"),
    ("形制", "立轴、手卷、册页、扇面等装裱形制。"),
    ("材质", "绢本、纸本、绫本等绘画材质。"),
    ("设色方式", "最具代表性的一种设色方式，例如青绿、水墨、浅绛、重彩。"),
    ("题跋", "画上题字、题诗、题记、款识等明确文字内容。"),
    ("印章", "作者印、鉴藏印、闲章等明确印章信息。"),
)

ACADEMIC_V2_SCHEMA: OrderedDict[str, OrderedDict[str, str]] = OrderedDict(
    [
        (
            "基本信息",
            OrderedDict(
                [
                    ("作品名称", "作品的正式名称、通行名称、题名信息。"),
                    ("尺寸规格", "纵横尺寸、幅面规格、计量单位等。"),
                    ("收藏地点", "现藏机构、馆藏单位、收藏地点。"),
                ]
            ),
        ),
        (
            "基础视觉理解层",
            OrderedDict(
                [
                    ("材质与形制", "绢本、纸本等材质，以及立轴、长卷、册页、扇面等形制。"),
                    ("构图与空间", "三远法、留白、主次关系、空间层次、主体位置等。"),
                    ("笔墨技法", "皴法、描法、墨法、运笔走势等具体技法要点。"),
                    ("色彩与设色", "设色体系与具体色彩运用，例如青绿、浅绛、水墨等。"),
                ]
            ),
        ),
        (
            "语义理解层",
            OrderedDict(
                [
                    ("画面主体与题材", "描绘物象、题材分类及场景内容。"),
                    ("形神表现", "气韵、生动程度、以形写神等表现力相关事实。"),
                    ("艺术风格", "工笔/写意、院体/文人、流派风格等。"),
                    ("意境与情感", "清远、旷达、苍凉等意境与画面情绪。"),
                    ("象征与隐喻", "托物言志、文化寓意、象征关系等。"),
                ]
            ),
        ),
        (
            "上下文推理层",
            OrderedDict(
                [
                    ("画家身份与背景", "画家姓名、身份、生平关键节点。"),
                    ("创作年代与语境", "朝代、年份及对应社会历史语境。"),
                    ("题跋与印章", "题画诗、钤印、鉴藏印及其内容。"),
                    ("艺术传承与影响", "师法对象、传承脉络、后世影响。"),
                    ("艺术史地位", "艺术史价值、独创性贡献、地位判断。"),
                ]
            ),
        ),
    ]
)

SCHEMA_PROFILES = ("legacy", "academic_v2", "simple_v1")
RELEVANCE_VALUES = {"强相关", "弱相关", "不相关"}


def default_entry() -> dict[str, str]:
    return {"关键词": "", "相关性": "不相关", "原句": "不相关"}


def default_academic_leaf() -> dict[str, Any]:
    return {"相关性": "不相关", "要素列表": []}


def default_schema(profile: str = "legacy") -> OrderedDict[str, Any]:
    if profile == "legacy":
        schema: OrderedDict[str, Any] = OrderedDict()
        for name, _ in LEGACY_CATEGORY_SPECS:
            schema[name] = [default_entry()]
        return schema

    if profile == "simple_v1":
        schema = OrderedDict()
        for name, _ in SIMPLE_V1_FIELD_SPECS:
            schema[name] = []
        return schema

    if profile == "academic_v2":
        schema = OrderedDict()
        for group_name, leaf_specs in ACADEMIC_V2_SCHEMA.items():
            group_payload: OrderedDict[str, Any] = OrderedDict()
            for leaf_name in leaf_specs:
                group_payload[leaf_name] = default_academic_leaf()
            schema[group_name] = group_payload
        return schema

    raise ValueError(f"Unsupported schema profile: {profile}")


def build_schema_description(profile: str = "legacy") -> str:
    if profile == "legacy":
        return "\n".join(f"- {name}：{description}" for name, description in LEGACY_CATEGORY_SPECS)

    if profile == "simple_v1":
        return "\n".join(f"- {name}：{description}" for name, description in SIMPLE_V1_FIELD_SPECS)

    if profile == "academic_v2":
        blocks: list[str] = []
        for group_name, leaf_specs in ACADEMIC_V2_SCHEMA.items():
            blocks.append(f"{group_name}：")
            for leaf_name, description in leaf_specs.items():
                blocks.append(f"- {leaf_name}：{description}")
        return "\n".join(blocks)

    raise ValueError(f"Unsupported schema profile: {profile}")


def iter_schema_leaves(profile: str = "legacy") -> list[tuple[str | None, str, str]]:
    if profile == "legacy":
        return [(None, name, description) for name, description in LEGACY_CATEGORY_SPECS]

    if profile == "simple_v1":
        return [(None, name, description) for name, description in SIMPLE_V1_FIELD_SPECS]

    if profile == "academic_v2":
        rows: list[tuple[str | None, str, str]] = []
        for group_name, leaf_specs in ACADEMIC_V2_SCHEMA.items():
            for leaf_name, description in leaf_specs.items():
                rows.append((group_name, leaf_name, description))
        return rows

    raise ValueError(f"Unsupported schema profile: {profile}")


def infer_schema_profile(payload: dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return "legacy"

    simple_keys = {name for name, _ in SIMPLE_V1_FIELD_SPECS}
    if simple_keys.issubset(payload.keys()):
        return "simple_v1"

    if all(key in payload for key in ACADEMIC_V2_SCHEMA):
        return "academic_v2"

    if any(isinstance(value, list) for value in payload.values()):
        return "legacy"

    if any(isinstance(value, dict) for value in payload.values()):
        return "academic_v2"

    return "legacy"
