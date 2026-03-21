from __future__ import annotations

import copy
import json
from typing import Any, ClassVar, Literal, get_args, get_origin

try:
    from pydantic import BaseModel, Field, field_validator
except ImportError:  # pragma: no cover - 本地无 pydantic 时走轻量兼容实现
    class _FieldInfo:
        def __init__(
            self,
            default: Any = ...,
            *,
            default_factory: Any | None = None,
            ge: float | None = None,
            le: float | None = None,
        ) -> None:
            self.default = default
            self.default_factory = default_factory
            self.ge = ge
            self.le = le


    def Field(  # type: ignore[misc]
        default: Any = ...,
        *,
        default_factory: Any | None = None,
        ge: float | None = None,
        le: float | None = None,
    ) -> _FieldInfo:
        return _FieldInfo(default, default_factory=default_factory, ge=ge, le=le)


    def field_validator(*field_names: str):  # type: ignore[misc]
        def decorator(func: Any) -> Any:
            setattr(func, "__field_validator_names__", field_names)
            return func

        return decorator


    class _BaseModelMeta(type):
        def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any]) -> Any:
            cls = super().__new__(mcls, name, bases, namespace)
            annotations: dict[str, Any] = {}
            fields: dict[str, _FieldInfo] = {}
            validators: dict[str, list[Any]] = {}

            for base in reversed(bases):
                annotations.update(getattr(base, "__annotations__", {}))
                fields.update(copy.deepcopy(getattr(base, "__fields__", {})))
                for key, value in getattr(base, "__validators__", {}).items():
                    validators.setdefault(key, []).extend(value)

            annotations.update(namespace.get("__annotations__", {}))

            for field_name in annotations:
                if field_name.startswith("_") or get_origin(annotations[field_name]) is ClassVar:
                    continue
                raw_default = namespace.get(field_name, getattr(cls, field_name, ...))
                if isinstance(raw_default, _FieldInfo):
                    fields[field_name] = raw_default
                elif raw_default is ...:
                    fields[field_name] = _FieldInfo(...)
                else:
                    fields[field_name] = _FieldInfo(raw_default)

            for value in namespace.values():
                field_names = getattr(value, "__field_validator_names__", None)
                if not field_names:
                    continue
                for field_name in field_names:
                    validators.setdefault(field_name, []).append(value)

            cls.__fields__ = fields
            cls.__validators__ = validators
            cls.__model_annotations__ = {
                key: value
                for key, value in annotations.items()
                if key in fields
            }
            return cls


    class BaseModel(metaclass=_BaseModelMeta):  # type: ignore[misc]
        __fields__: ClassVar[dict[str, _FieldInfo]]
        __validators__: ClassVar[dict[str, list[Any]]]
        __model_annotations__: ClassVar[dict[str, Any]]

        def __init__(self, **kwargs: Any) -> None:
            for field_name, annotation in self.__model_annotations__.items():
                field_info = self.__fields__[field_name]
                if field_name in kwargs:
                    value = kwargs[field_name]
                elif field_info.default_factory is not None:
                    value = field_info.default_factory()
                elif field_info.default is not ...:
                    value = copy.deepcopy(field_info.default)
                else:
                    raise TypeError(f"{self.__class__.__name__} 缺少必填字段: {field_name}")

                value = self._convert_value(annotation, value)
                value = self._run_validators(field_name, value)
                self._check_bounds(field_name, field_info, value)
                setattr(self, field_name, value)

        @classmethod
        def _run_validators(cls, field_name: str, value: Any) -> Any:
            for validator in cls.__validators__.get(field_name, []):
                value = validator(cls, value)
            return value

        @classmethod
        def _convert_value(cls, annotation: Any, value: Any) -> Any:
            origin = get_origin(annotation)
            args = get_args(annotation)

            if value is None:
                return None

            if origin is Literal:
                return value

            if origin in (list, list[Any]):
                item_type = args[0] if args else Any
                return [cls._convert_value(item_type, item) for item in value]

            if origin is dict:
                key_type, value_type = args if len(args) == 2 else (Any, Any)
                return {
                    cls._convert_value(key_type, key): cls._convert_value(value_type, item)
                    for key, item in value.items()
                }

            if origin in (tuple,):
                item_types = args or ()
                return tuple(
                    cls._convert_value(item_types[min(index, len(item_types) - 1)], item)
                    if item_types
                    else item
                    for index, item in enumerate(value)
                )

            if origin is None and isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if isinstance(value, annotation):
                    return value
                if isinstance(value, dict):
                    return annotation.model_validate(value)

            if isinstance(annotation, type) and issubclass(annotation, BaseModel):
                if isinstance(value, annotation):
                    return value
                if isinstance(value, dict):
                    return annotation.model_validate(value)

            return value

        @staticmethod
        def _check_bounds(field_name: str, field_info: _FieldInfo, value: Any) -> None:
            if isinstance(value, (int, float)):
                if field_info.ge is not None and value < field_info.ge:
                    raise ValueError(f"{field_name} 必须 >= {field_info.ge}")
                if field_info.le is not None and value > field_info.le:
                    raise ValueError(f"{field_name} 必须 <= {field_info.le}")

        @classmethod
        def model_validate(cls, data: Any) -> Any:
            if isinstance(data, cls):
                return data
            if not isinstance(data, dict):
                raise TypeError(f"{cls.__name__}.model_validate 只接受 dict")
            return cls(**data)

        def model_dump(self) -> dict[str, Any]:
            return {
                field_name: self._dump_value(getattr(self, field_name))
                for field_name in self.__model_annotations__
            }

        @classmethod
        def _dump_value(cls, value: Any) -> Any:
            if isinstance(value, BaseModel):
                return value.model_dump()
            if isinstance(value, list):
                return [cls._dump_value(item) for item in value]
            if isinstance(value, dict):
                return {key: cls._dump_value(item) for key, item in value.items()}
            return value

        def model_dump_json(self, indent: int | None = None) -> str:
            return json.dumps(self.model_dump(), ensure_ascii=False, indent=indent)

        def model_copy(self, update: dict[str, Any] | None = None) -> Any:
            data = self.model_dump()
            if update:
                data.update(update)
            return self.__class__.model_validate(data)


class SlotDefinition(BaseModel):
    slot_name: str
    description: str = ""
    covered_terms: list[str] = Field(default_factory=list)


class SlotsSpec(BaseModel):
    slots: list[SlotDefinition]

    @field_validator("slots")
    @classmethod
    def ensure_slots(cls, value: list) -> list:
        return value or []


class DuplicateCluster(BaseModel):
    cluster_id: int = Field(ge=0)
    sentence_ids: list[int] = Field(default_factory=list)
    sentences: list[str] = Field(default_factory=list)
    avg_similarity: float = Field(ge=0.0, le=1.0)


class SemanticTerm(BaseModel):
    term: str
    category: str
    detail: str = ""
    sentence_ids: list[int] = Field(default_factory=list)
    evidence_sentences: list[str] = Field(default_factory=list)


class SlotMatchRecord(BaseModel):
    slot_name: str
    matched_terms: list[str] = Field(default_factory=list)
    matched_categories: list[str] = Field(default_factory=list)
    sentence_ids: list[int] = Field(default_factory=list)
    reason: str = ""


class FidelityRecord(BaseModel):
    term: str
    category: str
    supported_by_ground_truth: bool
    reason: str = ""


class ContextMetrics(BaseModel):
    context_name: Literal["context_baseline", "context_enhanced"]
    sentence_count: int = Field(ge=0)
    token_count: int = Field(ge=0)
    similar_semantic_num: int = Field(ge=0)
    duplicate_sentence_num: int = Field(ge=0)
    unique_semantic_num: int = Field(ge=0)
    term_num: int = Field(ge=0)
    slots_match: int = Field(ge=0)
    accuracy: float = Field(ge=0.0, le=1.0)
    duplicate_rate: float = Field(ge=0.0, le=1.0)
    information_density: float = Field(ge=0.0)
    slot_coverage: float = Field(ge=0.0, le=1.0)
    duplicate_clusters: list[DuplicateCluster] = Field(default_factory=list)
    terms: list[SemanticTerm] = Field(default_factory=list)
    slot_matches: list[SlotMatchRecord] = Field(default_factory=list)
    fidelity_records: list[FidelityRecord] = Field(default_factory=list)
    duplicate_clusters_jsonl: str
    terms_jsonl: str
    slot_matches_jsonl: str
    fidelity_jsonl: str


class FinalJudgment(BaseModel):
    winner: Literal["context_baseline", "context_enhanced", "tie"]
    textual_loss_for: Literal["context_baseline", "context_enhanced", "tie"]
    reasoning: str
    textual_loss: str


class EvalV2Result(BaseModel):
    base_url: str
    embedding_model: str
    judge_model: str
    duplicate_threshold: float = Field(ge=0.0, le=1.0)
    slots_input: str
    slots_number: int = Field(ge=0)
    slots_spec: list[SlotDefinition] = Field(default_factory=list)
    image_context_v: str = ""
    context_baseline_metrics: ContextMetrics
    context_enhanced_metrics: ContextMetrics
    final_judgment: FinalJudgment
    output_dir: str
    result_json_path: str
    llm_tokens: int = Field(ge=0)


class TermExtractionOutput(BaseModel):
    terms: list[SemanticTerm]

    @field_validator("terms")
    @classmethod
    def ensure_terms(cls, value: list) -> list:
        return value or []


class SlotMatchOutput(BaseModel):
    matches: list[SlotMatchRecord]

    @field_validator("matches")
    @classmethod
    def ensure_matches(cls, value: list) -> list:
        return value or []


class FidelityOutput(BaseModel):
    fidelity: list[FidelityRecord]

    @field_validator("fidelity")
    @classmethod
    def ensure_fidelity(cls, value: list) -> list:
        return value or []


class FinalJudgmentOutput(BaseModel):
    winner: Literal["context_baseline", "context_enhanced", "tie"]
    textual_loss_for: Literal["context_baseline", "context_enhanced", "tie"]
    reasoning: str
    textual_loss: str
