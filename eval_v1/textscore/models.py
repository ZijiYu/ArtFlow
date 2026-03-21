from __future__ import annotations

from typing import List, Literal

from pydantic import BaseModel, Field, field_validator


class SlotAnalysis(BaseModel):
    sentence_ids: List[int]
    sentences: List[str] = Field(default_factory=list)
    slot_name: str
    slot_term: str
    score: int = Field(ge=0)


class SentenceScore(BaseModel):
    sentence_id: int = Field(ge=0)
    sentence_text: str = ""
    score: int = Field(ge=0)
    loss: int = Field(ge=0)
    logic_score: int = Field(ge=0)
    slot_relevance_score: int = Field(ge=0)
    redundancy_score: int = Field(ge=0)
    worth_optimizing: bool
    reasoning: str
    improvement_suggestion: str
    matched_slots: List[str] = Field(default_factory=list)
    matched_terms: List[str] = Field(default_factory=list)


class ContextOptimizationSummary(BaseModel):
    need_optimization_count: int = Field(ge=0)
    worth_optimizing_sentence_ids: List[int] = Field(default_factory=list)
    total_loss: int = Field(ge=0)


class TextScoreResult(BaseModel):
    context_1_score: int
    context_2_score: int
    context_1_slots_score: int
    context_2_slots_score: int
    context_1_slots_analysis: List[SlotAnalysis]
    context_2_slots_analysis: List[SlotAnalysis]
    context_1_sentence_scores: List[SentenceScore]
    context_2_sentence_scores: List[SentenceScore]
    context_1_optimization_summary: ContextOptimizationSummary
    context_2_optimization_summary: ContextOptimizationSummary
    context_more_to_optimize: Literal["context_1", "context_2", "tie"]
    tokens: int = Field(ge=0)


class SlotsLLMOutput(BaseModel):
    slots_analysis: List[SlotAnalysis]

    @field_validator("slots_analysis")
    @classmethod
    def ensure_slots_list(cls, value: list) -> list:
        return value or []


class SentenceScoresLLMOutput(BaseModel):
    sentence_scores: List[SentenceScore]

    @field_validator("sentence_scores")
    @classmethod
    def ensure_sentence_scores_list(cls, value: list) -> list:
        return value or []
