from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PipelineConfig:
    slots_file: str = "/Users/ken/MM/Pipeline/final_version/preception_layer/artifacts/slots.jsonl"
    max_pixel: int = 1_003_520
    concurrent_workers: int = 4
    rag_query_max_blocks: int = 2
    enable_rag_verification: bool = True
    retrieval_gain: bool = False
    enable_web_search: bool = False
    web_search_url: str | None = None
    web_search_api_key: str | None = None
    web_search_api_key_file: str | None = None
    web_search_api_key_line: int = 1
    web_search_timeout: int = 20
    web_search_top_k: int = 5
    web_search_fetch_top_n: int = 2
    web_search_use_llm_rerank: bool = True
    web_search_skip_llm_if_confident: bool = True
    web_search_fallback_on_empty_rag: bool = True
    max_spawn_rounds: int = 1
    max_dialogue_rounds: int = 4
    max_threads_per_round: int = 4
    thread_attempt_limit: int = 2
    convergence_patience: int = 2
    resize_image: bool = True
    domain_temperature: float = 0.2
    validation_temperature: float = 0.1
    domain_model: str | None = None
    validation_model: str | None = None
    final_prompt_model: str | None = None
    final_answer_model: str | None = None
    output_dir: str = "artifacts"


@dataclass(slots=True)
class SlotSchema:
    slot_name: str
    slot_term: str
    description: str
    specific_questions: list[str]
    metadata: dict[str, Any]
    controlled_vocabulary: list[str] = field(default_factory=list)


@dataclass(slots=True)
class QuestionCoverage:
    question: str
    answered: bool
    support: str = ""


@dataclass(slots=True)
class EvidenceItem:
    observation: str
    evidence: str = ""
    position: str = ""


@dataclass(slots=True)
class DecodingItem:
    term: str
    explanation: str
    status: str = "IDENTIFIED"
    reason: str = ""


@dataclass(slots=True)
class MappingItem:
    insight: str
    basis: str = ""
    risk_note: str = ""


@dataclass(slots=True)
class DomainCoTRecord:
    slot_name: str
    slot_term: str
    analysis_round: int
    controlled_vocabulary: list[str]
    visual_anchoring: list[EvidenceItem]
    domain_decoding: list[DecodingItem]
    cultural_mapping: list[MappingItem]
    question_coverage: list[QuestionCoverage]
    unresolved_points: list[str]
    generated_questions: list[str]
    statuses: list[str]
    confidence: float
    retrieval_gain_focus: str = ""
    retrieval_gain_terms: list[str] = field(default_factory=list)
    retrieval_gain_queries: list[str] = field(default_factory=list)
    retrieval_gain_mode: str = "rag"
    retrieval_gain_web_queries: list[str] = field(default_factory=list)
    retrieval_gain_reason: str = ""
    retrieval_gain_has_value: bool = False
    raw_response: str = ""


@dataclass(slots=True)
class CrossValidationIssue:
    issue_type: str
    severity: str
    slot_names: list[str]
    detail: str
    evidence: list[str] = field(default_factory=list)
    rag_terms: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CrossValidationResult:
    issues: list[CrossValidationIssue]
    semantic_duplicates: list[str]
    missing_points: list[str]
    rag_terms: list[str]
    removed_questions: list[str]
    llm_review: str = ""
    round_table_blind_spots: list[str] = field(default_factory=list)
    round_table_follow_up_questions: list[dict[str, Any]] = field(default_factory=list)
    round_table_rag_needs: list[dict[str, Any]] = field(default_factory=list)
    slot_lifecycle_reviews: list[dict[str, Any]] = field(default_factory=list)
    follow_up_task_reviews: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class SpawnTask:
    slot_name: str
    reason: str
    prompt_focus: str
    rag_terms: list[str] = field(default_factory=list)
    retrieval_mode: str = "rag"
    web_queries: list[str] = field(default_factory=list)
    retrieval_reason: str = ""
    priority: int = 1
    dispatch_target: str = "cot"
    requested_slot_name: str = ""
    source_thread_id: str | None = None


@dataclass(slots=True)
class RoutingDecision:
    action: str
    rationale: list[str]
    paused_slots: list[str]
    spawned_tasks: list[SpawnTask]
    removed_questions: list[str]
    merged_duplicates: list[str]
    converged: bool = False
    convergence_reason: str = ""
    answered_slots: list[str] = field(default_factory=list)


@dataclass(slots=True)
class CoTThread:
    thread_id: str
    slot_name: str
    slot_term: str
    focus: str
    reason: str
    rag_terms: list[str] = field(default_factory=list)
    priority: int = 1
    status: str = "OPEN"
    attempts: int = 0
    max_attempts: int = 2
    last_round: int = 0
    latest_confidence: float = 0.0
    evidence_count: int = 0
    answered_questions: list[str] = field(default_factory=list)
    unresolved_points: list[str] = field(default_factory=list)
    latest_summary: str = ""
    latest_new_info_gain: int = 0
    stale_rounds: int = 0
    pause_reason: str = ""
    parent_thread_id: str | None = None
    latest_record: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)


@dataclass(slots=True)
class DialogueTurn:
    round_index: int
    active_thread_ids: list[str]
    executed_thread_ids: list[str]
    spawned_thread_ids: list[str]
    answered_thread_ids: list[str]
    blocked_thread_ids: list[str]
    paused_thread_ids: list[str]
    merged_thread_ids: list[str]
    routing_action: str
    notes: list[str]
    new_information_count: int
    convergence_snapshot: dict[str, Any]


@dataclass(slots=True)
class DialogueState:
    conversation_history: list[str] = field(default_factory=list)
    turns: list[DialogueTurn] = field(default_factory=list)
    threads: list[CoTThread] = field(default_factory=list)
    resolved_questions: list[str] = field(default_factory=list)
    unresolved_questions: list[str] = field(default_factory=list)
    removed_questions: list[str] = field(default_factory=list)
    merged_duplicates: list[str] = field(default_factory=list)
    no_new_info_rounds: int = 0
    converged: bool = False
    convergence_reason: str = ""
    final_round_index: int = 0


@dataclass(slots=True)
class PreparedImage:
    path: str
    original_size: tuple[int, int] | None = None
    prepared_size: tuple[int, int] | None = None
    original_pixels: int | None = None
    prepared_pixels: int | None = None
    was_resized: bool = False
    note: str = ""


@dataclass(slots=True)
class PipelineResult:
    image_path: str
    prepared_image: PreparedImage
    slot_schemas: list[SlotSchema]
    domain_outputs: list[DomainCoTRecord]
    cross_validation: CrossValidationResult
    routing: RoutingDecision
    dialogue_state: DialogueState
    cot_threads: list[CoTThread]
    round_memory: dict[str, Any]
    final_appreciation_prompt: str
    api_logs: list[dict[str, Any]]
    execution_log: list[dict[str, Any]]
