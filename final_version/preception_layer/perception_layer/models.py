from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path


@dataclass(slots=True)
class TermCandidate:
    term: str
    description: str
    category_guess: str
    visual_evidence: list[str] = field(default_factory=list)
    text_evidence: list[str] = field(default_factory=list)


@dataclass(slots=True)
class RagDocument:
    source_id: str
    content: str
    score: float | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class GroundingQueryRecord:
    query_text: str
    duration_ms: float = 0.0
    initial_top_k_count: int = 0
    matched_count: int = 0


@dataclass(slots=True)
class GroundedTerm:
    candidate: TermCandidate
    documents: list[RagDocument] = field(default_factory=list)
    alignment_scores: list[float] = field(default_factory=list)
    search_queries: list[str] = field(default_factory=list)
    query_records: list[GroundingQueryRecord] = field(default_factory=list)


@dataclass(slots=True)
class SlotMetadata:
    confidence: float
    source_id: str
    extra: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SlotRecord:
    slot_name: str
    slot_term: str
    description: str
    specific_questions: list[str]
    metadata: SlotMetadata

    def to_dict(self) -> dict[str, object]:
        metadata = {
            "confidence": self.metadata.confidence,
            "source_id": self.metadata.source_id,
        }
        metadata.update(self.metadata.extra)
        return {
            "slot_name": self.slot_name,
            "slot_term": self.slot_term,
            "description": self.description,
            "specific_questions": self.specific_questions,
            "metadata": metadata,
        }


@dataclass(slots=True)
class OntologyLink:
    child: str
    parent: str
    relation: str
    rationale: str

    def to_markdown(self) -> str:
        return f"- `{self.child}` {self.relation} `{self.parent}`: {self.rationale}"


@dataclass(slots=True)
class PipelineResult:
    slots: list[SlotRecord]
    grounded_terms: list[GroundedTerm]
    ontology_links: list[OntologyLink]
    output_path: Path
    context_path: Path
    rag_search_record_path: Path
    llm_chat_record_path: Path

    def write_jsonl(self) -> None:
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        with self.output_path.open("w", encoding="utf-8") as handle:
            for slot in self.slots:
                handle.write(json.dumps(slot.to_dict(), ensure_ascii=False) + "\n")

    def to_dict(self) -> dict[str, object]:
        return {
            "slots": [slot.to_dict() for slot in self.slots],
            "grounded_terms": [
                {
                    "candidate": asdict(item.candidate),
                    "documents": [asdict(doc) for doc in item.documents],
                    "alignment_scores": item.alignment_scores,
                    "search_queries": item.search_queries,
                    "query_records": [asdict(record) for record in item.query_records],
                }
                for item in self.grounded_terms
            ],
            "ontology_links": [asdict(link) for link in self.ontology_links],
            "output_path": str(self.output_path),
            "context_path": str(self.context_path),
            "rag_search_record_path": str(self.rag_search_record_path),
            "llm_chat_record_path": str(self.llm_chat_record_path),
        }
