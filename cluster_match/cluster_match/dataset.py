from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


SOURCE_DIR_NAMES = {
    "baseline": "pilot30_baseline",
    "enhanced": "enhanced",
}


@dataclass(slots=True)
class SampleRecord:
    sample_id: str
    question: str


@dataclass(slots=True)
class ExtractionJob:
    sample_id: str
    question: str
    source: str
    source_dir_name: str
    input_path: Path


def clean_question(raw: str) -> str:
    question = (raw or "").strip()
    if question.endswith("/think"):
        question = question[: -len("/think")].rstrip()
    return question


def load_dataset(path: Path) -> list[SampleRecord]:
    rows: list[SampleRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            data = json.loads(text)
            sample_id = str(data.get("sample_id", "")).strip()
            if not sample_id:
                continue
            rows.append(
                SampleRecord(
                    sample_id=sample_id,
                    question=clean_question(str(data.get("question", ""))),
                )
            )
    return rows


def locate_text(base_dir: Path, sample_id: str) -> Path | None:
    candidates = [
        base_dir / f"{sample_id}.md",
        base_dir / f"{sample_id}.txt",
        base_dir / sample_id / "answer.md",
        base_dir / sample_id / "answer.txt",
        base_dir / sample_id / "final_appreciation_prompt.md",
        base_dir / "v_1" / f"{sample_id}.md",
        base_dir / "v_1" / f"{sample_id}.txt",
        base_dir / "v_1" / sample_id / "answer.md",
        base_dir / "v_1" / sample_id / "answer.txt",
        base_dir / "v_1" / sample_id / "final_appreciation_prompt.md",
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


def build_jobs(
    *,
    dataset_path: Path,
    baseline_dir: Path,
    enhanced_dir: Path,
    sources: list[str],
    sample_ids: set[str] | None = None,
    limit: int | None = None,
) -> list[ExtractionJob]:
    dataset = load_dataset(dataset_path)
    jobs: list[ExtractionJob] = []
    base_dirs = {
        "baseline": baseline_dir,
        "enhanced": enhanced_dir,
    }

    for record in dataset:
        if sample_ids and record.sample_id not in sample_ids:
            continue
        for source in sources:
            source_dir = base_dirs[source]
            input_path = locate_text(source_dir, record.sample_id)
            if input_path is None:
                continue
            jobs.append(
                ExtractionJob(
                    sample_id=record.sample_id,
                    question=record.question,
                    source=source,
                    source_dir_name=SOURCE_DIR_NAMES[source],
                    input_path=input_path,
                )
            )
        if limit is not None and len(jobs) >= limit:
            return jobs[:limit]
    return jobs
