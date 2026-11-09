from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    period: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    period: str
    text: str
    position: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {"score": round(self.score, 6), **self.chunk.to_dict()}


@dataclass(frozen=True)
class Answer:
    question: str
    answer: str
    citations: tuple[str, ...]
    evidence: tuple[SearchResult, ...]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "answer": self.answer,
            "citations": list(self.citations),
            "confidence": round(self.confidence, 4),
            "evidence": [item.to_dict() for item in self.evidence],
        }

